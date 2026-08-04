"""Per-token health monitoring for fine-tuning runs.

Drops into a Hugging Face ``Trainer`` as a ``TrainerCallback``. It watches a set
of token ids (newly added special tokens, plus important existing specials you
do NOT want clobbered) and logs, per eval step:

  * ``norm/in/<label>``    L2 norm of the input-embedding row
  * ``norm/out/<label>``   L2 norm of the LM-head (unembedding) row
  * ``drift/in/<label>``   distance of the input row from its value at init
  * ``drift/out/<label>``  distance of the LM-head row from its value at init
  * ``grad/in/<label>``    grad L2 on the input row, captured on the last
                           optimizer step before the eval (see on_pre_optimizer_step)
  * ``grad/out/<label>``   grad L2 on the LM-head row

The intent (from the token-training discussion):
  - New tokens: norms/grads should be large early, then settle. Zero grad on a
    new token == it never reached the loss (usually a label-masking bug).
  - Existing specials (EOS, turn delimiters): norms should stay ~flat. Drift here
    is the early signature of catastrophic forgetting -> lower their LR or freeze.

Tied vs untied embeddings are handled transparently: if the model ties input and
output embeddings, the same tensor is read for both and the ``out`` series simply
mirror ``in``.

Emission-probability probes (does the model actually *emit* the token when it
should) are a separate, data-dependent concern -- see ``emit_prob_hook`` for a
hook point, but they are intentionally not baked in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


@dataclass(frozen=True)
class TokenSpec:
    """A token to watch.

    Attributes:
        token_id: vocab index to monitor.
        label: short name used in the logged metric keys.
        is_new: True for tokens you added and are training in; False for existing
            specials you are watching for regression (only affects nothing in the
            math -- it is carried through so dashboards can split the two groups).
    """

    token_id: int
    label: str
    is_new: bool = False

    @classmethod
    def from_tokenizer(
        cls, tokenizer, token: str, *, is_new: bool = False
    ) -> "TokenSpec":
        """Resolve a token *string* to a spec, asserting it is atomic.

        Guards the classic footgun: a special token that silently re-splits into
        multiple subword pieces (bad ``add_special_tokens`` / template setup).
        """
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"{token!r} does not encode to a single token id (got {ids}); "
                "it is not registered as an atomic special token."
            )
        return cls(token_id=ids[0], label=token.strip("<|>") or token, is_new=is_new)


class TokenMonitorCallback(TrainerCallback):
    """Logs embedding-row norms, drift-from-init, and gradient norms per token.

    Args:
        tokens: the tokens to watch.
        log_grads: capture per-row gradient norms. Requires that the embedding /
            LM-head rows are actually trainable (e.g. with LoRA you must add the
            embedding + lm_head modules to ``modules_to_save`` or mark their
            ``requires_grad``; LoRA adapters alone do not touch these rows).
        emit_prob_hook: optional callback ``(model) -> dict[str, float]`` run at
            each eval to add behavioral probes (emission probability, false-emit
            rate). Kept external because it needs your probe dataset.
    """

    def __init__(
        self,
        tokens: Sequence[TokenSpec],
        *,
        log_grads: bool = True,
        emit_prob_hook: Optional[Callable[[torch.nn.Module], Dict[str, float]]] = None,
    ) -> None:
        if not tokens:
            raise ValueError("TokenMonitorCallback needs at least one TokenSpec.")
        self.tokens: List[TokenSpec] = list(tokens)
        self.log_grads = log_grads
        self.emit_prob_hook = emit_prob_hook

        self._ids = torch.tensor([t.token_id for t in tokens], dtype=torch.long)
        # Init snapshots, filled lazily on first access to the real weights.
        self._init_in: Optional[torch.Tensor] = None
        self._init_out: Optional[torch.Tensor] = None
        # Grad norms captured on the pre-optimizer-step hook, consumed at eval.
        self._pending_grad: Dict[str, float] = {}

    # --- weight access -----------------------------------------------------

    @staticmethod
    def _in_weight(model: torch.nn.Module) -> torch.Tensor:
        return model.get_input_embeddings().weight

    @staticmethod
    def _out_weight(model: torch.nn.Module) -> torch.Tensor:
        out = model.get_output_embeddings()
        # Tied embeddings (or no separate head): fall back to the input matrix.
        if out is None:
            return model.get_input_embeddings().weight
        return out.weight

    def _rows(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.detach().index_select(0, self._ids.to(weight.device)).float()

    def _snapshot_init(self, model: torch.nn.Module) -> None:
        if self._init_in is None:
            self._init_in = self._rows(self._in_weight(model)).cpu()
            self._init_out = self._rows(self._out_weight(model)).cpu()

    # --- gradient capture --------------------------------------------------

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """Grab per-row grad norms while grads still exist (pre-step, pre-zero)."""
        if not (self.log_grads and model is not None):
            return
        for name, weight in (
            ("in", self._in_weight(model)),
            ("out", self._out_weight(model)),
        ):
            grad = weight.grad
            if grad is None:
                continue
            rows = grad.index_select(0, self._ids.to(grad.device)).float()
            norms = rows.norm(dim=1)
            for spec, n in zip(self.tokens, norms.tolist()):
                self._pending_grad[f"grad/{name}/{spec.label}"] = n

    # --- eval-time logging -------------------------------------------------

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        if model is None:
            return
        self._snapshot_init(model)

        in_rows = self._rows(self._in_weight(model)).cpu()
        out_rows = self._rows(self._out_weight(model)).cpu()

        metrics: Dict[str, float] = {}
        for i, spec in enumerate(self.tokens):
            metrics[f"norm/in/{spec.label}"] = in_rows[i].norm().item()
            metrics[f"norm/out/{spec.label}"] = out_rows[i].norm().item()
            metrics[f"drift/in/{spec.label}"] = (
                (in_rows[i] - self._init_in[i]).norm().item()
            )
            metrics[f"drift/out/{spec.label}"] = (
                (out_rows[i] - self._init_out[i]).norm().item()
            )

        metrics.update(self._pending_grad)
        self._pending_grad.clear()

        if self.emit_prob_hook is not None:
            metrics.update(self.emit_prob_hook(model))

        # Route through the Trainer's own logger so it lands in W&B / TB / console.
        trainer = kwargs.get("trainer")
        if trainer is not None:
            trainer.log(metrics)
        else:
            # Fallback: attach to the reported eval metrics.
            reported = kwargs.get("metrics")
            if isinstance(reported, dict):
                reported.update(metrics)
