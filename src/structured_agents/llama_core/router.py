"""Multi-LoRA agent-router surface — the project-17 flagship as a library API.

One base model loaded once; one pinned-adapter ``llama_context`` per adapter (plus
an optional base context). A request carries an adapter name; the router routes it
to that adapter's context, batches concurrent requests within a context (own-batch
multi-sequence decode), and multiplexes across contexts. Optional grammar-constrained
decoding reuses :mod:`structured_agents.llama_core.grammar`.

Design boundaries (repo standing rule): Pydantic validates at the edges
(:class:`RouterConfig`, :class:`RouteRequest`, :class:`RouteResult`); the decode hot
path passes plain token ids and numpy logit views. This mirrors
:mod:`structured_agents.llama_core.decode`, extended from one sequence to a routed,
batched wave.

The research provenance (GPU-validated, exact-match vs isolated baselines) lives in
``benchmarks/project17/``; this module is the teaching-quality library form.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .decode import FINISH_LENGTH, FINISH_STOP
from .grammar import JsonSchemaGrammar, apply_packed_bitmask_inplace
from .models import EngineConfig, GenerationResult
from .seq_routing import NO_ADAPTER, SeqRoutingBinding, SeqRoutingUnavailable

BASE: None = None  # sentinel adapter name meaning "no adapter / raw base model"


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterSpec(_BoundaryModel):
    """One LoRA adapter in the router's pool."""

    name: str
    gguf_path: str
    scale: float = Field(default=1.0, gt=0)


Backend = Literal["context_pool", "seq_routed", "auto"]


class RouterConfig(_BoundaryModel):
    """Everything needed to stand up a router over one base model.

    ``backend`` selects how a mixed-adapter workload is served:

    * ``context_pool`` — one pinned-adapter ``llama_context`` per adapter; requests
      are grouped by adapter and each group multiplexed on its context (the
      always-available shipping path).
    * ``seq_routed`` — the P2 fork's true mixed-batch path: one context, the ordered
      adapter pool registered once, each sequence assigned its adapter, a single
      ``llama_decode`` carrying the mix. Requires a fork lib; raises at construction
      if the capability is absent (explicit opt-in).
    * ``auto`` (default) — ``seq_routed`` when the loaded lib reports the routing
      capability, else ``context_pool``. Fail-closed: a stock lib silently uses the
      context-pool path; missing capability is never an inference failure.
    """

    engine: EngineConfig
    adapters: tuple[AdapterSpec, ...]
    n_seq_max: int = Field(default=8, gt=0)
    include_base: bool = True
    backend: Backend = "auto"


class RouteRequest(_BoundaryModel):
    """A generation request tagged with the adapter it should route to."""

    prompt: str
    adapter: str | None = None  # None -> base context
    max_tokens: int = Field(default=64, gt=0)
    request_id: str | None = None


class RouteResult(GenerationResult):
    """A completed route: the generation plus which adapter served it."""

    adapter: str | None = None
    decision: dict[str, Any] | None = None  # parsed JSON when grammar-constrained


class MultiLoRARouter:
    """Shared base model, one pinned-adapter context per adapter, batched routing."""

    def __init__(self, config: RouterConfig) -> None:
        from llama_cpp import Llama, llama_cpp

        self.config = config
        self.C = llama_cpp
        eng = config.engine
        self.n_seq_max = config.n_seq_max
        self.n_batch = eng.n_batch
        self.llm = Llama(
            model_path=eng.model_path, n_ctx=eng.n_ctx, n_batch=eng.n_batch,
            n_gpu_layers=eng.n_gpu_layers, seed=eng.seed, verbose=False,
        )
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self._eos = int(self.llm.token_eos())

        self._adapter_ptr: dict[str, Any] = {}
        for spec in config.adapters:
            ptr = self.C.llama_adapter_lora_init(self.model, spec.gguf_path.encode("utf-8"))
            if not ptr:
                raise RuntimeError(f"failed to load adapter {spec.name!r} from {spec.gguf_path}")
            self._adapter_ptr[spec.name] = ptr
        self._scale = {s.name: s.scale for s in config.adapters}

        # ---- backend selection (fail-closed) ----
        # ``auto`` uses the fork path only when the loaded lib exports it; ``seq_routed``
        # is an explicit opt-in that raises here if the fork is absent; ``context_pool``
        # never touches the fork surface.
        self.backend = self._resolve_backend(config.backend)

        self._seq_binding: SeqRoutingBinding | None = None
        self._seq_ctx: Any = None
        # Ordered adapter pool: pool index == routing id. Fixed at construction so a
        # sequence's index is stable across waves. BASE maps to the -1 sentinel.
        self._seq_pool: tuple[str, ...] = tuple(s.name for s in config.adapters)
        self._seq_index: dict[str, int] = {name: i for i, name in enumerate(self._seq_pool)}

        self._ctx: dict[str | None, Any] = {}
        if self.backend == "seq_routed":
            self._setup_seq_routed()
        else:
            self._setup_context_pool()
        self._grammar: JsonSchemaGrammar | None = None

    # ---- backend setup ----

    def _resolve_backend(self, requested: Backend) -> str:
        """Pick the concrete backend, honoring the fail-closed rule for ``auto``."""
        lib = getattr(self.C, "_lib", None)
        from .seq_routing import library_supports_seq_routing

        capable = lib is not None and library_supports_seq_routing(lib)
        if requested == "seq_routed":
            if not capable:
                raise SeqRoutingUnavailable(
                    "backend='seq_routed' requires the P2 fork lib "
                    "(llama_set_seq_adapters/llama_set_seq_adapter absent)"
                )
            return "seq_routed"
        if requested == "auto":
            return "seq_routed" if capable else "context_pool"
        return "context_pool"

    def _setup_context_pool(self) -> None:
        names: list[str | None] = ([BASE] if self.config.include_base else []) + list(self._seq_pool)
        for name in names:
            ctx = self._make_ctx()
            if name is not BASE:
                self._pin_adapter(ctx, self._adapter_ptr[name], self._scale[name])
            self._ctx[name] = ctx

    def _setup_seq_routed(self) -> None:
        """One context; register the ordered adapter pool once via the fork binding.

        The fork's ``llama_set_seq_adapters`` takes no per-adapter scale — the masked
        path applies each adapter's built-in alpha scaling. So an ``AdapterSpec.scale``
        override is honored only on the ``context_pool`` backend; on ``seq_routed`` a
        non-default scale is ignored. Warn rather than silently diverge.
        """
        off_default = [s.name for s in self.config.adapters if s.scale != 1.0]
        if off_default:
            import warnings

            warnings.warn(
                f"seq_routed backend ignores AdapterSpec.scale overrides for {off_default} "
                "(the fork applies built-in adapter alpha); use backend='context_pool' to "
                "apply custom scales.",
                stacklevel=2,
            )
        self._seq_binding = SeqRoutingBinding(self.C)
        self._seq_ctx = self._make_ctx()
        self._seq_binding.set_seq_adapters(
            self._seq_ctx, [self._adapter_ptr[name] for name in self._seq_pool]
        )

    # ---- setup ----

    def _make_ctx(self) -> Any:
        p = self.C.llama_context_default_params()
        p.n_ctx = self.config.engine.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = self.n_seq_max
        ctx = self.C.llama_new_context_with_model(self.model, p)
        if not ctx:
            raise RuntimeError("llama_new_context_with_model returned NULL (out of VRAM?)")
        return ctx

    def _pin_adapter(self, ctx: Any, ptr: Any, scale: float) -> None:
        arr = (self.C.llama_adapter_lora_p_ctypes * 1)(ptr)
        scales = (ctypes.c_float * 1)(scale)
        if self.C.llama_set_adapters_lora(ctx, arr, 1, scales) != 0:
            raise RuntimeError("llama_set_adapters_lora failed")

    def enable_grammar(self, tokenizer: Any, schema: dict[str, Any]) -> None:
        """Compile a JSON-schema grammar; constrained runs mask to it per sequence."""
        self._grammar = JsonSchemaGrammar.from_huggingface(tokenizer, schema, vocab_size=self.n_vocab)

    # ---- decode primitives (plain tokens / numpy on the hot path) ----

    def _seq_rm(self, ctx: Any, seq_id: int) -> None:
        self.C.llama_memory_seq_rm(self.C.llama_get_memory(ctx), seq_id, -1, -1)

    def _prefill(self, ctx: Any, tokens: list[int], seq_id: int) -> None:
        n = len(tokens)
        for off in range(0, n, self.n_batch):
            chunk = tokens[off:off + self.n_batch]
            m = len(chunk)
            batch = self.C.llama_batch_init(m, 0, 1)
            batch.n_tokens = m
            for i, t in enumerate(chunk):
                batch.token[i] = t
                batch.pos[i] = off + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 1 if off + m == n and i == m - 1 else 0
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"prefill llama_decode rc={rc}")

    def _sample_row(self, ctx: Any, row: int, matcher: Any | None) -> int:
        ptr = self.C.llama_get_logits_ith(ctx, row)
        logits = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        if matcher is not None:
            import xgrammar as xgr

            bitmask = np.zeros(xgr.get_bitmask_shape(1, self.n_vocab), dtype=np.int32)
            if matcher.fill_next_token_bitmask(bitmask):
                apply_packed_bitmask_inplace(logits, bitmask[0], self.n_vocab)
        tok = int(np.argmax(logits))
        if matcher is not None and matcher.accept_token(tok) is False:
            raise RuntimeError(f"grammar rejected token {tok} (fail-closed)")
        return tok

    def _advance(self, out: list[int], tok: int, matcher: Any | None) -> str | None:
        """Record a sampled token; return a finish_reason if the sequence is done.

        Grammar completion or a sampled EOS is a clean ``stop``; otherwise the
        sequence continues (``None``) until ``max_tokens`` yields a ``length`` cut.
        """
        if matcher is not None:
            out.append(tok)
            return FINISH_STOP if matcher.is_terminated() else None
        if tok == self._eos:
            return FINISH_STOP  # EOS is the boundary, not content — do not emit it
        out.append(tok)
        return None

    def _generate_wave(self, ctx: Any, prompts: list[list[int]], max_tokens: int,
                       matchers: list[Any] | None) -> list[tuple[list[int], str]]:
        """Greedy batched decode of up to n_seq_max sequences; seq_id == row.

        Returns ``(tokens, finish_reason)`` per sequence so truncation is
        observable. Finished slots ride along (frozen) to keep the seq_id==row
        invariant until all finish or ``max_tokens`` is hit.
        """
        s = len(prompts)
        for i in range(s):
            self._seq_rm(ctx, i)
        out: list[list[int]] = [[] for _ in range(s)]
        cur: list[int] = [0] * s
        pos: list[int] = [0] * s
        finish: list[str | None] = [None] * s
        for i in range(s):
            self._prefill(ctx, prompts[i], i)
            cur[i] = self._sample_row(ctx, -1, matchers[i] if matchers else None)
            pos[i] = len(prompts[i])
            finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
        for _ in range(max_tokens - 1):
            if all(f is not None for f in finish):
                break
            batch = self.C.llama_batch_init(s, 0, 1)
            batch.n_tokens = s
            for i in range(s):
                batch.token[i] = cur[i]
                batch.pos[i] = pos[i]
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = i
                batch.logits[i] = 1
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"batched llama_decode rc={rc}")
            for i in range(s):
                pos[i] += 1
                if finish[i] is not None:
                    continue
                cur[i] = self._sample_row(ctx, i, matchers[i] if matchers else None)
                finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
        return [(out[i], finish[i] or FINISH_LENGTH) for i in range(s)]

    # ---- public API ----

    def _tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def run(self, requests: list[RouteRequest], *, constrained: bool = False) -> list[RouteResult]:
        """Route + batch a mixed workload; optionally constrain to the grammar.

        Same public surface on both backends. ``context_pool`` groups by adapter and
        multiplexes across contexts; ``seq_routed`` carries a mix of adapters in each
        single-``llama_decode`` wave.
        """
        if constrained and self._grammar is None:
            raise RuntimeError("constrained=True requires enable_grammar(...) first")
        for r in requests:
            if r.adapter is not BASE and r.adapter not in self._adapter_ptr:
                raise KeyError(f"unknown adapter {r.adapter!r}")
        if self.backend == "seq_routed":
            return self._run_seq_routed(requests, constrained=constrained)
        return self._run_context_pool(requests, constrained=constrained)

    def _finalize(self, r: RouteRequest, prompt: list[int], gen: tuple[list[int], str],
                  constrained: bool) -> RouteResult:
        toks, finish_reason = gen
        toks = toks[:r.max_tokens]
        text = self.llm.detokenize(toks).decode("utf-8", errors="replace")
        # Only parse when the grammar actually completed; a length cut leaves the
        # JSON truncated, and ``validated`` must reflect that.
        decision = None
        if constrained and finish_reason == FINISH_STOP:
            try:
                decision = json.loads(text)
            except ValueError:
                decision = None
        return RouteResult(
            text=text, token_ids=tuple(toks), prompt_token_count=len(prompt),
            completion_token_count=len(toks),
            finish_reason=finish_reason, request_id=r.request_id,
            adapter=r.adapter, decision=decision,
            validated=(decision is not None) if constrained else None,
        )

    def _run_context_pool(self, requests: list[RouteRequest], *, constrained: bool) -> list[RouteResult]:
        by_adapter: dict[str | None, list[int]] = {}
        for idx, r in enumerate(requests):
            by_adapter.setdefault(r.adapter, []).append(idx)
        results: dict[int, RouteResult] = {}
        for adapter, idxs in by_adapter.items():
            ctx = self._ctx[adapter]
            for w in range(0, len(idxs), self.n_seq_max):
                wave = idxs[w:w + self.n_seq_max]
                prompts = [self._tokenize(requests[j].prompt) for j in wave]
                matchers = [self._grammar.new_matcher() for _ in wave] if constrained else None
                gen = self._generate_wave(ctx, prompts, max(requests[j].max_tokens for j in wave), matchers)
                for slot, j in enumerate(wave):
                    results[j] = self._finalize(requests[j], prompts[slot], gen[slot], constrained)
        return [results[i] for i in range(len(requests))]

    def _run_seq_routed(self, requests: list[RouteRequest], *, constrained: bool) -> list[RouteResult]:
        """True mixed-batch: each wave is one ``llama_decode`` over a mix of adapters.

        Requests keep submission order; each wave of up to ``n_seq_max`` sequences
        assigns row ``i`` its own adapter (``-1`` for BASE) before decoding, so a
        single decode carries the mix — no per-adapter context, no ``n_seq_max`` split.
        """
        assert self._seq_binding is not None and self._seq_ctx is not None
        ctx = self._seq_ctx
        results: list[RouteResult] = []
        for w in range(0, len(requests), self.n_seq_max):
            wave = requests[w:w + self.n_seq_max]
            for i, r in enumerate(wave):
                idx = NO_ADAPTER if r.adapter is BASE else self._seq_index[r.adapter]
                self._seq_binding.set_seq_adapter(ctx, i, idx)
            prompts = [self._tokenize(r.prompt) for r in wave]
            matchers = [self._grammar.new_matcher() for _ in wave] if constrained else None
            gen = self._generate_wave(ctx, prompts, max(r.max_tokens for r in wave), matchers)
            for i, r in enumerate(wave):
                results.append(self._finalize(r, prompts[i], gen[i], constrained))
        return results

    def close(self) -> None:
        for ctx in self._ctx.values():
            self.C.llama_free(ctx)
        self._ctx.clear()
        if self._seq_ctx is not None:
            self.C.llama_free(self._seq_ctx)
            self._seq_ctx = None
        self.llm.close()

    def __enter__(self) -> MultiLoRARouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
