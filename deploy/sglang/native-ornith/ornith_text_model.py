"""A complete text-only top-level model for the Ornith-1.0-9B GGUF spike.

The class SGLang 0.5.14 registers/rewrites GGUF checkpoints to for
``model_type="qwen3_5"`` is ``sglang.srt.models.qwen3_5.Qwen3_5ForCausalLM``
(via ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES``). But despite its name, that class
is **not** a servable causal-LM: it is the flat decoder stack
(``embed_tokens`` / ``layers`` / ``norm``), its ``forward`` returns raw hidden
states, and it has no ``lm_head`` and no ``LogitsProcessor``. The only complete
top-level Qwen3.5 classes in this SGLang version are the multimodal
``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5MoeForConditionalGeneration``
wrappers, which build a full vision tower this text-only spike deliberately
excludes.

Two independent symptoms both trace to using the flat decoder as the top
model:

1. **Every GGUF weight fails to bind.** The GGUF->HF tensor-name map is built
   from a Transformers ``Qwen3_5ForCausalLM`` dummy, whose parameters are
   nested under ``model.`` (``model.layers.0.self_attn.q_proj.weight``, ...)
   and which owns an ``lm_head``. The flat SGLang decoder's ``params_dict`` has
   **no** ``model.`` prefix and **no** ``lm_head`` -- so all 629 incoming
   tensors miss, and the model loads at its random init (~1.96 GB of a ~6 GB
   checkpoint).
2. **No logits.** Even had the weights bound, the flat decoder cannot produce
   token logits to sample from.

This module provides ``OrnithTextForCausalLM``: a minimal, text-only complete
wrapper that owns the flat decoder as ``self.model``, adds a ``ParallelLMHead``
and a ``LogitsProcessor``, and reuses the *same* weight-loading semantics as
the real ``Qwen3_5ForConditionalGeneration`` (which keep the ``model.`` prefix
and handle ``lm_head`` / tied embeddings) minus the vision-only branches that
never fire for a text checkpoint. Registering this under the name SGLang
rewrites GGUF ``qwen3_5`` checkpoints to (``Qwen3_5ForCausalLM``) makes the
GGUF-derived names line up with a servable model, without pulling in the
multimodal stack.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

import ornith_gdn_transforms as gdn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM as Qwen3_5TextDecoder
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean env toggle (``0``/``false``/``no``/``off`` -> False).

    Used to gate the three GDN weight transforms individually so each can be
    validated in isolation against the llama.cpp oracle (guide S6.3). All
    default to on, so an unconfigured launch applies the full fix.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class OrnithTextForCausalLM(nn.Module):
    """Text-only servable wrapper around the flat Qwen3.5 decoder stack."""

    # Reuse the decoder's fusion metadata so the GGUF loader and any LoRA
    # plumbing see the same packed-module layout.
    packed_modules_mapping = Qwen3_5TextDecoder.packed_modules_mapping
    supported_lora_modules = Qwen3_5TextDecoder.supported_lora_modules

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # The flat decoder builds itself under the "model." prefix, matching
        # the GGUF-derived (Transformers-style) tensor names.
        self.model = Qwen3_5TextDecoder(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # Unlike qwen2.py, the Qwen3.5 decoder builds ``embed_tokens`` with no
        # quant config, so its param is a plain ``.weight`` and the GGUF's
        # quantized ``token_embd`` (yielded as ``qweight`` / ``qweight_type``)
        # has nowhere to bind. Rebuild the embedding with the quant config so
        # it exposes GGUF params, matching every other quantized tensor.
        if quant_config is not None and self.pp_group.is_first_rank:
            self.model.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("model.embed_tokens", prefix),
            )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # Diagnostic: log per-decoder-layer output stats on the first forward,
        # to localize where the hidden state diverges (GDN vs full-attn/rope).
        self._dbg_layers = _env_flag("ORNITH_DEBUG_LAYERS", default=False)
        self._dbg_done = False
        if self._dbg_layers:
            block_types = list(getattr(config, "layers_block_type", []) or [])
            for idx, layer in enumerate(getattr(self.model, "layers", [])):
                if not hasattr(layer, "register_forward_hook"):
                    continue
                bt = block_types[idx] if idx < len(block_types) else "?"

                def _hook(_m, _inp, out, _i=idx, _bt=bt):
                    if self._dbg_done:
                        return
                    h = out[0] if isinstance(out, tuple) else out
                    if not isinstance(h, torch.Tensor):
                        return
                    hf = h.float()
                    logger.warning(
                        "ORNITH LAYER %02d [%s] mean=%.4f std=%.4f absmax=%.4f nan=%s",
                        _i,
                        _bt,
                        hf.mean().item(),
                        hf.std().item(),
                        hf.abs().max().item(),
                        bool(torch.isnan(hf).any().item()),
                    )

                layer.register_forward_hook(_hook)

    # SGLang consults these on the top-level model class.
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        # Dense text checkpoint: no MoE expert-location tracking. Mirrors
        # ornith_gguf_compat.install_dense_expert_location_skip.
        return None

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not self.pp_group.is_last_rank:
            return hidden_states

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if self._dbg_layers and not self._dbg_done:
            self._dbg_done = True

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Bind GGUF-derived (``model.``-prefixed) weights to this wrapper.

        This is the ``Qwen3_5ForConditionalGeneration.load_weights`` logic with
        the vision-only branches removed: params_dict here is exactly
        ``model.*`` + ``lm_head`` + no ``visual.*``, so keeping the ``model.``
        prefix and the tied-embedding handling is all that is needed. The GDN
        and attention fused-projection stacked mappings are identical to the
        decoder's own.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        # GDN GGUF->HF numeric transforms (guide S2/S3). Each is individually
        # gated so it can be bisected against the llama.cpp oracle; all default
        # on so a normal launch applies the full fix.
        fix_norm = _env_flag("ORNITH_FIX_NORM")
        fix_alog = _env_flag("ORNITH_FIX_ALOG")
        fix_perm = _env_flag("ORNITH_FIX_PERM")

        _gdn_touched: dict[str, int] = {}

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                # This text-only spike's GGUF ships no vision tensors; guard
                # anyway so a stray tensor never lands on a random-init param.
                continue
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            # GGUF stores the GDN depthwise conv as a 2-D [conv_dim, kernel]
            # tensor, but SGLang materializes conv1d.weight as the 3-D Conv1d
            # shape [conv_dim, 1, kernel] (it unsqueezes dim 1 at build time).
            # Restore the singleton channel dim so the mamba conv weight loader
            # can place it.
            if name.endswith("linear_attn.conv1d.weight") and loaded_weight.dim() == 2:
                loaded_weight = loaded_weight.unsqueeze(1)

            # Apply the GGUF->HF numeric transforms before any weight_loader
            # runs, whether the tensor lands via the stacked mapping
            # (in_proj_qkv/z -> qweight rows; in_proj_a/b -> plain rows) or the
            # default else-branch (norms, A_log, dt_bias, conv1d, out_proj).
            # Quantized tensors arrive as ``.qweight`` (packed [out, bytes],
            # row-permute safe); ``.qweight_type`` scalars are never touched.
            _before = loaded_weight
            if name.endswith(".qweight"):
                loaded_weight = gdn.transform_quant_rows(
                    name, loaded_weight, fix_perm=fix_perm
                )
            elif not name.endswith(".qweight_type"):
                loaded_weight = gdn.transform_plain(
                    name,
                    loaded_weight,
                    fix_norm=fix_norm,
                    fix_alog=fix_alog,
                    fix_perm=fix_perm,
                )
            if loaded_weight is not _before:
                _tail = name.rsplit(".", 2)[-2] + "/" + name.rsplit(".", 1)[-1]
                _gdn_touched[_tail] = _gdn_touched.get(_tail, 0) + 1

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        if self._dbg_layers:
            logger.warning(
                "ORNITH GDN transforms applied (norm=%s alog=%s perm=%s): %s",
                fix_norm,
                fix_alog,
                fix_perm,
                dict(sorted(_gdn_touched.items())),
            )
        return loaded_params
