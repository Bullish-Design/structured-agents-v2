# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma 4 GGUF adapter.

Gemma 4's GGUF architecture name has not yet been registered by gguf-py.
The general adapter therefore cannot build a tensor table for
``gemma4_unified``.  Keep the small, stable naming delta here rather than
copying an in-tree vLLM loader whose API has since changed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


def build_gemma4_mapper() -> WeightsMapper:
    """Map llama.cpp's Gemma 4 tensor names to HF checkpoint names.

    The resulting names intentionally retain the HF wrapper prefixes.  The
    registered vLLM Gemma4 / Gemma4Unified model then applies its own
    ``hf_to_vllm_mapper`` and packed-QKV loader logic.
    """
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.language_model.embed_tokens.",
            "blk.": "model.language_model.layers.",
            "output_norm.": "model.language_model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr={
            "attn_output.": "self_attn.o_proj.",
            "attn_q.": "self_attn.q_proj.",
            "attn_k.": "self_attn.k_proj.",
            "attn_v.": "self_attn.v_proj.",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_norm.": "input_layernorm.",
            "post_attention_norm.": "post_attention_layernorm.",
            "ffn_gate.": "mlp.gate_proj.",
            "ffn_up.": "mlp.up_proj.",
            "ffn_down.": "mlp.down_proj.",
            "ffn_norm.": "pre_feedforward_layernorm.",
            "post_ffw_norm.": "post_feedforward_layernorm.",
            "layer_output_scale.weight": "layer_scalar",
        },
    )


def build_gemma4_moe_tensor_map(layer_idx: int) -> dict[str, str]:
    """Return the Gemma 4 MoE tensor exceptions used by GGUF exporters.

    The 12B deployment model is dense, but retain these mappings for Gemma 4
    MoE variants.  In particular, the handled names include ``.weight``:
    omitting it was the reviewed bug in vLLM PR #41589 and makes the general
    unmapped-parameter check reject otherwise valid expert tensors.
    """
    prefix = f"model.layers.{layer_idx}"
    return {
        f"blk.{layer_idx}.ffn_gate_up_exps.weight": (
            f"{prefix}.moe.gate_up_proj.weight"
        ),
        f"blk.{layer_idx}.ffn_down_exps.weight": f"{prefix}.moe.down_proj.weight",
        f"{prefix}.experts.gate_up_proj.weight": (
            f"{prefix}.experts.gate_up_proj.weight"
        ),
        f"{prefix}.experts.down_proj.weight": f"{prefix}.experts.down_proj.weight",
    }


class Gemma4GGUFAdapter(BaseGGUFWeightsAdapter):
    """Load Gemma 4 GGUF tensors through vLLM's native Gemma 4 model path."""

    mapper: WeightsMapper | None = None
    load_spec: GGUFLoadSpec | None = None
    # llama.cpp exports this RoPE-frequency helper for Gemma 4, but vLLM's
    # Gemma4Unified implementation constructs RoPE from configuration and has
    # no corresponding module or parameter.
    ignored_gguf_tensors = frozenset({"rope_freqs.weight"})

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return config.model_type in ("gemma4", "gemma4_unified")

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # This preserves the separately pinned local HF config supplied by
        # serve.sh while taking vocabulary/tied-embedding metadata from GGUF.
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        self.mapper = build_gemma4_mapper()
        unquantized = get_gguf_unquantized_params([model_path])
        self.load_spec = GGUFLoadSpec(
            weights_source=[model_path],
            unquantized_modules=list(set(self.mapper.apply_list(unquantized))),
        )
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        assert self.mapper is not None
        assert self.load_spec is not None
        weights = (
            (name, weight)
            for name, weight in gguf_quant_weights_iterator_multi(
                self.load_spec.weights_source
            )
            if name not in self.ignored_gguf_tensors
        )
        yield from self.mapper.apply(weights)
