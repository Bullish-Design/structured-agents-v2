# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.gemma4 import (
    Gemma4GGUFAdapter,
    build_gemma4_mapper,
    build_gemma4_moe_tensor_map,
)


def test_gemma4_unified_uses_dedicated_adapter():
    adapter = get_weights_adapter(SimpleNamespace(model_type="gemma4_unified"))
    assert isinstance(adapter, Gemma4GGUFAdapter)


def test_gemma4_key_tensor_names_map_to_hf_checkpoint_layout():
    mapper = build_gemma4_mapper()
    mapped = dict(
        mapper.apply(
            [
                ("token_embd.weight", None),
                ("blk.0.attn_q.weight", None),
                ("blk.0.layer_output_scale.weight", None),
                ("output_norm.weight", None),
            ]
        )
    )
    assert "model.language_model.embed_tokens.weight" in mapped
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in mapped
    assert "model.language_model.layers.0.layer_scalar" in mapped
    assert "model.language_model.norm.weight" in mapped


def test_gemma4_moe_handled_names_retain_weight_suffix():
    mappings = build_gemma4_moe_tensor_map(7)
    assert "model.layers.7.experts.gate_up_proj.weight" in mappings
    assert "model.layers.7.experts.down_proj.weight" in mappings


def test_gemma4_ignores_llamacpp_rope_frequency_helper():
    assert "rope_freqs.weight" in Gemma4GGUFAdapter.ignored_gguf_tensors
