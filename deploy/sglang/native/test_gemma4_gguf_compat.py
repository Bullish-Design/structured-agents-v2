"""Regression test for the local Gemma 4 GGUF compatibility layer.

This reads metadata from the exact immutable GGUF but never asks for tensors or
constructs a CUDA model.
"""

from __future__ import annotations

import os

import gguf
from transformers import AutoConfig
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

from gemma4_gguf_compat import prepare_sglang_gemma4_construction


MODEL_PATH = os.environ["MODEL_PATH"]

parsed = load_gguf_checkpoint(MODEL_PATH, return_tensors=False)["config"]
assert parsed["model_type"] == "gemma4_text"
assert parsed["num_key_value_heads"] == 8
assert parsed["num_global_key_value_heads"] == 1
assert parsed["hidden_size_per_layer_input"] == 0
assert parsed["attention_k_eq_v"] is True
assert parsed["layer_types"] == ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 5 + ["full_attention"]

config = AutoConfig.from_pretrained(MODEL_PATH)
assert config.num_key_value_heads == 8
assert config.num_global_key_value_heads == 1
assert config.hidden_size_per_layer_input == 0
assert config.attention_k_eq_v is True
assert "gemma4_text" in gguf.MODEL_ARCH_NAMES.values()

# The compatibility layer must request eager only for Transformers'
# construction-time validation. SGLang's selected Triton backend remains a
# separate server setting and is not replaced here.
config._attn_implementation = "sdpa"
assert prepare_sglang_gemma4_construction(config)._attn_implementation == "eager"

class NonGemmaConfig:
    model_type = "llama"
    _attn_implementation = "sdpa"


non_gemma = NonGemmaConfig()
assert prepare_sglang_gemma4_construction(non_gemma)._attn_implementation == "sdpa"
