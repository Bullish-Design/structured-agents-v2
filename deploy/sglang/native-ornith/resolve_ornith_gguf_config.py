#!/usr/bin/env python3
"""One-time offline resolver: publish a static config.json for the SGLang
Ornith-1.0-9B GGUF spike.

Unlike Gemma 4, there is no live-GGUF-derive path here at all: Transformers'
GGUF metadata parser (``GGUF_CONFIG_MAPPING``) does not recognize the
``qwen35`` architecture string and raises before a config can be built. This
instead copies the real ``config.json`` already published in the non-GGUF
``unsloth/Ornith-1.0-9B`` repo, validated by round-tripping it through
``AutoConfig`` so class resolution (``Qwen3_5Config``, registered by SGLang
at import time) is confirmed working before the server ever starts.

Run once per target GGUF/tokenizer directory, then point
SGLANG_GGUF_CONFIG_PATH at the "config.json" this writes.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ornith_gguf_compat import install_gguf_model_type_alias


def resolve(source_config_dir: str, output_dir: str) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    install_gguf_model_type_alias()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source_config_dir)
    if config.model_type != "qwen3_5":
        raise SystemExit(f"expected model_type=qwen3_5, got {config.model_type!r}")

    # `sglang.srt.utils.hf_transformers.config.get_config` unconditionally
    # rewrites a GGUF checkpoint's `architectures` to Transformers'
    # MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type] entry -- "Qwen3_5ForCausalLM"
    # for either "qwen3_5" or "qwen3_5_text" -- regardless of what this
    # config declares, so top-level architectures barely matters here. What
    # does matter: Qwen3_5ForCausalLM.get_model_config_for_expert_location
    # reads `config.num_hidden_layers` directly off whatever config object is
    # handed to it. The full multimodal Qwen3_5Config only has that field
    # nested under `.text_config`; the flat Qwen3_5TextConfig
    # (`get_text_config()`) has it at the top level, matching what the dense
    # CausalLM class expects. Publish the text-only config; see
    # ornith_gguf_compat.install_text_only_mm_processor_skip for why this
    # spike's known-text-only architecture never reaches a multimodal
    # processor lookup despite the source repo being multimodal.
    text_config = config.get_text_config()
    if text_config.model_type != "qwen3_5_text":
        raise SystemExit(f"expected text model_type=qwen3_5_text, got {text_config.model_type!r}")
    if not hasattr(text_config, "num_hidden_layers"):
        raise SystemExit("resolved text config is missing num_hidden_layers")
    text_config.architectures = ["Qwen3_5ForCausalLM"]

    # sglang.srt.models.qwen3_5.Qwen3_5ForCausalLM.get_model_config_for_expert_location
    # is shared by both the dense and MoE subclasses and unconditionally reads
    # config.num_experts. This dense checkpoint has none; this code path has
    # apparently never been exercised for the dense class (it was never
    # reachable through the model registry before install_causal_lm_registry_entry).
    # 0 experts is the correct value for a dense model, not a placeholder.
    if not hasattr(text_config, "num_experts"):
        text_config.num_experts = 0

    # sglang.srt.models.qwen3_5's decoder-layer builder reads
    # config.layers_block_type (values "attention" / "linear_attention"),
    # not Transformers' own field name for the same per-layer list
    # (config.layer_types, values "full_attention" / "linear_attention").
    # Translate rather than rename in place, since other code may still read
    # layer_types under its original name/values.
    if not hasattr(text_config, "layer_types") or not text_config.layer_types:
        raise SystemExit("resolved text config is missing layer_types")
    text_config.layers_block_type = [
        "attention" if t == "full_attention" else t for t in text_config.layer_types
    ]

    # sglang.srt.models.qwen3_5's linear-attention module unconditionally
    # reads config.output_gate_type, but this installed Transformers version's
    # Qwen3_5TextConfig never defines or defaults it, and the source repo's
    # config.json has no such field either -- likely drift between the
    # bleeding-edge Qwen3.5 config revision SGLang's model code was written
    # against and this one. SGLang's own code treats None as "no special
    # output-gate activation" (see the `if self.output_gate_type is not None`
    # branch just below where it's read), so None is not a guess -- it's
    # this checkpoint's genuine value under that code's own semantics.
    if not hasattr(text_config, "output_gate_type"):
        text_config.output_gate_type = None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "config.json")
    text_config.to_json_file(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_config_dir",
        help="Directory containing the real (non-GGUF) Ornith-1.0-9B config.json",
    )
    parser.add_argument("output_dir", help="Directory to write the resolved config.json into")
    args = parser.parse_args()
    output_path = resolve(args.source_config_dir, args.output_dir)
    print(f"wrote resolved config to {output_path}")


if __name__ == "__main__":
    main()
