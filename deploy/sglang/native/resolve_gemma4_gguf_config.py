#!/usr/bin/env python3
"""One-time offline resolver: Gemma 4 GGUF -> a static SGLang-native config.json.

Writes a config with SGLang's own head_dim/swa_head_dim/layer_types
convention already baked in, so the live server loads a plain config file
at startup instead of deriving one from the GGUF on every launch.

The earlier design re-derived this config live, inside the server process,
via a monkeypatch on ``sglang.srt.utils.hf_transformers.config.get_config``.
That hook silently never ran: ``sglang.srt.configs.model_config`` does
``from ...config import get_config`` at import time, which copies a
reference to the *original* function into its own module namespace.
Reassigning the attribute on the ``config`` module afterward doesn't reach
a caller that already imported the name -- so every server start built every
attention layer (including full-attention ones) with the un-swapped default
head_dim. Resolving the config once, offline, and handing the server a
plain file removes that whole class of live monkeypatch fragility.

Run once per target GGUF, then point SGLANG_GGUF_CONFIG_PATH at the
"config.json" this writes.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemma4_gguf_compat import (
    install_transformers_gguf_patch,
    normalize_sglang_gemma4_text_config,
)


def resolve(gguf_path: str, output_dir: str) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    install_transformers_gguf_patch()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gguf_path, gguf_file=gguf_path)
    config = normalize_sglang_gemma4_text_config(config)

    if config.model_type != "gemma4_text":
        raise SystemExit(f"expected model_type=gemma4_text, got {config.model_type!r}")
    if config.head_dim == getattr(config, "swa_head_dim", None):
        raise SystemExit(
            "head_dim == swa_head_dim after normalize; the sliding/full "
            "attention split did not apply -- refusing to write a config "
            "that would reproduce the original weight-shape crash"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "config.json")
    config.to_json_file(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_path", help="Path to the target Gemma 4 GGUF file")
    parser.add_argument(
        "output_dir", help="Directory to write the resolved config.json into"
    )
    args = parser.parse_args()
    output_path = resolve(args.gguf_path, args.output_dir)
    print(f"wrote resolved config to {output_path}")


if __name__ == "__main__":
    main()
