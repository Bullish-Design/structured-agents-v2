"""Narrow local-config adapter for SGLang GGUF startup experiments.

SGLang 0.5.14 asks Transformers to infer a config from a GGUF file. Transformers
does not yet parse the Gemma 4 GGUF metadata, although it can parse the pinned
upstream Gemma configuration. When explicitly opted in, keep the GGUF file as
SGLang's weight path but resolve its model config from that existing local file.
"""

from __future__ import annotations

import os

from gemma4_gguf_compat import (
    install_gguf_model_type_alias,
    install_transformers_gguf_patch,
    normalize_sglang_gemma4_text_config,
    prepare_sglang_gemma4_construction,
)


_config_path = os.environ.get("SGLANG_GGUF_CONFIG_PATH")

# Newer Transformers already owns a small number of model-type names that
# SGLang 0.5.14 registers at import time. Keep the upstream registration when
# it exists; Gemma 4 does not use either duplicate.
from transformers import AutoConfig

_original_register = AutoConfig.register

def _register_with_existing_ok(*args: object, **kwargs: object) -> None:
    kwargs.setdefault("exist_ok", True)
    _original_register(*args, **kwargs)

AutoConfig.register = _register_with_existing_ok
install_transformers_gguf_patch()
install_gguf_model_type_alias()

from sglang.srt.utils.hf_transformers import config as _config_module

_original_get_config = _config_module.get_config

if _config_path and not os.path.isfile(_config_path):
    raise RuntimeError(f"SGLANG_GGUF_CONFIG_PATH is not a file: {_config_path}")

def _get_config_with_gemma4_gguf_compat(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    model_override_args: dict | None = None,
    model_config_parser: str = "auto",
    **kwargs: object,
):
    if _config_path and os.path.isfile(model) and model.lower().endswith(".gguf"):
        config = _original_get_config(
                _config_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                model_override_args=model_override_args,
                model_config_parser="hf",
                **kwargs,
            )
    else:
        config = _original_get_config(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=model_override_args,
            model_config_parser=model_config_parser,
            **kwargs,
        )
    return normalize_sglang_gemma4_text_config(config)

_config_module.get_config = _get_config_with_gemma4_gguf_compat

# SGLang supplies its own Triton/RadixAttention implementation for Gemma 4.
# The HF base class still validates ``config._attn_implementation`` while the
# SGLang module is constructed, so override only that construction-time choice.
from sglang.srt.model_loader import loader as _loader

_original_initialize_model = _loader._initialize_model


def _initialize_model_with_gemma4_eager_construction(*args: object, **kwargs: object):
    model_config = args[0] if args else kwargs["model_config"]
    prepare_sglang_gemma4_construction(model_config.hf_config)
    return _original_initialize_model(*args, **kwargs)


_loader._initialize_model = _initialize_model_with_gemma4_eager_construction
