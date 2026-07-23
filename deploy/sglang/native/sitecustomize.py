"""Narrow local-config adapter for SGLang GGUF startup experiments.

Gemma 4's mixed sliding/full attention config can't come from a live GGUF
parse inside the server process: SGLang's model-config resolution is reached
through ``sglang.srt.configs.model_config``, which does
``from ...hf_transformers.config import get_config`` at import time. That
copies a reference to the original function into its own namespace, so a
later ``sglang.srt.utils.hf_transformers.config.get_config = wrapper``
reassignment (the previous design here) never reaches that caller -- the
live-derive path silently keeps building every attention layer with the
un-swapped default head_dim. See ``resolve_gemma4_gguf_config.py`` for the
full account and the offline resolver this replaced it with.

Instead, ``SGLANG_GGUF_CONFIG_PATH`` (produced once by
``resolve_gemma4_gguf_config.py``) is required, and is spliced in by
patching ``HfModelConfigParser.parse`` directly on the class. Unlike
``get_config``, that method is looked up fresh on every call (SGLang's
parser registry constructs a new instance from the registry dict each time),
so patching the class attribute is immune to the same import-order bug.
"""

from __future__ import annotations

import os

from gemma4_gguf_compat import (
    install_gguf_model_type_alias,
    prepare_sglang_gemma4_construction,
)


_config_path = os.environ.get("SGLANG_GGUF_CONFIG_PATH")
if _config_path and not os.path.isfile(_config_path):
    raise RuntimeError(f"SGLANG_GGUF_CONFIG_PATH is not a file: {_config_path}")

# Newer Transformers already owns a small number of model-type names that
# SGLang 0.5.14 registers at import time. Keep the upstream registration when
# it exists; Gemma 4 does not use either duplicate.
from transformers import AutoConfig

_original_register = AutoConfig.register

def _register_with_existing_ok(*args: object, **kwargs: object) -> None:
    kwargs.setdefault("exist_ok", True)
    _original_register(*args, **kwargs)

AutoConfig.register = _register_with_existing_ok
install_gguf_model_type_alias()

# Only patch the GGUF-config redirect when a resolved config is actually
# available: this same devenv/python is also used to *produce* that file
# (resolve_gemma4_gguf_config.py) and for other scripts that never touch
# SGLang's GGUF path at all. serve.sh is the actual launch-time gate that
# requires SGLANG_GGUF_CONFIG_PATH before a real server start.
if _config_path:
    _config_dir = os.path.dirname(_config_path)

    from sglang.srt.utils.hf_transformers.config import HfModelConfigParser

    _original_parse = HfModelConfigParser.parse

    def _parse_with_gemma4_gguf_compat(self, model, trust_remote_code, revision=None, **kwargs):
        if "gguf_file" in kwargs:
            # A GGUF weight path: SGLang's get_config() already rewrote
            # `model` to the GGUF's parent directory and stashed the GGUF
            # path in `gguf_file`. Ignore both and load the pre-resolved
            # static config instead of parsing the GGUF live.
            return AutoConfig.from_pretrained(
                _config_dir, trust_remote_code=trust_remote_code, revision=revision
            )
        return _original_parse(
            self, model, trust_remote_code, revision=revision, **kwargs
        )

    HfModelConfigParser.parse = _parse_with_gemma4_gguf_compat

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
