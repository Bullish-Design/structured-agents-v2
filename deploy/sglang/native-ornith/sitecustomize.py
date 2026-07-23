"""Narrow local-config adapter for the SGLang Ornith-1.0-9B GGUF spike.

See ``ornith_gguf_compat.py`` for why this is needed and why the patch
targets the ``HfModelConfigParser`` class rather than a module function.
"""

from __future__ import annotations

import os

from ornith_gguf_compat import (
    install_causal_lm_registry_entry,
    install_dense_expert_location_skip,
    install_gguf_arch_name_translation,
    install_gguf_model_type_alias,
    install_static_config_redirect,
    install_text_only_mm_processor_skip,
)

_config_path = os.environ.get("SGLANG_GGUF_CONFIG_PATH")
if _config_path and not os.path.isfile(_config_path):
    raise RuntimeError(f"SGLANG_GGUF_CONFIG_PATH is not a file: {_config_path}")

install_gguf_model_type_alias()

if _config_path:
    install_static_config_redirect(os.path.dirname(_config_path))
    install_causal_lm_registry_entry()
    install_text_only_mm_processor_skip()
    install_dense_expert_location_skip()
    install_gguf_arch_name_translation()
