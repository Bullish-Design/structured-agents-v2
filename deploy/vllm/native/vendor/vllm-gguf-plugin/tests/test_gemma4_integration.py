# SPDX-License-Identifier: Apache-2.0
"""Integration coverage for Structured Agents' pinned Gemma 4 artifact.

Set ``GEMMA4_GGUF_PATH`` to the downloaded
``gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`` file to run this test.  It intentionally
does not fetch a multi-gigabyte model in ordinary unit-test runs.
"""

import os
from pathlib import Path

import gguf
import pytest

from vllm_gguf_plugin.weights_adapter.gemma4 import build_gemma4_mapper


@pytest.mark.integration
def test_pinned_gemma4_gguf_tensor_table_is_supported():
    model_path = os.environ.get("GEMMA4_GGUF_PATH")
    if model_path is None:
        pytest.skip("set GEMMA4_GGUF_PATH to run against the pinned GGUF")

    reader = gguf.GGUFReader(Path(model_path))
    architecture = reader.get_field("general.architecture")
    assert architecture is not None
    assert bytes(architecture.parts[-1]).decode() == "gemma4"

    mapped_names = {
        name for name, _ in build_gemma4_mapper().apply(
            [(tensor.name, None) for tensor in reader.tensors]
        )
    }
    assert "model.language_model.embed_tokens.weight" in mapped_names
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in mapped_names
    assert "model.language_model.layers.47.layer_scalar" in mapped_names
