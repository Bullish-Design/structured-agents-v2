# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GGUF CPU-to-final-device loading."""

import torch

from vllm_gguf_plugin.quantization.params import (
    GGUFUninitializedWeightParameter,
    _store_gguf_loaded_weight,
)


def test_unsharded_weight_materializes_final_parameter_from_cpu_tensor():
    """The loader accepts a CPU GGUF tensor without a GPU staging allocation."""
    parameter = GGUFUninitializedWeightParameter(requires_grad=False)
    loaded_weight = torch.arange(12, dtype=torch.uint8).reshape(3, 4)

    _store_gguf_loaded_weight(parameter, loaded_weight)

    assert parameter.device.type == "cpu"
    assert torch.equal(parameter.data, loaded_weight)
    assert parameter.data.data_ptr() != loaded_weight.data_ptr()
