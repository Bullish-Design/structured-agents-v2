"""Boundary-model contracts for the multi-LoRA router surface.

No native library or GPU: this validates the Pydantic edges only, matching the
project rule that validation stops at the boundary and the hot path stays plain.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from structured_agents.llama_core.models import EngineConfig, GenerationResult
from structured_agents.llama_core.router import (
    BASE,
    AdapterSpec,
    RouteRequest,
    RouteResult,
    RouterConfig,
)


def _engine() -> EngineConfig:
    return EngineConfig(model_path="m.gguf", n_ctx=4096, n_batch=256, n_gpu_layers=-1, backend="cuda")


def test_router_config_validates_and_forbids_extra() -> None:
    config = RouterConfig(
        engine=_engine(),
        adapters=(AdapterSpec(name="a", gguf_path="a.gguf"),
                  AdapterSpec(name="b", gguf_path="b.gguf", scale=0.75)),
        n_seq_max=8,
    )
    assert [s.name for s in config.adapters] == ["a", "b"]
    assert config.adapters[1].scale == 0.75
    assert config.include_base is True
    with pytest.raises(ValidationError):
        RouterConfig(engine=_engine(), adapters=(), n_seq_max=8, bogus=1)  # type: ignore[call-arg]


def test_adapter_scale_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AdapterSpec(name="a", gguf_path="a.gguf", scale=0.0)


def test_route_request_defaults_and_base_sentinel() -> None:
    req = RouteRequest(prompt="route me", request_id="r0")
    assert req.adapter is BASE  # None -> base context
    assert req.max_tokens == 64
    with pytest.raises(ValidationError):
        RouteRequest(prompt="x", max_tokens=0)


def test_route_result_extends_generation_result() -> None:
    result = RouteResult(
        text='{"tool": "none"}', token_ids=(1, 2, 3), prompt_token_count=5,
        completion_token_count=3, finish_reason="stop", adapter="a",
        decision={"tool": "none"}, validated=True,
    )
    assert isinstance(result, GenerationResult)
    assert result.adapter == "a"
    assert result.decision == {"tool": "none"}
    assert result.finish_reason == "stop"
