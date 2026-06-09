"""Capability gating: a backend refuses to build agents it can't serve."""

from __future__ import annotations

import pytest

from structured_agents_v2 import (
    AgentProfile,
    Backend,
    BackendCapabilityError,
    BackendCaps,
    ConstrainedOutput,
    DecoderSpec,
)


class Cmd(ConstrainedOutput):
    value: str


def _backend(**caps: bool) -> Backend:
    return Backend(base_url="http://mock/v1", default_model="base", caps=BackendCaps(**caps))


@pytest.mark.parametrize("mode", ["grammar", "regex", "choice"])
def test_bare_string_modes_gated_on_xgrammar(mode: str) -> None:
    backend = _backend(xgrammar=False)
    spec = DecoderSpec(mode=mode, grammar='root ::= "x"', regex="x.*", choices=["a", "b"])
    profile = AgentProfile(name="g", instructions="x", decoder=spec)
    with pytest.raises(BackendCapabilityError, match="XGrammar"):
        backend.build(profile)


def test_adapter_gated_on_lora() -> None:
    backend = _backend(lora=False)
    profile = AgentProfile(
        name="a",
        adapter="some-lora",
        instructions="x",
        decoder=DecoderSpec(mode="json_schema"),
        output_type_ref="test_backend:Cmd",
    )
    with pytest.raises(BackendCapabilityError, match="LoRA"):
        backend.build(profile)


def test_json_schema_is_never_gated() -> None:
    # xgrammar off, but json_schema rides standard response_format → must still build.
    backend = _backend(xgrammar=False, lora=False)
    profile = AgentProfile(name="j", instructions="x", output_type_ref="test_backend:Cmd")
    agent = backend.build(profile)
    assert agent.profile.name == "j"


def test_model_for_uses_adapter_then_default() -> None:
    backend = Backend(base_url="http://mock/v1", default_model="base")
    assert backend.model_for(None).model_name == "base"
    assert backend.model_for("file-edit-lora").model_name == "file-edit-lora"
