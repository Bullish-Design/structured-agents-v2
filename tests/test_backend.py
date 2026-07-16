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
        output_type_ref="test_backend:Cmd",  # ConstrainedOutput carries its own decoder_spec
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


def test_backend_shares_one_client_across_agents() -> None:
    # Phase 5 item 2: one httpx client per Backend, reused by every built agent (so N agents
    # share one connection pool against the single server).
    backend = Backend(base_url="http://mock/v1", default_model="base", capture=True)
    backend.build(AgentProfile(name="a", instructions="x", output_type_ref="test_backend:Cmd"))
    backend.build(AgentProfile(name="b", adapter="lora-b", instructions="x", output_type_ref="test_backend:Cmd"))
    assert backend._shared_client() is backend._shared_client()  # single shared instance


def test_backend_aclose_closes_client() -> None:
    import asyncio

    backend = Backend(base_url="http://mock/v1", default_model="base")
    backend.build(AgentProfile(name="a", instructions="x", output_type_ref="test_backend:Cmd"))
    client = backend._shared_client()
    assert not client.is_closed
    asyncio.run(backend.aclose())
    assert client.is_closed
