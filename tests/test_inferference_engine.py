"""The engine that asks the serving plane what it can do instead of guessing."""

from __future__ import annotations

import json

import httpx
import pytest
from inferference.capabilities import AdapterInfo, Capabilities, ConstraintKind, ModelInfo
from pydantic import BaseModel

from structured_agents import AgentSpec, Backend, Choice, Grammar, Schema
from structured_agents.engine import select
from structured_agents.engine.inferference import InferferenceEngine
from structured_agents.errors import BackendCapabilityError


class Person(BaseModel):
    name: str


def _capabilities(
    *,
    models: tuple[str, ...] = ("base", "ner-json"),
    adapters: tuple[str, ...] = (),
    constraints: frozenset[ConstraintKind] = frozenset(ConstraintKind),
) -> Capabilities:
    return Capabilities(
        server="llama-server-router",
        models=tuple(ModelInfo(name=name) for name in models),
        adapters=tuple(AdapterInfo(name=name) for name in adapters),
        constraints=constraints,
    )


# ---- capability derivation ---------------------------------------------------


def test_constraint_support_comes_from_the_server_not_a_constant() -> None:
    full = InferferenceEngine(_capabilities())
    assert full.supports == frozenset({"schema", "choice", "grammar"})

    narrow = InferferenceEngine(_capabilities(constraints=frozenset({ConstraintKind.SCHEMA})))
    assert narrow.supports == frozenset({"schema"})


def test_wire_rendering_matches_the_llama_cpp_dialect() -> None:
    engine = InferferenceEngine(_capabilities())
    assert engine.render(Grammar('root ::= "a"')).extra_body == {"grammar": 'root ::= "a"'}
    assert engine.render(Choice("a", "b")).extra_body == {"grammar": 'root ::= "a" | "b"'}


# ---- naming a model on the wire ---------------------------------------------


def test_resident_adapter_resolves_onto_the_model_field() -> None:
    engine = InferferenceEngine(_capabilities())
    assert engine.resolve_model("ner-json", "base") == "ner-json"
    assert engine.resolve_model(None, "base") == "base"


def test_loaded_but_unaddressable_adapter_is_refused_with_the_reason() -> None:
    engine = InferferenceEngine(_capabilities(models=("base",), adapters=("acrouter",)))
    with pytest.raises(BackendCapabilityError, match="not addressable"):
        engine.resolve_model("acrouter", "base")


def test_a_default_model_the_server_does_not_hold_is_refused() -> None:
    engine = InferferenceEngine(_capabilities(models=("base",)))
    with pytest.raises(BackendCapabilityError, match="unknown model"):
        engine.resolve_model(None, "not-resident")


# ---- the regression this whole layer exists for ------------------------------


def test_an_adapter_bearing_spec_builds_when_the_server_publishes_the_adapter() -> None:
    """Before capability negotiation this raised unconditionally, always."""
    backend = Backend(
        engine=InferferenceEngine(_capabilities()),
        http_client=httpx.AsyncClient(),
    )
    agent = backend.build(AgentSpec("neg-adapter-ok", Schema(Person), "x", adapter="ner-json"))
    assert agent.spec.adapter == "ner-json"


def test_llama_cpp_engine_still_refuses_an_adapter_it_cannot_prove() -> None:
    with pytest.raises(BackendCapabilityError, match="LoRA"):
        select("llama_cpp").resolve_model("my-lora", "base")


# ---- negotiation -------------------------------------------------------------


async def test_negotiate_builds_a_backend_from_a_probed_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload: object = {"data": [{"id": "base"}, {"id": "ner-json"}]} if request.url.path == "/v1/models" else {}
        status = 200 if request.url.path == "/v1/models" else 404
        return httpx.Response(status, content=json.dumps(payload), headers={"content-type": "application/json"})

    backend = await Backend.negotiate(
        "http://rig.invalid:8081/v1",
        default_model="base",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert isinstance(backend.engine, InferferenceEngine)
    assert backend.engine.resolve_model("ner-json", "base") == "ner-json"
    await backend.aclose()
