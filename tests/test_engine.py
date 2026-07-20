from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai.output import NativeOutput

from structured_agents import AgentSpec, Backend, Choice, Grammar, Regex, Schema
from structured_agents.engine import select
from structured_agents.errors import BackendCapabilityError, ConfigError


class Person(BaseModel):
    name: str


def test_vllm_bytes_are_unchanged() -> None:
    """Regression guard: the vLLM engine must reproduce the pre-refactor wire exactly."""
    vllm = select("vllm")
    assert vllm.render(Regex(r"\d{4}")).extra_body == {"structured_outputs": {"regex": r"\d{4}"}}
    assert vllm.render(Choice("keep", "skip")).extra_body == {"structured_outputs": {"choice": ["keep", "skip"]}}
    assert vllm.render(Grammar('root ::= "a" | "b"')).extra_body == {
        "structured_outputs": {"grammar": 'root ::= "a" | "b"'}
    }
    schema_wire = vllm.render(Schema(Person))
    assert isinstance(schema_wire.output_type, NativeOutput)
    assert schema_wire.extra_body == {}


def test_sglang_dialect() -> None:
    sglang = select("sglang")
    assert sglang.render(Regex(r"\d{4}")).extra_body == {"regex": r"\d{4}"}
    assert sglang.render(Grammar('root ::= "a"')).extra_body == {"ebnf": 'root ::= "a"'}
    assert sglang.render(Choice("a.b", "c")).extra_body == {"regex": r"(a\.b|c)"}
    assert isinstance(sglang.render(Schema(Person)).output_type, NativeOutput)


def test_llama_cpp_narrow_caps() -> None:
    llama = select("llama_cpp")
    assert "regex" not in llama.supports and "lora" not in llama.supports
    assert llama.render(Grammar('root ::= "a"')).extra_body == {"grammar": 'root ::= "a"'}
    assert llama.render(Choice("a", "b")).extra_body == {"grammar": 'root ::= "a" | "b"'}
    with pytest.raises(BackendCapabilityError):
        llama.render(Regex(r"\d"))


def test_backend_gate_rejects_unsupported_constraint() -> None:
    backend = Backend(engine="llama_cpp", http_client=httpx.AsyncClient())
    with pytest.raises(BackendCapabilityError, match="regex"):
        backend.build(AgentSpec("r", Regex(r"\d"), "x"))


def test_backend_gate_rejects_lora_when_unsupported() -> None:
    backend = Backend(engine="llama_cpp", http_client=httpx.AsyncClient())
    with pytest.raises(BackendCapabilityError, match="LoRA"):
        backend.build(AgentSpec("s", Schema(Person), "x", adapter="my-lora"))


def test_unknown_engine_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown engine"):
        select("does-not-exist")
