from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from structured_agents import AgentSpec, Backend, Choice, Grammar, Regex, Schema
from structured_agents.engine import select
from structured_agents.errors import BackendCapabilityError, ConfigError


class Person(BaseModel):
    name: str


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
