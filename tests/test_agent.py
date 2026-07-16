"""`StructuredAgent.run` end-to-end against the in-process mock OpenAI server."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
import pytest

from structured_agents_v2 import AgentProfile, Backend, ConstrainedOutput, DecoderSpec


class Route(ConstrainedOutput):
    route: Literal["file_edit", "git_ops", "answer"]


class ActionOut(ConstrainedOutput):
    kind: Literal["no_action", "edit"]


def _backend(transport: httpx.ASGITransport, *, capture: bool = True) -> Backend:
    return Backend(base_url="http://mock/v1", default_model="base", capture=capture).attach_transport(transport)


def test_json_schema_run_returns_validated_model(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "git_ops"}'
    backend = _backend(transport)
    profile = AgentProfile(name="router", instructions="route", output_type_ref="test_agent:Route")
    agent = backend.build(profile)

    result = asyncio.run(agent.run("pick one"))

    assert isinstance(result.output, Route)
    assert result.output.route == "git_ops"
    # capture is on → request body present, native response_format on the wire
    assert result.request_body is not None
    assert result.request_body["response_format"]["type"] == "json_schema"
    assert "tools" not in result.request_body
    assert result.usage is not None
    assert result.raw is not None


def test_regex_run_returns_str(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: "git status ."
    backend = _backend(transport)
    profile = AgentProfile(
        name="git", instructions="cmd", decoder=DecoderSpec(mode="regex", regex=r"git (status|diff) [\w./-]*")
    )
    agent = backend.build(profile)

    result = asyncio.run(agent.run("show status"))

    assert result.output == "git status ."
    assert result.request_body is not None
    assert "structured_outputs" in result.request_body
    assert result.request_body["structured_outputs"] == {"regex": r"git (status|diff) [\w./-]*"}


def test_user_extra_body_merged_not_clobbered(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # A4 regression: a profile's own extra_body must survive alongside the decoder's
    # structured_outputs, not be overwritten by it.
    mock_openai.responder = lambda _req: "git status ."
    backend = _backend(transport)
    profile = AgentProfile(
        name="git",
        instructions="cmd",
        decoder=DecoderSpec(mode="regex", regex=r"git (status|diff) [\w./-]*"),
        model_settings={"extra_body": {"custom": 1}},
    )
    agent = backend.build(profile)

    result = asyncio.run(agent.run("show status"))

    assert result.request_body is not None
    assert result.request_body["custom"] == 1  # user key preserved
    assert result.request_body["structured_outputs"] == {"regex": r"git (status|diff) [\w./-]*"}  # decoder key present


def test_regex_guard_raises_on_nonmatching_output(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # B4: if the backend doesn't enforce the constraint (here: mock returns junk), the
    # client-side guard must catch it rather than leak unconstrained text.
    from structured_agents_v2 import ConstraintViolationError

    mock_openai.responder = lambda _req: "rm -rf /"  # does NOT match the git regex
    backend = _backend(transport)
    profile = AgentProfile(
        name="git", instructions="cmd", decoder=DecoderSpec(mode="regex", regex=r"git (status|diff) [\w./-]*")
    )
    agent = backend.build(profile)
    with pytest.raises(ConstraintViolationError, match="does not match declared regex"):
        asyncio.run(agent.run("show status"))


def test_regex_guard_passes_matching_output(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: "git status ."
    backend = _backend(transport)
    profile = AgentProfile(
        name="git", instructions="cmd", decoder=DecoderSpec(mode="regex", regex=r"git (status|diff) [\w./-]*")
    )
    agent = backend.build(profile)
    result = asyncio.run(agent.run("show status"))
    assert result.output == "git status ."


def test_choice_guard_raises_on_out_of_set_output(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    from structured_agents_v2 import ConstraintViolationError

    mock_openai.responder = lambda _req: "delete"  # not in the choice set
    backend = _backend(transport)
    profile = AgentProfile(
        name="pick", instructions="pick", decoder=DecoderSpec(mode="choice", choices=["keep", "skip"])
    )
    agent = backend.build(profile)
    with pytest.raises(ConstraintViolationError, match="not in declared choices"):
        asyncio.run(agent.run("choose"))


def test_choice_guard_passes_in_set_output(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: "keep"
    backend = _backend(transport)
    profile = AgentProfile(
        name="pick", instructions="pick", decoder=DecoderSpec(mode="choice", choices=["keep", "skip"])
    )
    agent = backend.build(profile)
    result = asyncio.run(agent.run("choose"))
    assert result.output == "keep"


def test_adapter_sets_wire_model_field(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "answer"}'
    backend = _backend(transport)
    profile = AgentProfile(
        name="router", adapter="router-lora", instructions="route", output_type_ref="test_agent:Route"
    )
    agent = backend.build(profile)

    result = asyncio.run(agent.run("hi"))

    assert result.request_body is not None
    assert result.request_body["model"] == "router-lora"


def test_capture_off_yields_no_request_body(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "answer"}'
    backend = _backend(transport, capture=False)
    profile = AgentProfile(name="router", instructions="route", output_type_ref="test_agent:Route")
    agent = backend.build(profile)

    result = asyncio.run(agent.run("hi"))

    assert result.request_body is None
    assert isinstance(result.output, Route)


def test_run_sync_returns_validated_model(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "file_edit"}'
    backend = _backend(transport)
    profile = AgentProfile(name="router", instructions="route", output_type_ref="test_agent:Route")
    agent = backend.build(profile)

    result = agent.run_sync("pick one")

    assert isinstance(result.output, Route)
    assert result.output.route == "file_edit"
    assert result.request_body is not None


def test_valid_output_constructs_agent_result_without_capture(
    mock_openai: Any, transport: httpx.ASGITransport
) -> None:
    """ISSUE.md regression: a valid json_schema completion must yield AgentResult
    without raising, with capture OFF (usage access must not depend on capture)."""
    mock_openai.responder = lambda _req: '{"kind": "no_action"}'
    backend = _backend(transport, capture=False)
    agent = backend.build(AgentProfile(name="p", instructions="i", output_type_ref="test_agent:ActionOut"))
    result = asyncio.run(agent.run("go"))
    assert result.output.kind == "no_action"
    assert result.usage is not None  # RunUsage value, not a bound method
    assert not callable(result.usage)  # guards against re-introducing .usage()
    assert result.request_body is None  # capture off -> no body


def test_valid_output_constructs_agent_result_with_capture(
    mock_openai: Any, transport: httpx.ASGITransport
) -> None:
    """Same ISSUE.md path with capture ON: result still constructs, body present."""
    mock_openai.responder = lambda _req: '{"kind": "no_action"}'
    backend = _backend(transport, capture=True)
    agent = backend.build(AgentProfile(name="p", instructions="i", output_type_ref="test_agent:ActionOut"))
    result = asyncio.run(agent.run("go"))
    assert result.output.kind == "no_action"
    assert result.usage is not None
    assert not callable(result.usage)
    assert result.request_body is not None


def test_agent_escape_hatch_exposed(transport: httpx.ASGITransport) -> None:
    from pydantic_ai import Agent

    backend = _backend(transport)
    profile = AgentProfile(name="router", instructions="route", output_type_ref="test_agent:Route")
    agent = backend.build(profile)

    assert isinstance(agent.agent, Agent)
