"""Wire shapes via the real build path (`Backend.build` → `StructuredAgent`).

The GPU-free analogue of the request-path spike: it confirms the production build path
(not an inline test builder) puts the intended bytes on the wire — `response_format:
json_schema` for json_schema mode, text-mode + `extra_body` for the bare-string modes,
and the adapter name as the wire `model` field.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx

from structured_agents_v2 import AgentProfile, Backend, ConstrainedOutput, DecoderSpec


class Route(ConstrainedOutput):
    route: Literal["file_edit", "git_ops", "answer"]


def _backend(transport: httpx.ASGITransport, *, default_model: str = "test-model") -> Backend:
    return Backend(base_url="http://mock/v1", default_model=default_model, capture=True).attach_transport(transport)


def test_json_schema_mode_emits_response_format(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "file_edit"}'
    profile = AgentProfile(name="router", instructions="x", output_type_ref="test_wire_shapes:Route")
    result = asyncio.run(_backend(transport).build(profile).run("pick one"))

    body = result.request_body
    assert body is not None
    assert body["response_format"]["type"] == "json_schema"
    assert "tools" not in body
    assert "structured_outputs" not in body
    # round-trip: validated into the model
    assert isinstance(result.output, Route)
    assert result.output.route == "file_edit"


def test_regex_mode_emits_text_plus_extra_body(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: "git status ."
    spec = DecoderSpec(mode="regex", regex=r"git (status|diff) [\w./-]*")
    profile = AgentProfile(name="git", instructions="x", decoder=spec)
    result = asyncio.run(_backend(transport).build(profile).run("show status"))

    body = result.request_body
    assert body is not None
    assert "response_format" not in body
    assert "tools" not in body
    assert body["structured_outputs"] == {"regex": r"git (status|diff) [\w./-]*"}
    # text-mode output comes back as a plain string
    assert result.output == "git status ."


def test_adapter_name_is_the_wire_model_field(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "answer"}'
    profile = AgentProfile(
        name="router", adapter="file-edit-lora", instructions="x", output_type_ref="test_wire_shapes:Route"
    )
    result = asyncio.run(_backend(transport).build(profile).run("hi"))

    assert result.request_body is not None
    assert result.request_body["model"] == "file-edit-lora"
