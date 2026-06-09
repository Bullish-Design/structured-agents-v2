"""Integration: assert the wire shape each decode mode produces, against the mock server.

This is the GPU-free analogue of the request-path spike: it confirms that applying a
`DecoderSpec` actually yields the intended bytes on the wire via PydanticAI -
`response_format: json_schema` for json_schema mode, and text-mode + `extra_body` for
the bare-string modes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from structured_agents_v2 import ConstrainedOutput, DecoderSpec, RequestCapture


class Route(ConstrainedOutput):
    route: Literal["file_edit", "git_ops", "answer"]


def _build_agent(
    capture: RequestCapture,
    transport: httpx.ASGITransport,
    spec: DecoderSpec,
    output_type: Any,
) -> Agent[None, Any]:
    app = spec.apply(output_type)
    client = capture.client(transport=transport)
    provider = OpenAIProvider(base_url="http://mock/v1", api_key="x", http_client=client)
    model = OpenAIChatModel("test-model", provider=provider)
    settings = OpenAIChatModelSettings(max_tokens=64, extra_body=app.extra_body)
    return Agent(model, output_type=app.output_type, model_settings=settings, instructions="x")


def test_json_schema_mode_emits_response_format(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "file_edit"}'
    cap = RequestCapture()
    agent = _build_agent(cap, transport, Route.decoder_spec(), Route)

    result = asyncio.run(agent.run("pick one"))

    # wire shape: native response_format, no tools, no extra_body
    assert cap.last.response_format is not None
    assert cap.last.response_format["type"] == "json_schema"
    assert cap.last.tools is None
    assert cap.last.extra_body_keys == []
    # round-trip: validated into the model
    assert isinstance(result.output, Route)
    assert result.output.route == "file_edit"


def test_regex_mode_emits_text_plus_extra_body(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: "git status ."
    cap = RequestCapture()
    spec = DecoderSpec(mode="regex", regex=r"git (status|diff) [\w./-]*")
    agent = _build_agent(cap, transport, spec, None)

    result = asyncio.run(agent.run("show status"))

    # wire shape: no response_format, structured_outputs in extra_body
    assert cap.last.response_format is None
    assert cap.last.tools is None
    assert "structured_outputs" in cap.last.extra_body_keys
    assert cap.last.body["structured_outputs"] == {"regex": r"git (status|diff) [\w./-]*"}
    # text-mode output comes back as a plain string
    assert result.output == "git status ."


def test_adapter_name_is_the_wire_model_field(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = lambda _req: '{"route": "answer"}'
    cap = RequestCapture()
    app = Route.decoder_spec().apply(Route)
    client = cap.client(transport=transport)
    provider = OpenAIProvider(base_url="http://mock/v1", api_key="x", http_client=client)
    # the adapter/LoRA selector is the model name:
    model = OpenAIChatModel("file-edit-lora", provider=provider)
    agent = Agent(model, output_type=app.output_type, instructions="x")

    asyncio.run(agent.run("hi"))

    assert cap.last.model == "file-edit-lora"
