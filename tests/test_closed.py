"""Regression proofs for the closed, one-request OpenAI-compatible path."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from structured_agents_v2.closed import ClosedBackend, ClosedBackendError


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str


class _OpenAI:
    def __init__(self, status: int = 200, content: str = '{"kind":"no_action"}') -> None:
        self.status = status
        self.content = content
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        assert scope["method"] == "POST"
        assert scope["path"] == "/v1/chat/completions"
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break
        request = json.loads(body)
        self.requests.append(request)
        if self.status != 200:
            await send({"type": "http.response.start", "status": self.status, "headers": []})
            await send({"type": "http.response.body", "body": b"safe error"})
            return
        payload = {
            "id": "opaque",
            "object": "chat.completion",
            "created": 0,
            "model": request["model"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}
            ],
        }
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


def _closed(monkeypatch: pytest.MonkeyPatch, app: _OpenAI) -> ClosedBackend:
    def client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            trust_env=False,
            transport=httpx.ASGITransport(app=app),
        )

    monkeypatch.setattr("structured_agents_v2.closed._new_http_client", client)
    return ClosedBackend(
        base_url="http://127.0.0.1:8000/v1",
        api_key="test-key",
        model="reviewed-local-model",
        timeout=5,
        output_type=Decision,
        instructions="Return only the fixed schema.",
    )


@pytest.mark.asyncio
async def test_closed_backend_returns_validated_output_in_one_request(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _OpenAI()
    closed = _closed(monkeypatch, app)

    try:
        output = await closed.run('{"profile_version":1}')
    finally:
        await closed.aclose()

    assert output == Decision(kind="no_action")
    assert len(app.requests) == 1
    request = app.requests[0]
    assert request["model"] == "reviewed-local-model"
    assert request["stream"] is False
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in request
    assert "tool_choice" not in request
    assert "store" not in request
    assert "user" not in request
    assert "logprobs" not in request


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", '{"kind":"no_action","extra":true}'])
async def test_closed_backend_does_not_retry_invalid_output(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    app = _OpenAI(content=content)
    closed = _closed(monkeypatch, app)

    try:
        with pytest.raises(ClosedBackendError):
            await closed.run('{"profile_version":1}')
    finally:
        await closed.aclose()

    assert len(app.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 500])
async def test_closed_backend_does_not_retry_http_failures(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    app = _OpenAI(status=status)
    closed = _closed(monkeypatch, app)

    try:
        with pytest.raises(ClosedBackendError):
            await closed.run('{"profile_version":1}')
    finally:
        await closed.aclose()

    assert len(app.requests) == 1


def test_closed_backend_has_no_legacy_escape_hatches() -> None:
    assert not hasattr(ClosedBackend, "agent")
    assert not hasattr(ClosedBackend, "run_sync")
    assert not hasattr(ClosedBackend, "build")
    assert not hasattr(ClosedBackend, "attach_transport")
    with pytest.raises(ValueError):
        ClosedBackend(
            base_url="https://provider.invalid/v1",
            api_key="test-key",
            model="reviewed-local-model",
            timeout=5,
            output_type=Decision,
            instructions="fixed",
        )
