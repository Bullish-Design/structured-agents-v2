"""Test fixtures: an in-process OpenAI-compatible mock server (no network, no GPU)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest


class MockOpenAI:
    """A minimal ASGI app implementing POST /v1/chat/completions.

    `responder` maps a parsed request body to the assistant message content string,
    letting each test control what the "model" returns.
    """

    def __init__(self) -> None:
        self.responder: Callable[[dict[str, Any]], str] = lambda _req: "ok"

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break
        req = json.loads(body) if body else {}
        content = self.responder(req)
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": data})


@pytest.fixture
def mock_openai() -> MockOpenAI:
    return MockOpenAI()


@pytest.fixture
def transport(mock_openai: MockOpenAI) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_openai)
