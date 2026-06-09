"""Wire capture — an httpx event hook that records outgoing requests.

Used to introspect exactly what PydanticAI puts on the wire (the technique the request
spike used). Attach to any `OpenAIProvider(http_client=...)`; results land in `.records`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

_KNOWN_KEYS = frozenset(
    {
        "model",
        "messages",
        "response_format",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "max_completion_tokens",
        "max_tokens",
    }
)


@dataclass
class RequestRecord:
    """A single captured outgoing request body."""

    url: str
    body: dict[str, Any]

    @property
    def model(self) -> str | None:
        return self.body.get("model")

    @property
    def response_format(self) -> dict[str, Any] | None:
        rf = self.body.get("response_format")
        return rf if isinstance(rf, dict) else None

    @property
    def tools(self) -> list[Any] | None:
        return self.body.get("tools")

    @property
    def extra_body_keys(self) -> list[str]:
        """Top-level keys that came from `extra_body` (i.e. not standard OpenAI fields)."""
        return [k for k in self.body if k not in _KNOWN_KEYS]


class RequestCapture:
    """Collects request bodies via an httpx request event hook."""

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []

    async def _hook(self, request: httpx.Request) -> None:
        raw = request.content
        try:
            body = json.loads(raw.decode()) if raw else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        self.records.append(RequestRecord(url=str(request.url), body=body))

    def client(self, *, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
        """An `httpx.AsyncClient` with this capture hook attached."""
        kwargs: dict[str, Any] = {"event_hooks": {"request": [self._hook]}}
        if transport is not None:
            kwargs["transport"] = transport
        return httpx.AsyncClient(**kwargs)

    @property
    def last(self) -> RequestRecord:
        if not self.records:
            raise IndexError("no requests captured")
        return self.records[-1]
