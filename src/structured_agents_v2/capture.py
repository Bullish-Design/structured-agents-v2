"""Wire capture — an httpx event hook that records outgoing requests.

Used to introspect exactly what PydanticAI puts on the wire (the technique the request
spike used). Attach to any `OpenAIProvider(http_client=...)`; results land in `.records`.
"""

from __future__ import annotations

import collections
import contextvars
import json
from dataclasses import dataclass
from typing import Any

import httpx

# Per-run sink: `agent.run` sets this to a list around the awaited call, so the httpx hook
# — which executes in the same task/context — appends that run's records to it. This gives
# each run its OWN request body even when the same agent runs concurrently (run_batch).
_run_sink: contextvars.ContextVar[list[RequestRecord] | None] = contextvars.ContextVar(
    "sav_capture_run_sink", default=None
)

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
        # ordinary sampling params — present in standard bodies, not extra_body
        "temperature",
        "top_p",
        "seed",
        "stop",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "user",
        "parallel_tool_calls",
        "reasoning_effort",
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
    """Collects request bodies via an httpx request event hook.

    `records` is a bounded ring buffer (default 1000) so long-lived `capture=True` fleets
    don't grow without bound. For correct per-run attribution, read the record via the
    contextvar sink `agent.run` installs rather than `.last`.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self.records: collections.deque[RequestRecord] = collections.deque(maxlen=max_records)

    async def _hook(self, request: httpx.Request) -> None:
        raw = request.content
        try:
            body = json.loads(raw.decode()) if raw else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        record = RequestRecord(url=str(request.url), body=body)
        self.records.append(record)
        sink = _run_sink.get()
        if sink is not None:
            sink.append(record)

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
