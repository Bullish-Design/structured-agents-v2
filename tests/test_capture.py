"""RequestCapture: the httpx request hook records bodies and classifies keys."""

from __future__ import annotations

import asyncio

import httpx

from structured_agents_v2 import RequestCapture


def test_hook_records_body_and_extra_body_keys() -> None:
    cap = RequestCapture()
    request = httpx.Request(
        "POST",
        "http://mock/v1/chat/completions",
        json={
            "model": "m",
            "messages": [],
            "response_format": {"type": "json_schema"},
            "structured_outputs": {"regex": "x"},
        },
    )
    asyncio.run(cap._hook(request))

    rec = cap.last
    assert rec.model == "m"
    assert rec.response_format == {"type": "json_schema"}
    assert rec.tools is None
    assert rec.extra_body_keys == ["structured_outputs"]


def test_last_raises_when_empty() -> None:
    cap = RequestCapture()
    try:
        _ = cap.last
    except IndexError:
        return
    raise AssertionError("expected IndexError on empty capture")
