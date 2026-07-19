"""Test fixtures: an in-process OpenAI-compatible mock server (no network, no GPU)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

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


@pytest.fixture(scope="session")
def dual_path_pg_url() -> str:
    """Return the PostgreSQL URL supplied by devenv/CI for dual-path tests.

    Database tests deliberately fail when this contract is absent: the supported
    commands provision PostgreSQL rather than silently testing against a local
    developer service or skipping persistence coverage.
    """
    url = os.environ.get("DUAL_PATH_TEST_PG_URL")
    if not url:
        pytest.fail(
            "DUAL_PATH_TEST_PG_URL is required for dual-path tests; run "
            "`devenv shell -- test-dual-path` or configure the CI PostgreSQL service."
        )
    return url


def _with_search_path(url: str, schema: str) -> str:
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params["options"] = f"-c search_path={schema}"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params, quote_via=quote), parts.fragment))


@pytest.fixture
def dual_path_isolated_pg_url(dual_path_pg_url: str) -> Iterator[str]:
    """Provide a per-test schema, safe for parallel tests and local applications."""
    psycopg = pytest.importorskip("psycopg")
    schema = f"dual_path_test_{uuid4().hex}"
    with psycopg.connect(dual_path_pg_url, autocommit=True) as conn:
        conn.execute(f'create schema "{schema}"')
    try:
        yield _with_search_path(dual_path_pg_url, schema)
    finally:
        with psycopg.connect(dual_path_pg_url, autocommit=True) as conn:
            conn.execute(f'drop schema if exists "{schema}" cascade')
