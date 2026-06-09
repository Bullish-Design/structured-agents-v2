"""Postgres + DBOS gated tests for the dual-path runtime/runner (`DualPathRuntime`/`DualPathRunner`).

Architecture C, GPU-free: two `DBOSAgent` legs against an in-process OpenAI-compatible ASGI mock
(keyed by the wire `model`), joined by a top-level `asyncio.gather`, persisted as `ComparisonRecord`s.

Gates: `dbos`/`psycopg` absent → skipped; Postgres not on 127.0.0.1:5433 → skipped. DBOS is a
process-global singleton, so the whole module shares **one** lifecycle (a module-scoped fixture:
init → register → launch; teardown: destroy) driven on a single dedicated event loop.

Run: `devenv shell -- uv run --extra dev --extra dual-path pytest tests/test_dual_path_runtime.py -q`
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("dbos")
pytest.importorskip("psycopg")
import psycopg  # noqa: E402

SPIKE_DIR = Path(__file__).resolve().parents[1] / ".scratch" / "projects" / "03-dual-path" / "spike"
sys.path.insert(0, str(SPIKE_DIR))  # so output_type_ref="schemas:Command" resolves (as in the spike)


from structured_agents_v2.backend import Backend, BackendCaps  # noqa: E402
from structured_agents_v2.dual_path import (  # noqa: E402
    ComparisonStore,
    DualPathConfig,
    DualPathConfigError,
    DualPathRuntime,
)
from structured_agents_v2.dual_path.runtime import DualPathRuntime as _RT  # noqa: E402  (static guard)
from structured_agents_v2.profile import AgentProfile  # noqa: E402

PG_URL = os.environ.get("DUAL_PATH_TEST_PG_URL", "postgresql://andrew@127.0.0.1:5433/dual_path")
PROMPT = "Create a file notes.txt"

_VALID = '{"action":"create","target":"notes.txt","reason":"create it"}'
_RESPONSES = {"cmd-adapter": _VALID, "frontier-sim": _VALID}


def _pg_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 5433), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _pg_up(), reason="dual-path Postgres not running on 127.0.0.1:5433")


class MockOpenAI:
    """ASGI mock: Command JSON keyed by the wire `model`; HTTP 500 for the `bad-frontier` model."""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break
        req = json.loads(body) if body else {}
        model = req.get("model", "cmd-adapter")
        if model == "bad-frontier":
            await send(
                {"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": b'{"error":"boom"}'})
            return
        content = _RESPONSES.get(model, _VALID)
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
        data = json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": data})


def _profile(adapter: str | None) -> AgentProfile:
    return AgentProfile(
        name="cmd", adapter=adapter, instructions="Emit one command object.", output_type_ref="schemas:Command"
    )


@dataclass
class Env:
    runtime: DualPathRuntime
    store: ComparisonStore
    cmd: Any  # DualPathRunner: vllm primary ‖ frontier reference, sample_rate 1.0
    errcmd: Any  # reference leg 500s
    skip: Any  # sample_rate 0.0
    primary_sa: Any  # the cmd runner's primary StructuredAgent (capture on)
    bad_agent: Any  # a non-json_schema StructuredAgent (for the guard test)
    run: Callable[[Coroutine[Any, Any, Any]], Any]


@pytest.fixture(scope="module")
def env() -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    transport = httpx.ASGITransport(app=MockOpenAI())

    def primary_be() -> Backend:
        return Backend(base_url="http://mock/v1", default_model="base", capture=True).attach_transport(transport)

    def reference_be(model: str) -> Backend:
        return Backend(
            base_url="http://mock/v1", default_model=model, caps=BackendCaps(xgrammar=False, lora=False)
        ).attach_transport(transport)

    runtime = DualPathRuntime(DualPathConfig(app_name="dual-path-test", pg_url=PG_URL, default_sample_rate=1.0))

    # NB: DBOS function registration is process-global and survives destroy(); these names must not
    # collide with any other DBOS-using test in the session (e.g. the spike's "cmd@primary").
    cmd_primary = primary_be().build(_profile("cmd-adapter"))
    cmd = runtime.register("rt_cmd", primary=cmd_primary, reference=reference_be("frontier-sim").build(_profile(None)))
    errcmd = runtime.register(
        "rt_err",
        primary=primary_be().build(_profile("cmd-adapter")),
        reference=reference_be("bad-frontier").build(_profile(None)),
    )
    skip = runtime.register(
        "rt_skip",
        primary=primary_be().build(_profile("cmd-adapter")),
        reference=reference_be("frontier-sim").build(_profile(None)),
        sample_rate=0.0,
    )

    # a non-json_schema agent (choice mode) for the json_schema guard test — never registered.
    from structured_agents_v2.decoder import DecoderSpec

    bad_profile = AgentProfile(name="pick", instructions="pick", decoder=DecoderSpec(mode="choice", choices=["a", "b"]))
    bad_agent = Backend(base_url="http://mock/v1", default_model="base").attach_transport(transport).build(bad_profile)

    runtime.launch()
    env_obj = Env(
        runtime=runtime,
        store=runtime.store,
        cmd=cmd,
        errcmd=errcmd,
        skip=skip,
        primary_sa=cmd_primary,
        bad_agent=bad_agent,
        run=loop.run_until_complete,
    )
    try:
        yield env_obj
    finally:
        runtime.shutdown()
        loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture
def clean(env: Env) -> Env:
    """Truncate the data table before each test for row-count isolation (DBOS tables untouched)."""
    with psycopg.connect(PG_URL) as conn:
        conn.execute("truncate comparison_records")
        conn.commit()
    return env


# --- §7.1 register / launch / guards ---------------------------------------------------


def test_register_non_json_schema_raises(env: Env) -> None:
    with pytest.raises(DualPathConfigError, match="json_schema"):
        _RT._require_json_schema("pick", "reference", env.bad_agent)


def test_register_after_launch_raises(env: Env) -> None:
    with pytest.raises(DualPathConfigError, match="after launch"):
        env.runtime.register(
            "late",
            primary=env.primary_sa,
            reference=env.primary_sa,
        )


# --- §7.2 wire shape survives the wrapper ----------------------------------------------


def test_run_returns_validated_output_and_wire_shape(clean: Env) -> None:
    record = clean.run(clean.cmd.run(PROMPT))
    assert record.primary_valid
    assert record.primary_output == {"action": "create", "target": "notes.txt", "reason": "create it"}

    body = clean.primary_sa._capture.last.body  # noqa: SLF001 - test introspection of the wire shape
    assert body["model"] == "cmd-adapter"
    assert (body.get("response_format") or {}).get("type") == "json_schema"


# --- §7.3 a dual run persists exactly one correlatable row -----------------------------


def test_dual_run_persists_one_row(clean: Env) -> None:
    record = clean.run(clean.cmd.run(PROMPT, force_reference=True))
    assert record.primary_valid and record.reference_valid
    assert not record.reference_skipped

    rows = clean.store.query(profile_version=record.profile_version)
    assert len(rows) == 1
    saved = rows[0]
    assert saved.run_id == record.run_id
    assert saved.primary_workflow_id == f"primary-{record.run_id}"
    assert saved.reference_workflow_id == f"reference-{record.run_id}"

    # the escape-hatch properties expose the underlying DBOSAgents
    from pydantic_ai.durable_exec.dbos import DBOSAgent

    assert isinstance(clean.cmd.primary, DBOSAgent)
    assert isinstance(clean.cmd.reference, DBOSAgent)


# --- §7.4 sampling ---------------------------------------------------------------------


def test_force_reference_false_skips_reference(clean: Env) -> None:
    record = clean.run(clean.cmd.run(PROMPT, force_reference=False))
    assert record.reference_skipped
    assert record.reference_output is None
    assert record.reference_model is None
    assert record.reference_workflow_id is None
    assert len(clean.store.query(profile_version=record.profile_version)) == 1


def test_sample_rate_zero_skips_reference(clean: Env) -> None:
    record = clean.run(clean.skip.run(PROMPT))  # sample_rate 0.0, no force
    assert record.reference_skipped
    assert record.reference_output is None
    assert record.primary_valid


def test_force_reference_true_runs_reference(clean: Env) -> None:
    record = clean.run(clean.skip.run(PROMPT, force_reference=True))  # override the 0.0 rate
    assert not record.reference_skipped
    assert record.reference_valid
    assert record.reference_output == {"action": "create", "target": "notes.txt", "reason": "create it"}


# --- §7.5 error path: reference leg fails, record still saved, primary OK ---------------


def test_reference_error_is_captured(clean: Env) -> None:
    record = clean.run(clean.errcmd.run(PROMPT, force_reference=True))
    assert record.primary_valid
    assert record.primary_output is not None
    assert not record.reference_valid
    assert record.reference_error is not None
    assert record.signal is None  # no agreement signal when a leg is invalid
    assert len(clean.store.query(profile_version=record.profile_version)) == 1
