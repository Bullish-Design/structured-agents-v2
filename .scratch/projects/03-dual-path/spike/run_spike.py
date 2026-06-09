"""Runnable dual-path spike — gates 1-3, GPU-free, against an in-process ASGI mock.

    devenv shell -- uv run --extra dev --extra dual-path python \
        .scratch/projects/03-dual-path/spike/run_spike.py

Proves Architecture C: two independent DBOSAgents joined by a top-level asyncio.gather, the
Phase-2 wire shape surviving the wrapper, and a versioned ComparisonRecord persisted to Postgres
jsonb. Writes artifacts under ./artifacts/.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `schemas` and output_type_ref="schemas:Command" resolve

import runner  # noqa: E402
import schemas  # noqa: E402
from dbos import DBOS, SetWorkflowID  # noqa: E402
from pydantic_ai.durable_exec.dbos import StepConfig  # noqa: E402
from structured_agents_v2.backend import Backend, BackendCaps  # noqa: E402
from structured_agents_v2.profile import AgentProfile  # noqa: E402

ARTIFACTS = HERE / "artifacts"
PROMPT = "Create a file notes.txt"

# --- in-process mock: returns Command JSON keyed by the wire `model` field --------------

_RESPONSES = {
    "cmd-adapter": '{"action":"create","target":"notes.txt","reason":"local model says create"}',
    "frontier-sim": '{"action":"create","target":"notes.txt","reason":"frontier teacher says create"}',
}


class MockOpenAI:
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break
        req = json.loads(body) if body else {}
        model = req.get("model", "cmd-adapter")
        content = _RESPONSES.get(model, _RESPONSES["cmd-adapter"])
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


def _write(name: str, obj: Any) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / name).write_text(json.dumps(obj, indent=2, default=str))


async def main() -> int:
    transport = httpx.ASGITransport(app=MockOpenAI())
    report: dict[str, Any] = {}

    # primary = local vLLM-like backend (xgrammar+lora); reference = frontier-like (neither)
    primary_backend = Backend(
        base_url="http://mock/v1", default_model="base", caps=BackendCaps(), capture=True
    ).attach_transport(transport)
    reference_backend = Backend(
        base_url="http://mock/v1", default_model="frontier-sim",
        caps=BackendCaps(xgrammar=False, lora=False), capture=True,
    ).attach_transport(transport)

    primary_profile = AgentProfile(
        name="cmd", adapter="cmd-adapter", instructions="Emit one command object.",
        output_type_ref="schemas:Command",
    )
    reference_profile = AgentProfile(
        name="cmd", adapter=None, instructions="Emit one command object.",
        output_type_ref="schemas:Command",
    )

    # --- DBOS lifecycle: register agents BEFORE launch -----------------------------------
    DBOS(config=runner.dbos_config())
    primary_sa, primary_dbos = runner.build_dbos_agent(
        primary_backend, primary_profile, dbos_name="cmd@primary",
        step_config=StepConfig(max_attempts=2),
    )
    reference_sa, reference_dbos = runner.build_dbos_agent(
        reference_backend, reference_profile, dbos_name="cmd@reference",
        step_config=StepConfig(max_attempts=3, retries_allowed=True),
    )
    DBOS.launch()

    try:
        # --- Gate 1: one DBOSAgent runs durably, output validates -------------------------
        g1 = await primary_dbos.run(PROMPT)
        gate1_ok = isinstance(g1.output, schemas.Command)
        report["gate1_one_durable_run"] = {
            "ok": gate1_ok, "output": g1.output.model_dump() if gate1_ok else None,
        }
        print(f"[gate 1] durable run -> {g1.output!r}  ok={gate1_ok}")

        # --- Gate 2: wire shape identical wrapped vs unwrapped ----------------------------
        recs = primary_sa._capture.records  # noqa: SLF001
        await primary_sa.run(PROMPT)            # unwrapped baseline
        unwrapped = recs[-1].body
        await primary_dbos.run(PROMPT)          # through the DBOS wrapper
        wrapped = recs[-1].body
        gate2_ok = unwrapped == wrapped
        report["gate2_wire_shape"] = {
            "ok": gate2_ok,
            "model": wrapped.get("model"),
            "response_format_type": (wrapped.get("response_format") or {}).get("type"),
            "extra_body_keys": [k for k in wrapped if k not in {"model", "messages", "response_format", "tools", "tool_choice", "stream", "stream_options", "max_completion_tokens", "max_tokens"}],
        }
        _write("captured_request.json", {"unwrapped": unwrapped, "wrapped": wrapped})
        print(f"[gate 2] wire shape wrapped==unwrapped ok={gate2_ok}  model={wrapped.get('model')}  rf={report['gate2_wire_shape']['response_format_type']}")

        # --- Gate 3: top-level gather over two DBOSAgents + ComparisonRecord --------------
        run_id = uuid.uuid4().hex[:12]
        pid, rid = f"primary-{run_id}", f"reference-{run_id}"

        async def _run(agent: Any, wid: str) -> Any:
            with SetWorkflowID(wid):
                return await agent.run(PROMPT)

        primary_res, reference_res = await asyncio.gather(_run(primary_dbos, pid), _run(reference_dbos, rid))

        record = runner.build_record(
            run_id=run_id, prompt=PROMPT, profile=primary_profile, output_type=schemas.Command,
            decode_mode="json_schema", primary_model="cmd-adapter", reference_model="frontier-sim",
            primary_result=primary_res, reference_result=reference_res,
            primary_workflow_id=pid, reference_workflow_id=rid,
        )

        store = runner.ComparisonStore()
        store.init_schema()
        row_id = store.save(record)

        primary_steps = await runner.workflow_steps(pid)
        reference_steps = await runner.workflow_steps(rid)
        gate3_ok = (
            record.primary_valid and record.reference_valid
            and record.agreement_exact is False and "reason" in (record.field_diff or {})
            and len(primary_steps) >= 1 and len(reference_steps) >= 1
        )
        report["gate3_dual_gather"] = {
            "ok": gate3_ok, "row_id": row_id, "agreement_exact": record.agreement_exact,
            "field_diff": record.field_diff, "primary_workflow_id": pid, "reference_workflow_id": rid,
            "primary_steps_count": len(primary_steps), "reference_steps_count": len(reference_steps),
        }
        _write("comparison_record.json", record.model_dump())
        _write("dbos_steps.json", {"primary": primary_steps, "reference": reference_steps})
        print(f"[gate 3] dual gather -> row#{row_id}  agreement={record.agreement_exact}  diff={record.field_diff}")
        print(f"[gate 3] DBOS persisted steps: primary={len(primary_steps)} reference={len(reference_steps)}")

        report["all_ok"] = bool(gate1_ok and gate2_ok and gate3_ok)
        _write("report.json", report)
        print(f"\nALL GATES OK: {report['all_ok']}  (artifacts in {ARTIFACTS})")
        return 0 if report["all_ok"] else 1
    finally:
        DBOS.destroy()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
