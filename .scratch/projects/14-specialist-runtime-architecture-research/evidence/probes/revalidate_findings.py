"""Minimal, read-only reproductions for the highest-impact review findings."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from dbos import DBOS, DBOSConfig, SetWorkflowID
from pydantic import BaseModel

from structured_agents import AgentSpec, Approval, Backend, Choice, Queue, Schema, Settings, all_of
from structured_agents import config as config_module
from structured_agents.config import constraint_from_config, register_constraint


class Plan(BaseModel):
    value: int


def respond(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "probe",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"value": 7}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


database_dir = Path(tempfile.mkdtemp(prefix="structured-agents-review-probe-"))
DBOS(
    config=DBOSConfig(
        name="structured-agents-review-probe",
        system_database_url=f"sqlite:///{database_dir / 'system.sqlite'}",
        use_listen_notify=False,
    )
)

approval = Approval[str]()
backend = Backend(http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
agent = backend.build(AgentSpec("review-real-queue-agent", Schema(Plan), "Return a plan."))
queue = Queue("review-real-agent-queue")


@DBOS.workflow(name="structured_agents.review.approval_truthiness")
async def approval_workflow() -> Any:
    return await approval.request("mutate", to="ops", timeout=30)


def config_concurrency_probe() -> list[str]:
    barrier = threading.Barrier(2)
    observed: list[str] = []
    lock = threading.Lock()
    kind = "review-concurrent-context"

    def factory(_: dict[str, Any]) -> Any:
        barrier.wait(timeout=5)
        active = sorted(config_module._active_allow_modules())
        barrier.wait(timeout=5)
        with lock:
            observed.append(active[0])
        return Choice("ok")

    register_constraint(kind, factory)

    def build(module: str) -> None:
        constraint_from_config({"kind": kind}, allow_modules=frozenset({module}))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build, "module_a"), executor.submit(build, "module_b")]
        for future in futures:
            future.result()
    return observed


async def wait_pending(workflow_id: str) -> None:
    for _ in range(200):
        status = await DBOS.get_workflow_status_async(workflow_id)
        if status is not None and status.status == "PENDING":
            return
        await asyncio.sleep(0.01)
    raise RuntimeError("approval probe did not become pending")


async def main() -> None:
    results: dict[str, Any] = {}
    results["empty_all_of"] = all_of().decide("anything").__dict__
    results["invalid_settings"] = Settings(temperature="cold", max_tokens=-1).__dict__  # type: ignore[arg-type]
    parsed = Schema(Plan).parse({"value": 7})
    results["schema_cast"] = {"runtime_type": type(parsed).__name__, "value": parsed}
    results["config_allowlist_observed"] = config_concurrency_probe()

    DBOS.launch()
    try:
        try:
            await queue.submit(agent, "queued")
        except Exception as exc:
            results["real_agent_queue"] = {"exception": type(exc).__name__, "message": str(exc)}
        else:
            results["real_agent_queue"] = {"exception": None}

        workflow_id = "review-malformed-approval"
        with SetWorkflowID(workflow_id):
            handle = await DBOS.start_workflow_async(approval_workflow)
        await wait_pending(workflow_id)
        await DBOS.send_async(workflow_id, {"allowed": "false", "reason": "string"}, topic="approval")
        decision = await handle.get_result()
        results["malformed_approval"] = decision.__dict__
    finally:
        await backend.aclose()
        DBOS.destroy(destroy_registry=True)

    print(json.dumps(results, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    asyncio.run(main())
