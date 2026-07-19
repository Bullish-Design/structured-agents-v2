from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace
from typing import cast

import httpx
from dbos import DBOS, SetWorkflowID

from structured_agents.agent import Agent, AgentSpec, Backend
from structured_agents.approval import Approval
from structured_agents.authority import Decision
from structured_agents.constraint import Choice
from structured_agents.plane import Queue, cancel, compare, fork, schedule, status, workflows

active = 0
peak_active = 0
successful_items: list[str] = []
step_runs = 0
scheduled_runs = 0
comparison_model_calls = 0
approval = Approval[str]()


@DBOS.workflow(name="structured_agents.tests.queue_agent")
async def queued_run(prompt: str) -> str:
    global active, peak_active
    active += 1
    peak_active = max(peak_active, active)
    try:
        await DBOS.sleep_async(0.05)
        if prompt == "fail":
            raise RuntimeError("isolated queue failure")
        successful_items.append(prompt)
        return prompt.upper()
    finally:
        active -= 1


class QueueAgent:
    raw = SimpleNamespace(run=queued_run)

    async def run(self, prompt: str) -> str:
        return await queued_run(prompt)


def comparison_response(request: httpx.Request) -> httpx.Response:
    global comparison_model_calls
    comparison_model_calls += 1
    body = json.loads(request.content)
    return httpx.Response(200, json={
        "id": "comparison", "object": "chat.completion", "created": 0, "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "COMPARE"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


comparison_primary = Backend(http_client=httpx.AsyncClient(transport=httpx.MockTransport(comparison_response))).build(
    AgentSpec("comparison-primary", Choice("COMPARE"), "Return COMPARE.")
)
comparison_reference = Backend(http_client=httpx.AsyncClient(transport=httpx.MockTransport(comparison_response))).build(
    AgentSpec("comparison-reference", Choice("COMPARE"), "Return COMPARE.")
)


@DBOS.step()
async def counted_step(value: str) -> str:
    global step_runs
    step_runs += 1
    return value


@DBOS.workflow(name="structured_agents.tests.forkable")
async def forkable(value: str) -> str:
    return await counted_step(value)


# DBOS 2.23 accepts a seconds field, keeping this real timer test CI-safe.
@schedule("*/2 * * * * *")
@DBOS.workflow(name="structured_agents.tests.scheduled")
async def scheduled_workflow(_scheduled_time: datetime, _actual_time: datetime) -> None:
    global scheduled_runs
    scheduled_runs += 1


@DBOS.workflow(name="structured_agents.tests.cancelled")
async def cancellable() -> None:
    await DBOS.recv_async("never-arrives")


@DBOS.workflow(name="structured_agents.tests.pending_for_observability")
async def pending_for_observability() -> Decision:
    return await approval.request("review me", to="ops", timeout=30)


async def wait_for(workflow_id: str, expected: str, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (await status(workflow_id)).status == expected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"{workflow_id} did not reach {expected}")


async def test_queue_caps_concurrency_and_isolates_batch_failures() -> None:
    global active, peak_active, successful_items
    active = peak_active = 0
    successful_items = []
    queue = Queue("structured-agents-tests-queue", concurrency=2, rate_limit=(100, 1))
    agent = cast(Agent[str], QueueAgent())

    handles = await queue.submit_batch(agent, ["one", "fail", "two", "three"])
    results = await asyncio.gather(*(handle.get_result() for handle in handles), return_exceptions=True)

    assert peak_active <= 2
    assert results[0] == "ONE"
    assert isinstance(results[1], RuntimeError)
    assert results[2:] == ["TWO", "THREE"]
    assert sorted(successful_items) == ["one", "three", "two"]


async def test_scheduled_workflow_fires() -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if scheduled_runs:
            break
        await asyncio.sleep(0.1)
    assert scheduled_runs >= 1
    assert any(item.status == "SUCCESS" for item in await workflows(name="structured_agents.tests.scheduled"))


async def test_observability_lists_status_forks_and_cancels() -> None:
    pending_id = "plane-pending-observability"
    with SetWorkflowID(pending_id):
        pending = await DBOS.start_workflow_async(pending_for_observability)
    await wait_for(pending_id, "PENDING")
    assert pending_id in {item.workflow_id for item in await workflows(status="PENDING")}
    assert (await status(pending_id)).status == "PENDING"
    await DBOS.send_async(pending_id, {"allowed": False, "reason": "done"}, topic="approval")
    assert await pending.get_result() == Decision(False, "done")

    global step_runs
    step_runs = 0
    original_id = "plane-fork-source"
    with SetWorkflowID(original_id):
        assert await forkable("value") == "value"
    replay = await fork(original_id, from_step=0)
    assert await replay.get_result() == "value"
    assert step_runs == 2

    cancel_id = "plane-cancel-source"
    with SetWorkflowID(cancel_id):
        await DBOS.start_workflow_async(cancellable)
    await wait_for(cancel_id, "PENDING")
    await cancel(cancel_id)
    await wait_for(cancel_id, "CANCELLED")


async def test_compare_is_keyed_and_durable() -> None:
    global comparison_model_calls
    comparison_model_calls = 0

    first = await compare(comparison_primary, comparison_reference, "compare", key="plane-comparison")
    second = await compare(comparison_primary, comparison_reference, "compare", key="plane-comparison")

    assert first.primary == first.reference == "COMPARE"
    assert second == first
    assert comparison_model_calls == 2
    assert first.primary_workflow_id == "plane-comparison:primary"
    assert first.reference_workflow_id == "plane-comparison:reference"
