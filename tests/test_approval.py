from __future__ import annotations

import asyncio

from dbos import DBOS, SetWorkflowID
from pydantic import BaseModel

from structured_agents.approval import Approval, ApprovalClient, PendingApproval
from structured_agents.authority import Decision, Denied, execute

approval = Approval[object]()
client = ApprovalClient()


@DBOS.workflow(name="structured_agents.tests.approval")
async def approval_workflow(command: str, timeout: float | None) -> Decision:
    return await approval.request(command, to="ops", timeout=timeout)


class Command(BaseModel):
    argv: list[str]


class Allow:
    def decide(self, command: Command) -> Decision:
        del command
        return Decision(True)


effect_calls = 0


class CountingEffect:
    @DBOS.step()
    async def run(self, command: Command) -> int:
        global effect_calls
        del command
        effect_calls += 1
        return effect_calls


@DBOS.workflow(name="structured_agents.tests.guarded_approval")
async def guarded_workflow(command: Command) -> Denied | int:
    decision = await approval.request(command, to="ops", timeout=30)
    if not decision.allowed:
        return Denied(decision.reason, command)
    return await execute(Allow(), CountingEffect(), command)


async def wait_for_pending(handle: object) -> None:
    for _ in range(100):
        status = await handle.get_status()  # ty: ignore[unresolved-attribute]
        if status.status == "PENDING":
            return
        await asyncio.sleep(0.01)
    raise AssertionError("workflow did not become PENDING")


async def test_pending_status_and_command_are_inspectable() -> None:
    workflow_id = "approval-pending-inspection"
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(approval_workflow, "deploy staging", 30)
    await wait_for_pending(handle)

    assert (await handle.get_status()).status == "PENDING"
    assert await DBOS.get_event_async(workflow_id, "pending_command", timeout_seconds=1) == "deploy staging"
    assert await client.pending() == [PendingApproval(workflow_id, "deploy staging", "ops")]
    await client.approve(workflow_id, reason="reviewed")
    assert await handle.get_result() == Decision(True, "reviewed")


async def test_approval_resumes_workflow_to_success() -> None:
    workflow_id = "approval-resumes-success"
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(approval_workflow, "deploy production", 30)
    await wait_for_pending(handle)
    await client.approve(workflow_id)
    assert await handle.get_result() == Decision(True)
    assert (await handle.get_status()).status == "SUCCESS"


async def test_denial_prevents_effect_execution() -> None:
    global effect_calls
    effect_calls = 0
    workflow_id = "approval-denial-prevents-effect"
    command = Command(argv=["deploy"])
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(guarded_workflow, command)
    await wait_for_pending(handle)
    await client.deny(workflow_id, reason="change window closed")
    assert await handle.get_result() == Denied("change window closed", command)
    assert effect_calls == 0


async def test_timeout_returns_denied_decision() -> None:
    result = await approval_workflow("deploy later", 0.01)
    assert result == Decision(False, "timeout")


async def test_malformed_approval_never_allows() -> None:
    workflow_id = "approval-malformed-string-false"
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(approval_workflow, "deploy unsafe", 30)
    await wait_for_pending(handle)
    await DBOS.send_async(workflow_id, {"allowed": "false", "reason": "malformed"}, topic="approval")

    assert await handle.get_result() == Decision(False, "invalid approval decision")
