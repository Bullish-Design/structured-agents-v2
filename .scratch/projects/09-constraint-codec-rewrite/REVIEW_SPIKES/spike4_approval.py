"""SPIKE 4 — Human-in-the-loop durable approval.

A @DBOS.workflow generates a command, publishes it via set_event, then durably BLOCKS on
DBOS.recv(topic="approval") until an out-of-process-style approver calls DBOS.send(...).
Also shows the timeout path (recv -> None -> denied) and the set_event/get_event variant.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dbos import DBOS, DBOSConfig, SetWorkflowID

TMP = Path(tempfile.mkdtemp(prefix="dbos-spike4-"))
DB = TMP / "dbos.sqlite"


@DBOS.workflow()
async def approval_wf(command: str, timeout_s: float) -> str:
    # Publish the pending command so an external approver can inspect it (workflow -> outside).
    await DBOS.set_event_async("pending_command", command)
    # Durably block waiting for the decision (outside -> workflow).
    decision = await DBOS.recv_async(topic="approval", timeout_seconds=timeout_s)
    if decision is None:
        return f"DENIED(timeout): {command}"
    if decision.get("allowed"):
        return f"EXECUTED: {command}"
    return f"DENIED(rejected): {command}"


async def status_of(handle) -> str:  # noqa: ANN001
    return (await handle.get_status()).status


async def main() -> None:
    cfg = DBOSConfig(name="spike4", system_database_url="sqlite:///" + str(DB), use_listen_notify=False)
    DBOS(config=cfg)
    DBOS.launch()
    try:
        # ---- (a)+(b) APPROVE PATH ----
        wid = "approve-1"
        with SetWorkflowID(wid):
            h = await DBOS.start_workflow_async(approval_wf, "rm -rf /tmp/foo", 30.0)
        await asyncio.sleep(0.3)  # let it reach the recv and block
        print("(a) status while blocked on recv =", await status_of(h))

        # External approver inspects the published command, then approves.
        cmd = await DBOS.get_event_async(wid, "pending_command", timeout_seconds=5)
        print("(a) approver read pending_command via get_event =", repr(cmd))
        await DBOS.send_async(wid, {"allowed": True}, topic="approval")

        result = await h.get_result()
        print("(b) status after send =", await status_of(h))
        print("(b) result =", repr(result))

        # ---- (c) TIMEOUT PATH ----
        wid2 = "timeout-1"
        with SetWorkflowID(wid2):
            h2 = await DBOS.start_workflow_async(approval_wf, "deploy prod", 1.0)
        r2 = await h2.get_result()  # nobody ever sends -> recv times out -> None
        print("(c) timeout-path result =", repr(r2))

        # ---- (d) DENY PATH (explicit reject via send) ----
        wid3 = "deny-1"
        with SetWorkflowID(wid3):
            h3 = await DBOS.start_workflow_async(approval_wf, "drop table users", 30.0)
        await asyncio.sleep(0.2)
        await DBOS.send_async(wid3, {"allowed": False}, topic="approval")
        r3 = await h3.get_result()
        print("(d) deny-path result =", repr(r3))
    finally:
        DBOS.destroy()


if __name__ == "__main__":
    asyncio.run(main())
