"""SPIKE 2 — DBOSAgent top-level vs nested inside a user workflow."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dbos import DBOS, DBOSConfig
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent
from pydantic_ai.models.test import TestModel

TMP = Path(tempfile.mkdtemp(prefix="dbos-spike2-"))
DB = TMP / "dbos.sqlite"


class Out(BaseModel):
    value: int


dbos_agent = DBOSAgent(Agent(TestModel(), output_type=Out, name="spike2agent"), name="spike2agent")


@DBOS.workflow()
async def user_wf() -> str:
    r = await dbos_agent.run("hi from inside a user workflow")
    return f"nested output={r.output!r} inside wf_id={DBOS.workflow_id}"


async def main() -> None:
    cfg = DBOSConfig(name="spike2", system_database_url="sqlite:///" + str(DB), use_listen_notify=False)
    DBOS(config=cfg)
    DBOS.launch()
    try:
        # (a) TOP LEVEL — no surrounding workflow, no explicit SetWorkflowID
        print("--- (a) top-level .run (no surrounding workflow, no SetWorkflowID) ---")
        try:
            r = await dbos_agent.run("hi from top level")
            print("TOPLEVEL OK output =", repr(r.output))
        except Exception as e:  # noqa: BLE001
            print("TOPLEVEL RAISED:", type(e).__name__, "-", e)

        # (b) INSIDE a user @DBOS.workflow()
        print("--- (b) nested inside a user @DBOS.workflow() ---")
        try:
            out = await user_wf()
            print("NESTED OK:", out)
        except Exception as e:  # noqa: BLE001
            print("NESTED RAISED:", type(e).__name__, "-", e)

        # Inspect what workflows DBOS recorded (proves each .run got its own durable wf id)
        wfs = await DBOS.list_workflows_async()
        print("--- recorded workflows (id, name, status) ---")
        for w in wfs:
            print(f"  {w.workflow_id}  name={w.name!r}  status={w.status}")
    finally:
        DBOS.destroy()


if __name__ == "__main__":
    asyncio.run(main())
