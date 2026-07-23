"""Worker for the crash-after-remote-commit ambiguity probe."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dbos import DBOS, DBOSConfig, SetWorkflowID

MODE, DATABASE_ARG, LEDGER_ARG, CRASH_MARKER_ARG, WORKFLOW_ID = sys.argv[1:]
DATABASE = Path(DATABASE_ARG)
LEDGER = Path(LEDGER_ARG)
CRASH_MARKER = Path(CRASH_MARKER_ARG)
DBOS(
    config=DBOSConfig(
        name="project15-crash-effect-probe",
        system_database_url=f"sqlite:///{DATABASE}",
        use_listen_notify=False,
    )
)


def ledger_entries() -> list[str]:
    return LEDGER.read_text().splitlines() if LEDGER.exists() else []


def remote_commit(key: str) -> None:
    entries = ledger_entries()
    if MODE == "idempotent" and key in entries:
        return
    with LEDGER.open("a") as handle:
        handle.write(f"{key}\n")
        handle.flush()
        os.fsync(handle.fileno())


@DBOS.step(retries_allowed=False)
async def ambiguous_external_effect(key: str) -> int:
    remote_commit(key)
    if not CRASH_MARKER.exists():
        CRASH_MARKER.write_text("remote commit completed; DBOS checkpoint not reached\n")
        os._exit(71)
    return len(ledger_entries())


@DBOS.workflow(name="project15.crash_effect")
async def workflow(key: str) -> int:
    return await ambiguous_external_effect(key)


async def main() -> None:
    DBOS.launch()
    try:
        with SetWorkflowID(WORKFLOW_ID):
            handle = await DBOS.start_workflow_async(workflow, "remote-effect-key")
        result = await handle.get_result()
        status = await handle.get_status()
        print(
            json.dumps(
                {
                    "mode": MODE,
                    "result": result,
                    "status": status.status,
                    "ledger": ledger_entries(),
                },
                sort_keys=True,
            )
        )
    finally:
        DBOS.destroy(destroy_registry=True)


if __name__ == "__main__":
    asyncio.run(main())
