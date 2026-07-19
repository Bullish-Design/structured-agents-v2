"""SPIKE 1 — Can DBOS run Postgres-free (SQLite) for local/dev/test?"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dbos import DBOS, DBOSConfig, SetWorkflowID

TMP = Path(tempfile.mkdtemp(prefix="dbos-spike1-"))
DB = TMP / "dbos.sqlite"

step_calls = 0


@DBOS.step()
def bump() -> int:
    global step_calls
    step_calls += 1
    return step_calls


@DBOS.workflow()
def wf() -> int:
    return bump()


def main() -> None:
    cfg = DBOSConfig(
        name="spike1",
        system_database_url="sqlite:///" + str(DB),
        use_listen_notify=False,
    )
    DBOS(config=cfg)
    DBOS.launch()
    try:
        with SetWorkflowID("fixed-wid-1"):
            r1 = wf()
        with SetWorkflowID("fixed-wid-1"):
            r2 = wf()
        print("RESULT r1 =", r1)
        print("RESULT r2 =", r2)
        print("STEP_CALLS (side-effect counter) =", step_calls)
        print("SQLITE FILE EXISTS =", DB.exists(), "size =", DB.stat().st_size if DB.exists() else 0)
        print("SIBLING FILES =", sorted(p.name for p in TMP.iterdir()))
    finally:
        DBOS.destroy()


if __name__ == "__main__":
    main()
