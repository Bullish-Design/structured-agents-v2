"""SPIKE 3 — SetWorkflowID isolation under asyncio concurrency.

Two legs run concurrently via asyncio.gather, each wrapping its durable call in
`with SetWorkflowID(f"wid-{i}")`. Each leg's workflow must be attributed to ITS OWN
id — ids must not cross between concurrent legs.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dbos import DBOS, DBOSConfig, SetWorkflowID

TMP = Path(tempfile.mkdtemp(prefix="dbos-spike3-"))
DB = TMP / "dbos.sqlite"


@DBOS.step()
async def observe_id() -> str:
    # Inside the durable context, report which workflow id DBOS attributes this to.
    await asyncio.sleep(0.05)  # force interleaving of the two legs
    return DBOS.workflow_id


@DBOS.workflow()
async def leg_wf() -> str:
    # Interleave again before + after the step to maximise the chance of a cross-leak.
    a = DBOS.workflow_id
    await asyncio.sleep(0.05)
    b = await observe_id()
    await asyncio.sleep(0.05)
    c = DBOS.workflow_id
    assert a == b == c, f"id changed mid-workflow: {a} {b} {c}"
    return c


async def leg(i: int) -> tuple[str, str]:
    expected = f"wid-{i}"
    with SetWorkflowID(expected):
        got = await leg_wf()
    return expected, got


async def main() -> None:
    cfg = DBOSConfig(name="spike3", system_database_url="sqlite:///" + str(DB), use_listen_notify=False)
    DBOS(config=cfg)
    DBOS.launch()
    try:
        results = await asyncio.gather(*(leg(i) for i in range(5)))
        ok = True
        for expected, got in results:
            match = expected == got
            ok = ok and match
            print(f"leg expected={expected!r}  attributed={got!r}  MATCH={match}")
        print("ALL LEGS ISOLATED (no id crossing) =", ok)
    finally:
        DBOS.destroy()


if __name__ == "__main__":
    asyncio.run(main())
