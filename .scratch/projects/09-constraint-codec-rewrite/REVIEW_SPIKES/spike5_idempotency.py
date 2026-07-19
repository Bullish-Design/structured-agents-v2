"""SPIKE 5 — Exactly-once effect idempotency ergonomics.

A @DBOS.step with a real side effect (append a line to a file + bump a module counter).
Run its workflow TWICE with the SAME workflow id -> the side effect must happen EXACTLY ONCE.
Then: keying exactly-once on a BUSINESS identity is just choosing SetWorkflowID as the natural key.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dbos import DBOS, DBOSConfig, SetWorkflowID

TMP = Path(tempfile.mkdtemp(prefix="dbos-spike5-"))
DB = TMP / "dbos.sqlite"
SIDE_EFFECT_LOG = TMP / "effects.log"

counter = 0


@DBOS.step()
def charge_card(amount: int) -> int:
    global counter
    counter += 1
    with SIDE_EFFECT_LOG.open("a") as f:
        f.write(f"charged {amount} (call #{counter})\n")
    return counter


@DBOS.workflow()
def payment_wf(amount: int) -> int:
    return charge_card(amount)


def main() -> None:
    cfg = DBOSConfig(name="spike5", system_database_url="sqlite:///" + str(DB), use_listen_notify=False)
    DBOS(config=cfg)
    DBOS.launch()
    try:
        # Same workflow id == same business identity (e.g. "order-42").
        biz_key = "order-42"
        with SetWorkflowID(biz_key):
            r1 = payment_wf(100)
        with SetWorkflowID(biz_key):
            r2 = payment_wf(100)  # replay: cached, step NOT re-executed

        # A DIFFERENT business identity runs the effect again (independent).
        with SetWorkflowID("order-99"):
            r3 = payment_wf(250)

        print("r1 =", r1, " r2 =", r2, " r3 =", r3)
        print("module counter (side-effect count) =", counter)
        log = SIDE_EFFECT_LOG.read_text()
        print("side-effect log lines =", len(log.strip().splitlines()))
        print("side-effect log:\n" + log.strip())
        assert counter == 2, f"expected exactly 2 charges (order-42 once, order-99 once), got {counter}"
        print("EXACTLY-ONCE PER BUSINESS KEY = CONFIRMED")
    finally:
        DBOS.destroy()


if __name__ == "__main__":
    main()
