"""Durable human approval for user-authored DBOS workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dbos import DBOS

from .authority import Decision

_PENDING_COMMAND = "pending_command"
_PENDING_TO = "pending_to"


@dataclass(frozen=True)
class PendingApproval:
    workflow_id: str
    command: Any
    to: str


class Approval[C]:
    """A durable pause whose result remains an authorization decision."""

    def __init__(self, *, topic: str = "approval") -> None:
        self._topic = topic

    async def request(self, command: C, *, to: str, timeout: float | None = None) -> Decision:
        """Publish a command then wait durably for an approver's decision.

        This must run inside a ``@DBOS.workflow`` because DBOS events and
        receives are workflow-scoped durable operations.
        """
        await DBOS.set_event_async(_PENDING_COMMAND, command)
        await DBOS.set_event_async(_PENDING_TO, to)
        if timeout is None:
            message = await DBOS.recv_async(topic=self._topic)
        else:
            message = await DBOS.recv_async(topic=self._topic, timeout_seconds=timeout)
        if message is None:
            return Decision(False, "timeout")
        if isinstance(message, Mapping):
            allowed = bool(message.get("allowed", False))
            reason = message.get("reason", "")
            return Decision(allowed, reason if isinstance(reason, str) else str(reason))
        return Decision(False, "invalid approval decision")


class ApprovalClient:
    """Out-of-workflow approval operations for a CLI, UI, or bot."""

    def __init__(self, *, topic: str = "approval") -> None:
        self._topic = topic

    async def approve(self, workflow_id: str, *, reason: str = "") -> None:
        await DBOS.send_async(workflow_id, {"allowed": True, "reason": reason}, topic=self._topic)

    async def deny(self, workflow_id: str, *, reason: str) -> None:
        await DBOS.send_async(workflow_id, {"allowed": False, "reason": reason}, topic=self._topic)

    async def pending(self) -> list[PendingApproval]:
        pending: list[PendingApproval] = []
        workflows = await DBOS.list_workflows_async(status="PENDING", load_input=False, load_output=False)
        for workflow in workflows:
            events = await DBOS.get_all_events_async(workflow.workflow_id)
            if _PENDING_COMMAND not in events or _PENDING_TO not in events:
                continue
            to = events[_PENDING_TO]
            if not isinstance(to, str):
                continue
            pending.append(PendingApproval(workflow.workflow_id, events[_PENDING_COMMAND], to))
        return pending
