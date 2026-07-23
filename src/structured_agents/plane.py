"""Lifecycle and operational services for the durable agent plane."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from dbos import DBOS, DBOSConfig, SetWorkflowID, WorkflowHandleAsync, WorkflowStatus
from dbos import Queue as DBOSQueue
from dbos._queue import QueueRateLimit

from .agent import Agent


def configure(*, database_url: str | None = None, app_name: str = "structured_agents", **dbos_config: Any) -> None:
    """Create the DBOS singleton; call this before registering durable work."""
    if database_url is None:
        database_url = f"sqlite:///{Path.cwd() / 'structured_agents.sqlite'}"
    DBOS(
        config=DBOSConfig(
            name=app_name,
            system_database_url=database_url,
            use_listen_notify=False if database_url.startswith("sqlite:") else None,
            **dbos_config,
        )
    )


def launch() -> None:
    """Launch DBOS after agents, effectors, and workflows are registered."""
    DBOS.launch()


async def shutdown() -> None:
    """Destroy the process-global DBOS singleton."""
    DBOS.destroy()


class Queue:
    """A durable DBOS queue for submitting agent runs."""

    def __init__(
        self,
        name: str,
        *,
        concurrency: int | None = None,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        limiter: QueueRateLimit | None = (
            None if rate_limit is None else {"limit": rate_limit[0], "period": rate_limit[1]}
        )
        self._queue = DBOSQueue(name, concurrency=concurrency, limiter=limiter)

    async def submit[T](self, agent: Agent[T], prompt: str, *, key: str | None = None) -> WorkflowHandleAsync[T]:
        """Enqueue one durable agent run, optionally keyed by business identity."""
        if key is None:
            return cast(WorkflowHandleAsync[T], await self._queue.enqueue_async(agent.workflow, prompt))
        with SetWorkflowID(key):
            return cast(WorkflowHandleAsync[T], await self._queue.enqueue_async(agent.workflow, prompt))

    async def submit_batch[T](
        self,
        agent: Agent[T],
        prompts: Sequence[str],
        *,
        keys: Sequence[str | None] | None = None,
    ) -> list[WorkflowHandleAsync[T]]:
        """Enqueue every item and return independent handles for their outcomes."""
        if keys is not None and len(keys) != len(prompts):
            raise ValueError("keys must have the same length as prompts")
        return await asyncio.gather(
            *(
                self.submit(agent, prompt, key=None if keys is None else keys[index])
                for index, prompt in enumerate(prompts)
            )
        )


def schedule(cron: str) -> Callable[[Any], Any]:
    """Decorate a user-authored DBOS workflow with a durable cron schedule."""
    return DBOS.scheduled(cron)


async def workflows(*, status: str | None = None, name: str | None = None) -> list[WorkflowStatus]:
    """List durable workflows through DBOS's async observability API."""
    return await DBOS.list_workflows_async(status=status, name=name)


async def status(workflow_id: str) -> WorkflowStatus:
    """Return a workflow's durable status or raise when it is absent."""
    result = await DBOS.get_workflow_status_async(workflow_id)
    if result is None:
        raise ValueError(f"Unknown workflow {workflow_id!r}")
    return result


async def fork(workflow_id: str, *, from_step: int) -> WorkflowHandleAsync[Any]:
    """Fork a workflow at a DBOS step number."""
    return await DBOS.fork_workflow_async(workflow_id, from_step)


async def cancel(workflow_id: str) -> None:
    """Cancel a running or pending workflow."""
    await DBOS.cancel_workflow_async(workflow_id)


@dataclass(frozen=True)
class Comparison[T]:
    prompt: str
    primary: T
    reference: T
    primary_workflow_id: str
    reference_workflow_id: str


async def compare[T](primary: Agent[T], reference: Agent[T], prompt: str, *, key: str | None = None) -> Comparison[T]:
    """Run two durable agent legs independently and return their recorded pair."""
    comparison_key = key or str(uuid4())
    primary_id = f"{comparison_key}:primary"
    reference_id = f"{comparison_key}:reference"

    async def run(agent: Agent[T], workflow_id: str) -> tuple[T, str]:
        with SetWorkflowID(workflow_id):
            value = await agent.run(prompt)
        return value, workflow_id

    (primary_value, primary_workflow_id), (reference_value, reference_workflow_id) = await asyncio.gather(
        run(primary, primary_id), run(reference, reference_id)
    )
    return Comparison(prompt, primary_value, reference_value, primary_workflow_id, reference_workflow_id)
