"""`StructuredAgent` — the runnable wrapper around a built `pydantic_ai.Agent`.

A `StructuredAgent` is produced by `Backend.build(profile)`; it owns the configured
`pydantic_ai.Agent` and exposes a lean `run`/`run_sync` returning an `AgentResult`. The
underlying agent stays reachable via `.agent` as an escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .capture import _run_sink

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from .capture import RequestCapture, RequestRecord
    from .profile import AgentProfile


@dataclass
class AgentResult[OutputT]:
    """The result of one agent run: the validated output plus escape hatches."""

    output: OutputT
    usage: Any  # PydanticAI RunUsage
    request_body: dict[str, Any] | None  # last captured request body, when capture is on
    raw: Any  # the underlying AgentRunResult


class StructuredAgent:
    """A constrained agent: a profile + the `pydantic_ai.Agent` built from it."""

    def __init__(
        self,
        profile: AgentProfile,
        agent: Agent[None, Any],
        *,
        capture: RequestCapture | None = None,
    ) -> None:
        self.profile = profile
        self._agent = agent
        self._capture = capture

    @property
    def agent(self) -> Agent[None, Any]:
        """The underlying `pydantic_ai.Agent` (escape hatch)."""
        return self._agent

    def _result(self, raw: Any, request_body: dict[str, Any] | None) -> AgentResult[Any]:
        return AgentResult(output=raw.output, usage=raw.usage, request_body=request_body, raw=raw)

    async def run(self, prompt: str, **kwargs: Any) -> AgentResult[Any]:
        """Run the agent and wrap the result; kwargs pass through to `Agent.run`."""
        if self._capture is None:
            return self._result(await self._agent.run(prompt, **kwargs), None)
        # Correlate this run's captured request via a per-run contextvar sink (robust under
        # same-agent concurrency), rather than reading the shared capture's `.last`.
        sink: list[RequestRecord] = []
        token = _run_sink.set(sink)
        try:
            raw = await self._agent.run(prompt, **kwargs)
        finally:
            _run_sink.reset(token)
        return self._result(raw, sink[-1].body if sink else None)

    def run_sync(self, prompt: str, **kwargs: Any) -> AgentResult[Any]:
        """Synchronous variant of `run`."""
        if self._capture is None:
            return self._result(self._agent.run_sync(prompt, **kwargs), None)
        sink: list[RequestRecord] = []
        token = _run_sink.set(sink)
        try:
            raw = self._agent.run_sync(prompt, **kwargs)
        finally:
            _run_sink.reset(token)
        return self._result(raw, sink[-1].body if sink else None)
