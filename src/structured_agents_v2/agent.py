"""`StructuredAgent` — the runnable wrapper around a built `pydantic_ai.Agent`.

A `StructuredAgent` is produced by `Backend.build(profile)`; it owns the configured
`pydantic_ai.Agent` and exposes a lean `run`/`run_sync` returning an `AgentResult`. The
underlying agent stays reachable via `.agent` as an escape hatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .capture import _run_sink
from .errors import ConstraintViolationError

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from .capture import RequestCapture, RequestRecord
    from .decoder import DecoderSpec
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
        spec: DecoderSpec | None = None,
        capture: RequestCapture | None = None,
    ) -> None:
        self.profile = profile
        self._agent = agent
        self._spec = spec
        self._capture = capture

    @property
    def agent(self) -> Agent[None, Any]:
        """The underlying `pydantic_ai.Agent` (escape hatch)."""
        return self._agent

    def _guard(self, output: Any) -> Any:
        """Client-side check that a bare-string output satisfies its declared constraint.

        Closes the "backend silently not enforcing" hole: against a mis-capped backend, a
        frontier API, or a test mock that ignores `extra_body`, unconstrained text would
        otherwise flow straight into an executor. For `regex`/`choice` modes this verifies
        the returned `str` and raises `ConstraintViolationError` on a mismatch.

        `grammar` mode is NOT client-checkable without xgrammar, so it is trusted here
        (server-side enforcement only); `json_schema` mode already returns a validated model.
        """
        spec = self._spec
        if spec is None or not isinstance(output, str):
            return output
        if spec.mode == "regex":
            if re.fullmatch(spec.regex or "", output) is None:
                raise ConstraintViolationError(
                    f"{self.profile.name!r}: output does not match declared regex "
                    f"(is the backend actually enforcing extra_body constraints?)"
                )
        elif spec.mode == "choice":
            if output not in (spec.choices or []):
                raise ConstraintViolationError(
                    f"{self.profile.name!r}: output {output!r} not in declared choices."
                )
        return output

    def _result(self, raw: Any, request_body: dict[str, Any] | None) -> AgentResult[Any]:
        return AgentResult(output=self._guard(raw.output), usage=raw.usage, request_body=request_body, raw=raw)

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
