"""Durable authorization decisions and exactly-once effects.

``Effector.run`` remains directly callable for advanced use.  ``execute`` is
the blessed path: it evaluates policy before entering the durable effect
workflow, so a denial is data and cannot run an effect.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from dbos import DBOS, SetWorkflowID
from pydantic import BaseModel, TypeAdapter

from .errors import AuthorityError


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class Denied:
    reason: str
    command: Any


@runtime_checkable
class Authorizer[C](Protocol):
    def decide(self, command: C) -> Decision: ...


@runtime_checkable
class Effector[C, R](Protocol):
    async def run(self, command: C) -> R: ...


class Allowlist[C]:
    """Default-deny policy made from named allow rules."""

    def __init__(self, rules: dict[str, Callable[[C], bool]]) -> None:
        self._rules = rules

    def decide(self, command: C) -> Decision:
        for name, rule in self._rules.items():
            try:
                if rule(command):
                    return Decision(True, name)
            except Exception as exc:
                return Decision(False, f"allow rule {name!r} failed: {exc}")
        return Decision(False, "no allow rule matched")


@dataclass(frozen=True)
class _AllOf[C]:
    authorizers: tuple[Authorizer[C], ...]

    def decide(self, command: C) -> Decision:
        for authorizer in self.authorizers:
            decision = authorizer.decide(command)
            if not decision.allowed:
                return decision
        return Decision(True)


@dataclass(frozen=True)
class _AnyOf[C]:
    authorizers: tuple[Authorizer[C], ...]

    def decide(self, command: C) -> Decision:
        reasons: list[str] = []
        for authorizer in self.authorizers:
            decision = authorizer.decide(command)
            if decision.allowed:
                return decision
            if decision.reason:
                reasons.append(decision.reason)
        return Decision(False, "; ".join(reasons) or "no authorizer allowed the command")


def all_of[C](*authorizers: Authorizer[C]) -> Authorizer[C]:
    return _AllOf(authorizers)


def any_of[C](*authorizers: Authorizer[C]) -> Authorizer[C]:
    return _AnyOf(authorizers)


class Null[C]:
    """A durable dry-run that records intent without an external effect."""

    @DBOS.step()
    async def run(self, command: C) -> None:
        del command


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


_argv_adapter = TypeAdapter(list[str])


def validated_argv(command: BaseModel) -> tuple[str, ...]:
    """Extract a non-empty, already validated argv field without shell parsing."""
    data = command.model_dump(mode="python")
    try:
        argv = tuple(_argv_adapter.validate_python(data["argv"], strict=True))
    except (KeyError, ValueError) as exc:
        raise AuthorityError("command must contain a validated non-empty argv") from exc
    if not argv:
        raise AuthorityError("command argv must not be empty")
    return argv


class Subprocess:
    """Run a validated argv as a durable step, never through a shell."""

    @DBOS.step()
    async def run(self, command: BaseModel) -> ProcessResult:
        completed = await asyncio.to_thread(
            subprocess.run,
            validated_argv(command),
            capture_output=True,
            check=False,
            text=True,
        )
        return ProcessResult(completed.returncode, cast(str, completed.stdout), cast(str, completed.stderr))


@DBOS.workflow(name="structured_agents.execute")
async def _run_effect[C, R](effector: Effector[C, R], command: C) -> R:
    return await effector.run(command)


async def execute[C, R](
    authorizer: Authorizer[C], effector: Effector[C, R], command: C, *, key: str | None = None
) -> Denied | R:
    """Authorize a command, then run its durable effect once per optional business key."""
    decision = authorizer.decide(command)
    if not decision.allowed:
        return Denied(decision.reason, command)
    if key is None:
        return await _run_effect(effector, command)
    with SetWorkflowID(key):
        return await _run_effect(effector, command)
