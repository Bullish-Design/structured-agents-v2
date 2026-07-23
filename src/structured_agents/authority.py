"""Durable authorization decisions and retry-aware effects.

``Effector.run`` remains directly callable for advanced use.  ``execute`` is
the blessed path: it evaluates policy before entering the durable effect
workflow, so a denial is data and cannot run an effect.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
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
    if not authorizers:
        raise AuthorityError("all_of requires at least one authorizer")
    return _AllOf(authorizers)


def any_of[C](*authorizers: Authorizer[C]) -> Authorizer[C]:
    if not authorizers:
        raise AuthorityError("any_of requires at least one authorizer")
    return _AnyOf(authorizers)


class AuthorityMode(StrEnum):
    ENFORCE = "enforce"
    AUTOMATED = "automated"
    PERMIT_ALL = "permit_all"
    BYPASS = "bypass"


class DecisionKind(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"
    REQUIRES_HUMAN = "requires_human"
    ERROR = "error"
    BYPASSED = "bypassed"


@dataclass(frozen=True)
class CommandBinding[C: BaseModel]:
    command: C
    subject: str
    action: str
    scope: str
    digest: str

    @classmethod
    def create(cls, command: C, *, subject: str, action: str, scope: str) -> CommandBinding[C]:
        if not subject or not action or not scope:
            raise AuthorityError("command binding subject, action, and scope must be non-empty")
        canonical = json.dumps(
            {
                "action": action,
                "command": command.model_dump(mode="json"),
                "scope": scope,
                "subject": subject,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return cls(command, subject, action, scope, hashlib.sha256(canonical).hexdigest())

    def validate(self) -> None:
        expected = type(self).create(self.command, subject=self.subject, action=self.action, scope=self.scope).digest
        if not self.digest or not hmac.compare_digest(self.digest, expected):
            raise AuthorityError("command binding digest does not match the command")


@dataclass(frozen=True)
class AuthorityRequest:
    mode: AuthorityMode
    actor: str
    scope: str
    bypass_reason: str = ""
    bypass_expires_at: datetime | None = None


@dataclass(frozen=True)
class ApprovalEvidence:
    kind: DecisionKind
    request_digest: str
    approver: str
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.kind in {DecisionKind.ALLOW, DecisionKind.BYPASSED}


@runtime_checkable
class AutomatedAuthorizer[C](Protocol):
    def decide(self, command: C) -> Decision | ApprovalEvidence | Any: ...


async def authorize[C: BaseModel](
    binding: CommandBinding[C],
    request: AuthorityRequest,
    authorizer: Authorizer[C] | AutomatedAuthorizer[C] | None = None,
    *,
    now: datetime | None = None,
) -> ApprovalEvidence:
    """Resolve explicit authority mode against an exact command binding."""
    binding.validate()
    if not request.actor or request.scope != binding.scope:
        return ApprovalEvidence(DecisionKind.DENY, binding.digest, request.actor, "authority scope mismatch")

    if request.mode is AuthorityMode.PERMIT_ALL:
        return ApprovalEvidence(DecisionKind.ALLOW, binding.digest, request.actor, "explicit permit-all policy")
    if request.mode is AuthorityMode.BYPASS:
        current = now or datetime.now(UTC)
        expires = request.bypass_expires_at
        if not request.bypass_reason:
            return ApprovalEvidence(DecisionKind.DENY, binding.digest, request.actor, "bypass reason required")
        if expires is None or expires.tzinfo is None or expires <= current:
            return ApprovalEvidence(DecisionKind.DENY, binding.digest, request.actor, "bypass is expired or unbounded")
        return ApprovalEvidence(DecisionKind.BYPASSED, binding.digest, request.actor, request.bypass_reason)
    if authorizer is None:
        return ApprovalEvidence(DecisionKind.ERROR, binding.digest, request.actor, "authorizer required")

    try:
        result = authorizer.decide(binding.command)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        return ApprovalEvidence(DecisionKind.ERROR, binding.digest, request.actor, f"authorizer failed: {exc}")
    if isinstance(result, ApprovalEvidence):
        if result.request_digest != binding.digest:
            return ApprovalEvidence(DecisionKind.ERROR, binding.digest, request.actor, "decision binding mismatch")
        return result
    if isinstance(result, Decision):
        kind = DecisionKind.ALLOW if result.allowed else DecisionKind.DENY
        return ApprovalEvidence(kind, binding.digest, request.actor, result.reason)
    return ApprovalEvidence(DecisionKind.ERROR, binding.digest, request.actor, "invalid authorizer decision")


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
    """Authorize a command, then run a durable, potentially retried effect.

    A completed DBOS step is replayed from storage. An external effect can still
    be attempted more than once if the process crashes after the remote commit
    but before DBOS records step completion, so effectors need their own
    idempotency, transaction/outbox, reconciliation, or compensation protocol.
    """
    decision = authorizer.decide(command)
    if not decision.allowed:
        return Denied(decision.reason, command)
    if key is None:
        return await _run_effect(effector, command)
    with SetWorkflowID(key):
        return await _run_effect(effector, command)
