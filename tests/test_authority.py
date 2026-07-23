from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from dbos import DBOS
from pydantic import BaseModel

from structured_agents.authority import (
    Allowlist,
    ApprovalEvidence,
    AuthorityMode,
    AuthorityRequest,
    CommandBinding,
    Decision,
    DecisionKind,
    Denied,
    ProcessResult,
    Subprocess,
    all_of,
    any_of,
    authorize,
    execute,
)


class Command(BaseModel):
    argv: list[str]


class StaticAuthorizer:
    def __init__(self, allowed: bool, reason: str = "policy") -> None:
        self.decision = Decision(allowed, reason)

    def decide(self, command: Command) -> Decision:
        del command
        return self.decision


effects = 0


class CountingEffector:
    @DBOS.step()
    async def run(self, command: Command) -> int:
        global effects
        del command
        effects += 1
        return effects


class FailingEffector:
    @DBOS.step(retries_allowed=False)
    async def run(self, command: Command) -> int:
        del command
        raise RuntimeError("effect failed")


async def test_denial_precedes_effect() -> None:
    global effects
    effects = 0
    result = await execute(StaticAuthorizer(False, "not approved"), CountingEffector(), Command(argv=["true"]))
    assert result == Denied("not approved", Command(argv=["true"]))
    assert effects == 0


def test_allowlist_fails_closed_and_composes() -> None:
    command = Command(argv=["git"])

    def indexes_second(value: Command) -> bool:
        return value.argv[1] == "status"

    def is_git(value: Command) -> bool:
        return value.argv == ["git"]

    def is_false(value: Command) -> bool:
        return value.argv == ["false"]

    raising = Allowlist[Command]({"argv-index": indexes_second})
    assert raising.decide(command).allowed is False
    assert "failed" in raising.decide(command).reason

    allow = Allowlist[Command]({"git": is_git})
    deny = Allowlist[Command]({"never": is_false})
    assert all_of(allow, deny).decide(command).allowed is False
    assert any_of(deny, allow).decide(command).allowed is True
    with pytest.raises(Exception, match="at least one"):
        all_of()
    with pytest.raises(Exception, match="at least one"):
        any_of()


async def test_completed_effect_is_replayed_per_business_key() -> None:
    global effects
    effects = 0
    allowed = StaticAuthorizer(True)
    command = Command(argv=["true"])
    effector = CountingEffector()
    assert await execute(allowed, effector, command, key="order-42") == 1
    assert await execute(allowed, effector, command, key="order-42") == 1
    assert effects == 1
    assert await execute(allowed, effector, command, key="order-99") == 2
    assert effects == 2


async def test_effect_failures_surface_and_subprocess_returns_result() -> None:
    command = Command(argv=["true"])
    with pytest.raises(RuntimeError, match="effect failed"):
        await execute(StaticAuthorizer(True), FailingEffector(), command, key="failure-1")
    assert await execute(StaticAuthorizer(True), Subprocess(), command, key="success-1") == ProcessResult(0, "", "")


class AutomatedApproval:
    def __init__(self, evidence: ApprovalEvidence) -> None:
        self.evidence = evidence

    async def decide(self, command: Command) -> ApprovalEvidence:
        del command
        return self.evidence


async def test_automated_approval_requires_exact_command_binding() -> None:
    binding = CommandBinding.create(
        Command(argv=["deploy", "staging"]), subject="release", action="deploy", scope="staging"
    )
    request = AuthorityRequest(AuthorityMode.AUTOMATED, actor="approver-agent", scope="staging")
    valid = ApprovalEvidence(DecisionKind.ALLOW, binding.digest, "approver-agent", "checks passed")

    assert (await authorize(binding, request, AutomatedApproval(valid))).allowed

    wrong = ApprovalEvidence(DecisionKind.ALLOW, "0" * 64, "approver-agent", "wrong command")
    result = await authorize(binding, request, AutomatedApproval(wrong))
    assert result.kind is DecisionKind.ERROR
    assert not result.allowed


async def test_scoped_bypass_is_explicit_bounded_and_bound() -> None:
    now = datetime(2026, 7, 21, tzinfo=UTC)
    binding = CommandBinding.create(
        Command(argv=["deploy", "staging"]), subject="release", action="deploy", scope="staging"
    )
    request = AuthorityRequest(
        AuthorityMode.BYPASS,
        actor="operator",
        scope="staging",
        bypass_reason="incident recovery",
        bypass_expires_at=now + timedelta(minutes=5),
    )

    evidence = await authorize(binding, request, now=now)
    assert evidence.kind is DecisionKind.BYPASSED
    assert evidence.request_digest == binding.digest

    wrong_scope = AuthorityRequest(
        AuthorityMode.BYPASS,
        actor="operator",
        scope="production",
        bypass_reason="incident recovery",
        bypass_expires_at=now + timedelta(minutes=5),
    )
    assert not (await authorize(binding, wrong_scope, now=now)).allowed

    expired = AuthorityRequest(
        AuthorityMode.BYPASS,
        actor="operator",
        scope="staging",
        bypass_reason="incident recovery",
        bypass_expires_at=now,
    )
    assert not (await authorize(binding, expired, now=now)).allowed

    tampered = CommandBinding(
        Command(argv=["deploy", "production"]), binding.subject, binding.action, binding.scope, binding.digest
    )
    with pytest.raises(Exception, match="digest"):
        await authorize(tampered, request, now=now)
