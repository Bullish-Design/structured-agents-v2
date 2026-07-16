"""Executor authority boundary: DryRunExecutor (no-op) + AllowlistExecutor (auto, default-deny)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_agents_v2 import (
    AllowlistExecutor,
    Decision,
    DryRunExecutor,
    ExecResult,
    Executor,
    Policy,
)
from structured_agents_v2.errors import PolicyError


class GitCommand(BaseModel):
    value: str


_SAFE = {"status", "diff", "log"}


def _git_policy(sink: list[str]) -> Policy:
    """A policy that allows read-only git verbs and 'runs' them by appending to a sink."""
    return Policy(
        name="git_safe_v1",
        allow=lambda c: c.value.split()[1] in _SAFE if c.value.startswith("git ") else False,
        action=lambda c: sink.append(c.value) or f"ran: {c.value}",
    )


# --- AllowlistExecutor: the autonomous, default-deny executor ---------------------------


def test_allowlist_allows_and_executes_matching_command() -> None:
    sink: list[str] = []
    ex = AllowlistExecutor([_git_policy(sink)])
    cmd = GitCommand(value="git status .")

    decision = ex.authorize("git_safe_v1", cmd)
    assert decision.allowed
    result = ex.execute("git_safe_v1", cmd)
    assert result.ok and result.output == "ran: git status ."
    assert sink == ["git status ."]  # the side effect actually happened


def test_allowlist_denies_unlisted_command_and_run_raises() -> None:
    sink: list[str] = []
    ex = AllowlistExecutor([_git_policy(sink)])
    cmd = GitCommand(value="git push --force")

    decision = ex.authorize("git_safe_v1", cmd)
    assert not decision.allowed
    with pytest.raises(PolicyError, match="not permitted"):
        ex.run("git_safe_v1", cmd)
    assert sink == []  # denied -> action never ran


def test_allowlist_unknown_policy_fails_closed() -> None:
    ex = AllowlistExecutor([_git_policy([])])
    with pytest.raises(PolicyError, match="unknown policy"):
        ex.authorize("does_not_exist", GitCommand(value="git status"))


def test_allowlist_policy_without_rule_denies() -> None:
    ex = AllowlistExecutor([Policy(name="p")])  # no allow-rule -> default deny
    assert ex.authorize("p", GitCommand(value="git status")).allowed is False


def test_allowlist_action_optional_is_noop() -> None:
    ex = AllowlistExecutor([Policy(name="p", allow=lambda _c: True)])  # allowed, but no action
    result = ex.execute("p", GitCommand(value="anything"))
    assert result.ok and result.output is None and "no action" in result.detail


def test_run_returns_execresult_when_allowed() -> None:
    sink: list[str] = []
    ex = AllowlistExecutor([_git_policy(sink)])
    result = ex.run("git_safe_v1", GitCommand(value="git diff HEAD"))
    assert isinstance(result, ExecResult) and result.ok
    assert sink == ["git diff HEAD"]


def test_allowlist_execute_rechecks_authority() -> None:
    # B1: execute() must fail closed on its own, even if the caller skips authorize().
    sink: list[str] = []
    ex = AllowlistExecutor([_git_policy(sink)])
    with pytest.raises(PolicyError, match="not permitted"):
        ex.execute("git_safe_v1", GitCommand(value="git push --force"))
    assert sink == []  # the action never ran


def test_allowlist_raising_allow_rule_fails_closed() -> None:
    # B2: a rule that raises must deny, not crash the pipeline.
    ex = AllowlistExecutor([Policy(name="p", allow=lambda c: c.value.split()[1] == "status")])
    decision = ex.authorize("p", GitCommand(value="git"))  # split()[1] -> IndexError inside the rule
    assert decision.allowed is False
    assert "fail closed" in decision.reason


def test_allowlist_action_exception_is_data_not_raise() -> None:
    # B5: an action that raises surfaces as ExecResult(ok=False), not an exception.
    def _boom(_c: BaseModel) -> str:
        raise RuntimeError("kaboom")

    ex = AllowlistExecutor([Policy(name="p", allow=lambda _c: True, action=_boom)])
    result = ex.execute("p", GitCommand(value="x"))
    assert result.ok is False
    assert "kaboom" in result.detail


# --- DryRunExecutor: authorizes but never performs a side effect ------------------------


def test_dry_run_raising_allow_rule_fails_closed() -> None:
    # B2 (DryRunExecutor path): a raising rule denies here too.
    ex = DryRunExecutor([Policy(name="p", allow=lambda c: c.value.split()[1] == "status")])
    decision = ex.authorize("p", GitCommand(value="git"))
    assert decision.allowed is False
    assert "fail closed" in decision.reason


def test_dry_run_authorizes_but_does_not_execute() -> None:
    sink: list[str] = []
    ex = DryRunExecutor([_git_policy(sink)])
    cmd = GitCommand(value="git status .")

    assert ex.authorize("git_safe_v1", cmd).allowed  # default_allow + rule both say yes
    result = ex.execute("git_safe_v1", cmd)
    assert result.ok and result.dry_run
    assert sink == []  # NO side effect, even though the policy has an action
    assert ex.log == [("git_safe_v1", cmd)]  # but the intent was recorded


def test_dry_run_honors_a_deny_rule() -> None:
    ex = DryRunExecutor([_git_policy([])])
    assert ex.authorize("git_safe_v1", GitCommand(value="git push origin")).allowed is False


def test_dry_run_default_allow_for_unknown_policy() -> None:
    ex = DryRunExecutor()  # no policies, default_allow=True
    assert ex.authorize("whatever", GitCommand(value="x")).allowed is True
    # default_allow=False flips it (so you can dry-run in fail-closed mode)
    assert DryRunExecutor(default_allow=False).authorize("whatever", GitCommand(value="x")).allowed is False


# --- Protocol -------------------------------------------------------------------------


def test_executors_satisfy_the_runtime_checkable_protocol() -> None:
    assert isinstance(DryRunExecutor(), Executor)
    assert isinstance(AllowlistExecutor(), Executor)


def test_decision_and_execresult_are_frozen() -> None:
    d = Decision(allowed=True)
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        d.allowed = False  # type: ignore[misc]
