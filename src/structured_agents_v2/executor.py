"""`Executor` — the authority boundary between a validated command and a real side effect.

The rest of the library guarantees only that an agent's output is *well-formed* (XGrammar
constrains the syntax; Pydantic validates it into a typed command). It says nothing about whether
that command is *allowed* or what running it does. An `Executor` is the explicit seam where that
second question lives: `authorize(policy, command)` decides, `execute(policy, command)` performs.

**Nothing here is ever called implicitly.** Generation never triggers an effect; the only place a
side effect happens is an `execute()`/`run()` (or `AgentSet.route_and_execute`) call *you* make.

Authority is policy, not necessarily a human. A `Policy` bundles an authorization rule (`allow`) and
an action (`action`); "human-in-the-loop" is one kind of policy, **automatic** is another:

- `DryRunExecutor` — the safe default: authorizes, but performs **no** side effect (records what it
  *would* do). Use it to watch a whole router → specialist → execute flow with nothing happening.
- `AllowlistExecutor` — the autonomous one: **default-deny**, but auto-approves and runs any command
  matching its policy's `allow` rule, with **no human in the loop** — the allowlist is the safety net.

Worked example (router → specialist → executor; the same flow `AgentSet.route_and_execute` automates).
The command is a **json_schema-mode** `ConstrainedOutput` so it arrives as a validated model with
a `.value` field (regex/choice-mode agents return a bare `str` instead — see `ConstrainedOutput`)::

    from structured_agents_v2 import AllowlistExecutor, Policy

    def run_git(cmd):  # the real side effect; cmd.value is a validated string like "git status"
        return subprocess.run(cmd.value.split(), capture_output=True, text=True).stdout

    def is_safe(c) -> bool:                                # length-guarded so bare "git" denies, not raises
        parts = c.value.split()
        return len(parts) > 1 and parts[1] in {"status", "diff", "log"}

    executor = AllowlistExecutor([
        Policy("git_safe_v1", allow=is_safe, action=run_git),
    ])
    routed = await fleet.route_and_run(user_msg)          # validated GitCommand — an intention
    decision = executor.authorize("git_safe_v1", routed.output)
    if decision.allowed:
        executor.execute("git_safe_v1", routed.output)    # the only line that *does* anything
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .errors import PolicyError


@dataclass(frozen=True)
class Decision:
    """The authority verdict for one command under one policy."""

    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ExecResult:
    """The outcome of performing (or declining to perform) a command."""

    ok: bool
    output: Any = None  # whatever the policy's action returned
    detail: str = ""  # human-readable note
    dry_run: bool = False  # True when no real side effect happened


@dataclass
class Policy:
    """A named authority rule + the action it gates. The unit an executor is configured with."""

    name: str
    allow: Callable[[BaseModel], bool] | None = None  # authorization rule (semantics per executor)
    action: Callable[[BaseModel], Any] | None = None  # what execute() performs; None => no-op
    description: str = ""


@runtime_checkable
class Executor(Protocol):
    """The authority boundary: decide if a command is allowed, then perform it."""

    def authorize(self, policy: str, command: BaseModel) -> Decision: ...
    def execute(self, policy: str, command: BaseModel) -> ExecResult: ...


class BaseExecutor:
    """Shared policy registry + the `run()` (authorize-then-execute) convenience."""

    def __init__(self, policies: Iterable[Policy] = ()) -> None:
        self._policies: dict[str, Policy] = {p.name: p for p in policies}

    def policy(self, name: str) -> Policy:
        """Look up a registered policy, or raise `PolicyError` (unknown policy = default-deny)."""
        try:
            return self._policies[name]
        except KeyError:
            raise PolicyError(f"unknown policy {name!r}") from None

    def authorize(self, policy: str, command: BaseModel) -> Decision:  # pragma: no cover - overridden
        raise NotImplementedError

    def execute(self, policy: str, command: BaseModel) -> ExecResult:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self, policy: str, command: BaseModel) -> ExecResult:
        """Authorize then execute in one call; raise `PolicyError` if the command is denied."""
        decision = self.authorize(policy, command)
        if not decision.allowed:
            raise PolicyError(f"{policy}: {decision.reason or 'denied by policy'}")
        return self.execute(policy, command)


class DryRunExecutor(BaseExecutor):
    """Authorizes per policy but performs **no** side effect — records intended executions in `.log`.

    The safe default: with `default_allow=True` it authorizes everything (so you can watch a full
    flow), yet `execute` never runs a policy's `action`. A policy's `allow` rule, if set, is still
    honored so refusals are visible.
    """

    def __init__(self, policies: Iterable[Policy] = (), *, default_allow: bool = True) -> None:
        super().__init__(policies)
        self.default_allow = default_allow
        self.log: list[tuple[str, BaseModel]] = []

    def authorize(self, policy: str, command: BaseModel) -> Decision:
        p = self._policies.get(policy)
        if p is None or p.allow is None:
            return Decision(self.default_allow, "" if self.default_allow else "no allow-rule (default deny)")
        ok = _safe_allow(p, command)
        if ok is None:
            return Decision(False, f"allow rule raised for policy {policy!r} (fail closed)")
        return Decision(ok, "" if ok else f"command not permitted by policy {policy!r}")

    def execute(self, policy: str, command: BaseModel) -> ExecResult:
        self.log.append((policy, command))
        return ExecResult(ok=True, detail=f"dry-run: would execute {policy!r}", dry_run=True)


class AllowlistExecutor(BaseExecutor):
    """Auto-approves commands matching a policy's `allow` rule and runs its `action` — **default-deny**.

    The autonomous executor: an unknown policy or a policy with no `allow` rule denies; otherwise the
    rule decides, with no human in the loop. The allowlist is the safety net.

    Unknown-policy handling differs from `DryRunExecutor` by design:

    | executor          | unknown policy        | no allow-rule    | allow-rule raises |
    |-------------------|-----------------------|------------------|-------------------|
    | `DryRunExecutor`  | allow (default_allow) | allow (default)  | deny (fail closed)|
    | `AllowlistExecutor`| `PolicyError`        | deny             | deny (fail closed)|
    """

    def authorize(self, policy: str, command: BaseModel) -> Decision:
        p = self.policy(policy)  # unknown policy -> PolicyError (fail closed)
        if p.allow is None:
            return Decision(False, f"policy {policy!r} has no allow-rule (default deny)")
        ok = _safe_allow(p, command)
        if ok is None:
            return Decision(False, f"allow rule raised for policy {policy!r} (fail closed)")
        return Decision(ok, "" if ok else f"command not permitted by policy {policy!r}")

    def execute(self, policy: str, command: BaseModel) -> ExecResult:
        # B1: execute re-checks authority so a caller who skips authorize() can't bypass the
        # default-deny safety net. authorize is a local callable; double-evaluating it is cheap.
        decision = self.authorize(policy, command)
        if not decision.allowed:
            raise PolicyError(f"{policy}: {decision.reason or 'denied by policy'}")
        p = self.policy(policy)
        if p.action is None:
            return ExecResult(ok=True, detail=f"policy {policy!r} has no action (no-op)")
        try:
            out = p.action(command)
        except Exception as exc:  # noqa: BLE001 - surface action failure as data (ok=False)
            return ExecResult(ok=False, detail=f"action for {policy!r} raised: {exc!r}")
        return ExecResult(ok=True, output=out, detail=f"executed {policy!r}")


def _safe_allow(p: Policy, command: BaseModel) -> bool | None:
    """Evaluate a policy's allow rule, fail-closed on exception.

    Returns the boolean verdict, or ``None`` if the rule raised — an authority boundary must
    never let a buggy/throwing rule crash the pipeline; a raise is treated as a denial.
    """
    if p.allow is None:  # pragma: no cover - callers guard this
        return False
    try:
        return bool(p.allow(command))
    except Exception:  # noqa: BLE001 - authority boundary: any rule failure denies
        return None
