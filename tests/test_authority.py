from __future__ import annotations

import pytest
from dbos import DBOS
from pydantic import BaseModel

from structured_agents.authority import (
    Allowlist,
    Decision,
    Denied,
    ProcessResult,
    Subprocess,
    all_of,
    any_of,
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


async def test_execute_is_exactly_once_per_business_key() -> None:
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
