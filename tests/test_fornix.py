from __future__ import annotations

import subprocess

import pytest
from pydantic import BaseModel

from structured_agents.authority import Allowlist, ProcessResult, execute
from structured_agents.errors import BackendCapabilityError
from structured_agents.integrations.fornix import FornixEffector


class Command(BaseModel):
    argv: list[str]


def allow(_: Command) -> bool:
    return True


async def test_fornix_uses_argv_and_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert "shell" not in kwargs
        return subprocess.CompletedProcess(argv, 0, '{"returncode": 0, "stdout": "ok", "stderr": ""}', "")

    monkeypatch.setattr("structured_agents.integrations.fornix.shutil.which", lambda _: "/bin/fornix")
    monkeypatch.setattr("structured_agents.integrations.fornix.subprocess.run", fake_run)
    result = await execute(
        Allowlist[Command]({"test": allow}), FornixEffector(), Command(argv=["git", "status"]), key="fornix-json"
    )
    assert result == ProcessResult(0, "ok", "")
    assert calls == [["/bin/fornix", "box", "--check", "--", "git", "status"]]


async def test_fornix_missing_binary_raises_capability_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("structured_agents.integrations.fornix.shutil.which", lambda _: None)
    with pytest.raises(BackendCapabilityError, match="fornix"):
        await execute(
            Allowlist[Command]({"test": allow}), FornixEffector(), Command(argv=["git"]), key="fornix-missing"
        )
