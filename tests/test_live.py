"""Live vLLM cutover checks; skipped unless the operator explicitly opts in."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import pytest
from dbos import DBOS
from pydantic import BaseModel

from structured_agents import (
    AgentSpec,
    Allowlist,
    Backend,
    Choice,
    Decision,
    Grammar,
    Regex,
    Schema,
    Settings,
    execute,
)

pytestmark = pytest.mark.live

if os.environ.get("SAV_LIVE") != "1":
    pytest.skip("set SAV_LIVE=1 to contact the configured vLLM endpoint", allow_module_level=True)


BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("LLM_API_KEY", "sk-none")
MODEL = os.environ.get("LLM_MODEL", "base")
LORA_NAME = os.environ.get("LORA_NAME")
SETTINGS = Settings(temperature=0, seed=7, max_tokens=32)


class LiveCommand(BaseModel):
    argv: tuple[Literal["echo"], Literal["phase7-live"]]


backend = Backend(base_url=BASE_URL, api_key=API_KEY, default_model=MODEL)
schema_agent = backend.build(AgentSpec(
    "live-schema", Schema(LiveCommand), "Return exactly the argv requested by the user.", settings=SETTINGS,
))
regex_agent = backend.build(AgentSpec(
    "live-regex", Regex(r"phase7-[0-9]{4}"), "Return a phase7 code with exactly four digits.", settings=SETTINGS,
))
choice_agent = backend.build(AgentSpec(
    "live-choice", Choice("phase7-allow", "phase7-deny"), "Choose phase7-allow.", settings=SETTINGS,
))
grammar_agent = backend.build(AgentSpec(
    "live-grammar", Grammar('root ::= "phase7-grammar"'), "Return phase7-grammar.", settings=SETTINGS,
))

effect_calls = 0


class LiveCountingEffector:
    @DBOS.step()
    async def run(self, command: LiveCommand) -> int:
        global effect_calls
        assert command.argv == ("echo", "phase7-live")
        effect_calls += 1
        return effect_calls


class LiveAuthorizer:
    def decide(self, command: LiveCommand) -> Decision:
        return Allowlist[LiveCommand]({"phase7": lambda value: value.argv == ("echo", "phase7-live")}).decide(command)


@DBOS.workflow()
async def live_durable_pipeline() -> int:
    command = await schema_agent.run("Return argv [\"echo\", \"phase7-live\"] exactly.")
    result = await execute(LiveAuthorizer(), LiveCountingEffector(), command, key="phase7-live-effect")
    assert isinstance(result, int)
    return result


async def test_live_health_and_model_identity() -> None:
    root_url = BASE_URL.removesuffix("/v1")
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    async with httpx.AsyncClient(timeout=30) as client:
        health = await client.get(f"{root_url}/health", headers=headers)
        models = await client.get(f"{BASE_URL}/models", headers=headers)
    assert health.status_code == 200
    assert models.status_code == 200
    model_ids = {item["id"] for item in models.json()["data"]}
    assert MODEL in model_ids
    if LORA_NAME:
        assert LORA_NAME in model_ids


async def test_live_schema_constraint() -> None:
    command = await schema_agent.run("Return argv [\"echo\", \"phase7-live\"] exactly.")
    assert command == LiveCommand(argv=("echo", "phase7-live"))


async def test_live_regex_constraint() -> None:
    assert await regex_agent.run("Return phase7-2048.") == "phase7-2048"


async def test_live_choice_constraint() -> None:
    assert await choice_agent.run("Choose phase7-allow.") == "phase7-allow"


async def test_live_grammar_constraint() -> None:
    assert await grammar_agent.run("Return phase7-grammar.") == "phase7-grammar"


async def test_live_lora_constraint() -> None:
    if not LORA_NAME:
        pytest.skip("set LORA_NAME when the tower exposes a LoRA adapter")
    agent = backend.build(AgentSpec(
        "live-lora", Choice("phase7-lora"), "Return phase7-lora.", adapter=LORA_NAME, settings=SETTINGS,
    ))
    assert await agent.run("Return phase7-lora.") == "phase7-lora"


async def test_live_durable_pipeline_executes_keyed_effect_once() -> None:
    global effect_calls
    effect_calls = 0
    assert await live_durable_pipeline() == 1
    assert await live_durable_pipeline() == 1
    assert effect_calls == 1


async def test_live_durable_workflow_recovers_after_worker_crash() -> None:
    """Prove DBOS recovery in a fresh worker process, never by restarting vLLM."""
    if os.environ.get("SAV_LIVE_CRASH") != "1":
        pytest.skip("set SAV_LIVE_CRASH=1 to run the destructive test-owned worker crash proof")

    artifact_root = Path(".scratch/projects/10-durable-agent-plane-build/artifacts")
    artifact = artifact_root / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-phase7-dbos-worker-crash-recovery"
    artifact.mkdir(parents=True)
    database = artifact / "dbos.sqlite"
    effects = artifact / "effects.log"
    worker = Path("tests/live_crash_worker.py")
    environment = {
        "LLM_BASE_URL": BASE_URL,
        "LLM_API_KEY": API_KEY,
        "LLM_MODEL": MODEL,
        "SAV_PHASE7_ARTIFACT": str(artifact.resolve()),
    }
    (artifact / "command.txt").write_text(
        "SAV_LIVE=1 SAV_LIVE_CRASH=1 devenv shell -- pytest -x "
        "tests/test_live.py::test_live_durable_workflow_recovers_after_worker_crash\n"
    )
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "git-status-before.txt").write_text(
        subprocess.run(["git", "status", "--short"], check=True, text=True, capture_output=True).stdout
    )
    (artifact / "versions.txt").write_text(
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.metadata, sys; print(sys.version); "
                "print('dbos', importlib.metadata.version('dbos')); "
                "print('pydantic-ai-slim', importlib.metadata.version('pydantic-ai-slim'))",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )

    async def start_worker(mode: str) -> asyncio.subprocess.Process:
        command = [sys.executable, str(worker), mode, str(database), str(effects), "phase7-worker-crash"]
        (artifact / f"worker-{mode}-command.txt").write_text(" ".join(command) + "\n")
        return await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **environment},
        )

    first = await start_worker("start")
    assert first.stdout is not None
    ready = await asyncio.wait_for(first.stdout.readline(), timeout=120)
    assert ready, "initial DBOS worker exited before reaching its durable pause"
    ready_data = json.loads(ready)
    (artifact / "worker-start-before-kill.json").write_text(json.dumps(ready_data, indent=2) + "\n")
    assert ready_data == {"event": "pending", "status": "PENDING", "effect_lines": 1}
    assert effects.read_text().splitlines() == ["phase7-worker-crash"]

    first.kill()
    first_stdout, first_stderr = await asyncio.wait_for(first.communicate(), timeout=30)
    (artifact / "worker-start.stdout.txt").write_bytes(ready + first_stdout)
    (artifact / "worker-start.stderr.txt").write_bytes(first_stderr)
    (artifact / "worker-start-returncode.txt").write_text(f"{first.returncode}\n")
    assert first.returncode is not None and first.returncode < 0
    (artifact / "effect-before-replacement.txt").write_text(effects.read_text())

    replacement = await start_worker("resume")
    replacement_stdout, replacement_stderr = await asyncio.wait_for(replacement.communicate(), timeout=120)
    (artifact / "worker-resume.stdout.txt").write_bytes(replacement_stdout)
    (artifact / "worker-resume.stderr.txt").write_bytes(replacement_stderr)
    (artifact / "worker-resume-returncode.txt").write_text(f"{replacement.returncode}\n")
    assert replacement.returncode == 0, replacement_stderr.decode()
    recovered = json.loads(replacement_stdout)
    (artifact / "worker-after-recovery.json").write_text(json.dumps(recovered, indent=2) + "\n")
    assert recovered == {
        "event": "success",
        "status": "SUCCESS",
        "result": {"argv": ["echo", "phase7-live"]},
        "effect_lines": 1,
    }
    assert effects.read_text().splitlines() == ["phase7-worker-crash"]
    (artifact / "effect-after-recovery.txt").write_text(effects.read_text())
    (artifact / "git-status-after.txt").write_text(
        subprocess.run(["git", "status", "--short"], check=True, text=True, capture_output=True).stdout
    )
