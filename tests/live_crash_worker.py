"""Test-owned DBOS worker used only by the opt-in Phase 7 crash proof."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal

import httpx
from dbos import DBOS, DBOSConfig, SetWorkflowID
from pydantic import BaseModel

from structured_agents import AgentSpec, Allowlist, Backend, Decision, Schema, Settings, execute


class LiveCommand(BaseModel):
    argv: tuple[Literal["echo"], Literal["phase7-live"]]


MODE, DATABASE_ARG, EFFECTS_ARG, WORKFLOW_ID = sys.argv[1:]
DATABASE = Path(DATABASE_ARG)
EFFECTS = Path(EFFECTS_ARG)
DBOS(config=DBOSConfig(
    name="phase7-worker-crash",
    system_database_url=f"sqlite:///{DATABASE}",
    use_listen_notify=False,
))


def line_count(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


class CountingEffector:
    @DBOS.step()
    async def run(self, command: LiveCommand) -> int:
        assert command.argv == ("echo", "phase7-live")
        with EFFECTS.open("a") as handle:
            handle.write(f"{WORKFLOW_ID}\n")
        return line_count(EFFECTS)


class Authorizer:
    def decide(self, command: LiveCommand) -> Decision:
        policy = Allowlist[LiveCommand]({
            "phase7": lambda value: value.argv == ("echo", "phase7-live"),
        })
        return policy.decide(command)


async def main() -> None:
    artifact = Path(os.environ["SAV_PHASE7_ARTIFACT"])
    raw_request = {
        "model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": "Return argv [\"echo\", \"phase7-live\"] exactly."}],
        "temperature": 0,
        "max_tokens": 32,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "phase7_worker_capture",
                "schema": {
                    "type": "object",
                    "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
                    "required": ["argv"],
                    "additionalProperties": False,
                },
            },
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        raw_response = await client.post(f"{os.environ['LLM_BASE_URL']}/chat/completions", json=raw_request)
    (artifact / "raw-vllm-request.json").write_text(json.dumps(raw_request, indent=2) + "\n")
    (artifact / "raw-vllm-response.json").write_text(raw_response.text + "\n")
    (artifact / "raw-vllm-status.txt").write_text(f"{raw_response.status_code}\n")
    raw_response.raise_for_status()
    settings = Settings(temperature=0, seed=7, max_tokens=32)
    backend = Backend(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        default_model=os.environ["LLM_MODEL"],
    )
    agent = backend.build(AgentSpec(
        "phase7-worker-crash-schema",
        Schema(LiveCommand),
        "Return exactly the argv requested by the user.",
        settings=settings,
    ))

    @DBOS.workflow(name="phase7.worker_crash")
    async def workflow() -> LiveCommand:
        command = await agent.run("Return argv [\"echo\", \"phase7-live\"] exactly.")
        assert command == LiveCommand(argv=("echo", "phase7-live"))
        result = await execute(Authorizer(), CountingEffector(), command, key="phase7-worker-crash-effect")
        assert result == 1
        await DBOS.set_event_async("constrained_output", command.model_dump(mode="json"))
        await DBOS.set_event_async("effect_completed", {"lines": line_count(EFFECTS)})
        message = await DBOS.recv_async("phase7-worker-crash")
        assert message == {"resume": True}
        return command

    DBOS.launch()
    try:
        if MODE == "start":
            with SetWorkflowID(WORKFLOW_ID):
                await DBOS.start_workflow_async(workflow)
            for _ in range(240):
                state = await DBOS.get_workflow_status_async(WORKFLOW_ID)
                if state is not None and state.status == "PENDING" and line_count(EFFECTS) == 1:
                    print(json.dumps({"event": "pending", "status": state.status, "effect_lines": 1}), flush=True)
                    await asyncio.Event().wait()
                await asyncio.sleep(0.25)
            raise RuntimeError("workflow did not reach its post-effect durable pause")

        if MODE == "resume":
            with SetWorkflowID(WORKFLOW_ID):
                handle = await DBOS.start_workflow_async(workflow)
            before = await handle.get_status()
            assert before.status == "PENDING"
            await DBOS.send_async(WORKFLOW_ID, {"resume": True}, topic="phase7-worker-crash")
            result = await handle.get_result()
            after = await handle.get_status()
            raw = await DBOS.get_event_async(WORKFLOW_ID, "constrained_output", timeout_seconds=5)
            (artifact / "recovered-constrained-output.json").write_text(json.dumps(raw, indent=2) + "\n")
            print(json.dumps({
                "event": "success",
                "status": after.status,
                "result": result.model_dump(mode="json"),
                "effect_lines": line_count(EFFECTS),
            }), flush=True)
            return
        raise ValueError(f"unknown mode {MODE!r}")
    finally:
        await backend.aclose()
        DBOS.destroy(destroy_registry=True)


if __name__ == "__main__":
    asyncio.run(main())
