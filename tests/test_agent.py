from __future__ import annotations

import json

import httpx
from dbos import DBOS
from pydantic import BaseModel

from structured_agents import AgentSpec, Backend, Schema
from structured_agents.plane import Queue


class Plan(BaseModel):
    value: int


def respond(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": '{"value": 7}'}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


agent = Backend(http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))).build(
    AgentSpec("plan-agent", Schema(Plan), "Return a plan.")
)


@DBOS.workflow()
async def parent() -> Plan:
    return await agent.run("nested")


async def test_top_level_run_is_durable() -> None:
    assert await agent.run("top") == Plan(value=7)
    assert any(w.status == "SUCCESS" for w in await DBOS.list_workflows_async(name="plan-agent.run"))
    assert any(w.status == "SUCCESS" for w in await DBOS.list_workflows_async(name="plan-agent.structured_run"))
    assert await parent() == Plan(value=7)
    assert any(w.parent_workflow_id for w in await DBOS.list_workflows_async(name="plan-agent.run"))


async def test_real_agent_queues_through_registered_typed_workflow() -> None:
    handle = await Queue("real-agent-queue", concurrency=1).submit(agent, "queued")

    result = await handle.get_result()

    assert result == Plan(value=7)
    assert isinstance(result, Plan)
    assert (await handle.get_status()).name == "plan-agent.structured_run"
