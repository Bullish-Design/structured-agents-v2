"""Verify how PydanticAI 1.87 puts different output_type choices on the wire.

Question being settled (open #1 / caveat #3): when we want a *bare-string* constrained
output (choice/grammar/regex) vs a *JSON* constrained output (json_schema), what does
PydanticAI actually emit? Capture the request body for each output_type variant.

The capture hook fires when the request is SENT, so we get the wire shape even if the
run later errors on parsing -- we only care about the shape here.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

BASE_URL = os.environ["LLM_BASE_URL"]
API_KEY = os.environ.get("LLM_API_KEY", "sk-none")
MODEL = os.environ["LLM_MODEL"]

CAP: list[dict[str, Any]] = []


async def _hook(req: httpx.Request) -> None:
    try:
        body = json.loads(req.content.decode())
    except Exception:
        body = {}
    CAP.append(body)


class Route(BaseModel):
    route: Literal["file_edit", "git_ops", "answer"]


def model(client: httpx.AsyncClient) -> OpenAIChatModel:
    return OpenAIChatModel(MODEL, provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY, http_client=client))


def describe(label: str, body: dict[str, Any]) -> None:
    rf = body.get("response_format")
    rf_type = rf.get("type") if isinstance(rf, dict) else rf
    tools = body.get("tools")
    tool_names = [t.get("function", {}).get("name") for t in tools] if tools else None
    print(f"[{label}]")
    print(f"   response_format.type = {rf_type!r}")
    print(f"   tools                = {tool_names}")
    print(f"   tool_choice          = {body.get('tool_choice')!r}")
    # show the enum location if any
    if isinstance(rf, dict):
        sch = rf.get("json_schema", {}).get("schema", {})
        print(f"   response_format schema keys = {list(sch.get('properties', {}).keys()) or list(sch.keys())}")
    if tools:
        params = tools[0].get("function", {}).get("parameters", {})
        print(f"   tool[0] params props = {list(params.get('properties', {}).keys())}")
    print()


async def run_variant(label: str, output_type: Any, settings: OpenAIChatModelSettings | None = None) -> None:
    CAP.clear()
    async with httpx.AsyncClient(event_hooks={"request": [_hook]}) as client:
        agent = Agent(model(client), output_type=output_type,
                      model_settings=settings or OpenAIChatModelSettings(max_tokens=32),
                      instructions="Pick file_edit.")
        try:
            await agent.run("Edit config.py to bump the timeout.")
        except Exception as e:
            print(f"[{label}] (run raised {type(e).__name__}: {str(e)[:80]} -- shape still captured)")
        if CAP:
            describe(label, CAP[0])
        else:
            print(f"[{label}] no request captured\n")


async def main() -> None:
    print(f"# output_type wire shapes vs {BASE_URL} ({MODEL})\n")
    # 1. plain text
    await run_variant("output_type=str", str)
    # 2. Literal (bare-choice intent)
    await run_variant('output_type=Literal[...]', Literal["file_edit", "git_ops", "answer"])
    # 3. plain Model (PydanticAI default output mode)
    await run_variant("output_type=Route (default)", Route)
    # 4. NativeOutput(Model)
    await run_variant("NativeOutput(Route)", NativeOutput(Route))


if __name__ == "__main__":
    asyncio.run(main())
