"""In-process wire capture of pydantic-ai 2.11 /chat/completions request bodies.

No network: an httpx event_hooks request hook records the outbound JSON body, and a
MockTransport returns a canned OpenAI chat.completions response so Agent.run completes.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

CAPTURED: list[dict[str, Any]] = []


class Person(BaseModel):
    name: str
    age: int
    tags: list[str]


def _canned_response(request: httpx.Request) -> httpx.Response:
    # Record the request body.
    body = json.loads(request.content.decode())
    CAPTURED.append(body)
    # Decide content: json for native/tool paths, plain string for text.
    rf = body.get("response_format")
    tools = body.get("tools")
    if rf is not None or tools is not None:
        content = json.dumps({"name": "Ada", "age": 36, "tags": ["a", "b"]})
        tool_calls = None
        # If this is the function-calling output-tool path, answer via tool_call.
        if tools is not None:
            tool_name = tools[0]["function"]["name"]
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": content},
                }
            ]
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        else:
            message = {"role": "assistant", "content": content}
    else:
        message = {"role": "assistant", "content": "keep"}

    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "test-model"),
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx.Response(200, json=payload)


def _build_agent(output_type: Any, model_settings: dict | None) -> Agent:
    transport = httpx.MockTransport(_canned_response)
    http_client = httpx.AsyncClient(transport=transport)
    client = AsyncOpenAI(base_url="http://test/v1", api_key="x", http_client=http_client)
    model = OpenAIChatModel("test-model", provider=OpenAIProvider(openai_client=client))
    kwargs: dict[str, Any] = {"output_type": output_type}
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    return Agent(model, **kwargs)


def run_case(label: str, output_type: Any, model_settings: dict | None) -> dict:
    CAPTURED.clear()
    agent = _build_agent(output_type, model_settings)
    result = agent.run_sync("hello")
    body = CAPTURED[-1]
    print(f"\n===== {label} =====")
    print("output:", repr(result.output))
    print(json.dumps(body, indent=2, sort_keys=False))
    return body


def main() -> None:
    bodies = {}
    # 1. Schema mode via NativeOutput
    bodies["schema"] = run_case("1. SCHEMA (NativeOutput)", NativeOutput(Person), None)

    # 1b. plain model output_type (default path) for comparison
    bodies["plain_model"] = run_case("1b. PLAIN MODEL (default)", Person, None)

    # 2. Regex
    bodies["regex"] = run_case(
        "2. REGEX",
        str,
        {"extra_body": {"structured_outputs": {"regex": r"\d{4}-\d{2}-\d{2}"}}},
    )

    # 3. Choice
    bodies["choice"] = run_case(
        "3. CHOICE",
        str,
        {"extra_body": {"structured_outputs": {"choice": ["keep", "skip"]}}},
    )

    # 4. Grammar
    bodies["grammar"] = run_case(
        "4. GRAMMAR",
        str,
        {"extra_body": {"structured_outputs": {"grammar": 'root ::= "a" | "b"'}}},
    )

    # bare str, no extra_body (text mode baseline)
    bodies["bare_str"] = run_case("5. BARE STR (text baseline)", str, None)

    with open("bodies.json", "w") as f:
        json.dump(bodies, f, indent=2)


if __name__ == "__main__":
    main()
