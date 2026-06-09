"""Request-path spike for the structured-agents-v2 / xgrammar concept.

Goal: verify, against a live OpenAI-compatible inference server, exactly what
PydanticAI puts on the wire when we ask a per-agent model for constrained,
schema-validated output -- and whether concurrent agent runs actually batch.

What this DOES verify (works against any OpenAI-compatible server, incl. the
current llama.cpp remora-server and a future vLLM container):
  1. Native structured output -> `response_format: {type: json_schema, ...}` on the wire.
  2. `extra_body` passthrough -> vLLM XGrammar / guided-decoding params reach the server.
  3. Per-agent model selection -> the `model` field is how a LoRA adapter is chosen.
  4. Concurrency -> N agent.run() calls dispatch in parallel (server-side batching).

What this does NOT verify (needs the real vLLM+GPU box): that XGrammar actually
constrains decoding, or that LoRA weights load. The current server is llama.cpp
(GBNF grammars, no LoRA) -- see FINDINGS.md.

Run:  devenv shell -- spike-run
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

BASE_URL = os.environ.get("LLM_BASE_URL", "http://remora-server:8000/v1")
API_KEY = os.environ.get("LLM_API_KEY", "sk-no-key-required")
MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-9B-UD-Q6_K_XL.gguf")


# --- Per-agent output schemas (from the 01-xgrammar concept) ------------------
class FilePatch(BaseModel):
    op: Literal["replace", "insert_after", "delete"]
    path: str
    target: str | None = None
    content: str | None = None


class FileEditPlan(BaseModel):
    action: Literal["edit_file", "refuse", "needs_clarification"]
    patches: list[FilePatch] = Field(default_factory=list)
    reason: str


class GitCommand(BaseModel):
    action: Literal["status", "diff", "add", "commit", "checkout_new_branch", "refuse"]
    paths: list[str] = Field(default_factory=list)
    message: str | None = None
    reason: str


class TestRunRequest(BaseModel):
    action: Literal["run_tests", "run_single_test", "refuse"]
    command: Literal["pytest", "python -m pytest"]
    path: str | None = None
    test_name: str | None = None
    reason: str


# --- Wire capture -------------------------------------------------------------
CAPTURED: list[dict[str, Any]] = []


async def _capture_request(request: httpx.Request) -> None:
    try:
        body = json.loads(request.content.decode())
    except Exception:
        body = {"_raw": request.content.decode(errors="replace")[:500]}
    CAPTURED.append(
        {
            "t": time.monotonic(),
            "url": str(request.url),
            "model": body.get("model"),
            "response_format": body.get("response_format"),
            # anything we injected via extra_body lands at the top level of the JSON body:
            "extra_body_keys": [
                k
                for k in body
                if k
                not in {"model", "messages", "response_format", "stream", "stream_options"}
            ],
            "body": body,
        }
    )


def build_model(http_client: httpx.AsyncClient, model_name: str = MODEL) -> OpenAIChatModel:
    """One model object per 'agent'. `model_name` is the LoRA-adapter selector."""
    provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY, http_client=http_client)
    return OpenAIChatModel(model_name, provider=provider)


# --- Spike steps --------------------------------------------------------------
async def step_native_json_schema(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Agent 1: native structured output -> response_format json_schema on the wire."""
    agent = Agent(
        build_model(client),
        output_type=NativeOutput(FileEditPlan),
        instructions="You produce file-edit plans only. Be terse.",
    )
    res = await agent.run("Change the retry timeout in config.py from 5s to 10s.")
    ok = isinstance(res.output, FileEditPlan)
    return ok, repr(res.output)


async def step_extra_body_passthrough(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Agent 2: inject vLLM XGrammar guided-decoding params via extra_body.

    On llama.cpp these keys are ignored, but the capture proves PydanticAI puts
    them on the wire verbatim -- which is the mechanism a vLLM backend needs.
    """
    settings = OpenAIChatModelSettings(
        max_tokens=512,
        extra_body={
            # vLLM structured-outputs (current) form:
            "structured_outputs": {"json": GitCommand.model_json_schema()},
            # vLLM guided-decoding backend selector:
            "guided_decoding_backend": "xgrammar",
        },
    )
    agent = Agent(
        build_model(client),
        output_type=NativeOutput(GitCommand),
        model_settings=settings,
        instructions="You translate requests into a single safe git command.",
    )
    res = await agent.run("Stage all changes and commit them with a sensible message.")
    ok = isinstance(res.output, GitCommand)
    return ok, repr(res.output)


async def step_concurrency(client: httpx.AsyncClient, n: int = 4) -> dict[str, Any]:
    """Fire N agent runs concurrently; compare against sequential to prove batching."""
    agent = Agent(
        build_model(client),
        output_type=NativeOutput(TestRunRequest),
        model_settings=OpenAIChatModelSettings(max_tokens=256),
        instructions="Decide how to run the requested tests.",
    )
    prompts = [f"Run the tests in tests/test_mod{i}.py" for i in range(n)]

    t0 = time.monotonic()
    results = await asyncio.gather(*(agent.run(p) for p in prompts))
    concurrent_s = time.monotonic() - t0

    t0 = time.monotonic()
    for p in prompts:
        await agent.run(p)
    sequential_s = time.monotonic() - t0

    return {
        "n": n,
        "all_valid": all(isinstance(r.output, TestRunRequest) for r in results),
        "concurrent_s": round(concurrent_s, 2),
        "sequential_s": round(sequential_s, 2),
        "speedup": round(sequential_s / concurrent_s, 2) if concurrent_s else None,
    }


async def main() -> None:
    print(f"# Request-path spike against {BASE_URL} (model={MODEL})\n")
    async with httpx.AsyncClient(event_hooks={"request": [_capture_request]}) as client:
        print("## Step 1: native json_schema output")
        try:
            ok, out = await step_native_json_schema(client)
            print(f"  validated={ok}  output={out}\n")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}\n")

        print("## Step 2: extra_body XGrammar passthrough")
        try:
            ok, out = await step_extra_body_passthrough(client)
            print(f"  validated={ok}  output={out}\n")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}\n")

        print("## Step 3: concurrency / batching")
        try:
            stats = await step_concurrency(client)
            print(f"  {stats}\n")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}\n")

    # --- Wire-shape report ---
    print("## Captured request shapes (what PydanticAI sent)")
    for i, c in enumerate(CAPTURED[:3]):
        rf = c["response_format"]
        rf_type = rf.get("type") if isinstance(rf, dict) else rf
        print(f"  req[{i}] model={c['model']!r} response_format.type={rf_type!r} "
              f"extra_body_keys={c['extra_body_keys']}")
    # dump one full body for the record
    if CAPTURED:
        sample = next((c for c in CAPTURED if c["extra_body_keys"]), CAPTURED[0])
        out_path = os.path.join(os.path.dirname(__file__), "captured_request.json")
        with open(out_path, "w") as f:
            json.dump(sample["body"], f, indent=2)
        print(f"\n  full sample body -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
