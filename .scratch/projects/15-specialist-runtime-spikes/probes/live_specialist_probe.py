"""Exercise constraints, adapters, concurrency, and engine metrics on one server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx


def constraint_body(engine: str, marker: str, index: int) -> dict[str, Any]:
    mode = index % 3
    if mode == 0:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"item_{index}",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"marker": {"type": "string", "const": marker}},
                        "required": ["marker"],
                        "additionalProperties": False,
                    },
                },
            }
        }
    if mode == 1:
        regex = f"{marker}-[0-9]{{48}}"
        return {"structured_outputs": {"regex": regex}} if engine == "vllm" else {"regex": regex}
    grammar = f'root ::= "{marker}-grammar-abcdefghijklmnopqrstuvwxyz0123456789"'
    return {"structured_outputs": {"grammar": grammar}} if engine == "vllm" else {"ebnf": grammar}


def validate_content(content: str, marker: str, index: int) -> None:
    mode = index % 3
    if mode == 0:
        assert json.loads(content) == {"marker": marker}
    elif mode == 1:
        prefix, digits = content.rsplit("-", 1)
        assert prefix == marker and len(digits) == 48 and digits.isdigit()
    else:
        assert content == f"{marker}-grammar-abcdefghijklmnopqrstuvwxyz0123456789"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="base")
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    root = args.base_url.removesuffix("/v1")
    models = [args.base_model, args.adapter_a, args.adapter_b]
    metrics_samples: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=180) as client:
        # SGLang's /health drives an internal generation and returns 503 until the
        # first flashinfer decode kernel finishes its one-time JIT compile. Poll
        # readiness instead of failing on a transient 503.
        health = await client.get(f"{root}/health")
        deadline = time.time() + 600
        while health.status_code != 200 and time.time() < deadline:
            await asyncio.sleep(5)
            health = await client.get(f"{root}/health")
        model_response = await client.get(f"{args.base_url}/models")
        (args.evidence_dir / "health.txt").write_text(f"{health.status_code}\n")
        (args.evidence_dir / "models.json").write_text(model_response.text + "\n")
        health.raise_for_status()
        model_response.raise_for_status()

        stop = asyncio.Event()

        async def sample_metrics() -> None:
            while not stop.is_set():
                try:
                    response = await client.get(f"{root}/metrics", timeout=10)
                    interesting = [
                        line
                        for line in response.text.splitlines()
                        if "running" in line or "waiting" in line or "lora" in line.lower()
                    ]
                    metrics_samples.append({"time": time.time(), "status": response.status_code, "lines": interesting})
                except Exception as exc:
                    metrics_samples.append({"time": time.time(), "error": repr(exc)})
                await asyncio.sleep(0.02)

        async def request(index: int) -> None:
            marker = f"item{index:02d}"
            model = models[index % len(models)]
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Return only the constrained output for marker {marker}.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 96,
                **constraint_body(args.engine, marker, index),
            }
            started = time.time()
            response = await client.post(f"{args.base_url}/chat/completions", json=body)
            finished = time.time()
            record: dict[str, Any] = {
                "index": index,
                "marker": marker,
                "requested_model": model,
                "status": response.status_code,
                "started": started,
                "finished": finished,
                "request": body,
                "response": response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else response.text,
            }
            outcomes.append(record)
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            validate_content(content, marker, index)

        sampler = asyncio.create_task(sample_metrics())
        try:
            await asyncio.gather(*(request(index) for index in range(args.requests)))
        finally:
            stop.set()
            await sampler

    outcomes.sort(key=lambda item: item["index"])
    (args.evidence_dir / "outcomes.json").write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    (args.evidence_dir / "metrics-samples.json").write_text(
        json.dumps(metrics_samples, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "engine": args.engine,
        "requests": len(outcomes),
        "models": models,
        "all_validated": len(outcomes) == args.requests,
        "wall_seconds": max(item["finished"] for item in outcomes) - min(item["started"] for item in outcomes),
        "metrics_samples": len(metrics_samples),
    }
    (args.evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
