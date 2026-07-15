#!/usr/bin/env python3
"""Generate and preserve a like-for-like essay from the vLLM and llama.cpp APIs."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


PROMPT = """Write an essay of exactly 1,000 words in exactly five paragraphs about how
specialized inference engines shape the practical use of local language models. Compare
latency, throughput, memory management, and operational trade-offs. Do not include a
title, headings, bullet lists, notes, or citations. Return only the five essay paragraphs.
Silently count the visible essay words before responding and make the total exactly 1,000."""
MAX_TOKENS = 1_800
ENDPOINTS = {
    "vllm": "http://127.0.0.1:8000/v1",
    "llama_cpp": "http://127.0.0.1:8001/v1",
}

requested_engines = os.environ.get("LLM_ENGINES")
if requested_engines:
    names = [name.strip() for name in requested_engines.split(",") if name.strip()]
    unknown = set(names) - set(ENDPOINTS)
    if unknown:
        raise SystemExit(f"unknown LLM_ENGINES value(s): {', '.join(sorted(unknown))}")
    ENDPOINTS = {name: ENDPOINTS[name] for name in names}


def request(engine: str, base_url: str) -> tuple[str, dict, float]:
    payload = {
        "model": "base",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    if engine == "llama_cpp":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    return engine, result, time.perf_counter() - started


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = os.environ.get("RUN_LABEL", "").strip()
    output = Path("artifacts/head-to-head") / f"{timestamp}{('-' + label) if label else ''}"
    output.mkdir(parents=True)

    performance: dict[str, object] = {
        "timestamp_utc": timestamp,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "engines": {},
    }
    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
        futures = {pool.submit(request, engine, url): engine for engine, url in ENDPOINTS.items()}
        results = []
        for future in as_completed(futures):
            engine = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                performance["engines"][engine] = {"endpoint": ENDPOINTS[engine], "error": str(error)}

    for engine, response, elapsed in results:
        choice = response["choices"][0]
        message = choice["message"]
        essay = message.get("content") or ""
        thinking = message.get("reasoning_content") or ""
        usage = response.get("usage", {})
        engine_stats = {
            "endpoint": ENDPOINTS[engine],
            "wall_time_s": round(elapsed, 3),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "visible_word_count": word_count(essay),
            "visible_paragraph_count": len([p for p in re.split(r"\n\s*\n", essay.strip()) if p]),
            "thinking_word_count": word_count(thinking),
            "visible_output_tokens_per_s": round(usage.get("completion_tokens", 0) / elapsed, 3),
            "server_timings": response.get("timings"),
        }
        performance["engines"][engine] = engine_stats
        (output / f"{engine}-essay.txt").write_text(essay + "\n")
        (output / f"{engine}-thinking.txt").write_text(thinking + "\n")
        (output / f"{engine}-response.json").write_text(json.dumps(response, indent=2) + "\n")

    (output / "performance.json").write_text(json.dumps(performance, indent=2) + "\n")
    print(output)
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"generation failed: {error}", file=sys.stderr)
        raise
