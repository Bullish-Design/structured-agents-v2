#!/usr/bin/env python3
"""Issue the fixed single-slot MTP comparison requests and preserve raw JSON."""

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Explain why a binary search needs a sorted input, using a small example.",
    "Write a concise Python function that returns the nth Fibonacci number.",
    "Compare optimistic and pessimistic locking in two short paragraphs.",
    "Give three practical steps for investigating a slow SQL query.",
    "Explain dependency injection to a programmer new to backend services.",
    "Draft a short incident update for a five-minute API outage.",
    "Describe how a hash table resolves collisions, including one trade-off.",
    "Summarize the difference between a process and a thread.",
    "Give a compact checklist for reviewing a pull request before merge.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    args = p.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, prompt in enumerate(PROMPTS, 1):
        payload = {
            "model": "base",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 192,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "stream": False,
        }
        request = urllib.request.Request(
            args.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode()
        latency = time.perf_counter() - started
        result = json.loads(raw)
        (args.artifact_dir / f"response-{i:02d}.json").write_text(raw + "\n")
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        rows.append({
            "variant": args.variant,
            "request": i,
            "completion_tokens": tokens,
            "latency_seconds": round(latency, 6),
            "output_tok_s": round(tokens / latency, 6) if latency else 0,
        })

    with (args.artifact_dir / "requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
