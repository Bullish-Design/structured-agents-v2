#!/usr/bin/env python3
"""Profile sustained generation throughput for a local OpenAI-compatible vLLM server.

Runs 1 through 5 simultaneous requests.  Every request asks for a substantive
response and uses min_tokens=1024/max_tokens=1536, avoiding misleading TPS from
short answers or an early EOS.  The script uses only the Python standard library.

Usage:
  deploy/vllm/profile_tps.py
  LLM_BASE_URL=http://tower:8000/v1 LLM_MODEL=base BENCH_ROUNDS=5 deploy/vllm/profile_tps.py
  BENCH_OUTPUT=/tmp/gemma-tps.csv deploy/vllm/profile_tps.py
"""

import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "base")
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))
DISABLE_THINKING = os.environ.get("BENCH_DISABLE_THINKING", "").lower() in {"1", "true", "yes"}
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "3"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "1536"))
MIN_TOKENS = int(os.environ.get("BENCH_MIN_TOKENS", "1024"))
OUTPUT = Path(os.environ.get("BENCH_OUTPUT", Path(__file__).with_name("profile_tps.csv")))

PROMPTS = [
    "Explain how a modern web browser loads and renders a page, from DNS lookup through painting. Use several concise paragraphs.",
    "Explain how a lithium-ion battery stores and releases energy. Cover the electrodes, electrolyte, and charge cycle in several concise paragraphs.",
    "Describe how a database transaction provides atomicity and isolation. Include a practical example in several concise paragraphs.",
    "Explain the major stages of compiling a programming language, from source code to an executable. Use several concise paragraphs.",
    "Describe how TCP reliably transfers a byte stream across an unreliable network. Cover sequencing, acknowledgements, and congestion control.",
]

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"


def payload(prompt: str) -> dict:
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
    }
    if DISABLE_THINKING:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    return request


def request(prompt: str) -> tuple[int, float]:
    started = time.perf_counter()
    body = json.dumps(payload(prompt)).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    return result["usage"]["completion_tokens"], elapsed


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def run_level(concurrency: int) -> tuple[int, float, list[float]]:
    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(concurrency)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(request, prompts))
    return sum(tokens for tokens, _ in results), time.perf_counter() - started, [latency for _, latency in results]


def main() -> None:
    if MIN_TOKENS > MAX_TOKENS:
        raise SystemExit("BENCH_MIN_TOKENS cannot exceed BENCH_MAX_TOKENS")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not OUTPUT.exists() or OUTPUT.stat().st_size == 0
    output_file = OUTPUT.open("a", newline="")
    writer = csv.DictWriter(
        output_file,
        fieldnames=[
            "run_utc", "endpoint", "model", "rounds", "concurrency", "requests", "completion_tokens",
            "wall_s", "output_tps", "mean_latency_s", "p50_latency_s", "p95_latency_s",
        ],
    )
    if needs_header:
        writer.writeheader()
    print(f"endpoint: {BASE_URL}  model: {MODEL}  rounds: {ROUNDS}  output tokens: {MIN_TOKENS}-{MAX_TOKENS}")
    print(f"CSV output: {OUTPUT}")
    print("Warming up...")
    request(PROMPTS[0])
    print(f"{'conc':>4} {'reqs':>4} {'tokens':>7} {'wall_s':>7} {'out_tok/s':>10} {'mean_lat':>9} {'p50_lat':>8} {'p95_lat':>8}")
    for concurrency in range(1, 6):
        token_total = 0
        wall_total = 0.0
        latencies: list[float] = []
        try:
            for _ in range(ROUNDS):
                tokens, wall, wave_latencies = run_level(concurrency)
                token_total += tokens
                wall_total += wall
                latencies.extend(wave_latencies)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError) as error:
            print(f"{concurrency:>4} ERROR: {error}")
            continue
        row = {
            "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": BASE_URL,
            "model": MODEL,
            "rounds": ROUNDS,
            "concurrency": concurrency,
            "requests": len(latencies),
            "completion_tokens": token_total,
            "wall_s": f"{wall_total:.3f}",
            "output_tps": f"{token_total / wall_total:.3f}",
            "mean_latency_s": f"{statistics.mean(latencies):.3f}",
            "p50_latency_s": f"{percentile(latencies, 0.50):.3f}",
            "p95_latency_s": f"{percentile(latencies, 0.95):.3f}",
        }
        print(
            f"{concurrency:>4} {len(latencies):>4} {token_total:>7} {wall_total:>7.2f} "
            f"{token_total / wall_total:>10.1f} {statistics.mean(latencies):>9.2f} "
            f"{percentile(latencies, 0.50):>8.2f} {percentile(latencies, 0.95):>8.2f}"
        )
        writer.writerow(row)
        output_file.flush()
    output_file.close()


if __name__ == "__main__":
    main()
