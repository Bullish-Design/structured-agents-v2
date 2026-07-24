"""Produce a local benchmark artifact without requiring llama.cpp or a GPU."""

from __future__ import annotations

from pathlib import Path

from structured_agents.llama_core.benchmark import BenchmarkTimer, write_benchmark_record


def main() -> None:
    timer = BenchmarkTimer("synthetic-owned-loop", metadata={"backend": "synthetic", "purpose": "harness-smoke"})
    timer.add_ns("prefill", 2_000_000)
    for _ in range(3):
        timer.add_ns("decode", 500_000)
        timer.add_ns("sample", 20_000)
        timer.add_ns("accept", 5_000)
        timer.record_token_latency_ns(525_000)
    artifact = write_benchmark_record(timer.record(prompt_tokens=12, completion_tokens=3), Path("artifacts/benchmarks"))
    print(artifact)


if __name__ == "__main__":
    main()
