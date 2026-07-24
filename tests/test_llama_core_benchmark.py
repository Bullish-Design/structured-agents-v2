from __future__ import annotations

import json

import pytest

from structured_agents.llama_core.benchmark import BenchmarkTimer, write_benchmark_record


def test_benchmark_record_emits_all_timing_fields_and_metrics(tmp_path) -> None:
    timer = BenchmarkTimer("owned-loop", metadata={"backend": "cpu"})
    timer.add_ns("prefill", 100)
    timer.add_ns("decode", 80)
    timer.add_ns("sample", 15)
    timer.record_token_latency_ns(30)
    timer.record_token_latency_ns(50)

    artifact = write_benchmark_record(timer.record(prompt_tokens=20, completion_tokens=2), tmp_path)
    payload = json.loads(artifact.read_text())

    assert payload["tokens"] == {"prompt": 20, "completion": 2}
    assert set(payload["timings_ns"]) == {
        "prefill",
        "decode",
        "mask_creation",
        "mask_application",
        "sample",
        "accept",
        "detokenize",
        "validation",
    }
    assert payload["metrics"]["prefill_tokens_per_second"] == 200_000_000
    assert payload["metrics"]["decode_tokens_per_second"] == 25_000_000
    assert payload["metrics"]["ttft_ns"] == 130
    assert payload["metrics"]["token_latency_p50_ns"] == 30
    assert payload["metrics"]["token_latency_p95_ns"] == 50
    assert payload["metadata"] == {"backend": "cpu"}


def test_benchmark_timer_rejects_unknown_or_negative_durations() -> None:
    timer = BenchmarkTimer("test")
    with pytest.raises(ValueError, match="unknown timing"):
        timer.add_ns("kv_restore", 1)
    with pytest.raises(ValueError, match="non-negative"):
        timer.record_token_latency_ns(-1)
