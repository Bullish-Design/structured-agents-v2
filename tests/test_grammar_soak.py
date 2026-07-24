"""CPU-free accounting tests for the Project 17 grammar soak harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _soak_module() -> object:
    path = Path(__file__).parents[1] / "examples" / "soak_grammar.py"
    spec = importlib.util.spec_from_file_location("soak_grammar", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outcome(status: str, *, completion_tokens: int, timings: dict[str, int]) -> dict[str, object]:
    return {
        "status": status,
        "prompt_tokens": 5,
        "completion_tokens": completion_tokens,
        "timings_ns": timings,
    }


def test_aggregate_outcomes_counts_failures_cutoffs_and_tokens() -> None:
    soak = _soak_module()
    summary = soak.aggregate_outcomes(
        [
            _outcome("valid", completion_tokens=2, timings={"decode": 10, "mask_creation": 2, "mask_application": 3}),
            _outcome(
                "malformed",
                completion_tokens=4,
                timings={"decode": 20, "mask_creation": 4, "mask_application": 6},
            ),
            _outcome("rejected", completion_tokens=1, timings={"decode": 30}),
            _outcome("cutoff", completion_tokens=5, timings={"decode": 40}),
        ]
    )

    assert summary["valid_count"] == 1
    assert summary["invalid_count"] == 2
    assert summary["cutoff_count"] == 1
    assert summary["failure_count"] == 3
    assert summary["status_counts"] == {"valid": 1, "malformed": 1, "rejected": 1, "cutoff": 1}
    assert summary["token_counts"] == {"prompt": 20, "completion": 12, "total": 32}
    assert summary["phase_timings_ns"]["decode"]["total_ns"] == 100.0
    assert summary["mask_overhead"]["total_ns"] == 15.0
    assert summary["mask_overhead"]["per_completion_token_ns"] == 1.25


def test_decode_comparison_uses_per_token_totals() -> None:
    soak = _soak_module()
    constrained = soak.aggregate_outcomes([_outcome("valid", completion_tokens=2, timings={"decode": 40})])
    baseline = soak.aggregate_outcomes([_outcome("valid", completion_tokens=4, timings={"decode": 40})])

    comparison = soak.compare_decode_overhead(constrained, baseline)

    assert comparison == {
        "constrained_decode_ns_per_token": 20.0,
        "baseline_decode_ns_per_token": 10.0,
        "delta_percent": 100.0,
    }
