"""Repeat-run soak harness for the owned llama.cpp + XGrammar JSON path.

The harness compiles exactly one grammar, creates a fresh matcher for every
request, and fails closed on every non-stop, malformed, or rejected result.
All raw output is deliberately restricted to an ignored ``artifacts/project17-*``
directory.

Example (run in the pinned Project 17 environment)::

    python examples/soak_grammar.py \
      --model /path/Ornith-1.0-9B.gguf --requests 10 \
      --artifacts artifacts/project17-grammar-soak
"""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from structured_agents.llama_core.benchmark import (
    TIMING_FIELDS,
    BenchmarkRecord,
    BenchmarkTimer,
    write_benchmark_record,
)
from structured_agents.llama_core.decode import FINISH_STOP, OwnedLlamaDecoder
from structured_agents.llama_core.grammar import JsonSchemaGrammar


class CapitalAnswer(BaseModel):
    """The small deterministic JSON contract used by the default smoke."""

    city: str
    country: str


DEFAULT_PROMPT = "Return only a JSON object naming the capital city and country of France."
DEFAULT_SCHEMA = CapitalAnswer.model_json_schema()
FAILURE_STATUSES = frozenset({"malformed", "rejected", "runtime_error", "error"})


def _percentiles(values: list[int]) -> dict[str, float | None]:
    """Return deterministic nearest-rank p95 values without NumPy."""
    if not values:
        return {"p50_ns": None, "p95_ns": None, "mean_ns": None, "total_ns": 0.0}
    ordered = sorted(values)
    return {
        "p50_ns": float(statistics.median(ordered)),
        "p95_ns": float(ordered[min(len(ordered) - 1, (len(ordered) * 95 + 99) // 100 - 1)]),
        "mean_ns": statistics.fmean(ordered),
        "total_ns": float(sum(ordered)),
    }


def aggregate_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate request outcomes; intentionally GPU- and llama-free.

    ``invalid_count`` excludes cleanly identified cutoffs so an interrupted
    completion cannot disappear into a generic validation-failure bucket.
    A cutoff is a structured-correctness failure, but it is still a completed
    performance observation and must not invalidate throughput aggregates.
    """
    counts: dict[str, int] = {}
    phase_values = {field: [] for field in TIMING_FIELDS}
    wall_values = {"generation": [], "request": []}
    completion_tokens = 0
    prompt_tokens = 0
    mask_values: list[int] = []

    for outcome in outcomes:
        status = str(outcome["status"])
        counts[status] = counts.get(status, 0) + 1
        prompt_tokens += int(outcome.get("prompt_tokens", 0))
        completion_tokens += int(outcome.get("completion_tokens", 0))
        timings = outcome.get("timings_ns", {})
        for field in TIMING_FIELDS:
            phase_values[field].append(int(timings.get(field, 0)))
        mask_values.append(int(timings.get("mask_creation", 0)) + int(timings.get("mask_application", 0)))
        wall_timings = outcome.get("wall_timings_ns", {})
        for field in wall_values:
            wall_values[field].append(int(wall_timings.get(field, 0)))

    valid_count = counts.get("valid", 0)
    cutoff_count = counts.get("cutoff", 0)
    invalid_count = sum(count for status, count in counts.items() if status in FAILURE_STATUSES)
    mask_total_ns = sum(mask_values)
    generation_total_ns = sum(wall_values["generation"])
    request_total_ns = sum(wall_values["request"])
    return {
        "request_count": len(outcomes),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "cutoff_count": cutoff_count,
        "hard_failure_count": invalid_count,
        "failure_count": len(outcomes) - valid_count,
        "performance_result": "completed",
        "structured_correctness_result": "pass" if not invalid_count and not cutoff_count else "incomplete",
        "status_counts": counts,
        "token_counts": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "phase_timings_ns": {field: _percentiles(values) for field, values in phase_values.items()},
        "wall_timings_ns": {field: _percentiles(values) for field, values in wall_values.items()},
        "end_to_end": {
            "generation_tokens_per_second": (
                completion_tokens * 1_000_000_000 / generation_total_ns
                if generation_total_ns and completion_tokens
                else None
            ),
            "request_tokens_per_second": (
                completion_tokens * 1_000_000_000 / request_total_ns if request_total_ns and completion_tokens else None
            ),
            "generation_ns_per_completion_token": (
                generation_total_ns / completion_tokens if generation_total_ns and completion_tokens else None
            ),
            "request_ns_per_completion_token": (
                request_total_ns / completion_tokens if request_total_ns and completion_tokens else None
            ),
        },
        "mask_overhead": {
            **_percentiles(mask_values),
            "per_completion_token_ns": (mask_total_ns / completion_tokens if completion_tokens else None),
            "percent_of_generation_wall_ns": (
                mask_total_ns * 100 / generation_total_ns if generation_total_ns else None
            ),
        },
    }


def compare_decode_overhead(constrained: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    """Compare synchronized generation wall time per emitted token."""
    constrained_tokens = constrained["token_counts"]["completion"]
    baseline_tokens = baseline["token_counts"]["completion"]
    constrained_decode = constrained["phase_timings_ns"]["generation_wall"]["total_ns"]
    baseline_decode = baseline["phase_timings_ns"]["generation_wall"]["total_ns"]
    if not constrained_tokens or not baseline_tokens:
        return {"constrained_decode_ns_per_token": None, "baseline_decode_ns_per_token": None, "delta_percent": None}
    constrained_per_token = constrained_decode / constrained_tokens
    baseline_per_token = baseline_decode / baseline_tokens
    return {
        "constrained_decode_ns_per_token": constrained_per_token,
        "baseline_decode_ns_per_token": baseline_per_token,
        "delta_percent": ((constrained_per_token - baseline_per_token) * 100 / baseline_per_token)
        if baseline_per_token
        else None,
    }


def compare_end_to_end_overhead(constrained: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    """Compare synchronized generation wall time per emitted token."""
    constrained_per_token = constrained["end_to_end"]["generation_ns_per_completion_token"]
    baseline_per_token = baseline["end_to_end"]["generation_ns_per_completion_token"]
    if constrained_per_token is None or baseline_per_token is None:
        return {
            "constrained_generation_ns_per_token": None,
            "baseline_generation_ns_per_token": None,
            "delta_percent": None,
        }
    return {
        "constrained_generation_ns_per_token": constrained_per_token,
        "baseline_generation_ns_per_token": baseline_per_token,
        "delta_percent": ((constrained_per_token - baseline_per_token) * 100 / baseline_per_token)
        if baseline_per_token
        else None,
    }


def _synchronize_gpu() -> None:
    """Wait for queued CUDA work before recording an outer wall duration."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _model_for_schema(schema: dict[str, Any]) -> type[BaseModel]:
    """Build a small Pydantic object validator for a user-supplied JSON schema.

    The grammar can express more than this boundary validator.  Keep the CLI
    explicit: the dynamic form supports primitive object properties, while a
    richer contract should be added as a named Pydantic model instead of being
    silently under-validated.
    """
    if schema == DEFAULT_SCHEMA:
        return CapitalAnswer
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError("--schema-json/--schema-file must be an object schema with primitive properties")
    type_map: dict[str, type[Any]] = {"string": str, "integer": int, "number": float, "boolean": bool}
    required = set(schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    for name, property_schema in schema["properties"].items():
        json_type = property_schema.get("type")
        python_type = type_map.get(json_type)
        if python_type is None:
            raise ValueError(f"schema property {name!r} has unsupported type {json_type!r}")
        fields[name] = (python_type, ... if name in required else None)
    return create_model("SoakSchema", __config__=ConfigDict(extra="forbid"), **fields)


def _load_schema(args: argparse.Namespace) -> dict[str, Any]:
    if args.schema_json is not None:
        return json.loads(args.schema_json)
    if args.schema_file is not None:
        return json.loads(args.schema_file.read_text(encoding="utf-8"))
    return DEFAULT_SCHEMA


def _require_project17_artifacts(directory: Path) -> None:
    """Prevent accidental raw runtime output outside the project's ignored area."""
    normalized = directory.as_posix().rstrip("/")
    if not normalized.startswith("artifacts/project17-"):
        raise ValueError("--artifacts must be an ignored artifacts/project17-* directory")


def _resolve_tokenizer(tokenizer: str, *, allow_network: bool) -> str:
    """Resolve a repository ID to a local snapshot before Transformers loads it.

    Transformers 4.57 checks remote model metadata for some tokenizer classes
    even when all tokenizer files are cached.  Passing the resolved snapshot
    makes it an explicitly local load and keeps a soak reproducible offline.
    """
    if Path(tokenizer).exists():
        return tokenizer
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=tokenizer,
        allow_patterns=("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "*.jinja"),
        local_files_only=not allow_network,
    )


def _run_once(
    *,
    llm: Any,
    grammar: JsonSchemaGrammar | None,
    validator: type[BaseModel],
    prompt: str,
    prompt_tokens_count: int,
    request_index: int,
    max_tokens: int,
    constrained: bool,
    synchronize_gpu: bool,
) -> tuple[dict[str, Any], BenchmarkRecord]:
    """Run and classify one request, preserving the benchmark even on failure."""
    benchmark = BenchmarkTimer(
        "ornith-grammar-soak",
        metadata={"request_index": request_index, "constrained": constrained, "vocab_size": llm.n_vocab()},
    )
    outcome: dict[str, Any] = {
        "request_index": request_index,
        "status": "error",
        "detail": None,
        "prompt_tokens": prompt_tokens_count,
        "completion_tokens": 0,
        "finish_reason": None,
    }
    request_started_ns = perf_counter_ns()
    generation_started_ns: int | None = None
    try:
        logits_hook = None
        token_hook = None
        if constrained:
            assert grammar is not None
            with benchmark.measure("matcher_creation"):
                matcher = grammar.new_matcher()  # Never share state across requests.
            logits_hook = grammar.logits_hook(matcher, benchmark=benchmark)
            token_hook = grammar.token_hook(matcher)
        generation_started_ns = perf_counter_ns()
        with OwnedLlamaDecoder(llm) as decoder:
            generated = decoder.generate_text(
                prompt,
                max_tokens=max_tokens,
                logits_hook=logits_hook,
                token_hook=token_hook,
                benchmark=benchmark,
            )
        if synchronize_gpu:
            _synchronize_gpu()
        outcome["wall_timings_ns"] = {"generation": perf_counter_ns() - generation_started_ns}
        outcome["finish_reason"] = generated.finish_reason
        outcome["completion_tokens"] = generated.completion_token_count
        outcome["text"] = generated.text
        if generated.finish_reason != FINISH_STOP:
            outcome["status"] = "cutoff"
            outcome["detail"] = f"finish_reason={generated.finish_reason!r}"
        else:
            with benchmark.measure("validation"):
                validator.model_validate_json(generated.text)
            outcome["status"] = "valid"
    except ValidationError as exc:
        outcome["status"] = "malformed"
        outcome["detail"] = str(exc)
    except RuntimeError as exc:
        outcome["status"] = "rejected" if "rejected" in str(exc).lower() else "runtime_error"
        outcome["detail"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- a soak must account for every request.
        outcome["detail"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    wall_timings = outcome.setdefault("wall_timings_ns", {})
    if generation_started_ns is not None and "generation" not in wall_timings:
        wall_timings["generation"] = perf_counter_ns() - generation_started_ns
    wall_timings["request"] = perf_counter_ns() - request_started_ns
    record = benchmark.record(prompt_tokens=prompt_tokens_count, completion_tokens=outcome["completion_tokens"])
    outcome["timings_ns"] = dict(record.timings_ns)
    return outcome, record


def _run_batch(
    *,
    llm: Any,
    grammar: JsonSchemaGrammar | None,
    validator: type[BaseModel],
    prompt: str,
    prompt_tokens_count: int,
    request_count: int,
    max_tokens: int,
    constrained: bool,
    artifacts: Path,
    synchronize_gpu: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for request_index in range(request_count):
        outcome, record = _run_once(
            llm=llm,
            grammar=grammar,
            validator=validator,
            prompt=prompt,
            prompt_tokens_count=prompt_tokens_count,
            request_index=request_index,
            max_tokens=max_tokens,
            constrained=constrained,
            synchronize_gpu=synchronize_gpu,
        )
        outcomes.append(outcome)
        write_benchmark_record(
            record=replace(record, metadata={**record.metadata, **outcome}),
            directory=artifacts / "records",
        )
        print(f"[{'grammar' if constrained else 'baseline'}] {request_index + 1}/{request_count} {outcome['status']}")
    return outcomes, aggregate_outcomes(outcomes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", default="deepreinforce-ai/Ornith-1.0-9B")
    parser.add_argument(
        "--allow-network", action="store_true", help="allow downloading a tokenizer snapshot when it is not cached"
    )
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument("--schema-json", help="JSON object schema; defaults to CapitalAnswer")
    schema_group.add_argument("--schema-file", type=Path, help="path to a JSON object schema")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--seed", type=int, default=1234, help="llama.cpp seed (the default greedy sampler is deterministic)"
    )
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="llama.cpp layers to offload; defaults to -1 for full GPU offload",
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/project17-grammar-soak"))
    parser.add_argument("--no-baseline", action="store_true", help="skip the equal-size unconstrained comparison batch")
    parser.add_argument("--baseline-only", action="store_true", help="run only the unconstrained timing batch")
    args = parser.parse_args()
    if args.requests <= 0 or args.max_tokens <= 0 or args.n_ctx <= 0 or args.n_threads <= 0:
        raise ValueError("--requests, --max-tokens, --n-ctx, and --n-threads must be positive")
    _require_project17_artifacts(args.artifacts)
    if args.baseline_only and args.no_baseline:
        raise ValueError("--baseline-only and --no-baseline cannot be used together")
    schema = _load_schema(args)
    validator = _model_for_schema(schema)

    from llama_cpp import Llama
    from transformers import AutoTokenizer

    args.artifacts.mkdir(parents=True, exist_ok=True)
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=128,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        logits_all=False,
        verbose=False,
    )
    prompt_tokens_count = len(llm.tokenize(args.prompt.encode(), add_bos=False, special=True))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": str(args.model),
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "prompt": args.prompt,
        "schema": schema,
    }
    if args.baseline_only:
        _, baseline = _run_batch(
            llm=llm,
            grammar=None,
            validator=validator,
            prompt=args.prompt,
            prompt_tokens_count=prompt_tokens_count,
            request_count=args.requests,
            max_tokens=args.max_tokens,
            constrained=False,
            artifacts=args.artifacts,
            synchronize_gpu=args.n_gpu_layers != 0,
        )
        summary["baseline"] = baseline
        summary["result"] = "completed"
    else:
        tokenizer = AutoTokenizer.from_pretrained(_resolve_tokenizer(args.tokenizer, allow_network=args.allow_network))
        # Compile once for this entire constrained batch. _run_once creates only matchers.
        grammar = JsonSchemaGrammar.from_huggingface(tokenizer, schema, vocab_size=llm.n_vocab())
        _, constrained = _run_batch(
            llm=llm,
            grammar=grammar,
            validator=validator,
            prompt=args.prompt,
            prompt_tokens_count=prompt_tokens_count,
            request_count=args.requests,
            max_tokens=args.max_tokens,
            constrained=True,
            artifacts=args.artifacts,
            synchronize_gpu=args.n_gpu_layers != 0,
        )
        summary["constrained"] = constrained
        summary["result"] = "pass" if constrained["hard_failure_count"] == 0 else "fail"
        summary["structured_correctness_result"] = constrained["structured_correctness_result"]
        if not args.no_baseline:
            _, baseline = _run_batch(
                llm=llm,
                grammar=None,
                validator=validator,
                prompt=args.prompt,
                prompt_tokens_count=prompt_tokens_count,
                request_count=args.requests,
                max_tokens=args.max_tokens,
                constrained=False,
                artifacts=args.artifacts,
                synchronize_gpu=args.n_gpu_layers != 0,
            )
            summary["baseline"] = baseline
            summary["decode_comparison"] = compare_decode_overhead(constrained, baseline)
            summary["end_to_end_comparison"] = compare_end_to_end_overhead(constrained, baseline)

    summary_path = args.artifacts / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    return 0 if args.baseline_only or summary["constrained"]["hard_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
