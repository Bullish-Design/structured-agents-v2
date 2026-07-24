"""GPU-only sequential benchmark runner for Project 17's JSON workload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.soak_grammar import _model_for_schema, _resolve_tokenizer, _run_once, aggregate_outcomes
from workload import read_jsonl, select, sha256, validate_corpus

from structured_agents.llama_core.benchmark import write_benchmark_record
from structured_agents.llama_core.diagnostics import collect_runtime_diagnostics
from structured_agents.llama_core.grammar import JsonSchemaGrammar


def gpu_snapshot() -> dict[str, Any]:
    query = "index,name,uuid,driver_version,memory.total,memory.used"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True
    )
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def require_gpu0_only(snapshot: dict[str, Any]) -> None:
    lines = snapshot["stdout"].splitlines()
    if len(lines) < 2:
        raise RuntimeError("expected two GPUs in nvidia-smi snapshot")
    values = [line.split(", ") for line in lines]
    used0 = int(values[0][-1].split()[0])
    used1 = int(values[1][-1].split()[0])
    if used0 < 4000 or used1 > 200:
        raise RuntimeError(f"GPU isolation rejected: GPU0={used0} MiB GPU1={used1} MiB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--schema-registry", type=Path, default=Path("benchmarks/project17/schema_registry_v1.json"))
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--same-grammar-repeated", action="store_true")
    parser.add_argument("--tokenizer", default="deepreinforce-ai/Ornith-1.0-9B")
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--n-ctx", type=int, default=512)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")
    artifact_path = args.artifacts.resolve()
    if artifact_path.parent.name != "artifacts" or not artifact_path.name.startswith("project17-"):
        raise ValueError("artifacts must be an artifacts/project17-* directory")
    registry = json.loads(args.schema_registry.read_text(encoding="utf-8"))
    entries = read_jsonl(args.corpus)
    validate_corpus(entries, registry["schemas"])
    selected = select(entries, args.requests)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "gpu-before.json").write_text(json.dumps(gpu_snapshot(), indent=2) + "\n")

    from llama_cpp import Llama
    from transformers import AutoTokenizer

    load_started = perf_counter_ns()
    llm = Llama(
        model_path=str(args.model), n_ctx=args.n_ctx, n_batch=128, n_gpu_layers=-1, seed=args.seed, verbose=False
    )
    import torch

    torch.cuda.synchronize()
    load_wall_ns = perf_counter_ns() - load_started
    after_load = gpu_snapshot()
    (args.artifacts / "gpu-after-load.json").write_text(json.dumps(after_load, indent=2) + "\n")
    require_gpu0_only(after_load)
    tokenizer = AutoTokenizer.from_pretrained(_resolve_tokenizer(args.tokenizer, allow_network=False))
    grammars: dict[str, JsonSchemaGrammar] = {}
    compile_ns: dict[str, int] = {}
    if not args.baseline_only:
        for schema_id in sorted({str(item["schema_id"]) for item in selected}):
            schema = registry["schemas"][schema_id]
            started = perf_counter_ns()
            grammars[schema_id] = JsonSchemaGrammar.from_huggingface(tokenizer, schema, vocab_size=llm.n_vocab())
            compile_ns[schema_id] = perf_counter_ns() - started

    outcomes: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        schema_id = selected[0]["schema_id"] if args.same_grammar_repeated else item["schema_id"]
        started = perf_counter_ns()
        prompt_tokens = llm.tokenize(item["prompt"].encode("utf-8"), add_bos=False, special=True)
        prep_ns = perf_counter_ns() - started
        validator: type[BaseModel] = _model_for_schema(registry["schemas"][schema_id])
        outcome, record = _run_once(
            llm=llm,
            grammar=None if args.baseline_only else grammars[schema_id],
            validator=validator,
            prompt=item["prompt"],
            prompt_tokens_count=len(prompt_tokens),
            request_index=index,
            max_tokens=item["max_tokens"],
            constrained=not args.baseline_only,
            synchronize_gpu=True,
        )
        timings = dict(record.timings_ns)
        timings["tokenizer_preparation"] = prep_ns
        timings["grammar_compile"] = (
            compile_ns.get(schema_id, 0) if schema_id not in {o.get("schema_id") for o in outcomes} else 0
        )
        outcome.update(
            {
                "id": item["id"],
                "category": item["category"],
                "schema_id": schema_id,
                "input_metadata": {
                    "utf8_bytes": len(item["prompt"].encode("utf-8")),
                    "utf8_chars": len(item["prompt"]),
                    "prompt_tokens": len(prompt_tokens),
                },
            }
        )
        outcome["timings_ns"] = timings
        outcomes.append(outcome)
        write_benchmark_record(
            replace(record, timings_ns=timings, metadata={**record.metadata, **outcome}), args.artifacts / "records"
        )
        print(f"{index + 1}/{len(selected)} {outcome['status']} {item['id']}", flush=True)

    summary = aggregate_outcomes(outcomes)
    summary.update(
        {
            "corpus": str(args.corpus),
            "corpus_sha256": sha256(args.corpus),
            "runtime": collect_runtime_diagnostics().model_dump(),
            "model_load_and_gpu_warmup_ns": load_wall_ns,
            "grammar_compile_ns_by_schema": compile_ns,
            "mode": "baseline"
            if args.baseline_only
            else ("same-grammar-repeated" if args.same_grammar_repeated else "corpus"),
            "gpu_after_load": after_load,
            "gpu_after_run": gpu_snapshot(),
            "async_enqueue_warning": (
                "llama_decode enqueue timings are deliberately not reported as throughput; "
                "rates use synchronized wall durations."
            ),
        }
    )
    (args.artifacts / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    # Length cutoffs are retained as structured-correctness evidence but are
    # valid observed samples for the token-normalized performance baseline.
    return 0 if summary["hard_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
