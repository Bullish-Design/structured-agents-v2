"""Validate + benchmark the context-pool multi-LoRA router (guide 14 §10, P1).

Gates (fail-closed, deterministic greedy):
  1. Equivalence: every request's routed (batched, multiplexed) continuation is
     token-exact vs its isolated single-seq baseline on the same adapter.
  2. Routing is real: the SAME prompt under base / probe-a / probe-b yields
     distinct continuations (adapters are actually applied per-context).
Throughput: aggregate tok/s of the routed multiplex vs the sequential per-request
baselines (the win the router is supposed to buy).

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

from context_pool_router import BASE, ContextPoolRouter, Request


def gpu_snapshot() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def sync() -> None:
    import torch
    torch.cuda.synchronize()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--n-seq-max", default=8, type=int)
    p.add_argument("--max-tokens", default=32, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()
    print("[router] loading base + adapters + context pool", flush=True)
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
    )

    prompts = [
        "The capital of France is",
        "In a distant galaxy,",
        "def add(a, b):",
        "The three primary colors are",
        "Once upon a time",
        "The chemical symbol for gold is",
    ]
    # Mixed workload across the pool: base, probe-a, probe-b interleaved.
    adapters_cycle = [BASE, "probe-a", "probe-b"]
    requests = [
        Request(rid=f"r{i}", prompt=prompts[i % len(prompts)],
                adapter=adapters_cycle[i % len(adapters_cycle)],
                max_tokens=args.max_tokens)
        for i in range(9)
    ]

    # --- routed (batched + multiplexed) ---
    print(f"[router] running {len(requests)} routed requests", flush=True)
    sync(); t0 = perf_counter_ns()
    gens = router.run(requests)
    sync(); routed_ns = perf_counter_ns() - t0
    routed_tokens = sum(len(g.tokens) for g in gens)

    # --- baselines (isolated single-seq, sequential) ---
    print("[router] running isolated baselines", flush=True)
    sync(); t0 = perf_counter_ns()
    baselines = [router.baseline(r) for r in requests]
    sync(); base_ns = perf_counter_ns() - t0
    base_tokens = sum(len(g.tokens) for g in baselines)

    # --- gate 1: equivalence ---
    per_request = []
    all_match = True
    for g, b in zip(gens, baselines):
        match = g.tokens == b.tokens
        all_match = all_match and match
        first_div = next((i for i, (x, y) in enumerate(zip(g.tokens, b.tokens)) if x != y), None)
        per_request.append({
            "rid": g.rid, "adapter": g.adapter, "match": match,
            "first_divergence": first_div,
            "routed_head": g.tokens[:8], "baseline_head": b.tokens[:8],
            "text": router.detokenize(g.tokens),
        })

    # --- gate 2: routing distinctiveness (same prompt, three adapters) ---
    probe_prompt = "The capital of France is"
    distinct_reqs = [
        Request("d-base", probe_prompt, BASE, args.max_tokens),
        Request("d-a", probe_prompt, "probe-a", args.max_tokens),
        Request("d-b", probe_prompt, "probe-b", args.max_tokens),
    ]
    dgen = router.run(distinct_reqs)
    d_base, d_a, d_b = (g.tokens for g in dgen)
    routing_distinct = (d_base != d_a) and (d_a != d_b) and (d_base != d_b)

    verdict = "GO" if (all_match and routing_distinct) else "NO-GO"
    result = {
        "verdict": verdict,
        "gates": {
            "equivalence_routed_eq_baseline": all_match,
            "routing_distinct_base_a_b": routing_distinct,
        },
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "max_tokens": args.max_tokens, "seed": args.seed,
                   "n_requests": len(requests)},
        "throughput": {
            "routed_tokens": routed_tokens,
            "routed_ms": round(routed_ns / 1e6, 1),
            "routed_tok_s": round(routed_tokens / (routed_ns / 1e9), 1),
            "baseline_tokens": base_tokens,
            "baseline_ms": round(base_ns / 1e6, 1),
            "baseline_tok_s": round(base_tokens / (base_ns / 1e9), 1),
            "speedup": round((base_ns / base_tokens) / (routed_ns / routed_tokens), 2),
        },
        "per_request": per_request,
        "routing_distinctiveness": {
            "prompt": probe_prompt,
            "base_head": d_base[:8], "probe_a_head": d_a[:8], "probe_b_head": d_b[:8],
        },
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "context_pool_router.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "gates": result["gates"],
                      "throughput": result["throughput"]}, indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
