"""P2 fork validation: true mixed-batch multi-LoRA — one llama_decode, per-seq adapters.

The gate (guide 14 §7.6): ground truth = each sequence decoded ALONE on a
single-adapter context (proven, unmodified uniform-LoRA path). Then ONE mixed-batch
decode where different sequences use different adapters (probe-a / probe-b / none)
must be token-exact greedy for EVERY sequence. That proves the forked
build_lora_mm routes per-sequence correctly and in isolation.

Requires the P2 fork lib (LLAMA_CPP_LIB_PATH -> out-cuda-3060-p2fork/lib).
GPU-only: CUDA_VISIBLE_DEVICES in {0,1}, n_gpu_layers=-1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from context_pool_router import BASE, ContextPoolRouter, Request


def gpu_snapshot() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--n-seq-max", default=4, type=int)
    p.add_argument("--max-tokens", default=32, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES in {0,1} and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
        include_base=True,
    )
    router.enable_seq_routing(["probe-a", "probe-b"])

    prompts = ["The capital of France is", "In a distant galaxy,",
               "def add(a, b):", "Once upon a time"]
    # One mixed batch: each sequence a different adapter (incl. the no-adapter sentinel).
    adapters = ["probe-a", "probe-b", BASE, "probe-a"]
    requests = [Request(rid=f"r{i}", prompt=prompts[i], adapter=adapters[i],
                        max_tokens=args.max_tokens) for i in range(len(prompts))]

    print("[p2] mixed-batch routed decode (one context, per-seq adapters)", flush=True)
    routed = router.run_seq_routed(requests)
    print("[p2] isolated single-adapter baselines (unmodified path)", flush=True)
    baselines = [router.baseline(r) for r in requests]

    rows = []
    all_match = True
    for g, b in zip(routed, baselines):
        match = g.tokens == b.tokens
        all_match = all_match and match
        rows.append({
            "rid": g.rid, "adapter": "base" if g.adapter is BASE else g.adapter,
            "match": match,
            "first_div": next((i for i, (x, y) in enumerate(zip(g.tokens, b.tokens)) if x != y), None),
            "routed_head": g.tokens[:10], "baseline_head": b.tokens[:10],
        })

    # Also confirm the mixed batch is not trivially uniform: the 3 distinct adapters
    # on this batch should not all produce identical continuations.
    distinct = len({tuple(g.tokens) for g in routed}) > 1

    verdict = "GO" if (all_match and distinct) else "NO-GO"
    result = {
        "verdict": verdict,
        "gates": {"routed_eq_isolated_baseline": all_match, "outputs_distinct": distinct},
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "max_tokens": args.max_tokens, "seed": args.seed},
        "batch_adapters": [r["adapter"] for r in rows],
        "per_request": rows,
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "p2_mixed_batch.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "gates": result["gates"],
                      "per_request": [{k: r[k] for k in ("rid", "adapter", "match", "first_div")}
                                      for r in rows]}, indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
