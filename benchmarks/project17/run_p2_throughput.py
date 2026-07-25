"""P2 throughput eval: fork mixed-batch vs no-fork router vs sequential.

Serves the SAME S-request mixed-adapter workload three ways and measures aggregate
tokens/sec, sweeping S. Also cross-checks that all three modes produce token-exact
identical output (correctness), so the throughput numbers compare like for like.

  sequential : each request alone (batch-1), one at a time
  router     : P1 context-pool — batched within each adapter's context, then
               multiplexed across the K adapter contexts (K decodes of ~S/K)
  fork       : P2 — one context, one mixed-batch decode of all S (per-seq adapters)

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
from time import perf_counter_ns

from context_pool_router import ContextPoolRouter, Request


def gpu_snapshot() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def sync() -> None:
    import torch
    torch.cuda.synchronize()


PROMPTS = [
    "The capital of France is", "In a distant galaxy,", "def add(a, b):",
    "Once upon a time", "The three primary colors are", "Water boils at",
    "The opposite of hot is", "To make tea you first",
]
POOL = ["probe-a", "probe-b"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--n-seq-max", default=8, type=int)
    p.add_argument("--max-tokens", default=32, type=int)
    p.add_argument("--batch-sizes", default="2,4,8")
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES in {0,1} and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed, include_base=False,
    )
    router.enable_seq_routing(POOL)

    def make_requests(S: int) -> list[Request]:
        return [Request(rid=f"r{i}", prompt=PROMPTS[i % len(PROMPTS)],
                        adapter=POOL[i % len(POOL)], max_tokens=args.max_tokens)
                for i in range(S)]

    def timed(fn) -> tuple[float, list]:
        fn()  # warmup (graph build / cache)
        best = None
        out = None
        for _ in range(args.reps):
            sync(); t0 = perf_counter_ns()
            out = fn()
            sync(); dt = perf_counter_ns() - t0
            best = dt if best is None else min(best, dt)
        return best / 1e6, out  # ms

    def toks(gens) -> list[tuple]:
        return [tuple(g.tokens) for g in sorted(gens, key=lambda g: g.rid)]

    rows = []
    for S in (int(x) for x in args.batch_sizes.split(",")):
        if S > args.n_seq_max:
            rows.append({"S": S, "skipped": f"S>{args.n_seq_max}"})
            continue
        reqs = make_requests(S)
        gen_tokens = S * args.max_tokens

        seq_ms, seq_out = timed(lambda: [g for r in reqs for g in router.run_seq_routed([r])])
        rt_ms,  rt_out  = timed(lambda: router.run(reqs))
        fk_ms,  fk_out  = timed(lambda: router.run_seq_routed(reqs))

        # correctness cross-check: identical tokens across all three modes
        identical = toks(seq_out) == toks(rt_out) == toks(fk_out)

        rows.append({
            "S": S, "gen_tokens": gen_tokens, "outputs_identical": identical,
            "sequential_ms": round(seq_ms, 1), "sequential_tok_s": round(gen_tokens / (seq_ms / 1e3), 1),
            "router_ms": round(rt_ms, 1),      "router_tok_s": round(gen_tokens / (rt_ms / 1e3), 1),
            "fork_ms": round(fk_ms, 1),        "fork_tok_s": round(gen_tokens / (fk_ms / 1e3), 1),
            "fork_vs_router": round(rt_ms / fk_ms, 2),
            "fork_vs_sequential": round(seq_ms / fk_ms, 2),
            "router_vs_sequential": round(seq_ms / rt_ms, 2),
        })
        print(f"[thru] S={S}: seq={seq_ms:.0f}ms/{rows[-1]['sequential_tok_s']}tps "
              f"router={rt_ms:.0f}ms/{rows[-1]['router_tok_s']}tps "
              f"fork={fk_ms:.0f}ms/{rows[-1]['fork_tok_s']}tps "
              f"(fork/router x{rows[-1]['fork_vs_router']}) identical={identical}", flush=True)

    all_identical = all(r.get("outputs_identical", True) for r in rows if "skipped" not in r)
    result = {
        "all_outputs_identical": all_identical,
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "max_tokens": args.max_tokens, "reps": args.reps,
                   "pool": POOL, "seed": args.seed},
        "rows": rows,
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "model": str(args.model), "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "p2_throughput.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_outputs_identical": all_identical,
                      "rows": [{k: r.get(k) for k in ("S", "sequential_tok_s", "router_tok_s",
                               "fork_tok_s", "fork_vs_router", "fork_vs_sequential")}
                               for r in rows if "skipped" not in r]}, indent=2), flush=True)
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
