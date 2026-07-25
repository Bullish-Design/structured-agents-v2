"""Is the S>=4 cross-mode mismatch a fork bug or batch-size FP nondeterminism?

Ground truth = single-sequence isolated decode (batch-1, one adapter, UNMODIFIED
uniform-LoRA path = router.baseline). For each mode, compare its per-request output
to that baseline and report exact-match count + first divergence. If fork == baseline
for all requests, the fork is correct and the cross-mode diff is pure batch-size FP
tie-breaking in the other modes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from context_pool_router import ContextPoolRouter, Request

PROMPTS = ["The capital of France is", "In a distant galaxy,", "def add(a, b):",
           "Once upon a time", "The three primary colors are", "Water boils at",
           "The opposite of hot is", "To make tea you first"]
POOL = ["probe-a", "probe-b"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--sizes", default="4,8")
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--n-seq-max", default=8, type=int)
    p.add_argument("--max-tokens", default=32, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES in {0,1} and LLAMA_CPP_LIB_PATH")

    router = ContextPoolRouter(
        str(args.model), adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed, include_base=False)
    router.enable_seq_routing(POOL)

    def firstdiv(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if a == b else min(len(a), len(b)))

    out = {"config": {"max_tokens": args.max_tokens, "n_seq_max": args.n_seq_max}, "sizes": []}
    for S in (int(x) for x in args.sizes.split(",")):
        reqs = [Request(rid=f"r{i}", prompt=PROMPTS[i % len(PROMPTS)],
                        adapter=POOL[i % len(POOL)], max_tokens=args.max_tokens) for i in range(S)]
        base = {g.rid: g.tokens for g in (router.baseline(r) for r in reqs)}
        fork = {g.rid: g.tokens for g in router.run_seq_routed(reqs)}
        rout = {g.rid: g.tokens for g in router.run(reqs)}
        seql = {}
        for r in reqs:
            seql.update({g.rid: g.tokens for g in router.run_seq_routed([r])})

        def summarize(mode):
            per = [{"rid": rid, "match": mode[rid] == base[rid], "first_div": firstdiv(mode[rid], base[rid])}
                   for rid in base]
            return {"match_count": sum(1 for x in per if x["match"]), "total": len(per), "per": per}

        rec = {"S": S, "fork_vs_baseline": summarize(fork),
               "router_vs_baseline": summarize(rout), "sequential_vs_baseline": summarize(seql)}
        out["sizes"].append(rec)
        print(f"[chk] S={S}: fork {rec['fork_vs_baseline']['match_count']}/{S} "
              f"router {rec['router_vs_baseline']['match_count']}/{S} "
              f"sequential {rec['sequential_vs_baseline']['match_count']}/{S} match baseline", flush=True)

    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "p2_correctness_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
