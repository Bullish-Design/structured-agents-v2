"""Where does cached-prefix restore beat cold re-prefill? A length sweep.

The router composition (run_context_pool_router_cached.py) proved cached restore is
token-EXACT vs cold, but at a 223-token prefix the wall-clock win was ~1.03x because
GPU prefill is cheap and generation dominates. This sweep isolates the mechanism:
for each prefix length P, over S sequences sharing that prefix, measure

  * PREFILL-only : S x prefill(P + suffix)          (cold, no generation)
  * RESTORE-only : S x set_data(prefix blob) + S x prefill(suffix)

Generation is identical in both, so excluding it exposes the crossover directly.
Also reports an end-to-end cached-vs-cold wall time with a short generation.

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


FILLER = (
    "The routing policy considers intent, available tools, prior turns, latency, "
    "and cost before committing to a single deterministic decision every time. "
)
SUFFIXES = [
    " Request: book a flight. Decision:",
    " Request: compute 2+2. Decision:",
    " Request: summarize the doc. Decision:",
    " Request: lights off. Decision:",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=8192, type=int)
    p.add_argument("--n-seq-max", default=4, type=int)
    p.add_argument("--prefix-lengths", default="128,256,512,1024,1536")
    p.add_argument("--gen-tokens", default=8, type=int)
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")

    adapter = "probe-a"  # representative: sweep runs inside a pinned-adapter context
    Plist = [int(x) for x in args.prefix_lengths.split(",")]
    n_ctx_seq = args.n_ctx // args.n_seq_max
    S = args.n_seq_max

    gpu_before = gpu_snapshot()
    print("[sweep] loading base + adapters + pool", flush=True)
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
    )
    C = router.C
    ctx = router.ctx[adapter]

    # A long token pool to slice exact-length prefixes from (BOS + filler).
    filler_tokens = router.tokenize(FILLER * 200)
    suffix_tokens = [router.tokenize_suffix(s) for s in SUFFIXES[:S]]
    max_suffix = max(len(t) for t in suffix_tokens)

    def timed(fn) -> float:
        best = None
        for _ in range(args.reps):
            sync(); t0 = perf_counter_ns(); fn(); sync()
            dt = perf_counter_ns() - t0
            best = dt if best is None else min(best, dt)
        return best / 1e6  # ms, best-of-reps

    rows = []
    for P in Plist:
        if P + max_suffix + args.gen_tokens >= n_ctx_seq:
            rows.append({"prefix": P, "skipped": f"need {P+max_suffix+args.gen_tokens} >= budget {n_ctx_seq}"})
            continue
        prefix = filler_tokens[:P]
        cache = router.cache_prefix(adapter, prefix)

        def do_prefill():
            for i in range(S):
                router._seq_rm(ctx, i)
                router._decode_prefill(ctx, prefix + suffix_tokens[i], seq_id=i, start_pos=0)

        def do_restore():
            for i in range(S):
                router._seq_rm(ctx, i)
                if router._set_seq(ctx, cache.blob, i) == 0:
                    raise RuntimeError("set_seq failed")
                router._decode_prefill(ctx, suffix_tokens[i], seq_id=i, start_pos=cache.n)

        prefill_ms = timed(do_prefill)
        restore_ms = timed(do_restore)

        # End-to-end wall time incl. short generation (batched), best-of-reps.
        reqs = [Request(rid=f"p{P}-{i}", prompt=SUFFIXES[i], adapter=adapter,
                        max_tokens=args.gen_tokens) for i in range(S)]
        full_tok = [cache.tokens + suffix_tokens[i] for i in range(S)]
        e2e_cold = timed(lambda: router._generate_wave(ctx, full_tok, args.gen_tokens))
        e2e_cached = timed(lambda: router.run_cached(adapter, cache, reqs))

        rows.append({
            "prefix": P, "seqs": S, "blob_bytes": len(cache.blob),
            "prefill_only_ms": round(prefill_ms, 1),
            "restore_only_ms": round(restore_ms, 1),
            "prefill_vs_restore_speedup": round(prefill_ms / restore_ms, 2),
            "e2e_cold_ms": round(e2e_cold, 1),
            "e2e_cached_ms": round(e2e_cached, 1),
            "e2e_speedup": round(e2e_cold / e2e_cached, 2),
        })
        print(f"[sweep] P={P}: prefill={prefill_ms:.1f}ms restore={restore_ms:.1f}ms "
              f"(x{prefill_ms/restore_ms:.2f}) | e2e cold={e2e_cold:.1f} cached={e2e_cached:.1f} "
              f"(x{e2e_cold/e2e_cached:.2f})", flush=True)

    measured = [r for r in rows if "skipped" not in r]
    crossover = next((r["prefix"] for r in measured if r["prefill_vs_restore_speedup"] > 1.0), None)
    result = {
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "n_ctx_seq": n_ctx_seq, "gen_tokens": args.gen_tokens,
                   "reps": args.reps, "adapter": adapter, "seed": args.seed},
        "prefill_beats_restore_at_prefix": crossover,
        "rows": rows,
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "model": str(args.model),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "prefix_restore_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prefill_beats_restore_at_prefix": crossover,
                      "rows": [{k: r.get(k) for k in ("prefix", "prefill_only_ms",
                               "restore_only_ms", "prefill_vs_restore_speedup",
                               "e2e_speedup")} for r in measured]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
