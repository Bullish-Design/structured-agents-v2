"""Validate + benchmark cached shared-prefix restore in the context-pool router.

Composes the proven per-sequence KV-state reuse (run_seq_reuse.py) into the
router: a shared prefix is prefilled ONCE per adapter, its KV blob captured, and
restored into every seq slot so each request only decodes its own suffix.

Gates (fail-closed, deterministic greedy), swept over base / probe-a / probe-b:
  1. Equivalence: cached-restore continuation is token-exact vs the full
     (prefix+suffix) baseline on the same adapter — for EVERY sequence in the wave
     (also proves neighbour isolation across restored slots).
  2. Adapter-specificity is respected implicitly: each adapter caches/restores in
     its OWN context (layer-3 K/V are adapter-perturbed).
Throughput: cached wave (restore + suffix only) vs cold wave (full re-prefill).

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


# A long shared prefix (well past the ~192-token restore break-even) so restore
# is expected to beat cold re-prefill. Deterministic, no external data.
PREFIX_TEXT = (
    "You are a routing assistant for a tool-calling agent fleet. Follow these "
    "rules exactly. Always return a single well-formed decision. Consider the "
    "user's intent, the available tools, and the conversation so far. Do not "
    "invent tools that were not provided. Prefer the most specific tool. When no "
    "tool applies, say so plainly. Be concise, deterministic, and consistent. "
) * 3

SUFFIXES = [
    " User request: book a flight to Paris. Decision:",
    " User request: what is 2 plus 2? Decision:",
    " User request: summarize the attached document. Decision:",
    " User request: turn off the living room lights. Decision:",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=4096, type=int)
    p.add_argument("--n-seq-max", default=8, type=int)
    p.add_argument("--max-tokens", default=24, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()
    print("[cached] loading base + adapters + context pool", flush=True)
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
    )

    prefix_tokens = router.tokenize(PREFIX_TEXT)
    # Per-sequence context budget is n_ctx / n_seq_max (NOT n_ctx). prefix + suffix
    # + generated tokens for a single sequence must fit, else llama_decode rc=1.
    n_ctx_seq = args.n_ctx // args.n_seq_max
    max_suffix = max(len(router.tokenize_suffix(s)) for s in SUFFIXES)
    need = len(prefix_tokens) + max_suffix + args.max_tokens
    print(f"[cached] prefix={len(prefix_tokens)} tok, per-seq budget n_ctx_seq={n_ctx_seq}, "
          f"worst-case need={need}", flush=True)
    if need >= n_ctx_seq:
        raise SystemExit(f"per-seq budget {n_ctx_seq} too small for need {need}; "
                         f"raise --n-ctx or lower --n-seq-max/prefix")

    per_adapter = []
    all_match = True
    for adapter in (BASE, "probe-a", "probe-b"):
        name = "base" if adapter is BASE else adapter
        print(f"[cached] adapter={name}: cache prefix", flush=True)
        cache = router.cache_prefix(adapter, prefix_tokens)

        reqs = [Request(rid=f"{name}-{i}", prompt=SUFFIXES[i], adapter=adapter,
                        max_tokens=args.max_tokens) for i in range(len(SUFFIXES))]

        # cached path (restore + suffix-only), timed
        sync(); t0 = perf_counter_ns()
        cached = router.run_cached(adapter, cache, reqs)
        sync(); cached_ns = perf_counter_ns() - t0

        # cold path (full re-prefill of prefix+suffix in one wave), timed
        ctx = router.ctx[adapter]
        full_token_lists = [cache.tokens + router.tokenize_suffix(r.prompt) for r in reqs]
        sync(); t0 = perf_counter_ns()
        cold = router._generate_wave(ctx, full_token_lists, args.max_tokens)
        sync(); cold_ns = perf_counter_ns() - t0

        # equivalence baselines (isolated single-seq, full prompt)
        seqs = []
        for i, r in enumerate(reqs):
            full = cache.tokens + router.tokenize_suffix(r.prompt)
            base = router.baseline_from_tokens(adapter, full, args.max_tokens, rid=r.rid)
            c = cached[i].tokens
            match_cached = c == base.tokens
            match_cold = cold[i][:args.max_tokens] == base.tokens
            all_match = all_match and match_cached and match_cold
            seqs.append({
                "rid": r.rid,
                "match_cached_vs_baseline": match_cached,
                "match_cold_vs_baseline": match_cold,
                "first_div_cached": next((k for k, (x, y) in enumerate(zip(c, base.tokens)) if x != y), None),
                "cached_text": router.detokenize(c),
            })

        gen_tokens = sum(len(g.tokens) for g in cached)
        per_adapter.append({
            "adapter": name,
            "n_requests": len(reqs),
            "generated_tokens": gen_tokens,
            "cached_ms": round(cached_ns / 1e6, 1),
            "cold_ms": round(cold_ns / 1e6, 1),
            "cached_speedup_vs_cold": round(cold_ns / cached_ns, 2),
            "sequences": seqs,
        })

    verdict = "GO" if all_match else "NO-GO"
    result = {
        "verdict": verdict,
        "gate_equivalence_all_seqs": all_match,
        "prefix_tokens": len(prefix_tokens),
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "max_tokens": args.max_tokens, "seed": args.seed},
        "per_adapter": per_adapter,
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "context_pool_router_cached.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "verdict": verdict,
        "gate_equivalence_all_seqs": all_match,
        "prefix_tokens": len(prefix_tokens),
        "speedups": {a["adapter"]: a["cached_speedup_vs_cold"] for a in per_adapter},
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
