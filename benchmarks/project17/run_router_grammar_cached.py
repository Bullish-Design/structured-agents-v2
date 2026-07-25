"""Full path-(a) MVP: cached shared prefix + grammar-constrained routing.

Composes the two proven halves: a long shared router prompt is prefilled ONCE per
adapter and its KV restored into every seq slot; each request decodes only its
short suffix and emits a grammar-guaranteed JSON routing decision.

Gates (fail-closed, deterministic greedy):
  1. Validity: every cached+constrained output is schema-valid JSON.
  2. Equivalence: cached+constrained == cold+constrained (full prefix re-prefill),
     token-exact — the cached-prefix path does not change the constrained result.
Throughput: cached vs cold wall time.

GPU-only: CUDA_VISIBLE_DEVICES in {0,1}, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from pydantic import BaseModel, ValidationError

from context_pool_router import BASE, ContextPoolRouter, Request


class Route(BaseModel):
    tool: Literal["search", "calculator", "calendar", "smart_home", "none"]
    confidence: Literal["low", "medium", "high"]


# Long shared router prompt (the part every request has in common -> cache once).
PREFIX_TEXT = (
    "You are the routing brain of a tool-calling agent fleet. Your only job is to "
    "select exactly one tool for each incoming user request and rate your confidence. "
    "Available tools: search (web/knowledge lookup), calculator (arithmetic and math), "
    "calendar (scheduling, reminders, availability), smart_home (device and light "
    "control), none (no tool is appropriate). Weigh the user's intent, the tool "
    "descriptions, and prior context. Always answer with a single JSON object of the "
    "form {\"tool\": <one of the tools>, \"confidence\": <low|medium|high>} and nothing "
    "else. Be deterministic and consistent across similar requests. "
) * 2

SUFFIXES = [
    " Request: book a meeting for tomorrow at 3pm. Decision:",
    " Request: what is 17 times 23? Decision:",
    " Request: turn off the kitchen lights. Decision:",
    " Request: find the latest news about mars. Decision:",
]


def gpu_snapshot() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def sync() -> None:
    import torch
    torch.cuda.synchronize()


def validate(text: str) -> dict:
    out = {"json_ok": False, "schema_ok": False, "value": None, "raw": text}
    try:
        obj = json.loads(text)
        out["json_ok"] = True
    except Exception:
        return out
    try:
        out["value"] = Route.model_validate(obj).model_dump()
        out["schema_ok"] = True
    except ValidationError:
        pass
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=4096, type=int)
    p.add_argument("--n-seq-max", default=8, type=int)
    p.add_argument("--max-tokens", default=48, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES in {0,1} and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()
    print("[mvp] loading router + enabling xgrammar", flush=True)
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
    )
    router.enable_grammar(str(args.tokenizer_dir))
    grammar = router.compile_json_schema(Route.model_json_schema())

    prefix_tokens = router.tokenize(PREFIX_TEXT)
    n_ctx_seq = args.n_ctx // args.n_seq_max
    max_suffix = max(len(router.tokenize_suffix(s)) for s in SUFFIXES)
    need = len(prefix_tokens) + max_suffix + args.max_tokens
    print(f"[mvp] prefix={len(prefix_tokens)} tok, budget n_ctx_seq={n_ctx_seq}, need={need}", flush=True)
    if need >= n_ctx_seq:
        raise SystemExit(f"per-seq budget {n_ctx_seq} too small for need {need}")

    per_adapter = []
    all_valid = True
    all_match = True
    for adapter in (BASE, "probe-a"):
        name = "base" if adapter is BASE else adapter
        print(f"[mvp] adapter={name}: cache prefix + cached-constrained run", flush=True)
        cache = router.cache_prefix(adapter, prefix_tokens)
        reqs = [Request(rid=f"{name}-{i}", prompt=SUFFIXES[i], adapter=adapter,
                        max_tokens=args.max_tokens) for i in range(len(SUFFIXES))]

        sync(); t0 = perf_counter_ns()
        cached = router.run_constrained_cached(adapter, cache, reqs, grammar)
        sync(); cached_ns = perf_counter_ns() - t0

        # cold constrained: full prefix+suffix re-prefill, same token stream.
        ctx = router.ctx[adapter]
        full_tok = [cache.tokens + router.tokenize_suffix(r.prompt) for r in reqs]
        sync(); t0 = perf_counter_ns()
        cold = router._generate_wave_constrained(ctx, full_tok, grammar, args.max_tokens)
        sync(); cold_ns = perf_counter_ns() - t0

        seqs = []
        for i, g in enumerate(cached):
            text = router.detokenize(g.tokens)
            v = validate(text)
            match = g.tokens == cold[i][:len(g.tokens)]
            all_valid = all_valid and v["schema_ok"]
            all_match = all_match and match
            seqs.append({"rid": g.rid, "schema_ok": v["schema_ok"], "value": v["value"],
                         "raw": v["raw"], "cached_eq_cold": match})
        per_adapter.append({
            "adapter": name,
            "cached_ms": round(cached_ns / 1e6, 1),
            "cold_ms": round(cold_ns / 1e6, 1),
            "cached_speedup_vs_cold": round(cold_ns / cached_ns, 2),
            "sequences": seqs,
        })

    verdict = "GO" if (all_valid and all_match) else "NO-GO"
    result = {
        "verdict": verdict,
        "gates": {"all_schema_valid": all_valid, "cached_eq_cold": all_match},
        "prefix_tokens": len(prefix_tokens),
        "schema": Route.model_json_schema(),
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "max_tokens": args.max_tokens, "seed": args.seed},
        "per_adapter": per_adapter,
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "model": str(args.model),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "router_grammar_cached.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "gates": result["gates"],
                      "speedups": {a["adapter"]: a["cached_speedup_vs_cold"] for a in per_adapter},
                      "decisions": [s["value"] for a in per_adapter for s in a["sequences"]]},
                     indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
