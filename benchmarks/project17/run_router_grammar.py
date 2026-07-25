"""Grammar-constrained routing in the context-pool router (guide 14 flagship half).

xgrammar masks each sequence's logits to a JSON schema so every routed decision is
guaranteed well-formed and schema-valid. Per-sequence GrammarMatcher, batched
within the adapter's context.

Gates (fail-closed, deterministic greedy):
  1. Constrained validity: EVERY constrained output parses as JSON and validates
     against the routing schema (across base + an adapter).
  2. Grammar earns its keep: same prompts run UNCONSTRAINED are reported for
     schema-validity, showing what the grammar buys.

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from context_pool_router import BASE, ContextPoolRouter, Request


class Route(BaseModel):
    tool: Literal["search", "calculator", "calendar", "smart_home", "none"]
    confidence: Literal["low", "medium", "high"]


def gpu_snapshot() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


USER_TASKS = [
    "book a meeting for tomorrow at 3pm",
    "what is 17 times 23",
    "turn off the kitchen lights",
    "find the latest news about mars",
    "sing me a song",
]
INSTR = ("You are a tool router. For the user request, respond with ONLY a JSON "
         "object {\"tool\": ..., \"confidence\": ...}. User request: ")


def validate(text: str) -> dict:
    """Return {'json_ok','schema_ok','value'} for a decoded completion."""
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
    print("[grammar] loading router + enabling xgrammar", flush=True)
    router = ContextPoolRouter(
        str(args.model),
        adapters={"probe-a": str(args.lora_a), "probe-b": str(args.lora_b)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
    )
    router.enable_grammar(str(args.tokenizer_dir))
    grammar = router.compile_json_schema(Route.model_json_schema())

    # Mixed workload across base + probe-a (grammar is adapter-agnostic).
    adapters = [BASE, "probe-a"]
    requests = [Request(rid=f"{'base' if a is BASE else a}-{i}",
                        prompt=INSTR + USER_TASKS[i], adapter=a, max_tokens=args.max_tokens)
                for a in adapters for i in range(len(USER_TASKS))]

    print(f"[grammar] constrained run: {len(requests)} requests", flush=True)
    constrained = router.run_constrained(requests, grammar)
    print("[grammar] unconstrained run (comparison)", flush=True)
    unconstrained = router.run(requests)

    rows = []
    all_valid = True
    unconstrained_valid = 0
    for cg, ug in zip(constrained, unconstrained):
        ct = router.detokenize(cg.tokens)
        ut = router.detokenize(ug.tokens)
        cv = validate(ct)
        uv = validate(ut)
        all_valid = all_valid and cv["schema_ok"]
        unconstrained_valid += 1 if uv["schema_ok"] else 0
        rows.append({
            "rid": cg.rid, "adapter": cg.adapter,
            "constrained_schema_ok": cv["schema_ok"], "constrained_value": cv["value"],
            "constrained_raw": cv["raw"],
            "unconstrained_schema_ok": uv["schema_ok"], "unconstrained_raw": ut[:120],
        })

    verdict = "GO" if all_valid else "NO-GO"
    result = {
        "verdict": verdict,
        "gate_all_constrained_schema_valid": all_valid,
        "constrained_valid": sum(1 for r in rows if r["constrained_schema_ok"]),
        "unconstrained_valid": unconstrained_valid,
        "total": len(rows),
        "schema": Route.model_json_schema(),
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "max_tokens": args.max_tokens, "seed": args.seed},
        "rows": rows,
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "model": str(args.model),
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "router_grammar.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": verdict,
                      "constrained_valid": result["constrained_valid"],
                      "unconstrained_valid": unconstrained_valid,
                      "total": len(rows),
                      "sample": [r["constrained_value"] for r in rows[:5]]}, indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
