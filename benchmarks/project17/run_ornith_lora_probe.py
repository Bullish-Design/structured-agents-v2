"""§9 go/no-go: does llama.cpp apply a LoRA delta to Ornith's hybrid arch?

Greedy-decodes a fixed prompt three ways on the SAME base model and compares the
token continuations:

  * base            (no adapter)
  * base + probe-a  (synthetic large-perturbation adapter, layer-3 attn)
  * base + probe-b  (a second, distinct synthetic adapter)

Verdict (deterministic greedy, temperature 0):
  GO   if base != probe-a AND probe-a != probe-b  (delta applied + per-adapter distinct)
  NO-GO if base == probe-a                         (delta silently dropped on this arch)

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def gpu_snapshot() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def greedy_tokens(model_path: str, lora_path: str | None, prompt: str,
                  n_tokens: int, n_ctx: int, seed: int) -> list[int]:
    from llama_cpp import Llama

    kwargs = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1,
                  seed=seed, verbose=False)
    if lora_path is not None:
        kwargs["lora_path"] = lora_path
    llm = Llama(**kwargs)
    try:
        toks = list(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        out: list[int] = []
        for tok in llm.generate(toks, temp=0.0, top_k=1):
            out.append(int(tok))
            if len(out) >= n_tokens:
                break
        return out
    finally:
        llm.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n-tokens", default=48, type=int)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")

    def note(msg: str) -> None:
        print(f"[probe] {msg}", flush=True)

    gpu_before = gpu_snapshot()
    note("run 1/4: base")
    base = greedy_tokens(str(args.model), None, args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("run 2/4: base rerun")
    base2 = greedy_tokens(str(args.model), None, args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("run 3/4: base + probe-a")
    a = greedy_tokens(str(args.model), str(args.lora_a), args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("run 4/4: base + probe-b")
    b = greedy_tokens(str(args.model), str(args.lora_b), args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("all runs complete; comparing")

    base_reproducible = base == base2
    delta_applied = base != a
    per_adapter_distinct = a != b
    verdict = "GO" if (delta_applied and per_adapter_distinct and base_reproducible) else "NO-GO"

    result = {
        "verdict": verdict,
        "checks": {
            "base_reproducible": base_reproducible,
            "delta_applied_base_vs_a": delta_applied,
            "per_adapter_distinct_a_vs_b": per_adapter_distinct,
        },
        "prompt": args.prompt,
        "n_tokens": args.n_tokens,
        "seed": args.seed,
        "tokens": {"base": base, "base_rerun": base2, "probe_a": a, "probe_b": b},
        "first_divergence_base_vs_a": next(
            (i for i, (x, y) in enumerate(zip(base, a)) if x != y), None),
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "lora_a": str(args.lora_a),
        "lora_b": str(args.lora_b),
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "ornith_lora_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
