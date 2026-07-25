"""Task-aware multi-LoRA router eval on the TINY Qwen3.5-0.8B base.

Swaps the synthetic layer-3 probes (run_context_pool_router.py) for THREE real
fine-tuned task LoRAs, so we can eyeball that each adapter actually *does its job*
inside the shared-base context-pool router, and that a batch mixing all three
keeps the adapters isolated (no bleed-through).

Adapters (all base Qwen/Qwen3.5-0.8B):
  ner-json  Mike0021/qwen35-0.8b-ner-json-lora   sentence -> strict JSON.
            NOTE: targets the hybrid in_proj_* / out_proj modules -> this run is
            also the runtime proof that our loader applies LoRA on those.
  dolly-qa  Neural-Hacker/qwen3.5-0.8b-dolly-qa-lora   instruction -> answer.
  acrouter  Lance1573/acrouter-qwen35-08b-router-lora  coding task -> backend name.

Gates (deterministic greedy):
  1. Equivalence: each routed (batched+multiplexed) continuation is token-exact
     vs its isolated single-seq baseline on the same adapter.
  2. Task behaviour: ner-json emits parseable JSON w/ the expected keys; acrouter
     emits one of its backend model names; dolly-qa answer differs from base.
  3. Isolation: the SAME prompt under base vs each adapter yields distinct output.

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

from context_pool_router import BASE, ContextPoolRouter, Request

BACKENDS = ["claude-sonnet-4-6", "claude-opus-4-6", "kimi-k2.5", "gpt-5.4",
            "MiniMax-M2.7", "qwen3.5-plus", "glm-5", "Qwen3-Max"]

# Verbatim training prompt from LanceZPF/agent-as-a-router src/routing/prompts.py
# (ROUTER_SYSTEM_PROMPT). The adapter was SFT'd to emit JSON {"model","reasoning"}.
ROUTER_SYSTEM_PROMPT = """\
You are a coding task router. Your objective is to maximize the performance-cost \
trade-off: choose the model that achieves the best quality for its cost on this task.

## Available Models (sorted by cost, high to low)

1. **claude-opus-4-6**: Premium. Excels at code completion, bug fixing, code generation, \
and multi-language tasks. Strong on complex tasks requiring deep reasoning.

2. **claude-sonnet-4-6**: High. Good at code completion, bug fixing, and multi-language tasks. \
Good balance of speed and quality.

3. **gpt-5.4**: High. Strong at code refactoring and test generation. \
Good overall capabilities with competitive performance.

4. **glm-5**: Mid. Strong at algorithm design and bug fixing. \
Good cost-performance balance for algorithmic and code generation tasks.

5. **MiniMax-M2.7**: Mid. Strong at code refactoring and code completion. \
Cost-efficient option with good overall balance.

6. **kimi-k2.5**: Low. Competitive on data science and code understanding tasks. \
Very cost-efficient. Good for straightforward tasks.

7. **qwen3.5-plus**: Low. Exceptional at algorithm and competitive programming tasks. \
Also good at code completion. Best choice for algorithmic challenges.

8. **Qwen3-Max**: Low. Strong at test generation and algorithm tasks. \
Best quality-cost ratio for test generation scenarios.

## Instructions

Analyze the task and choose the model that maximizes quality relative to cost.
Consider the task's dimension, difficulty, language, and complexity.
Prefer cheaper models when quality is comparable.

Respond with ONLY a JSON object:
{"model": "<model_name>", "reasoning": "<brief explanation>"}
"""


def route_task(dimension: str, difficulty: str, language: str, prompt: str) -> str:
    """Build the zero-shot 'Task to Route' user message (prompts.py format)."""
    return (f"## Task to Route\n\n"
            f"**Dimension**: {dimension}\n"
            f"**Difficulty**: {difficulty}\n"
            f"**Language**: {language}\n\n"
            f"**Prompt**:\n{prompt}")


def chat(system: str, user: str) -> str:
    """Qwen3.5 chat scaffold, non-thinking branch (matches the base template)."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


NER_SYS = ("Extract named entities from the sentence. Return strict JSON only "
           "with keys people, places, dates. Each value must be an array of "
           "strings. Use [] when empty.")


def ner_req(rid: str, sentence: str) -> Request:
    return Request(rid, chat(NER_SYS, f"Sentence: {sentence}"), "ner-json", 96)


def dolly_req(rid: str, question: str) -> Request:
    return Request(rid, f"### Instruction:\n{question}\n\n### Response:\n",
                   "dolly-qa", 80)


def route_req(rid: str, dimension: str, difficulty: str, language: str,
              prompt: str) -> Request:
    user = route_task(dimension, difficulty, language, prompt)
    return Request(rid, chat(ROUTER_SYSTEM_PROMPT, user), "acrouter", 64)


def gpu_snapshot() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def sync() -> None:
    import torch
    torch.cuda.synchronize()


def first_json(text: str):
    """Best-effort extract the first balanced {...} object from text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-ner", required=True, type=Path)
    p.add_argument("--lora-dolly", required=True, type=Path)
    p.add_argument("--lora-acrouter", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--n-seq-max", default=8, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()
    print("[router] loading base + 3 task adapters + context pool", flush=True)
    router = ContextPoolRouter(
        str(args.model),
        adapters={"ner-json": str(args.lora_ner),
                  "dolly-qa": str(args.lora_dolly),
                  "acrouter": str(args.lora_acrouter)},
        n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed,
    )

    # A mixed workload: all three adapters interleaved in one submit.
    requests = [
        ner_req("ner1", "Barack Obama visited Berlin in July 2008."),
        dolly_req("qa1", "What is the capital of India?"),
        route_req("rt1", "code_generation", "hard", "python",
                  "Refactor this 2000-line legacy module and add a full test suite."),
        ner_req("ner2", "Alice met Bob near the Eiffel Tower on Monday."),
        dolly_req("qa2", "Name three primary colors."),
        route_req("rt2", "algorithm", "easy", "python",
                  "Write a one-line function that returns the sum of two integers."),
    ]

    print(f"[router] running {len(requests)} routed requests", flush=True)
    sync(); t0 = perf_counter_ns()
    gens = router.run(requests)
    sync(); routed_ns = perf_counter_ns() - t0
    routed_tokens = sum(len(g.tokens) for g in gens)

    print("[router] running isolated baselines", flush=True)
    sync(); t0 = perf_counter_ns()
    baselines = [router.baseline(r) for r in requests]
    sync(); base_ns = perf_counter_ns() - t0

    # gate 1: equivalence routed == isolated baseline
    per_request = []
    all_match = True
    by_rid = {r.rid: r for r in requests}
    for g, b in zip(gens, baselines):
        match = g.tokens == b.tokens
        all_match = all_match and match
        text = router.detokenize(g.tokens)
        per_request.append({
            "rid": g.rid, "adapter": g.adapter, "match_baseline": match,
            "prompt": by_rid[g.rid].prompt[:80],
            "output": text,
        })

    # gate 2: task behaviour
    out_by_rid = {pr["rid"]: pr["output"] for pr in per_request}
    ner_json1 = first_json(out_by_rid["ner1"])
    ner_json2 = first_json(out_by_rid["ner2"])
    ner_ok = all(j is not None and set(j) >= {"people", "places", "dates"}
                 for j in (ner_json1, ner_json2))
    # acrouter emits JSON {"model","reasoning"}; parse it and check the model field.
    route_parsed = {rid: first_json(out_by_rid[rid]) for rid in ("rt1", "rt2")}
    route_hit = [(route_parsed[rid] or {}).get("model") in BACKENDS for rid in ("rt1", "rt2")]
    route_ok = all(route_hit)

    # gate 3: isolation (same prompt, base vs each adapter)
    iso = {}
    iso_probes = {
        "ner-json": chat(NER_SYS, "Sentence: Barack Obama visited Berlin in July 2008."),
        "dolly-qa": "### Instruction:\nWhat is the capital of India?\n\n### Response:\n",
        "acrouter": chat(ROUTER_SYSTEM_PROMPT, route_task(
            "code_generation", "hard", "python",
            "Refactor this 2000-line legacy module and add a full test suite.")),
    }
    isolation_ok = True
    for name, prompt in iso_probes.items():
        base_g = router.run([Request("iso-base", prompt, BASE, 48)])[0]
        ad_g = router.run([Request("iso-ad", prompt, name, 48)])[0]
        distinct = base_g.tokens != ad_g.tokens
        isolation_ok = isolation_ok and distinct
        iso[name] = {
            "distinct_from_base": distinct,
            "base_output": router.detokenize(base_g.tokens),
            "adapter_output": router.detokenize(ad_g.tokens),
        }

    verdict = "GO" if (all_match and ner_ok and route_ok and isolation_ok) else "NO-GO"
    result = {
        "verdict": verdict,
        "gates": {
            "equivalence_routed_eq_baseline": all_match,
            "ner_json_parseable_keys": ner_ok,
            "acrouter_emits_backend_name": route_ok,
            "isolation_adapter_differs_from_base": isolation_ok,
        },
        "ner_parsed": {"ner1": ner_json1, "ner2": ner_json2},
        "acrouter_parsed": route_parsed,
        "acrouter_hits": dict(zip(("rt1", "rt2"), route_hit)),
        "config": {"n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max,
                   "seed": args.seed, "n_requests": len(requests)},
        "throughput": {
            "routed_tokens": routed_tokens,
            "routed_ms": round(routed_ns / 1e6, 1),
            "routed_tok_s": round(routed_tokens / (routed_ns / 1e9), 1),
            "baseline_ms": round(base_ns / 1e6, 1),
        },
        "per_request": per_request,
        "isolation": iso,
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "adapters": {"ner-json": str(args.lora_ner), "dolly-qa": str(args.lora_dolly),
                     "acrouter": str(args.lora_acrouter)},
    }
    router.close()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "qwen35_task_loras.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "gates": result["gates"],
                      "ner_parsed": result["ner_parsed"],
                      "acrouter_hits": result["acrouter_hits"],
                      "throughput": result["throughput"]}, indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
