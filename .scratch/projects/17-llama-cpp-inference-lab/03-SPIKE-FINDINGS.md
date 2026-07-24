# Spike findings — API feasibility (llama-cpp-python 0.3.34, CPU probe)

> Historical diagnostic only. Project 17 now runs all inference soaks,
> benchmarks, and evaluations on the CUDA/GPU build; this CPU probe must not be
> used as current runtime or performance guidance.

Method: isolated venv (`.venv-spike`), CPU wheel from the official index. On
NixOS the wheel needs nix `libstdc++.so.6` on `LD_LIBRARY_PATH` (stored in
`.stdcxx_dir`). Introspection only — no model load needed to answer the API
questions. GPU only matters for later perf numbers.

## Q1 — Logit access + owned sampler → GREEN ✅
- `Llama(logits_all=True)`, `_scores` buffer, `logits_to_logprobs` present.
- Low-level `llama_decode`, `llama_get_logits_ith`, `llama_batch_init` present.
- Verdict: we can own the logits→mask→sample loop. xgrammar integration surface
  is there (`llama_sampler_init_logit_bias`, `llama_sampler_apply`,
  `llama_sampler_chain_init`). Pillars 1–3 unblocked.

## Q2 — Multi-LoRA → CONFIRMS PATH (a), mixed-batch NOT supported ⚠️
- High-level `Llama` takes only a single `lora_path` at init.
- Low-level `llama_adapter_lora_init(model, path)` + `llama_set_adapter_lora`
  exist, but `set_adapter_lora` is **context-level** (ctx, adapter, scale).
- CORRECTION (from the cffi bindgen spike, verified against the shipped
  `include/llama.h`): the current primary symbol is the PLURAL
  `llama_set_adapters_lora(ctx, adapters**, n, scales*)` — `set_adapter_lora`
  (singular) is now a **deprecated Python compat shim** in the binding
  (`llama_cpp.py:2210`) forwarding to the plural with n=1. This spike recorded
  the deprecated alias. The plural sets/stacks N adapters on **one context** with
  per-adapter scales — still context-level, so it does NOT enable per-sequence
  mixed-adapter batching; the conclusion below is unchanged. Phase 3 should call
  `llama_set_adapters_lora` directly (and can compose multiple adapters per
  context).
- **`llama_batch` fields = [n_tokens, token, embd, pos, n_seq_id, seq_id,
  logits] — NO per-token adapter field.** => Cannot mix adapters in a single
  `llama_decode`. vLLM-style punica/SGMV mixed-batch is not available.
- **Architecture forced by this:** load base model ONCE, maintain a POOL of
  contexts, each `set_adapter_lora`-pinned to one router adapter; our scheduler
  batches requests *within* a context and multiplexes *across* contexts.
- This is exactly Decision D2 path (a). The gap vs. vLLM is now a measured
  teaching artifact, not a surprise.

## Q3 — KV save / restore / reuse → GREEN ✅
- `Llama.save_state` / `load_state`; low-level per-sequence
  `llama_state_seq_get_data` / `set_data` and `llama_memory_seq_cp(ctx, src,
  dst, p0, p1)` present (modern `memory` API, not old `kv_cache_seq_cp`).
- Verdict: LMCache-style prefix/KV reuse + cross-sequence copy feasible.

## Consequences for the plan
- Reframe the flagship: "massively parallel multi-LoRA" = **context-pool
  multiplexing on one shared base model**, not single-batch mixed adapters. Own
  the scheduler; measure adapter-swap / context-pool cost vs. a big general
  model. Still exercises every pillar.
- Perf numbers (adapter-swap latency, tok/s, context-pool VRAM on 2×3060 with
  Ornith-1.0-9B) still need a CUDA build — deferred; no `nvcc` on host, so use
  the prebuilt CUDA wheel index or build inside devenv. Not on the feasibility
  critical path.
- Reusable env recipe captured for the real library setup.
