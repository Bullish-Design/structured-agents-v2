# Project 17 — llama.cpp Inference Lab (working title)

**Status:** Brainstorm / interrogation
**Date opened:** 2026-07-23

## The pivot

Drop the sglang/vLLM OpenAI-compatible-engine framing. Rebuild the repo around
**llama.cpp via `llama-cpp-python`** as the single inference substrate.

New mission (two audiences, one artifact):
1. **A learning/teaching library about LLM inference optimization** — readable,
   inspectable, well-documented; the code is the lesson.
2. **A demonstrator** of what small, specialized models can do *once you own the
   low-level control loop* (logits, KV cache, batching, sampling, grammar).

## Pillars

### 1. Pythonic + Pydantic surface over llama-cpp-python
- Pydantic model representations of the inference primitives (params, requests,
  batches, KV state, sampler config, grammar spec, results).
- Motivation is partly *editor ergonomics*: LSP, go-to-def, go-to-ref,
  autocomplete, typed refactors — even where it's "just" a wrapper.
- Goal: make the inference process itself **easy to interact with**.

### 2. XGrammar integration, pythonic + pydantic
- Constrained decoding driven from Pydantic models (reuse the structured-agents
  heritage — projects 01, 09 constraint codec work).
- Grammar compiled from types, not hand-written GBNF.

### 3. Inference-as-a-Workflow
- Represent the inference process as a **workflow** you can add to, intercept,
  modify — hooks at every stage (pre-tokenize, per-step logits, post-sample,
  KV events, batch admission).
- Composable/pluggable middleware model.

### 4. Custom batched processing (plugin)
- Our own continuous/dynamic batching layer over llama.cpp, not the built-in
  server batching.

### 5. LMCache-style KV reuse (plugin)
- Prefix/KV caching, offload, reuse — our own take, as a plugin.

### 6. Benchmarking as a first-class citizen
- We tweak constantly, so measurement must be cheap and central.
- Run **established industry benchmarks** AND our own.
- **Local, one-at-a-time** for iteration.
- **Massively parallel on runpod (or similar)** for bulk verification / release
  CI — kick off every standard benchmark across models on a pod fleet at once,
  quality comparisons across models, etc.

## Open questions — see 01-INTERROGATION.md
