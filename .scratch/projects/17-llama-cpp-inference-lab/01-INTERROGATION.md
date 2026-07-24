# Interrogation — Project 17

Sharpest tensions to resolve before building. Grouped by risk.

## A. The batching pillar vs. what llama.cpp actually is
llama-cpp-python is a thin ctypes binding. Batching lives in C++
(`llama_batch` / `llama_decode`). A pure-Python scheduler CANNOT out-throughput
llama.cpp's own C++ continuous batching on the token hot path — Python per-token
overhead dominates.
- **Reframe needed:** we build a *request scheduler / admission-control /
  KV-sharing* layer around `llama_decode`, not a faster inner loop. That is a
  legitimate and teachable artifact; "beat llama.cpp perf from Python" is not.
- Decide: is the win **throughput** (unlikely from Python) or
  **controllability + understanding + fairness/scheduling policy** (yes)?

## B. Teaching vs. performance pull opposite directions
Readable, inspectable teaching code ≠ squeezing max perf from a C++ engine.
- Proposal: **teaching wins on conflict.** Performance is the *subject*, not the
  shipping SLA. Every optimization is a lesson with before/after benchmarks.
- Implication: correctness + clarity + measurement first; raw speed second.

## C. Pydantic on the hot path is a trap
Pydantic validation per-token/per-logit will wreck throughput and muddy the
lesson.
- Boundary rule: **Pydantic at config/request/result boundaries only.** Hot path
  (logits, KV, per-step) uses plain typed structures / numpy / ctypes views.
- The "editor ergonomics" goal is fully satisfied by boundary-layer typing.

## D. Why xgrammar when llama.cpp has native GBNF grammar sampling?
llama.cpp already constrains sampling with GBNF. Choosing xgrammar means we
**bypass llama's sampler and own the logits→mask→sample loop ourselves.**
- Upside: that ownership IS the teaching payoff (and enables our hooks/workflow).
- Cost: we reimplement the sampling loop in Python around `llama_decode` with
  `logits_all`/logit access. Confirm this is intended, not accidental.
- xgrammar rationale: faster mask compute, richer than GBNF, industry-aligned.

## E. Benchmarking: reuse vs. build
"Every industry-standard benchmark" = MMLU, GSM8K, HumanEval, IFEval, ... each
with existing harnesses (EleutherAI lm-eval-harness, etc.).
- Reuse `lm-eval-harness` as a backend + thin adapter? Or build our own runner?
  Recommend: **wrap the standard harness for industry benches; build our own
  lightweight runner only for bespoke/inference-optimization benches** (latency,
  TTFT, tok/s, KV-hit-rate, grammar-overhead — things standard harnesses don't
  measure).
- Parallel runpod orchestration is a durable-jobs system. We ALREADY have DBOS
  durable infra (projects 10, 11). **Reuse it** for fan-out benchmark runs
  instead of inventing a new orchestrator.

## F. What survives from the current repo?
- KEEP + repurpose: constraint codec work (01, 09), durable plane (10) for
  bench orchestration.
- RETIRE: sglang/vLLM provider abstraction (11), OpenAI-compat engine framing,
  Ornith GGUF sglang spike (16) — unless the GGUF loader learnings transfer to
  llama.cpp model loading.
- Decide: greenfield new top-level package, or in-place refactor of `src/`?

## G. MVP spine (proposed sequencing)
1. Typed llama-cpp-python wrapper (boundary Pydantic) + a single clean
   generate() with hookable **workflow/middleware** stages.
2. xgrammar-driven constrained decode from Pydantic models (owned sampler loop).
3. Benchmark harness: local single-run first (tok/s, TTFT, correctness).
4. THEN advanced plugins: custom batch scheduler, LMCache-style KV reuse.
5. THEN runpod parallel bench fan-out (on DBOS).
Rationale: pillars 1–3 give the interactive/teaching core and the measurement
loop; 4–5 are the "look what low-level control buys you" payoff, measured by 3.

## H. The flagship demo (the "so what")
The demonstrator needs ONE concrete, compelling story:
small specialized model + grammar + owned loop **beating** a big general model at
a real task (on the 2×3060 rig — see [[gpu-runner-topology]]).
- What task? What small model? What's the headline number? Undecided — needs a
  pick to anchor the whole library.
