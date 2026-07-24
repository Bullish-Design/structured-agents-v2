# Project 17 — Consolidated PLAN

**Goal:** a teaching/learning library about LLM inference optimization, built on
llama.cpp via llama-cpp-python, demonstrating what small specialized models can
do when you own the low-level control loop. Flagship: a multi-LoRA agent-router
fleet on Ornith-1.0-9B.

**Sequencing decision:** shared core first, then pillars (grammar → cache →
flagship). **Gating spikes** (must pass before the dependent build effort):
Ornith hybrid-KV restore, sampler double-accept, tokenizer equivalence. CUDA
build + cache break-even are NOT upfront gates; they live inside the cache
pillar.

**Standing rules** (from 02-DECISIONS): teaching wins over raw perf; Pydantic at
boundaries only, plain/numpy on the hot path; xgrammar means we own
logits→mask→sample; batching win = control/understanding, not beating C++
throughput; reuse lm-eval-harness + DBOS rather than rebuild. Every optimization
ships with a before/after benchmark — that IS the lesson.

**In-place refactor:** grow the llama.cpp core inside existing `src/`; retire the
sglang/vLLM provider abstraction; keep constraint codecs + DBOS durable plane.

---

## Phase 0 — Shared core + gating spikes

Nothing pillar-specific is built until the three gates pass. The shared core is
the single spine both pillars and the flagship reuse (prevents the two intern
docs' parallel-stack duplication).

### 0.a Environment + pinned-version tuple
- Reproducible env: CPU spike venv already works (nix `libstdc++` on
  LD_LIBRARY_PATH — see `.stdcxx_dir`). Fold into devenv/uv the way prior spikes
  did (`launch-spike.sh` pattern).
- Record the lockstep tuple in runtime diagnostics: llama-cpp-python release +
  vendored llama.cpp commit + xgrammar release + (only if used) torch. Verify
  the actual vendored commit rather than trusting the intern doc's hash.
- Deliverable: `versions.py` emitting the tuple; a `devenv`-runnable entrypoint.

### 0.a2 llama.cpp build/integration workflow (see 06-LLAMACPP-BUILD-WORKFLOW)
- Own the substrate: build llama-cpp-python against any chosen llama.cpp — new
  release, our fork, or a rig-tailored lightweight build. Two modes verified:
  Mode A source rebuild (pinned/distributable), Mode B `LLAMA_CPP_LIB_PATH`
  runtime swap (fast iteration).
- Load-bearing constraint: bindings are hand-maintained ctypes → the built
  commit must be ABI-compatible with the installed binding's anchor. Fork off
  the anchor; bump llama-cpp-python + re-anchor to ride a newer llama.cpp.
- Deliverables: `build-llamacpp.sh` (scaffolded; ref + flag profile →
  `cpu-light`/`cuda-3060`, emits lib set + manifest), the ABI smoke-gate script
  (surface probe + Ornith gen + tokenizer round-trip), devenv target exposing
  gcc/cmake/CUDA toolkit (no nvcc on host). Feeds the version tuple in 0.a.
- This unblocks the CUDA build the cache pillar (Phase 2) needs, without being an
  upfront gate.

### 0.b Engine fingerprint module (shared)
- One `LlamaEngineFingerprint` (Pydantic, frozen) covering model digest,
  tokenizer digest, llama.cpp commit, context/KV/RoPE/SWA config, active LoRA
  digests, backend. Used by grammar cache keys, prefix-cache keys, and adapter
  selection alike.
- Model digest strategy: hash once at registration, key by file identity
  (size/inode/mtime + select GGUF metadata); don't rehash multi-GB on startup.

### 0.c Benchmark harness skeleton (shared)
- Local single-run first. Nanosecond timing breakdown: prefill, decode,
  fill-mask, apply-mask, sample, accept, detokenize, validate.
- Metrics surface: tok/s, TTFT, p50/p95 token latency, grammar-overhead %, (later)
  KV break-even, router accuracy. Reuse for every pillar. Parallel fan-out on
  DBOS is future (parked).

### 0.d GATE 1 — sampler double-accept — ✅ RESOLVED (see 07-GATE1-DOUBLE-ACCEPT)
- Answer: YES, `llama_sampler_sample` accepts the chain internally (pinned
  `llama.h:1488` documents it). Intern's hot path double-accepts stateful
  samplers — confirmed bug.
- Canonical contract chosen: Option B (own the candidate array; apply mask →
  `llama_sampler_apply` → select → `llama_sampler_accept` once →
  `matcher.accept_token` once). Never call chain-accept after
  `llama_sampler_sample`. Encode as a tested helper in Phase 1.

### 0.e GATE 2 — tokenizer equivalence on Ornith — ✅ RESOLVED (see 09-GATE2-TOKENIZER-EQUIV)
- PASS: llama.cpp GGUF vs HF base tokenizer (deepreinforce-ai/Ornith-1.0-9B)
  agree token-for-token on 26/26 probes + 600/600 fuzz, incl. special/tool
  markers. Reference tokenizer identified from GGUF metadata (qwen35, gpt2 BPE).
- KEY note: model logits dim (248320) > tokenizer vocab (248077) — grammar
  bitmask MUST be sized to n_vocab, not tokenizer vocab (padding hazard).
- xgrammar `TokenizerInfo` construction (consumes this same HF tokenizer) is a
  Phase-1 step; keep the numpy mask backend (avoid torch).

### 0.f GATE 3 — Ornith hybrid-KV restore — ✅ RESOLVED (see 08-GATE3-ORNITH-RESTORE)
- PASS: Ornith hybrid (GatedDeltaNet) state restores bit-exact (greedy),
  same-instance AND cross-instance (restart). Also cleared the prerequisite:
  Ornith loads + generates COHERENTLY in llama.cpp (overturns the sglang
  gibberish result).
- Cache pillar CAN use Ornith as base (no plain-transformer forced for
  correctness). State is large (~3.5 MB/token) → break-even TBD on GPU.
- Phase 2 contract: restore prefix KV then DECODE the suffix (logits not in saved
  state; get_logits after load_state is stale). Per-seq `llama_state_seq_*` and
  partial-prefix restore still to confirm in Phase 2.

**Phase 0 exit:** three gates resolved with evidence; shared fingerprint +
version tuple + bench skeleton runnable.

---

## Phase 1 — Grammar pillar (own the decode loop)

MVP = XGRAMMAR concept Phase 0+1 only, teaching-first.

- Own the loop: `llama_decode` → `llama_get_logits_ith` (zero-copy host view) →
  xgrammar `fill_next_token_bitmask` → `apply_token_bitmask_inplace` (**numpy
  backend, not torch**) → native `llama_sampler_sample` → `matcher.accept_token`.
  Accept contract per GATE 1.
- Mask BEFORE top-k/top-p/temperature; bitmask covers full logits dim (vocab
  padding), not tokenizer vocab.
- Pydantic → JSON Schema → one persistent `GrammarCompiler` per model, compiled-
  grammar cache keyed by the canonical (schema+options+tokenizer+version) hash.
  One matcher per sequence.
- Pydantic at the boundary: request/result/config models; hot path stays plain.
- Teaching artifact: a runnable notebook/script showing the decode loop step by
  step with the timing breakdown; "valid-by-construction" JSON from a small model.
- Benchmark: unconstrained vs. native GBNF vs. xgrammar-owned path; report
  grammar-overhead %.

**Exit:** 1,000 constrained generations, zero malformed outputs, zero matcher
rejections after masked sampling; measured overhead vs. native sampling; clean
cleanup under repeat runs.

---

## Phase 2 — Persistent prefix-KV cache (renamed, honest)

MVP = LMCACHE concept Phase 0+1 only. **Not** "LMCache integration" — a homegrown
persistent exact-prefix KV snapshot cache. Base model per GATE 3.

- Capture → persist → restore → suffix-only prefill, guarded by the strict
  engine fingerprint. Store exact token IDs, never restore on hash match alone.
- Checkpoint-boundary hash-map index (no radix tree yet). Filesystem blob store +
  SQLite/SQLModel catalog; atomic writes + checksums; fallback never fails
  inference.
- CUDA build lands here (prebuilt CUDA wheel or devenv build; no nvcc on host).
- **Break-even benchmark is the deliverable, not an afterthought:** on the 3060s,
  does read+restore beat re-prefill, and at what context length? Publish the
  curve. If negative at all realistic lengths, say so and scope the pillar down
  to a teaching demonstration of *why*.
- Teaching artifact: the break-even curve + a walkthrough of state-size scaling
  and the whole-sequence-duplication cost.

**Exit:** survives process restart with correct continuation; rejects
incompatible states; no cross-sequence leakage; reproducible break-even data.

---

## Phase 3 — Multi-LoRA agent-router flagship

Ties the pillars together into the demonstrator. Architecture forced by the spike
(03-SPIKE-FINDINGS): llama_batch has no per-token adapter field, so **one shared
base model + a pool of adapter-pinned contexts**; our scheduler batches within a
context and multiplexes across. NOT vLLM-style mixed-adapter single-batch.

- Context-pool scheduler over `llama_set_adapter_lora`; measure adapter-swap /
  context-pool cost and VRAM footprint per live adapter on 12GB.
- Each router emits valid-by-construction tool calls via the Phase 1 grammar path.
- KV-reuse (Phase 2) shares the base-prefix KV across contexts — this is what
  makes N adapters fit on 12GB, so the pillars are interlocked by design.
- Benchmark: router accuracy + latency + tok/s of the small-router fleet vs. a
  big general model on tool-call routing. This is the headline number.
- Native sampler (parked P-X1) is likely needed here for throughput — trigger the
  decision with Phase 3 profiling, don't pre-build.

**Exit:** the flagship demo runs end-to-end on the rig with a published
accuracy/latency comparison; the "own the low-level loop" thesis is demonstrated
and measured.

---

## Out of scope (see 05-FUTURE-PARKED)
Native C++ sampler bridge, speculative/jump-forward decoding, GPU-resident
masking, structural tags, real LMCache connector, distributed/multi-tier cache,
upstream llama.cpp KV-block API, multi-tenant security, compression. Each has a
revival trigger recorded.

## Open decisions to revisit
- Cache-pillar base model — RESOLVED by GATE 3 (Ornith OK).
- Whether to shed `transformers`/HF-tokenizer dep via the GGUF-derived path —
  after GATE 2 proves correctness (GATE 2 passed via HF path).
- When to trigger the native sampler — Phase 3 profiling.

## xgrammar / torch finding (2026-07-24)
- **xgrammar HARD-requires torch to install** — its `requires_dist` lists
  `torch>=1.10.0` (mandatory, not an extra), plus `apache-tvm-ffi`, `triton`
  (Linux x86_64), `transformers<5,>=4.38.0`, numpy, pydantic. So Phase 1 pulls
  torch. It stays OFF the hot path (numpy mask backend per standing rules) — torch
  is installed-but-idle.
- **Version conflict to handle:** xgrammar pins `transformers<5`; we installed
  transformers 5.14.1 for Gate 2. Installing xgrammar will downgrade transformers.
  Gate 2 tokenization is version-insensitive here, but pin deliberately.
- **Torch-free escape hatch (future):** build a cffi binding to xgrammar's C++
  core (matcher.h/compiler.h) the same way as the llama binding — drops the torch
  wheel entirely and aligns with the "own the substrate" thesis. Park until/unless
  the torch footprint becomes a real problem.
