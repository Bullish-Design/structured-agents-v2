# Kickoff prompt — should benchmarking become its own library and repo?

> Hand this document to an investigation agent verbatim (it is self-contained).
> Read-only investigation: study, analyze, and write a recommendation. Do not edit,
> build, or run benchmarks. Cite file paths + line numbers as evidence. Be honest
> about costs; do not rubber-stamp the hypothesis below.

---

## Mission

Investigate two coupled questions and return a written recommendation:

1. **Is pulling the benchmarking capability out into its own library + repo a good
   idea?** When does it pay off, and what does it cost? If yes, what is the cleanest
   possible architecture for the benchmarking library and its repo?
2. **What is the best, cleanest, most elegant architecture for the overall
   comprehensive custom pythonic llama.cpp library** we are carving out in parallel —
   so that the benchmarking library, the llama.cpp fork repos, and the bindings fork
   all slot together without drift, duplication, or circular coupling?

We are at the carving-out moment for everything (see Context §5); boundaries decided
now are cheap to change later and expensive to fix then. The quality bar is explicit:
*elegant* means minimal public surface, single source of truth for shared contracts,
no circular dependencies, capability-gated optional surfaces, and no hidden state —
not merely "works."

---

## Context

### 1. The rig and the discipline

- Two NVIDIA RTX 3060 (sm_86 / compute capability 8.6), driver 595.84, 12 GiB each.
  **GPU 1 is often occupied by a vLLM runner — benchmarks pin GPU 0 and record
  nvidia-smi before/during/after.**
- Models are small (~0.5–9B), GGUF, via llama.cpp. Primary research model:
  `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (hybrid attention + GatedDeltaNet — this matters:
  `llama_state_seq_*` has `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` recurrent-state
  semantics; byte-count state success ≠ semantic success).
- **Standing rule: never propose vLLM/SGLang as an answer.** The project owns the
  decode path in Python via llama-cpp-python's low-level ctypes bindings, on purpose
  (teaching/experimentation: control and understanding, not beating C++ throughput).
- Benchmark numbers are only comparable on this hardware; every artifact must carry a
  full fingerprint (lib manifest, model sha, GPU, driver version).

### 2. The library today: `src/structured_agents/llama_core/` (18 modules)

- `decode.py` — owned decode loop (`llama_decode` + sampler C API, logits hooks)
- `middleware.py` — composable `DecodeMiddleware` pipeline
- `batching.py` — own continuous/dynamic batching over one `llama_context`, `n_seq_max`
  slots (Pillar 4)
- `router.py` — multi-LoRA agent-router (context-pool + seq-routed backends)
- `seq_routing.py` — ctypes surface for the P2 fork's two C entry points;
  capability-guarded (`SeqRoutingUnavailable` → auto-fallback)
- `grammar.py` — XGrammar integration, torch-free hot path
- `prefix_cache.py` / `prefix_cache_live.py` — exact-prefix KV state snapshot cache
- `node_delta.py` / `node_delta_live.py` / `node_blend_live.py` / `lsp_tree.py` —
  codebase context-tree over KV state
- `fingerprint.py` — strict immutable compatibility keys for runtime artifacts
- `diagnostics.py` — runtime version diagnostics without heavy imports
- `benchmark.py` — `BenchmarkRecord`/`BenchmarkTimer`; deliberately **no GPU or
  llama-cpp-python dependency**
- `models.py` — Pydantic boundary types (`EngineConfig`, `GenerationRequest`,
  `GenerationResult`)
- Boundary philosophy (standing rules): Pydantic validates at the edges only; the hot
  path passes plain token ids + numpy views; a missing capability is never an
  inference failure.

### 3. The fork: a private, validated llama.cpp modification

- ABI anchor: llama.cpp commit `c588c4f47` (= build `b10103`, ggml 0.16.0) = the exact
  llama.cpp commit `llama-cpp-python 0.3.34`'s hand-written ctypes bindings match.
  **ABI-anchor rule: bindings are not generated; a struct/enum change is silent
  memory corruption. A cffi-bindgen gate (compile failure == ABI drift) is the
  tripwire.**
- P2 fork (`feat/p2-mixed-batch-lora`): 267-line patch, 5 files
  (`include/llama.h`, `src/llama-context.{h,cpp}`, `src/llama-graph.{h,cpp}`), adds
  `llama_set_seq_adapters` / `llama_set_seq_adapter` — mixed-batch per-sequence LoRA
  routing in one `llama_decode`. **Validated**: token-exact 4/4 vs isolated
  baselines; throughput up to 2.24× sequential / 1.12–1.46× the no-fork router.
- Nanbeige arch fork exists as a separate lineage (`Nanbeige/llama.cpp@nanbeige42`,
  anchor + 48 commits); port plan says port the arch ONTO the anchor.
- llama-cpp-python 0.3.34 is installed **only** in the spike venv
  (`.scratch/projects/17-llama-cpp-inference-lab/.venv-spike`); the devenv venv has
  no `llama_cpp`. Runtime lib swap via `LLAMA_CPP_LIB_PATH` (Mode B) for iteration;
  source rebuild (Mode A) for shipping.

### 4. The benchmarking estate today (in structured-agents-v2)

- `benchmarks/project17/` — 16 runners, each a fail-closed "gate" with deterministic
  greedy + token-exact equivalence checks:
  `run_seq_state_spike`, `run_seq_reuse`, `run_seq_batch_breakeven`, `run_prefix_cache`,
  `run_prefix_restore_sweep`, `run_context_pool_router`(+`_cached`), `run_p2_mixed_batch`,
  `run_p2_correctness_check`, `run_p2_throughput`, `run_router_grammar`(+`_cached`),
  `run_json_workload`, `run_ornith_lora_probe`, `run_qwen35_task_loras`,
  `run_native_state_decompose`, plus `workload.py`, `state_blob_model.py`,
  `context_pool_router.py` (the spike's no-fork router), `json_workload_manifest.json`
  (100/1000 entries, 12 categories, sha256-pinned, seed 17001, schema registry).
- `benchmarks/project20/` — `abi_smoke_gate.py` (behavioral gate for fork libs) and
  `run_library_throughput.py` (throughput through the shipped `MultiLoLARouter`
  library: sequential vs `context_pool` vs `seq_routed`).
- **Artifact schema v1** (per-run dir): `command.txt`, `git-status-before.txt`,
  `runtime-environment.txt`, `gpu-before/after/during` csv, `processes-*.txt`,
  `stdout-stderr.log`, `summary.json` + `records/*.json` (per-request `timings_ns`
  breakdown: prefill/decode/sampler/mask/matcher/candidate_array/tokenizer/detokenize/
  validation; metrics: decode+prefill tps, token-latency p50/p95, TTFT).
- GPU test env: devenv entrypoints `project17-gpu-pytest` / `project20-gpu-pytest`
  (export `LLAMA_CPP_LIB_PATH`, `CUDA_VISIBLE_DEVICES`, `LLAMA_TEST_MODEL`,
  `LD_LIBRARY_PATH` with `/run/opengl-driver/lib` first so the real libcuda wins).
  Pytest skipif gates on those env vars.
- **No CI checked in at repo root.** One draft GitHub-Actions workflow in scratch
  (`ci/llama-cpp-bindgen.yml`: BUILD → BINDGEN → SMOKE → BENCH, self-hosted
  `[cuda, rtx3060]` leg, nightly canary cron). **No Act configuration exists.**
- Known gate caveat: batched-greedy FP nondeterminism flips rare near-tied argmaxes
  as batch grows (proven by `21-P2-THROUGHPUT.md` correctness section) — token-exact
  gates need batch-1 ground truth + known-flip reporting.

### 5. The carve-out plan already drafted

`.scratch/projects/22-llama-cpp-fork-reorg/01-RECOMMENDATION.md` proposes three repos:

1. `llama-cpp-fork` — clean git fork of ggml-org/llama.cpp; branches
   `anchor/b10103`, `feat/p2-mixed-batch-lora`, `feat/nanbeige-arch`, `integration`.
2. `python-llama-cpp-fork` — fork of abetlen/llama-cpp-python; `vendor/llama.cpp`
   submodule → `llama-cpp-fork@integration`.
3. `llama-core` — the pulled-out library (`llama_core/`), per-branch test suites,
   benchmark harness, build/CI tooling.

**This investigation may revise repo 3's shape** (e.g., split the benchmark harness
into its own repo, or split differently). The per-branch suite system (suite.json
manifests + capability probes, canary running every suite against fresh upstream) is
part of repo 3's design and is in scope for the taxonomy question below.

### 6. Key references (read these)

- `.scratch/projects/17-llama-cpp-inference-lab/` — `00-CONCEPT.md` (pillars),
  `06-LLAMACPP-BUILD-WORKFLOW.md` (ABI-anchor rule, Mode A/B, cffi bindgen),
  `14-PER-SEQUENCE-LORA-GUIDE.md`, `19-P2-FORK-DESIGN.md`,
  `20-P2-MIXED-BATCH-GO.md`, `21-P2-THROUGHPUT.md` (incl. FP-nondeterminism
  correctness), `22-P2B-FUSION-TRADEOFF.md`, `ci/llama-cpp-bindgen.yml`,
  `build-llamacpp.sh`
- `.scratch/projects/19-moe-moa-reactive-inference/PORT-PLAN-NANBEIGE-P2.md`
  (two-fork merge analysis; the hard sub-problem: threading a loop-step index into
  adapter selection in `build_lora_mm`)
- `.scratch/projects/22-llama-cpp-fork-reorg/01-RECOMMENDATION.md` (the 3-repo plan)
- `src/structured_agents/llama_core/` (all 18 modules — read `benchmark.py`,
  `fingerprint.py`, `seq_routing.py`, `models.py` fully)
- `benchmarks/project17/*.py`, `benchmarks/project20/*.py`
- `tests/` (31 files; GPU-gated ones: `test_node_delta.py`, `test_prefix_cache_live.py`,
  `test_seq_routing_gpu.py`), `tests/conftest.py`
- `devenv.nix` (entrypoints listed above), `pyproject.toml` (note: no llama-cpp-python
  dependency — it lives in the spike venv only)

---

## Questions to answer, with evidence

### Q1 — Verdict: standalone benchmarking library + repo?

Compare at least three shapes and give a recommendation with a tradeoff matrix:

- **(a) Bench as a subpackage of the `llama-core` library** (e.g. `llama_core.bench`).
- **(b) Fully standalone `benchkit`-style library + repo** that depends on
  `llama-core` as a pinned version.
- **(c) Hybrid split**: *record-producing primitives* (BenchmarkRecord/timer, artifact
  schema, fingerprinting) stay in the library as public contract; the *harness*
  (runners, workload definitions, A/B tooling, baselines, regression gates, CI/Act
  workflows, GPU env plumbing) lives in a separate repo that pins the library.

Evaluate on at least: drift risk between lib and bench (how fast do internal surfaces
churn?), version-pairing cost, CI locality, dependency cleanliness of the library
(no pytest/CI baggage in the lib), developer ergonomics for iterating on gates vs on
the library, and where "benchmark" ends and "test" begins. State explicitly when (b)
or (a) would be the right call instead of your recommendation.

### Q2 — Taxonomy: where do the three kinds of checks live?

Classify the existing 16+ runners into: (i) **validation gates** (token-exact
correctness of a mechanism), (ii) **performance benchmarks** (throughput/latency,
A/B-able), (iii) **soak/robustness runs** (grammar soak, GPU smoke). Recommend which
repo owns each class, and how the per-branch `suite.json` system from the 3-repo plan
interacts (e.g., does `feat/p2-mixed-batch-lora`'s validation gate live in the fork
suite or the bench repo?).

### Q3 — The benchmarking library's ideal architecture (if standalone)

Design the minimal public API and module tree. Requirements it must satisfy:

- Versioned, declarative **workloads** (the JSON workload manifest pattern: sha256-
  pinned corpora, schema registry, deterministic seed).
- One canonical **artifact schema** (evolve v1 or replace? justify), fingerprintable
  to lib+model+GPU+driver; single source of truth for the schema.
- **A/B harness**: run one workload against two builds, produce a side-by-side diff,
  exit non-zero on regression beyond threshold.
- **Baseline store** + regression gates.
- **CI/Act integration**: GitHub-Actions workflows executed locally by Act; GPU jobs
  serialized (2×3060s), labeled `[self-hosted, cuda, rtx3060]`; nightly canary that
  runs every fork branch's suite against fresh upstream.
- Zero dependency on fork-specific symbols at import time (capability-gated at run
  time, like `seq_routing.py`).

### Q4 — The `llama-core` library's ideal architecture

Given the carve-out, redesign the library's boundaries (you may keep or move modules):
module tree, public vs internal surface, where `benchmark.py`/`fingerprint.py`/
`diagnostics.py` live, how the Pydantic-at-the-edges rule and the
capability-gating pattern hold, and how the library relates to the two fork repos and
the bench repo (dependencies, version pairing, no circular coupling). Address how the
"owned decode loop" philosophy (Pillar 3 middleware, Pillar 4 batching) maps onto the
public API.

### Q5 — Elegance criteria

Define what "clean/elegant" means operationally for this system (e.g., minimal public
API, no circular deps, single source of truth per contract, capability-gated
surfaces, no hidden state, artifacts reproducible from a fingerprint alone). Then
score your own recommendation against it and state where elegance loses to pragmatism.

---

## Deliverables

1. Verdict on Q1 with tradeoff matrix and explicit conditions under which the other
   options win.
2. Recommended module trees for the benchmarking library/repo AND for `llama-core`
   (text-tree diagrams + 1-line purpose per module; moved modules called out).
3. Taxonomy table: every existing runner → class → target repo.
4. Artifact schema decision (v1 keep/evolve) with a concrete schema sketch.
5. Versioning/pairing strategy between bench repo ↔ library ↔ fork repos.
6. CI/Act workflow sketch (jobs, labels, serialization, canary).
7. Phased implementation plan (what to do first, what can wait).
8. Risks + things the requester is likely not appreciating.

## Ground rules

- Read-only. No edits, no builds, no benchmark runs.
- Cite file:line evidence for every claim about the existing code.
- Be adversarial with the hybrid hypothesis in §Context 5 — prove it right or wrong
  with evidence, and say what you would need to believe to flip.
- The deliverable is a single written document (markdown), self-contained enough that
  a different engineer could execute phase 1 without re-reading this prompt.
