# Project 20 — Mixed-Batch Multi-LoRA Runtime (productionize the P2 fork)

**Goal:** promote the validated Project-17 P2 fork — true single-`llama_decode`
mixed-adapter batching (the vLLM/Punica capability) — from a benchmark spike into
a first-class `llama_core` primitive: a fork-backed backend for `MultiLoRARouter`
selected automatically by engine-fingerprint capability, fed by the continuous
batch scheduler, and guarded by the same fail-closed correctness gate the spike
used.

This is **not** new inference research. The capability is already proven GO. This
project is the productionization: reproducible fork build, capability-aware
fingerprinting, bindings lifted out of `benchmarks/`, scheduler integration, and
a maintenance decision.

## Established evidence (do not re-derive)

- **Capability proven** (`17-…/20-P2-MIXED-BATCH-GO.md`): one context, one batch of
  4 seqs with adapters `[probe-a, probe-b, base(-1), probe-a]`, greedy 32 tok —
  **4/4 token-exact** vs isolated single-adapter baselines, outputs distinct,
  `-1` "no adapter" sentinel correct.
- **Throughput** (`17-…/21-P2-THROUGHPUT.md`): fork fastest at every batch size —
  up to **2.24× sequential**, **1.12–1.46× the shipping context-pool router**;
  one context, no per-adapter VRAM / `n_seq_max` split. Margin over the router
  shrinks as requests-per-adapter grows (1.46×→1.12× as S 2→8).
- **Correctness caveat** (`21`): residual token flips at larger batch are
  base-GEMM FP nondeterminism (present in the no-fork path too), **not** routing
  bugs. `run_p2_correctness_check.py` already separates the two.
- **Fusion deferred** (`17-…/22-P2B-FUSION-TRADEOFF.md`): masked path is K× LoRA
  FLOPs, only ~1.5% overhead at K=2/r=16 (`overhead ≈ 2·K·r/d`). Stacked
  `mul_mat_id` fusion earns its keep only at K ≈ low dozens (r=16) AND when the
  LoRA step is compute- not bandwidth-bound. Out of scope here.

## Existing assets to reuse

- **Patch:** `.scratch/projects/17-llama-cpp-inference-lab/patches/p2-mixed-batch-lora.patch`
  (5 files, base llama.cpp `c588c4f47`). Touch points: `llama.h` (2 symbols),
  `llama-context.{h,cpp}`, `llama-graph.{h,cpp}` (`build_lora_mm` routes +
  `llm_graph_input_seq_lora_mask`). No `llama_batch` ABI change.
- **Reference bindings:** `benchmarks/project17/context_pool_router.py`
  `enable_seq_routing()` / `run_seq_routed()` — ctypes on the loaded lib, guarded
  by `hasattr(lib, "llama_set_seq_adapters")`.
- **Correctness/throughput harnesses:** `benchmarks/project17/run_p2_mixed_batch.py`,
  `run_p2_correctness_check.py`, `run_p2_throughput.py`.
- **Library seams:** `llama_core/router.py` (`MultiLoRARouter`, `AdapterSpec`,
  `RouterConfig`, `RouteRequest`, `RouteResult`), `llama_core/fingerprint.py`
  (`LlamaEngineFingerprint`, `cache_key`), `llama_core/batching.py`
  (`ContinuousBatchScheduler`, `LlamaContinuousBatchEngine`, `BatchConfig`).

## Standing rules (inherited from 02-DECISIONS)

- Teaching wins over raw perf; the fork is only worth shipping if it earns a clear
  structural or measured win over the already-shipping context-pool router — and
  that trade is an explicit deliverable, not an assumption.
- Pydantic at boundaries only; hot path stays plain/ctypes/numpy.
- Fail closed: a non-fork lib must transparently fall back to the context-pool
  path. Missing capability is never an inference failure.
- The fork is a **private** build of the pinned commit; NOT upstreamed. Bindings
  are hand-maintained ctypes → the built commit must stay ABI-compatible with the
  installed llama-cpp-python anchor.

## Non-goals

- P2b stacked-`mul_mat_id` fusion (parked; trigger = measured K× overhead).
- Upstreaming the fork.
- Mixed-rank / rank-padding support beyond what the masked path already gives.
- Any adapter count / accuracy claim beyond the K≤8 already measured, until
  re-measured on the fork-backed library path.
- Speculative decode, native sampler bridge, GPU-resident masking (all parked).

---

## Workstream A — Reproducible fork build

Turn the one-off patched build into a repeatable artifact the library can point at.

### Tasks
- Extend `build-llamacpp.sh` (or add a sibling `build-p2fork.sh`) with a
  `p2fork` profile: fetch pinned `c588c4f47`, apply
  `patches/p2-mixed-batch-lora.patch`, build the CUDA-3060 target, emit lib set +
  manifest recording base commit + patch sha256.
- Re-verify incremental rebuild cost (headers touched → ~164 CXX objects, CUDA
  kernels cached) and the driver-stub env (`llama-cpp-gpu-driver-stub-fix`).
- Confirm the patch still applies against the currently-pinned llama-cpp-python
  anchor; if the anchor moved, re-anchor the patch and re-run the ABI smoke gate.

### Exit
- One documented command produces a `libllama.so` exporting
  `llama_set_seq_adapters` / `llama_set_seq_adapter`, with a manifest.
- ABI smoke gate (surface probe + Ornith gen + tokenizer round-trip) passes on it.

## Workstream B — Capability-aware fingerprint

The fork is a non-stock ABI. Cache keys, diagnostics, and backend selection must
reflect that so a fork lib and a stock lib never share cache state or get confused.

### Tasks
- Add a capability descriptor to `LlamaEngineFingerprint` — e.g.
  `seq_adapter_routing: bool` (and/or a `build_id` that already differs) — so
  `cache_key()` distinguishes fork from stock.
- Detect capability at engine construction: probe the loaded lib for
  `llama_set_seq_adapters` (mirror the benchmark's `hasattr` guard) and record it.
- Surface it in `RuntimeDiagnostics` / `collect_runtime_diagnostics` alongside the
  pinned version tuple.

### Exit
- A fork-built engine and a stock engine over the same model produce **different**
  `cache_key()`s; the routing capability is visible in diagnostics.

## Workstream C — Bindings into `llama_core`

Lift the ctypes routing bindings out of `benchmarks/` into the library, behind a
narrow typed surface.

### Tasks
- New module (e.g. `llama_core/seq_routing.py`): a `SeqRoutingBinding` that binds
  `llama_set_seq_adapters(ctx, adapters, n)` and
  `llama_set_seq_adapter(ctx, seq_id, idx)` on a loaded lib, with the same
  `hasattr` capability guard and clear error if absent.
- Keep argtypes/restypes exactly as in `context_pool_router.py:523-535`.
- Pydantic only at the config boundary; the binding itself is plain ctypes.

### Exit
- The binding is unit-testable without a GPU (capability-absent path raises the
  documented error); the GPU-gated path is exercised by Workstream E.

## Workstream D — Router backend + scheduler integration

Give `MultiLoRARouter` a fork-backed backend and wire it to the continuous batch
scheduler so a single decode can carry a mix of adapters.

### Design
- `RouterConfig` gains a `backend: "context_pool" | "seq_routed" | "auto"` field.
  `auto` picks `seq_routed` when the fingerprint reports routing capability, else
  `context_pool`. Same `RouteRequest` / `RouteResult` surface either way.
- `seq_routed` backend: one context (`n_seq_max ≥ wave size`), register the ordered
  adapter pool once via the binding, assign each admitted request a `seq_id` +
  `llama_set_seq_adapter`, single `_generate_wave` over the mixed batch. Model the
  `-1` "no adapter / base" sentinel explicitly.
- `ContinuousBatchScheduler` admission: with the fork backend, a wave may mix
  adapters (the structural win — no per-adapter context, no `n_seq_max`
  fragmentation). Assign pool index per admitted `BatchRequest`.
- Reuse the Phase-1 grammar path per sequence (`enable_grammar`) unchanged — grammar
  is per-matcher, orthogonal to adapter routing.

### Exit
- `MultiLoRARouter(backend="auto")` runs the identical example on stock (context-
  pool) and fork (seq-routed) libs, same public API, no caller change.
- Scheduler admits a mixed-adapter wave into one decode on the fork backend.

## Workstream E — Correctness gate + regression tests

Port the spike's fail-closed gate into the library test suite.

### Tasks
- GPU-gated test (skip pattern per `tests/test_llama_core_batching.py:229` etc.):
  ground truth = each seq decoded alone on a single-adapter context (proven path);
  one mixed-batch fork decode must be **token-exact greedy** per seq. Sweep 2–4
  adapters, adapters-per-seq permutations, the `-1` sentinel, mixed with base.
- Reuse `run_p2_correctness_check.py`'s divergence classifier so benign batched-
  greedy FP tie-flips are not reported as routing failures.
- CPU-only tests: capability-absent fallback (fork binding missing → context_pool
  path selected, no error surfaced to inference); fingerprint key divergence
  (Workstream B).

### Exit
- Fork-backed router passes token-exact per-seq gate on GPU; capability-absent
  fallback proven on CPU; no cross-sequence adapter leakage.

## Workstream F — Throughput re-measure + maintenance decision

The open D2/P4 call: is 1.12–1.46× over the shipping router worth maintaining a
private fork?

### Tasks
- Re-run the fork-vs-router-vs-sequential sweep through the **library** path (not
  the benchmark script) so the shipping code is what's measured. Same Ornith 9B /
  1×3060 / K, best-of-3; separate cold vs warm.
- Extend to larger adapter pools (K up to the low dozens) to find where the
  structural wins (one context, no VRAM split) dominate and where the masked K×
  LoRA overhead starts to show (sanity-check vs `overhead ≈ 2·K·r/d`).
- Publish the curve + a written recommendation: ship the fork backend as `auto`,
  ship it as opt-in only, or keep it a documented teaching artifact.

### Exit
- Reproducible benchmark record + explicit maintenance recommendation. If the fork
  is adopted, `auto` is defensible; if not, the reasons are recorded and the
  backend stays opt-in.

---

## Implementation sequence

1. **A** — reproducible fork build + ABI smoke (unblocks everything GPU).
2. **B** — fingerprint capability + diagnostics (CPU-testable).
3. **C** — bindings into `llama_core` (CPU-testable capability-absent path).
4. **D** — router backend + scheduler wiring.
5. **E** — correctness gate (GPU) + fallback/fingerprint regressions (CPU).
6. **F** — throughput re-measure + maintenance decision.
7. Run focused pytest, Ruff, formatter, and type checks after each meaningful
   change; GPU tests only in the CUDA facility.

## Completion criteria

- One documented command builds the fork lib with a manifest; ABI smoke passes.
- `MultiLoRARouter(backend="auto")` transparently uses seq-routing on a fork lib
  and context-pool on a stock lib, same public surface, fallback never fails
  inference.
- Fork and stock engines never collide in the cache (distinct `cache_key()`).
- Token-exact per-seq correctness gate passes on GPU; benign batched-greedy FP
  flips are classified as such, not as failures.
- A reproducible throughput record and an explicit ship / opt-in / park
  recommendation for maintaining the private fork.

## References

- `17-…/19-P2-FORK-DESIGN.md` — design + touch points.
- `17-…/20-P2-MIXED-BATCH-GO.md` — masked-path GO evidence.
- `17-…/21-P2-THROUGHPUT.md` — fork vs router vs sequential + FP analysis.
- `17-…/22-P2B-FUSION-TRADEOFF.md` — when the stacked fusion earns its keep.
- `17-…/14-PER-SEQUENCE-LORA-GUIDE.md` §7 — stacked layout / gate origin.
- Patch: `17-…/patches/p2-mixed-batch-lora.patch`.
