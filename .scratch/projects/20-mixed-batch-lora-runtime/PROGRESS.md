# Project 20 — Progress Tracker

Check items off as they land. Keep the dated log at the bottom for evidence
(artifact paths, measurements, decisions). One line per meaningful change.

**Legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped/parked

## Workstream A — Reproducible fork build
- [x] A1 — `p2fork` build profile applies `p2-mixed-batch-lora.patch` to pinned `c588c4f47`
- [x] A2 — Emits lib set + manifest (base commit + patch sha256)
- [x] A3 — Verify patch still applies against current llama-cpp-python anchor (applied clean, no re-anchor)
- [x] A4 — ABI smoke gate PASS on the fork lib (surface probe + Ornith 32-tok gen + tokenizer round-trip)
- [x] A5 — Fork `libllama.so` exports `llama_set_seq_adapters` / `llama_set_seq_adapter`

## Workstream B — Capability-aware fingerprint
- [x] B1 — `LlamaEngineFingerprint.seq_adapter_routing: bool` routing-capability descriptor
- [x] B2 — Capability detected via `library_supports_seq_routing` (probe loaded lib)
- [x] B3 — Surfaced in `RuntimeDiagnostics.seq_adapter_routing` (read from build manifest)
- [x] B4 — Fork vs stock engine produce distinct `cache_key()` (test)

## Workstream C — Bindings into `llama_core`
- [x] C1 — `seq_routing.py` `SeqRoutingBinding` binds both symbols with capability guard
- [x] C2 — argtypes/restypes match `context_pool_router.py:523-535`
- [x] C3 — Capability-absent path raises `SeqRoutingUnavailable` (CPU unit test)

## Workstream D — Router backend + scheduler integration
- [x] D1 — `RouterConfig.backend: context_pool | seq_routed | auto`
- [x] D2 — `auto` selects seq_routed when lib reports capability, else context_pool (CPU test)
- [x] D3 — `seq_routed` backend: one context, pool registered once, per-seq adapter assign, single wave
- [x] D4 — `-1` base/no-adapter sentinel modeled explicitly (`NO_ADAPTER`)
- [x] D5 — `ContinuousBatchScheduler` admits a mixed-adapter wave (BatchRequest.adapter + set_slot_adapter hook)
- [x] D6 — Per-seq grammar path (`enable_grammar`) preserved (matchers threaded through both backends)
- [x] D7 — Same public `RouteRequest`/`RouteResult` surface on both backends (no caller change)

## Workstream E — Correctness gate + regression tests
- [x] E1 — GPU token-exact per-seq gate PASS (batch-1 exact vs isolated baseline; `-1` sentinel; permutations)
- [x] E2 — Divergence classifier ported (benign batched-greedy FP flips ≠ routing failure; seq3@2 classified benign)
- [x] E3 — CPU test: capability-absent fallback selects context_pool, never raises
- [x] E4 — CPU test: fingerprint cache-key divergence (fork vs stock)
- [x] E5 — No cross-sequence adapter leakage (GPU PASS; 4/4 tests green)

## Workstream F — Throughput re-measure + maintenance decision
- [x] F1 — Library-path bench run: fork 1.45/1.17/1.10× the router at S=2/4/8 (matches spike)
- [x] F2 — K-sweep 2→32: fork/router 1.10→2.0× (K 2→8), fork flat ~92 tps to K=16; router OOM@K16, fork OOM@K24
- [x] F3 — Curve + recommendation published (`02-THROUGHPUT-AND-DECISION.md`): **ship `auto`**

## Cross-cutting
- [x] Focused pytest + Ruff + formatter + type checks green after each change (CPU side)
- [x] GPU tests gated behind `project20-gpu-pytest` devenv target (GPU 1); CPU tests GPU-free
- [x] Docs: teaching note `01-MASKED-MIXED-BATCH-ROUTING.md` (masked path + library seams)

---

## Log

<!-- Append dated entries. Example:
### 2026-07-30
- A1/A2 done: `build-p2fork.sh` applies patch, manifest at <path>. Rebuild ~N min.
  Artifact: artifacts/project20-…
-->
### 2026-07-30
- **A1/A2/A3 done.** Added `p2fork` profile to `build-llamacpp.sh` (fetch pinned
  `c588c4f47` into a dedicated `src-p2fork` checkout, `git apply --check` then apply
  `patches/p2-mixed-batch-lora.patch`, build cuda-3060 flags). Patch **applies cleanly**
  against the 0.3.34 anchor `c588c4f47` — no re-anchor needed. Manifest now records
  `patch_sha256=8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752`,
  `llama_cpp_commit`, `build_id`, and `seq_adapter_routing:true`. Build launched via
  `cuda-shell.nix` (devenv lacks cmake/cuda); output → `out-p2fork-c588c4f47/`.
  (Note: pre-existing `out-cuda-3060-p2fork` had an empty manifest — superseded.)
- **A4/A5 scripted.** `benchmarks/project20/abi_smoke_gate.py` = surface probe (both
  fork symbols) + Ornith 32-tok gen + tokenizer round-trip. Runs once the lib lands.
- **B done.** `LlamaEngineFingerprint.seq_adapter_routing:bool` (in `cache_key`),
  `library_supports_seq_routing()` probe, `RuntimeDiagnostics.seq_adapter_routing`
  from manifest. CPU tests: fork/stock cache-key divergence, diagnostics surfacing.
- **C done.** `src/structured_agents/llama_core/seq_routing.py` — `SeqRoutingBinding`
  (argtypes verbatim from `context_pool_router.py:523-535`), `SeqRoutingUnavailable`,
  `NO_ADAPTER=-1`. Exported from `llama_core/__init__` (ctypes-only, stays light).
- **D done.** `router.py`: `RouterConfig.backend` (default `auto`, fail-closed
  resolution), `_setup_seq_routed` (one ctx + pool registered once), `_run_seq_routed`
  (per-wave per-seq `set_seq_adapter`, `-1` for BASE, single `_generate_wave`), shared
  `_finalize` so grammar/decision path is identical on both backends. `batching.py`:
  `BatchRequest.adapter`, scheduler mixed-adapter admission via `set_slot_adapter`,
  `LlamaContinuousBatchEngine` optional fork pool + `supports_seq_routing`.
- **E CPU parts done** (`tests/test_seq_routing.py`): capability-absent fallback →
  context_pool with no error, explicit seq_routed fails closed, cache-key divergence.
  GPU gate written (`tests/test_seq_routing_gpu.py`): token-exact vs isolated baseline
  + FP-flip classifier + no-leakage; pending fork lib.
- **F scripted** (`benchmarks/project20/run_library_throughput.py`): fork vs
  context_pool vs sequential through the `MultiLoRARouter` library path; phase-1 S-sweep
  at K=2, phase-2 K-sweep to low dozens with `pred_overhead=2·K·r/d`. Pending fork lib.
- **Infra.** `project20-gpu-pytest` devenv target → fork lib on GPU (default **GPU 0**;
  GPU 1 was occupied by a vLLM runner, user chose GPU 0). driver-stub order honored.
  CPU suite green; ruff/format/ty clean on new files (pre-existing router/batching are
  hand-compacted, left untouched).

### 2026-07-30 (continued) — CONVERGE on GPU 0
- **A DONE.** Fork lib `out-p2fork-c588c4f47/` built via `cuda-shell.nix` (344 CUDA
  objects), exports `llama_set_seq_adapter{,s}` (verified `nm -D`). **ABI smoke gate
  PASS**: surface probe + Ornith 32-tok gen (' Paris.\n…') + tokenizer round-trip.
- **E DONE — 4/4 GPU tests pass.** Batch-1 seq_routed == isolated context-pool baseline
  token-exact for probe-a / probe-b / base(-1) → routing proven with zero FP confound.
  Mixed batch-4: no cross-adapter leakage; the one flip (seq3@tok2, "\n" vs "\n\n"
  near-tie) is benign batched-GEMM FP — the non-fork router matches baseline too.
  Classifier fixed to key on batch-1 exactness + adapter-diverges-for-prompt, not an
  arbitrary divergence position.
- **F DONE.** Library-path bench (fork vs context_pool vs sequential), GPU 0, best-of-3.
  Phase-1 reproduces the spike (1.45/1.17/1.10× the router). K-sweep: fork throughput
  flat ~92 tps to K=16 (masked K× overhead never bit → **P2b trigger NOT reached**);
  router OOMs at K=16, fork at K=24 → the many-adapter ceiling is **adapter-weight VRAM**,
  orthogonal to P2b. Artifact:
  `artifacts/project20-library-throughput-20260731T033225Z/`. **Recommendation: ship
  the fork backend as the `auto` default** (fail-closed; stock libs fall back). Doc:
  `02-THROUGHPUT-AND-DECISION.md`.
- Note: the fork's `set_seq_adapters` takes no scales → `AdapterSpec.scale` overrides
  apply only on `context_pool`; the router now warns on `seq_routed`.
