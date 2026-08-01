# Project 22 — Benchmarking as a library: investigation + recommendation

> 2026-08-01. Answer to `02-BENCHMARK-LIBRARY-KICKOFF-PROMPT.md`. Read-only
> investigation; every claim about existing code cites `file:line`. This document
> is self-contained: a different engineer can execute Phase 1 of §7 without
> re-reading the kickoff prompt. The verdict revises repo 3's shape from
> `01-RECOMMENDATION.md` §0 as explicitly permitted by the kickoff ("This
> investigation may revise repo 3's shape").

---

## Q1 — Verdict: standalone benchmarking library + repo?

### TL;DR

**Adopt a hybrid split — but with the harness as a separate package *inside the
`llama-core` repo*, not a separate repo.** The *record-producing primitives*
(`benchmark.py`, `fingerprint.py`, `diagnostics.py`, boundary `models.py`) stay in
the library as public contract. The *harness* (runners, workloads, A/B tooling,
baselines, regression gates, CI/Act workflows, GPU env plumbing) lives in a
top-level `bench/` package in the same repo, is **never shipped in the wheel**, and
pins the library package via lockfile. This is option (c) with the "separate repo"
half downgraded to "separate package boundary" — and an explicit, near-zero-cost
promotion path (`git subtree split` + a new repo) if the flip conditions in §1.4
occur.

The load-bearing reason is **environment coupling, not code coupling**: the entire
GPU test/bench environment is repo-scoped today, and splitting it across repos buys
nothing while costing real ops duplication.

### 1.1 The evidence that decides the shape

**What is already clean (stays in the library):**

- `llama_core/benchmark.py:4-7` — the record primitives have *no* GPU or
  llama-cpp-python dependency, "so examples and unit tests can exercise artifact
  production on any development machine." They are CPU-testable today:
  `tests/test_llama_core_benchmark.py:7` imports `BenchmarkTimer,
  write_benchmark_record` and asserts the exact artifact field set
  (`test_llama_core_benchmark.py:10-31`) with no GPU.
- The record schema is already versioned and stable: `schema_version: int = 1`
  (`benchmark.py:58`), a frozen `TIMING_FIELDS` tuple (`benchmark.py:24-38`),
  atomic writer (`benchmark.py:163`).
- `fingerprint.py` is not bench-owned: `LlamaEngineFingerprint.cache_key()`
  (`fingerprint.py:99`) is part of the library's *runtime* contract (grammar and
  prefix-cache keys), and it already tracks fork-ness via `seq_adapter_routing`
  (`fingerprint.py:81`).
- The library already draws the public/internal line correctly:
  `__init__.py:20-23` keeps the shared import path lightweight; heavy native
  modules (`router`, `decode`) are imported directly by callers.
- The wheel is already guaranteed clean by packaging: `pyproject.toml:37-38`
  (`[tool.hatch.build.targets.wheel] packages = ["src/structured_agents"]`).
  pytest/CI files cannot enter the wheel today, so "no pytest/CI baggage in the
  lib" does **not** require a separate repo.

**What is actually coupled to the repo (the decisive fact):**

- The harness runs the library from its source tree, not as an installed package:
  `benchmarks/project20/run_library_throughput.py:32`
  (`sys.path.insert(0, .../src)`), `benchmarks/project17/run_json_workload.py:17`.
- The harness reaches *into the repo's examples/* — a coupling wart that proves how
  repo-shaped this code is: `run_json_workload.py:19`
  (`from examples.soak_grammar import _model_for_schema, ...`).
- The GPU environment is repo-scoped plumbing: `devenv.nix:131-182` (`project17-` /
  `project20-gpu-pytest` entrypoints exporting `CUDA_VISIBLE_DEVICES`,
  `LLAMA_CPP_LIB_PATH`, `LLAMA_TEST_MODEL`, and the libcuda-first
  `LD_LIBRARY_PATH` fix at `devenv.nix:147,179`). The bindings live only in the
  spike venv (`.venv-spike`); `pyproject.toml:8-21` has **no** llama-cpp-python
  dependency. Any repo that runs benches must own this environment story. One repo
  = one story; two repos = two stories to keep in lockstep (the exact drift the
  01 doc flags as real work, `01-RECOMMENDATION.md` §9.9).
- Runners also import each other (workload modules are local imports:
  `run_json_workload.py:20` `from workload import ...`), so the harness is
  currently one tightly-interwoven directory, not a reusable package.

**Drift-risk facts (Q1's central question):**

- The *internal* surface (decode/batching/router/grammar) churns fast — the whole
  `llama_core/` set was re-touched on 2026-07-30/31 and Aug 1 (file mtimes;
  corroborated by `17-.../24-ROUTER-FOLDED-INTO-CORE.md`). The *contract* surface
  (`benchmark.py`, `fingerprint.py`, `models.py`) has been stable since Jul 24-25.
- So the drift-risk between lib and bench is **localized to the stable contract**,
  and it is a one-way edge: harness → library contract. Nothing in the library
  imports the harness. The hybrid split exactly follows that one-way edge.
- "Where does benchmark end and test begin": a validation gate that asserts
  token-exact equivalence is a **test** (runs in CI on every relevant change,
  belongs to the suite system). A perf run asserts nothing — it produces artifacts,
  and only becomes a *gate* in the presence of a baseline + threshold. That
  baseline-threshold layer is the bench package's only test-like surface.

### 1.2 Tradeoff matrix

| Criterion | (a) `llama_core.bench` subpackage | (b) standalone repo | (c′) hybrid, same repo, non-shipped `bench/` package |
|---|---|---|---|
| Drift risk lib↔bench | lowest (same package) — but churn and contract share a namespace | needs version-pairing ceremony; A/B across lib versions requires wheel installs | one-way edge harness→contract; same commit day-to-day; A/B via pinned wheels |
| Version-pairing cost | zero | highest (bench repo pins lib releases) | zero for normal runs; explicit for A/B |
| CI locality (Act) | library CI carries bench jobs | two runners/envs, two `.actrc` | one rig, one env, path-filtered jobs |
| Lib dependency cleanliness | **fails** — harness drags pytest/GPU env into the lib package | passes | passes (hatch already ships only `src/`, `pyproject.toml:37-38`) |
| Iteration ergonomics (gates) | best | worst (reinstall cycle per library change) | best (same source tree) |
| Iteration ergonomics (A/B) | fine | good (installs two wheels) | good (harness installs two wheels into venvs) |
| Env duplication (devenv/venv/model paths) | n/a | **worst** — duplicates `devenv.nix:131-182` + `.venv-spike` + model mgmt | single story |
| Repo cleanliness (fork stays surgical) | n/a | n/a | ✓ (fork repos unchanged by this decision) |

### 1.3 Why not (a) or (b)

- **(a) fails dependency cleanliness.** The harness needs pytest, a GPU environment
  contract, workloads, baselines, and CI/Act files. Shipped inside the library
  package, those become the library's problem for every consumer (the deploy/
  endpoints in this repo import `structured_agents`, not benches). The brief's
  elegance bar ("minimal public surface") rules this out.
- **(b) fails ops reality.** The bench cannot run without the library's environment
  (GPU entrypoints, spike venv, `LD_LIBRARY_PATH` fix, model paths). A second repo
  means two environments that must never drift, and a reinstall cycle for every
  library change — the exact "developer ergonomics for iterating on gates" cost
  the brief asks about. It also strands the 16+ validation gates, which are
  mechanism *tests*, not benches, in a bench repo (see Q2).

### 1.4 When the other options win (flip conditions)

- **Promote (c′) → (b)** if any of: (1) the bench harness gains consumers outside
  the rig/library (e.g., a public benchmark suite, or benchmarking *other*
  engines); (2) you want baseline reproducibility to be independent of library
  commit cadence *and* the install cycle becomes the bottleneck (it won't at this
  scale — a llama-core wheel build is minutes); (3) you open-source the harness.
  Cost of flipping: `git subtree split --prefix=bench` + a repo + env copy — cheap
  while bench is a self-contained directory, so the decision is reversible.
- **Fall back to (a)** only if the harness ever shrinks to a handful of smoke
  checks used solely inside the library's own test suite. Not the case (18
  runners).

---

## Q2 — Taxonomy: where the three kinds of checks live

**Classification rule.** (i) *validation gate* = asserts a token-exact/behavioral
equivalence, fail-closed, deterministic greedy; (ii) *performance benchmark* =
produces throughput/latency artifacts, A/B-able; (iii) *soak/robustness* = long or
adversarial runs without a correctness verdict. Several runners are deliberately
dual (validate *and* measure) — the taxonomy table splits them, because the two
halves land in different places (see the "dual" rows).

**Ownership rule.** Fork-branch capability gates → the per-branch suite system in
the `llama-core` repo (`suites/<branch>/`, per `01-RECOMMENDATION.md` §4) — the C++
fork repo stays python-free and surgical. Mechanism-level correctness of the
*library itself* → the library's own (CPU) pytest suite. Pure performance →
`bench/`. Soak/smoke → CI jobs (`bench.yml` soak leg / `bindgen.yml` SMOKE stage).

| Runner | Class | Target repo/package | Evidence (docstring) |
|---|---|---|---|
| `project17/run_seq_state_spike.py` | (i) validation — seq-state semantics | `suites/state-partial/` (or new `feat/` branch suite) | "Byte-count success is not semantic success" |
| `project17/run_seq_reuse.py` | (i) validation — restore into live multi-seq context | `suites/` (prefix-cache suite) | "restore a cached prefix into a *nonzero* sequence slot ... without disturbing its neighbours" |
| `project17/run_prefix_cache.py` | (i)+(ii) dual — correctness + break-even sweep | validation half → `suites/`; sweep half → `bench/` | "exact-prefix whole-state cache correctness and break-even sweep" |
| `project17/run_prefix_restore_sweep.py` | (ii) performance — length sweep | `bench/` | "Where does cached-prefix restore beat cold re-prefill?" |
| `project17/run_seq_batch_breakeven.py` | (ii) performance — break-even + `n_seq_max` throughput | `bench/` | "break-even ... + n_seq_max batched throughput" |
| `project17/run_context_pool_router.py` | (i)+(ii) dual | gates → `suites/router/`; throughput → `bench/` | "Gates (fail-closed, deterministic greedy) ... Throughput: aggregate tok/s" |
| `project17/run_context_pool_router_cached.py` | (i)+(ii) dual | gates → `suites/router/`; sweep → `bench/` | "Validate + benchmark cached shared-prefix restore" |
| `project17/run_p2_correctness_check.py` | (i) validation — FP-nondeterminism attribution | `suites/p2-mixed-batch-lora/` | "Is the S>=4 cross-mode mismatch a fork bug or batch-size FP nondeterminism?" |
| `project17/run_p2_mixed_batch.py` | (i) validation — fork capability | `suites/p2-mixed-batch-lora/` | "P2 fork validation: true mixed-batch multi-LoRA" |
| `project17/run_p2_throughput.py` | (ii) perf (+ cross-check) | `bench/` | "P2 throughput eval ... cross-checks that all three modes produce token-exact identical output" |
| `project17/run_router_grammar.py` | (i)+(ii) dual | gates → `suites/router/`; perf → `bench/` | "Grammar-constrained routing ... Gates (fail-closed, deterministic greedy)" |
| `project17/run_router_grammar_cached.py` | (i) validation — full path-(a) MVP | `suites/router/` | "Full path-(a) MVP ... Gates" |
| `project17/run_json_workload.py` | (ii) perf baseline (+ soak mode) | `bench/` | "GPU-only sequential benchmark runner ... JSON workload"; `--baseline-only` |
| `project17/run_ornith_lora_probe.py` | (i) validation — go/no-go probe | `suites/ornith-lora/` (one-off; archive to docs) | "go/no-go: does llama.cpp apply a LoRA delta to Ornith's hybrid arch?" |
| `project17/run_qwen35_task_loras.py` | (i) validation — adapter isolation | `suites/router/` | "eyeball that each adapter actually *does its job* ... no bleed-through" |
| `project17/run_native_state_decompose.py` | (ii) measurement | `bench/` | "decomposition of whole-state restore cost" |
| `project20/abi_smoke_gate.py` | (iii) smoke/robustness — behavioral lib gate | CI gate (`bindgen.yml` SMOKE stage) | "ABI smoke gate for the P2 fork lib" (`abi_smoke_gate.py:5`) |
| `project20/run_library_throughput.py` | (ii) perf — shipped library path | `bench/` | "throughput re-measure through the shipping library path" |
| `tests/test_grammar_soak.py` | (iii) soak | `bench/` soak leg / nightly | (pytest soak over grammar) |

**Interaction with `suite.json` (the 3-repo plan's per-branch suite system):**
`feat/p2-mixed-batch-lora`'s validation gates (`run_p2_mixed_batch`,
`run_p2_correctness_check`) live in the **fork suite** — `suites/p2-mixed-batch-lora/`
in the `llama-core` repo — because the suite manifest's capability probe is exactly
a fork-surface question (`nm -D libllama.so | grep llama_set_seq_adapter`, per
`01-RECOMMENDATION.md` §4.1). They are *executed* against that fork branch's build
by `suites.yml` in CI; the C++ fork repo itself contains no Python. Pure-perf
benches are *not* part of any branch suite — they run from `bench/` against
whatever build the pipeline staged (stock or `integration`), and the A/B harness
reuses the same `suite.json`-style capability probe to decide what is runnable.

---

## Q3 — The bench package's ideal architecture

### 3.1 Module tree (`bench/` inside the llama-core repo; promoted to its own repo only under §1.4 conditions)

```
bench/
  pyproject.toml              # bench-only deps: pytest, numpy, torch (hard, for CUDA sync), llama-core pin
  benchkit/                   # import package — NEVER shipped to PyPI / never in the wheel
    __init__.py               # public: run(), ab(), gate(); capability-guarded imports
    env.py                    # nvidia-smi snapshot, require_gpu0_only, driver_version, env fingerprint
                              #   (port: run_json_workload.py:35-43 require_gpu0_only)
    runenv.py                 # isolated run dir: command.txt, runtime-environment.txt, stdout-stderr.log,
                              #   git-status-before; subprocess orchestration with fail-closed exit
    envelope.py               # RunEnvelope v2 (Pydantic) — the canonical run contract (§4)
    workload.py               # Workload registry: versioned declarative workloads (manifest + sha256 +
                              #   schema registry + seed) — generalize json_workload_manifest.json +
                              #   workload.py:13-38 validate/select; add class registry so runs are
                              #   `bench run <workload>@<version>`
    ab.py                     # A/B harness: same workload vs two builds → side-by-side diff
                              #   (aggregate tps, p50/p95 token latency, TTFT, breakeven points);
                              #   exit non-zero on regression beyond threshold
    baselines.py              # baseline store + regression gate: artifacts/baselines/<workload>@<anchor>-<featureset>.json
    gates.py                  # the token-exact validation gates (ported run_*.py), fail-closed, batch-1
                              #   ground truth + known-flip reporting (21-P2-THROUGHPUT.md:35-43)
    runners/                  # thin CLIs over the library: throughput, workload, state sweeps, soak
  workloads/                  # versioned corpora: json_workload/{manifest.json,100.jsonl,1000.jsonl,
                              #   schema_registry_v1.json}; future workloads as new versioned dirs
  ci/                         # Act/GHA workflows: build/bindgen/suites/bench/canary (§6) + .actrc + labels
```

### 3.2 Requirements coverage

- **Versioned declarative workloads** — proven pattern to keep:
  `json_workload_manifest.json` (`corpus_version`, per-file `sha256`, `generation_seed`,
  `prefix_contract`, `schema_registry`) + `workload.py:13-38` (deterministic
  validation + stable-prefix `select()`). Tested by `tests/test_project17_workload.py`
  (100-line prefix of the 1000 corpus is byte-identical, sha256 matches manifest).
- **One canonical artifact schema** — see §4 (keep records v1; replace the
  *directory* contract with envelope v2).
- **A/B harness** — `bench ab --a <build> --b <build> --workload X`; the same
  workload driven through two libs/builds; diff + threshold exit code. (The
  pattern already exists piecemeal in `run_p2_throughput.py` / `run_library_throughput.py`;
  `bench/ab.py` generalizes it.)
- **Baseline store + regression gates** — keyed by workload + fingerprint
  (model sha + lib manifest + GPU + driver), not by date. The library already
  provides the fingerprint type (`fingerprint.py:69-99`).
- **CI/Act integration** — workflows under `bench/ci/` (§6); GPU jobs serialized
  (2×3060s); labeled `[self-hosted, cuda, rtx3060]`; nightly canary.
- **Zero fork-symbol dependency at import time** — same pattern as
  `seq_routing.py:30,37-47` (`SeqRoutingUnavailable` on a stock lib; capability probe
  at bind time, never at import). The bench package probes `library_supports_seq_routing`
  before scheduling any fork-only runner; a fork-only bench on a stock build
  *skips*, never crashes.

---

## Q4 — The `llama-core` library's ideal architecture

```
llama-core/
  src/llama_core/                 # the wheel (only this ships; pyproject.toml:37-38 pattern)
    __init__.py                   # lightweight public surface (unchanged: __init__.py:20-23)
    models.py                     # boundary types EngineConfig/GenerationRequest/GenerationResult  [STAYS]
    benchmark.py                  # BenchmarkRecord/BenchmarkTimer/write_benchmark_record           [STAYS]
    fingerprint.py                # ArtifactIdentity/LlamaEngineFingerprint (runtime cache keys)    [STAYS]
    diagnostics.py                # collect_runtime_diagnostics (manifest-aware, no native import)  [STAYS]
    decode.py                     # OwnedLlamaDecoder — owned decode loop (Pillar 3)
    middleware.py                 # DecodeMiddleware/MiddlewarePipeline (Pillar 3)
    batching.py                   # ContinuousBatchScheduler/LlamaContinuousBatchEngine (Pillar 4)
    grammar.py                    # JsonSchemaGrammar + compiler cache (torch-free hot path)
    router.py                     # MultiLoRARouter (backend: context_pool | seq_routed | auto)
    seq_routing.py                # P2 ctypes surface, capability-guarded [UNCHANGED surface]
    prefix_cache.py / prefix_cache_live.py
    node_delta.py / node_delta_live.py / node_blend_live.py / lsp_tree.py
  suites/<branch>/                # per-branch suite manifests + tests (01-RECOMMENDATION.md §4)
  bench/                          # the harness package from §3 (same repo, not shipped)
  vendor/llama.cpp                # submodule -> llama-cpp-fork@integration (Mode A builds)
  vendor/llama-cpp-python         # submodule -> python-llama-cpp-fork (installable bindings)
  ci/                             # workflows, build profiles, .actrc
  tests/                          # CPU-only library tests (records, boundary models, middleware,
                                  #   batching without GPU); GPU-gated tests move to suites/ (§6.2)
```

**Moved modules, called out:** none of the 18 library modules move out of the
library. What moves is the *harness* — the 18 `benchmarks/project{17,20}` files
become `bench/` (with `workload.py`, `state_blob_model.py`,
`context_pool_router.py`, `schema_registry_v1.json` + the two corpus files moving
with their runners; the corpus files also stay byte-copied for a transition
period). `abi_smoke_gate.py` becomes a CI gate script under `ci/`. The library
keeps `benchmark.py`/`fingerprint.py`/`diagnostics.py`/`models.py` as the public
contract — this is the whole point of the hybrid: **the artifact contract ships
with the library, the machinery that produces artifacts does not.**

**Boundary rules preserved:** Pydantic at the edges only (hot path passes plain
token ids + numpy views — `router.py:9-13`, `models.py:3-7`); capability-gated
optional surfaces (`seq_routing.py:30,37-47`; `router.py:48,63-65` `auto` backend);
no hidden state; fail-closed.

**Repo relations (no circular coupling):**
`llama-cpp-fork` (no deps) ← `python-llama-cpp-fork` (`vendor/llama.cpp` submodule)
← `llama-core` (depends on the python fork's wheel + vendors llama.cpp for Mode A).
`bench/` depends only on the `llama-core` package (same repo, same commit) and its
own deps. Nothing in the C++ fork or bindings fork imports Python.

**One honest cleanup this carve-out should perform:** today the library depends on
`dbos` and `transformers` (`pyproject.toml:8-21`) and its test suite launches a
DBOS session for every run (`tests/conftest.py:8-25`). The pull-out is the moment
to give `llama_core` a minimal dependency set — it is currently welded to the
structured-agents-v2 app stack, which is exactly the "library has app baggage" the
brief's elegance bar forbids. This is the one place where the recommendation
*loses to pragmatism* if done now (see Q5 score); the safe sequencing is: keep the
dependency set unchanged through Phase 1-3, prune in Phase 4 when the last
structured-agents-v2-only consumer is gone.

---

## Q5 — Elegance criteria, operationalized + scored

| Criterion | Operational test | Score of this recommendation |
|---|---|---|
| Minimal public surface | `import llama_core` is light (`__init__.py:20-23`); wheel = `src/` only | ✓ preserved; bench never enters the wheel |
| Single source of truth per contract | one record schema (`benchmark.py:58`), one fingerprint type (`fingerprint.py:69`), one envelope (§4), one workload manifest format | ✓ records stay; envelope replaces ad-hoc summaries; manifest generalized |
| No circular deps | graph acyclic: fork → bindings → library → bench | ✓ |
| Capability-gated surfaces | missing capability = skip/fallback, never failure (`seq_routing.py:30,37-47`) | ✓ bench probes before scheduling fork-only runners |
| No hidden state | artifact reproducible from fingerprint + workload + commit alone | ✓ envelope v2 embeds full fingerprint + baseline ref |
| Reproducible artifacts | same fingerprint + workload + build ⇒ comparable numbers | ✓ (this is already the standing rule; envelope makes it machine-checkable) |

**Where elegance loses to pragmatism:** (1) the `bench/` package shares the library
repo rather than being a pure dependency-free repo — a concession to env
duplication; (2) the library keeps its app-stack dependencies through Phase 4
(§4, last paragraph); (3) the corpus files are byte-duplicated during the
transition (cheap, temporary); (4) validation gates that are *also* perf benches
are split across `suites/` and `bench/` — the split is by half, so a dual runner
must be carved into two files during the port (mechanical, contract-tested).

---

## Deliverable 4 — Artifact schema decision

**Keep records v1; evolve the run directory to envelope v2.** Justification:

- Records v1 (`benchmark.py:58`) are stable, CPU-tested, and byte-level
  comparable across every existing artifact (`records/*.json` in
  `artifacts/project17-*`). Changing them buys nothing.
- The *directory* contract v1 is where the weaknesses are, all observed in the
  wild: `artifacts/project17-seq-batch-20260724T202826Z/` contains
  `command.txt`, `git-commit.txt`, `gpu-before.json`, `gpu-after-load.json`,
  `gpu-during.csv`, `runtime-environment.txt`, `stdout-stderr.log`, `summary.json`
  — but (a) `summary.json` shapes are ad-hoc per runner (`run_json_workload.py:156`
  writes its own shape; the seq-batch dir has its own keys), (b) the fingerprint is
  **not** embedded in the artifacts (only `runtime-environment.txt`),
  (c) there is no baseline linkage, (d) there is no machine-readable exit/gate
  verdict, (e) `git-status-before` (named in 01) vs `git-commit.txt` (observed)
  naming drift.

**Envelope v2 sketch** (`envelope.json`, one per run dir):

```json
{
  "schema_version": 2,
  "run_id": "<uuid>",
  "workload": {"name": "json_workload", "version": "project17-json-workload-v1",
               "corpus_sha256": "82870ff2...", "requests": 1000},
  "environment": {
    "fingerprint": { ...LlamaEngineFingerprint.model_dump()... },
    "nvidia_smi": {"driver_version": "595.84", "gpus": ["..."]},
    "env": {"CUDA_VISIBLE_DEVICES": "0", "LLAMA_CPP_LIB_PATH": "..."}
  },
  "baseline_ref": "artifacts/baselines/json_workload@b10103-p2.json",
  "gate": {"kind": "validation|perf|soak|smoke", "verdict": "pass|fail|skip",
           "known_flips": [{"request": "r5", "token": 17, "reason": "near-tied argmax"}]},
  "summary": { ...canonicalized aggregate metrics (single Pydantic model)... },
  "records": "records/",
  "command": ["bench", "run", "json_workload@v1", "--requests", "1000"]
}
```

`records/*.json` keep `schema_version: 1` (backward compatible with the ~20
existing artifact dirs); a small migration script stamps an envelope onto old
dirs from their `runtime-environment.txt` + `git-commit.txt` so baselines can
reference them.

---

## Deliverable 5 — Versioning / pairing strategy

| Edge | Pairing mechanism |
|---|---|
| llama-cpp-fork branches | one branch = one modification off the anchor; `release-<anchor>` tags (01 §2, §3) |
| llama-cpp-fork → python-llama-cpp-fork | `vendor/llama.cpp` submodule pin = the anchor commit; bindings never bump without a fork tag (01 §2) |
| python-llama-cpp-fork → llama-core | llama-core lockfile pins the python fork release; Mode A wheel + library venv are one unit (06-LLAMACPP-BUILD-WORKFLOW.md:108-128) |
| llama-core → bench (same repo) | same commit for normal runs; A/B across lib versions = `bench ab` installs two pinned wheels into isolated venvs |
| artifacts → baselines | baseline key = `workload@<anchor>-<featureset>` + fingerprint (model sha, GPU, driver) — never a bare date |
| ABI guard | cffi bindgen compile-failure = ABI drift tripwire (06-LLAMACPP-BUILD-WORKFLOW.md:18-30; ci/llama-cpp-bindgen.yml) — unchanged, promoted to first-class |

---

## Deliverable 6 — CI/Act workflow sketch

```
ci/workflows/
  build.yml        # matrix (ref × profile): cpu-light anywhere, cuda-3060 on the rig
  bindgen.yml      # cffi gate — compile failure == ABI drift (port of the draft, keep stage
                   #   graph BUILD -> BINDGEN -> SMOKE -> BENCH; draft already path-filters:
                   #   + nightly cron: ci/llama-cpp-bindgen.yml:26-32)
  suites.yml       # every branch suite × {stock build, integration build}; capability probes
                   #   decide run vs skip (skip_on_stock)
  bench.yml        # bench/ package: perf benchmarks + A/B + regression gate
  canary.yml       # nightly cron "0 6 * * *" (draft's cadence): build latest upstream, run ALL
                   #   suites -> per-branch pass/fail/skip report; ALSO run soak legs here
```

Rules: GPU jobs `runs-on: [self-hosted, cuda, rtx3060]`, **exclusive/serialized**
(lock file or queue job — 2×3060s; `run_json_workload.py:35-43` already enforces
GPU-0-only isolation at runtime); model paths/cache via `act --secret` / `.actrc`;
bake the libcuda-first `LD_LIBRARY_PATH` fix (`devenv.nix:147,179`) into the
runner env, don't rediscover it (01 §7.2); CPU legs run on any machine, CUDA legs
rig-only — benchmark numbers are only comparable on this hardware anyway.

---

## Deliverable 7 — Phased implementation plan

**Phase 1 — carve out + suite skeleton (½–2 days).** Create `llama-cpp-fork`
(`anchor/b10103`, `feat/p2-mixed-batch-lora`, `feat/nanbeige-arch`), verify the P2
branch builds and passes `abi_smoke_gate.py` + `run_p2_mixed_batch.py` before
anything else (01 §8 Phase 0). Create `llama-core` repo shell: `src/llama_core/`
(import as-is), `tests/` (CPU tests only, incl. `test_llama_core_benchmark.py`),
`suites/p2-mixed-batch-lora/`, `ci/build.yml` + `bindgen.yml` + `suites.yml`,
`build-llamacpp.sh` + `abi_smoke_gate.py` under `ci/`. Get CPU legs green under
Act; CUDA leg second. **A different engineer can execute this with only
`01-RECOMMENDATION.md` + this document.**

**Phase 2 — python-llama-cpp fork (1–2 days).** Fork abetlen/llama-cpp-python;
point `vendor/llama.cpp` at `llama-cpp-fork@integration`; Mode A build wired;
cffi bindgen output integrated; library suite in CI.

**Phase 3 — bench package (2–3 days).** Port `bench/` per §3: envelope v2 +
migration stamp for old artifact dirs; workload registry (generalize the JSON
manifest); split the dual runners (Q2); `ab.py` + `baselines.py` + regression
gate; `bench.yml` serialized GPU job. Every runner's gate semantics preserved
byte-for-byte (they are already fail-closed; the port is a refactor with contract
tests, not a rewrite).

**Phase 4 — canary + pruning (2–3 days).** `canary.yml` first run → first
per-branch impact report; retire the patches from structured-agents-v2; prune
`llama_core`'s app-stack dependencies (dbos/transformers if unused by the core);
structured-agents-v2 imports `llama-core` as a dependency instead of owning it.

---

## Deliverable 8 — Risks + what the requester is likely not appreciating

1. **The harness is import-coupling soup, not a codebase.** `run_json_workload.py:19`
   reaches into `examples/soak_grammar`; runners import each other via bare local
   imports (`run_json_workload.py:20`) and the spike's `context_pool_router.py`.
   The port is a **refactor with contract tests**, not a copy. Budget for it in
   Phase 3; the CPU-testable record contract (`test_llama_core_benchmark.py`) is
   the net that catches regressions.
2. **Two GPU-testing truths already in the repo, easy to lose in the move:**
   (a) `run_json_workload.py:35-43` rejects any run where GPU 1 is busy (a vLLM
   runner often owns it) — the bench env must preserve this isolation check;
   (b) `tests/conftest.py:8-25` launches a DBOS session for *every* pytest run —
   the library's test suite carries app-stack baggage today, and the carve-out
   decides whether that follows the library or dies with it.
3. **The spike venv vs devenv venv split must be resolved, not inherited.**
   llama-cpp-python 0.3.34 is not in `pyproject.toml`/`uv.lock`
   (`pyproject.toml:8-21`); it exists only in `.venv-spike`, and `LLAMA_CPP_LIB_PATH`
   swaps it at runtime. The new repo needs **one** environment story, or every CI
   job re-derives this split and the bindings fall out of sync with the lockfile.
4. **`torch` is optional in the newest perf runner** (`run_library_throughput.py:55-61`
   swallows ImportError for `torch.cuda.synchronize`). Timing rigor then depends on
   whether torch is installed. The bench package should hard-require torch (or a
   `cudaEvent`-based sync) or document the fallback — a silent optional-sync is a
   reproducibility hole in a throughput benchmark.
5. **The FP-nondeterminism rule is load-bearing for every gate.**
   `21-P2-THROUGHPUT.md:35-43`: batched-greedy flips rare near-tied argmaxes;
   gates need batch-1 ground truth + known-flip reporting or CI cries wolf on
   every batch-size change. This is already designed into the suite system
   (01 §4.3) — but the bench package's A/B diff must apply the *same* rule to
   perf deltas, or A/B runs will false-fail on noise.
6. **Old artifacts can't be baseline-referenced without the envelope migration.**
   ~20 artifact dirs predate fingerprint embedding and baseline linkage. The
   migration stamp (§Deliverable 4) must land in Phase 3 or the first regression
   gate has no valid baseline to compare against.
7. **Storage and serialization are CI risks, not afterthoughts.** Builds (CPU +
   CUDA × refs), GGUFs (multi-GB), and artifacts grow without curation; 2 GPUs
   serializes to ~1-2 concurrent GPU jobs. Enforce `artifacts/` retention + the
   lock-file queue in Phase 3, not after the disk dies mid-benchmark.
8. **xgrammar is pinned `==0.2.1` (`pyproject.toml:16`)** — an ABI-sensitive
   native dependency that also gates the grammar benches. The fork reorg's CI must
   version-pin it like the llama.cpp anchor; upstream xgrammar moves will break
   the grammar runners, and that is a *scheduling* event for the bench queue.
9. **The interrupted-session lesson is real:** the original study died to a spend
   limit with the recommendation unwritten; the recovery cost was re-reading two
   subagent transcripts. The deliverable discipline (write the doc first, in
   phases, as you go) should be baked into how the Phase 3 port is executed —
   incremental markdown, not a single end-of-session dump.
10. **What you are likely not appreciating: the benchmark question is 90% decided
    by the artifact contract, not the repo topology.** The expensive-to-change
    decision is "what must every artifact contain to be comparable" (envelope v2 +
    fingerprint + baseline key). Whether the harness is a repo or a directory is
    cheap to flip later (§1.4). Spend the design budget on the envelope; the
    directory question is nearly free to revisit.

---

*End of investigation. All claims about existing code cite `file:line` in this
repo at commit `3898efd`; behavioral facts (FP nondeterminism, ABI anchor) cite
the scratch research docs listed in the kickoff's §Context 6.*
