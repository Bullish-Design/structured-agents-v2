# Project 22 — Pulling the llama.cpp fork + bindings into their own repo: recommendation

> 2026-07-31. This is the recommendation the 2026-07-31 study session never finished
> (spend limit hit mid-flight). It is grounded in recovered research: the two Explore
> subagent transcripts of that session, plus follow-up reads of
> `17-llama-cpp-inference-lab/` (`06-LLAMACPP-BUILD-WORKFLOW.md`,
> `19-P2-FORK-DESIGN.md`, `20-P2-MIXED-BATCH-GO.md`, `21-P2-THROUGHPUT.md`,
> `22-P2B-FUSION-TRADEOFF.md`), `PORT-PLAN-NANBEIGE-P2.md`, `devenv.nix`,
> `build-llamacpp.sh`, and the benchmark/test suites.

---

## 0. TL;DR

Your instinct is right and the timing is good: the fork work has outgrown a patch
inside structured-agents-v2. But the load-bearing constraint is **not** git branch
management — it is the **ABI anchor**: llama-cpp-python's bindings are hand-written
ctypes that match exactly one llama.cpp `llama.h` ABI. Every llama.cpp version bump
is an ABI question first, a feature question second. Design the whole system around
that fact and your branch-per-modification idea works beautifully; ignore it and you
get silent memory corruption on update day.

Recommended shape:

1. **`llama-cpp-fork`** — a real git fork of `ggml-org/llama.cpp`. One branch per
   modification (`feat/p2-mixed-batch-lora`, `feat/nanbeige-arch`, …), all merging
   down to `integration`. Keep this repo **clean** — upstream tree + our commits only.
2. **`python-llama-cpp-fork`** — a real git fork of `abetlen/llama-cpp-python`. Its
   `vendor/llama.cpp` submodule points at `llama-cpp-fork@integration`. This is where
   our custom bindings (the `seq_routing` ctypes surface, cffi bindgen output) live.
3. **`llama-core`** — the pulled-out Python library (`src/structured_agents/llama_core/`),
   the per-branch test suites, the benchmark harness, and the build/CI tooling. One
   repo; llama.cpp and python-llama-cpp enter as submodules.

Each modification branch carries its own **suite manifest** (capability probe + test
files). The nightly canary rebuilds against latest upstream and runs every branch's
suite — so when upstream moves, you get a per-branch "this one broke, fix it" report
instead of one undifferentiated breakage. CI is GitHub-Actions-formatted workflows
run by **Act** on the rig (CPU legs anywhere, CUDA legs on the 3060s), with the GPU
benchmark suite as a first-class serialized job.

---

## 1. What you actually have today (recovered research)

| Thing | Current state | Move to |
|---|---|---|
| P2 fork (mixed-batch multi-LoRA) | 267-line patch, 5 files, on anchor `c588c4f47` | `llama-cpp-fork` branch `feat/p2-mixed-batch-lora` (real commits) |
| Nanbeige arch | separate lineage `Nanbeige/llama.cpp@nanbeige42` (anchor +48 commits) | `feat/nanbeige-arch` ported onto the anchor |
| Build tooling | `build-llamacpp.sh` (profiles cpu-light/cuda-3060/p2fork) + build-manifest.json + abi_smoke_gate.py | `llama-core/ci/build/` |
| Bindings | `src/structured_agents/llama_core/seq_routing.py` (ctypes, capability-guarded) | `python-llama-cpp-fork` + `llama-core/src/llama_core/` |
| ABI safety | cffi bindgen spike (`llama_cffi_build.py`, `_llama_cffi.c`, `README-cffi-bindgen.md`) + draft workflow `ci/llama-cpp-bindgen.yml` | promote from spike to first-class gate |
| llama-cpp-python | 0.3.34 installed **only** in `.venv-spike`; not in pyproject/uv.lock | pinned via git submodule in `llama-core` |
| GPU test env | devenv scripts `project17-gpu-pytest`, `project20-gpu-pytest` (env exports + skipif gates) | port into `llama-core` CI jobs |
| Benchmarks | ~20 gate-runners in `benchmarks/project17`, artifact schema v1 (records/*.json, summary.json, gpu before/during/after, git-status-before) | port as `llama-core/bench/`, keep schema |
| CI | **none checked in at repo root**; one draft workflow in scratch; **no Act anywhere** | this project builds it |

The P2 fork is **validated** (token-exact 4/4 at batch 4; up to 2.24× sequential /
1.12–1.46× router throughput; cross-mode diffs proven to be batch-size FP
nondeterminism, not routing bugs). It is a proven asset — the pull-out must not
change its semantics, only its provenance.

---

## 2. Repo topology — why three repos, and the alternatives

### Recommended: three repos

```
llama-cpp-fork/                      # real fork of ggml-org/llama.cpp
  branches:
    upstream-track                   # moves with upstream (rebases onto tags)
    anchor/b10103                    # c588c4f47 == ABI anchor for llama-cpp-python 0.3.34
    feat/p2-mixed-batch-lora         # one branch per modification — YOUR plan
    feat/nanbeige-arch
    integration                      # merge-down target; what gets built & shipped
    release-<anchor>                 # tags, not branches: every released lib
  # deliberately NO harness, NO benchmarks, NO python — keep the fork surgical

python-llama-cpp-fork/               # real fork of abetlen/llama-cpp-python
  vendor/llama.cpp -> llama-cpp-fork@integration   (git submodule)
  llama_cpp/llama_cpp.py             # our additions (seq_routing surface if vendored)
  bindings/                          # cffi bindgen sources + generated binding
  # the fork pins ONE llama.cpp ref at a time = one ABI anchor = one release line

llama-core/                          # the pulled-out library + everything around it
  src/llama_core/                    # from structured-agents-v2 (decode, middleware,
                                     #   batching, grammar, router, prefix_cache,
                                     #   node_delta, lsp_tree, fingerprint, ...)
  vendor/llama.cpp -> llama-cpp-fork@integration   (submodule, for Mode A builds)
  vendor/llama-cpp-python -> python-llama-cpp-fork (submodule, installable package)
  suites/                            # per-branch test manifests (see §4)
  bench/                             # benchmark runners + A/B harness + baselines
  ci/build/                          # build-llamacpp.sh, profiles, manifest schema
  ci/workflows/                      # GitHub Actions yaml run by Act
  artifacts/                         # benchmark results, baselines (schema v1)
```

**Why keep the fork repo clean:** the whole point of "individualized test suites per
branch" is that a branch = one change against upstream. If the fork repo grows
harness code, every upstream rebase drags unrelated diffs through your suites. Upstream
llama.cpp moves fast; keep the fork surface minimal.

**Why python-llama-cpp as its own fork:** the ABI anchor is *per release*. When you
bump llama.cpp anchors, you also bump the binding release. Two moving parts, two
repos, one submodule edge between them — the pairing is explicit and versionable.

### Alternatives you might prefer

- **Two repos** (fold `llama-core` into `python-llama-cpp-fork`): fewer repos, but
  your benchmark harness and CI live in a fork whose upstream history churns. Don't.
- **One monorepo with `fork/` directories**: branches can't cleanly track "one aspect"
  across C++ and Python simultaneously, and the fork becomes unrebaseable. Don't.
- **Keep llama-core inside structured-agents-v2, only fork llama.cpp**: viable
  first step (see §8 Phase 0) but doesn't meet the "pull out" goal. The library and
  harness are currently welded to structured-agents-v2's devenv; pull them out.

---

## 3. Branch topology and the update workflow (your signature idea)

### Per-modification branches

Every modification you've made or will make becomes a branch off the anchor:

| Branch | Contents (as commits, not patches) | Suite |
|---|---|---|
| `feat/p2-mixed-batch-lora` | the 5-file P2 change (llama.h, llama-context.{h,cpp}, llama-graph.{h,cpp}) | `suites/p2-mixed-batch-lora/` |
| `feat/nanbeige-arch` | nanbeige GGUF arch + conversion tools, ported onto the anchor per `PORT-PLAN-NANBEIGE-P2.md` §1.2 direction | `suites/nanbeige-arch/` |
| `feat/<future>` | e.g. `seq-state-partial-flags` (Ornith's GatedDeltaNet partial state), MTP drafter tweaks | one suite each |

Rules:

1. **Every branch's base is the current anchor** (or `integration` if it needs another
   feature). The nanbeige port plan already proved the pattern: `merge-base` must be
   the anchor so the ABI never moves under the branch.
2. **A branch is mergeable only when its suite is green against the current
   `integration` build.** No green suite → no merge-down. This is the gate that makes
   the rest of the system safe.
3. **The anchor never moves silently.** `anchor/b10103` is a tagged commit. Bumping
   the anchor is a deliberate act that triggers the full pipeline (§5), because it
   potentially invalidates the binding ABI.

### The update workflow (upstream version bump)

```
1. canary job: fetch upstream master/tag, build stock, run EVERY branch's suite
   against that stock build.
2. Output: table of branch -> {pass, fail, skip(capability absent)}.
   - pass on stock      -> upstream adopted equivalent semantics; branch may be droppable
   - fail on stock      -> this modification broke against new upstream: fix it (suite
                           tells you which contract changed, usually a symbol/semantics)
   - skip on stock      -> capability absent (normal): suite's probe correctly gates
   A suite that PASSES on stock is your signal the branch's patch is redundant —
   run the branch's own tests against integration to confirm before deleting.
3. Fix failing branches one at a time — this is exactly your "fix them one by one".
4. When all green: bump anchor tag, rebase branches, merge down, build, release.
```

This converts "upstream moved and everything broke" into "these three branches broke,
here's which contract each one depends on". The per-branch suite design (§4) is what
makes that possible; without it you get one undifferentiated wall of failures.

---

## 4. Per-branch suite system — design

Each suite is a directory with three parts:

```
suites/p2-mixed-batch-lora/
  suite.json          # manifest: name, required env, capability probe, test globs
  probes/capability.sh   # e.g. nm -D libllama.so | grep -q llama_set_seq_adapter
  tests/              # pytest files (ported from tests/test_seq_routing*.py)
```

**suite.json schema** (extend the repo's existing fingerprint/benchmark conventions):

```json
{
  "suite": "p2-mixed-batch-lora",
  "branch": "feat/p2-mixed-batch-lora",
  "capability": {
    "probe": "probes/capability.sh",
    "required": true
  },
  "env": { "GPU": true, "MODEL": "Ornith-1.0-9B-UD-Q4_K_XL.gguf", "LIB": true },
  "tests": ["tests/test_seq_routing.py", "tests/test_seq_routing_gpu.py"],
  "skip_on_stock": false
}
```

Design rules (all already proven in the repo's fail-closed culture):

1. **Capability-gated, fail-closed.** The probe runs against the loaded lib. `required:
   true` + probe fails → suite **fails** (the fork is missing its capability — merge
   block). `required: false` + probe fails → suite **skips** (like
   `SeqRoutingUnavailable` → auto-fallback in `seq_routing.py`). A missing capability
   is never a silent pass.
2. **Stock-passthrough is meaningful.** Every suite must also run against a stock
   build. `skip_on_stock` marks suites whose tests fundamentally require the fork
   (they skip on stock); everything else must run and — when upstream has adopted the
   semantics — pass. This is the drift detector.
3. **Token-exact gates need a batch-1 ground truth.** The P2 correctness findings
   proved batched-greedy FP nondeterminism flips rare near-tied argmaxes as batch
   grows. Suites must compare against batch-1 isolated decode and tolerate known
   near-tie flips (report them, don't fail on them) — otherwise every batch-size
   change false-alarms the gate.
4. **State-restore branches need the partial-state flag.** Ornith is hybrid
   attention + GatedDeltaNet; `llama_state_seq_*` has
   `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`. Byte-count success ≠ semantic success
   (proven by `run_seq_state_spike.py`). Any suite touching state capture/restore
   must verify continuation equivalence, not byte counts.
5. **GPU tests pin GPU 0 and record the environment.** The rig's GPU 1 often hosts a
   vLLM runner. Every GPU suite records `gpu-before/during/after` csv and
   `git-status-before` — the artifact schema already does this; keep it.

---

## 5. Merge-down + build pipeline

```
merge-down: for each feature branch, suite green on current integration?
            -> git merge --no-ff into integration (keeps a merge commit per feature)
build:      ci/build/build-llamacpp.sh --ref integration --profile cuda-3060
            -> out-cuda-3060-<sha>/lib + headers + build-manifest.json
            manifest now records: base anchor, feature branches merged,
            patch_sha256, cmake flags, seq_adapter_routing flag
gate 1:     ABI smoke gate (abi_smoke_gate.py): fork symbols resolve, model loads,
            ~32 tokens generate, tokenizer round-trips
gate 2:     cffi bindgen: compile the binding against the REAL headers+lib of this
            build. Compile failure == ABI drift == STOP (no silent segfault path)
bindings:   python-llama-cpp-fork built from source (Mode A) with
            vendor/llama.cpp = integration  ->  one wheel, bindings+lib paired
library:    install wheel into llama-core venv, run full library suite
bench:      run benchmark suite, record artifacts against baseline, regression gate
release:    tag release-<anchor>-<featureset> in both forks
```

**Mode A vs Mode B discipline** (from `06-LLAMACPP-BUILD-WORKFLOW.md`):
- Mode B (`LLAMA_CPP_LIB_PATH` swap) = fast iteration, **same-ABI experiments only**.
  That's what the per-branch suites use to test many builds quickly.
- Mode A (rebuild python-llama-cpp from source) = what ships. The fork's
  `vendor/llama.cpp` submodule **is** the anchor. Never release a Mode-B-only
  artifact.

---

## 6. Benchmarking as a standard capability

The repo already has 80% of this. Port, don't rebuild:

1. **Runners**: port `benchmarks/project17/*` (seq-state/reuse/breakeven, prefix-cache
   + restore sweep, context-pool router, P2 mixed-batch/correctness/throughput,
   grammar-constrained, JSON workload, LoRA probes) into `llama-core/bench/` verbatim.
   They are already fail-closed gates with deterministic greedy.
2. **Artifact schema v1 is the contract**: `records/*.json` + `summary.json` +
   gpu/during/after + git-status-before + runtime-environment. Fingerprint every
   artifact with `llama_core.fingerprint` (strict immutable compat keys) so an
   artifact is provably tied to one lib+model+GPU.
3. **A/B harness**: `bench/ab.sh --a <manifest|libdir> --b <manifest|libdir> --workload X`
   runs the same workload against two builds (stock vs fork, masked vs fused P2a/P2b,
   old anchor vs new anchor), writes a side-by-side diff (aggregate tps, p50/p95 token
   latency, TTFT, breakeven points), and exits non-zero on regression beyond threshold.
   This is the decision-data machine for the A/B testing you want.
4. **Baselines**: stored under `artifacts/baselines/<workload>@<anchor>-<featureset>.json`.
   CI benchmark job fails if a p50 decode-tps regression exceeds threshold vs baseline
   **for the same model + GPU + driver** (record nvidia-smi driver version — you
   already do).
5. **Tiering**: smoke (fast, every PR), soak (longer, nightly), benchmark (GPU,
   serialized). The repo's existing smoke/soak/bench artifact naming maps directly.

---

## 7. CI/CD with Act (local, on the rig)

No CI exists at repo root today — you are not migrating, you are building. Use
GitHub Actions-format workflows executed by **Act** (works on the rig where the GPUs
are; CPU-only legs can also run on any machine).

```
ci/workflows/
  build.yml        # matrix (ref × profile): cpu-light on any runner, cuda-3060 on rig
  bindgen.yml      # cffi gate, requires build.yml output
  suites.yml       # every branch suite × {stock build, integration build}
  bench.yml        # GPU benchmark suite + A/B harness; serialized, label [cuda,rtx3060]
  canary.yml       # nightly: build latest upstream, run ALL suites → per-branch report
```

Design points:

1. **Label strategy**: jobs that need a GPU get `runs-on: [self-hosted, cuda, rtx3060]`.
   Act maps labels to local runners; the rig is where Act runs anyway. GPU jobs must
   be **exclusive** (one at a time) — two concurrent 9B-CUDA jobs on 2×3060s contend.
   Serialize with a lock file or a queue job.
2. **Act specifics**: keep workflows to constructs Act supports (no services/docker
   compose unless needed); pass model paths and cache dirs via secrets/`act --secret`/
   `.actrc`; `act -l` to list, `act --matrix` supported for the ref×profile matrix.
   GPU driver stubs: the repo already solved the libcuda-first `LD_LIBRARY_PATH`
   problem in devenv — bake that into the runner environment, don't rediscover it.
3. **Bindgen failure = ABI drift** semantics (from the draft `ci/llama-cpp-bindgen.yml`):
   the compile failure IS the signal. This job is the tripwire for the whole system.
4. **Canary cadence**: nightly cron (the draft uses `0 6 * * *`). The canary's output
   (per-branch pass/fail against fresh upstream) is the "should we bump?" decision
   input. On non-trivial upstream moves, also run the A/B bench (old anchor vs new
   anchor) before committing to the bump.
5. **No GitHub-hosted GPU**: the CUDA legs are rig-only. That's fine — your benchmark
   data is only comparable on your hardware anyway (that's the whole point of
   self-hosted benchmarking).

---

## 8. Phased plan

**Phase 0 — carve out (half a day).** Create `llama-cpp-fork` by cloning
`ggml-org/llama.cpp` and adding branches: `anchor/b10103` (c588c4f47) from the
existing `.llamacpp-builds/src`, `feat/p2-mixed-batch-lora` (apply the patch as
commits, preserving the manifest's patch_sha256), `feat/nanbeige-arch` (port per
`PORT-PLAN-NANBEIGE-P2.md` direction: nanbeige arch onto the anchor). Verify with
`build-llamacpp.sh` + `abi_smoke_gate.py` that `feat/p2-mixed-batch-lora` builds and
passes before touching anything else.

**Phase 1 — suite system + CI skeleton (2–3 days).** Build `llama-core` shell:
port `build-llamacpp.sh`, artifact schema, `llama_core` library, and the P2 suites
(port `test_seq_routing*.py` into `suites/p2-mixed-batch-lora/`). Write `build.yml`,
`bindgen.yml`, `suites.yml` and get them green under Act on the rig (CPU legs first,
CUDA leg second).

**Phase 2 — python-llama-cpp fork (1–2 days).** Fork `abetlen/llama-cpp-python`;
point its `vendor/llama.cpp` at `llama-cpp-fork@integration`; wire Mode A build into
the pipeline; integrate the cffi bindgen output; library suite in CI.

**Phase 3 — benchmarking + canary (2–3 days).** Port all benchmark runners, the A/B
harness, baseline store, and regression gate. Add `bench.yml` (serialized GPU job)
and `canary.yml` (nightly upstream sweep). First canary run against current upstream
produces your first per-branch impact report.

**Phase 4 — retire the patches.** Once the fork repos are live and green, delete
`patches/*.patch` and the `src-p2fork`/`src-nanbeige` vendored trees from
structured-agents-v2 (keep the docs; update devenv to point at the new repos).
structured-agents-v2 keeps importing `llama-core` as a dependency instead of owning it.

---

## 9. What you're not thinking about (honest list)

1. **ABI anchoring is the whole ballgame — not branch management.** Hand-written
   ctypes vs one `llama.h`. A struct field reorder is silent memory corruption, not
   a compile error. Your entire update story lives or dies on the cffi-bindgen gate.
   Promote it from spike to tripwire on day one.
2. **Bindings and lib are one paired unit.** Never ship a lib without its binding
   version recorded next to it (the manifest already records the anchor — extend it
   to record the binding ref). The submodule edge between the two forks is where
   version skew will bite; pin it in CI, not by convention.
3. **Every llama-cpp-python bump changes the anchor.** 0.3.34 → next release =
   different `llama.h` ABI. Your branches rebase onto the new anchor, and the P2
   patch is 267 lines touching `llama-graph.cpp` internals — expect real rebase work
   each bump. The per-branch suites are what make that work *measurable* instead of
   chaotic, not what makes it free.
4. **Token-exact gates are a floor, not a ceiling.** Batched-greedy FP flips near
   ties. Design gates with batch-1 ground truth and known-flip reporting or your CI
   will cry wolf on every batching change (P2 already hit this; don't re-learn it).
5. **Ornith's state is partial-flag territory.** Hybrid GatedDeltaNet means
   `llama_state_seq_*` has recurrent/partial semantics; byte-count success ≠ semantic
   success. State-touching branches need their own suite (this is a great first
   candidate for a new `feat/` branch — the research already documented it).
6. **One rig, serialized GPU.** CI concurrency on the 3060s is effectively 1–2 jobs.
   Design GPU jobs to be exclusive and idempotent, or your CI queue becomes your
   bottleneck. Storage also grows fast (builds + GGUF + artifacts) — curate or the
   disk dies quietly mid-benchmark.
7. **Upstream moves fast and its AGENTS.md forbids upstream contribution from AI.**
   You are building a permanently-private fork. That's fine (MIT), but expect
   perpetual rebase tax; the canary turns it from a surprise into a schedule.
8. **The hard part of combining features is inside `build_lora_mm`.** The nanbeige
   port plan already identified it: threading a loop-step index into per-sequence
   adapter selection (the layer knows nothing about the loop step today). If you plan
   nanbeige hats on the P2 fork, that is the design risk to spike early — not the git
   plumbing.
9. **Environment plumbing is real work.** The devenv venv has no `llama_cpp`; the
   bindings live in `.venv-spike`; `LD_LIBRARY_PATH` needs `/run/opengl-driver/lib`
   first. This does not magically become clean in a new repo — port the devenv
   entrypoints (`project17-gpu-pytest`, `project20-gpu-pytest`) as-is, then improve.
10. **Benchmarks are only comparable on your hardware.** That's the justification for
    self-hosted Act CI; it is also a trap if you ever run "the same benchmark" on a
    different machine and compare numbers. Record the full fingerprint (GPU, driver
    version from nvidia-smi, lib manifest, model sha) in every artifact — the
    fingerprint module already exists, use it as the artifact name.

---

## 10. Concrete first actions (today)

1. Create `llama-cpp-fork` with `anchor/b10103`, `feat/p2-mixed-batch-lora`,
   `feat/nanbeige-arch` from the existing `.llamacpp-builds` trees; verify the P2
   branch builds and passes `abi_smoke_gate.py` and `run_p2_mixed_batch.py`.
2. Port `build-llamacpp.sh` + `abi_smoke_gate.py` + the artifact schema into a new
   `llama-core` repo; add the P2 suite manifest.
3. Write `build.yml` + `bindgen.yml` + `suites.yml` and run them under Act on the rig.
4. Write the nanbeige-hats spike against the P2 fork (the `build_lora_mm` loop-step
   problem) — it's the earliest technical unknown and it gates the flagship use case.

Everything else (python fork, A/B harness, canary, baselines) is mechanical once 1–4
are real.
