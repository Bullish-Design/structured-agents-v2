# Kickoff prompt — `inferference`: the llama.cpp fork + build repo

> Hand this document to the first agent session in the new `inferference` repo
> verbatim (it is self-contained).
> This repo is the **fork-building** leg of the three-repo carve-out
> (`01-RECOMMENDATION.md`): `inferference` = the private llama.cpp fork AND its
> build pipeline. The python bindings fork and the python library are separate
> repos; this prompt does not create them.
>
> Working style: study first, then build, then verify, then document. Read-only
> against the source repo (`structured-agents-v2`) — it is the reference, never
> edit it. Cite file paths + line numbers as evidence. Build and run the gates;
> do not write benchmarks.

---

## Mission

Initialize `inferference` as the single source of truth for the private llama.cpp
fork and everything needed to build and validate it. Three coupled tasks:

1. **Real git fork of `ggml-org/llama.cpp`** with the branch topology from
   `01-RECOMMENDATION.md` §2-3: `upstream-track`, `anchor/b10103`,
   `feat/p2-mixed-batch-lora`, `feat/nanbeige-arch`, `integration`, and
   `release-<anchor>` tags. The two existing modification patches become **real
   commits** (never stay patches).
2. **Port the build pipeline** from `structured-agents-v2`: `build-llamacpp.sh`
   (profiles `cpu-light` / `cuda-3060` / `p2fork`), the `build-manifest.json`
   schema, the ABI smoke gate, the cffi bindgen gate, and the CI/Act workflows.
   The manifest schema is a **cross-repo contract** — `llama-core`'s
   `diagnostics.py` reads its keys; do not rename them.
3. **Verify, in order**: `feat/p2-mixed-batch-lora` MUST build and pass the smoke
   + bindgen gates end-to-end before anything else is "done". `feat/nanbeige-arch`
   must build and be arch-recognized; its *semantic* validation is explicitly
   deferred (see Q6).

The ABI anchor is the load-bearing constraint of this entire repo. Everything you
build is an ABI question first: llama-cpp-python 0.3.34's hand-written ctypes
match **exactly one** llama.cpp `llama.h`. A struct change is silent memory
corruption. Design every workflow around that fact (the cffi bindgen compile
failure IS the drift signal).

---

## Context

### 1. The rig and the standing discipline

- Two NVIDIA RTX 3060 (sm_86), 12 GiB each. **GPU 1 is often occupied by a vLLM
  runner — any GPU smoke pins GPU 0 (`CUDA_VISIBLE_DEVICES=0`).**
- **Standing rule: never propose vLLM/SGLang as an answer.** This project owns
  the decode path via llama-cpp-python's low-level bindings on purpose.
- Research model: `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (hybrid attention +
  GatedDeltaNet) at
  `/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf`.
- Every build artifact must carry a fingerprint: lib manifest + model sha + GPU +
  driver. The manifest schema already does this (see §6).

### 2. What this repo is — and what it deliberately is NOT

**Is:** the fork (upstream tree + our commits, nothing else on the branches), the
build tooling, the build/ABI gates, and the build CI. A branch = one modification
off the anchor. `integration` = merge-down target; `release-<anchor>` tags = every
shipped lib.

**Is NOT:** no python library code (`llama_core` is a separate repo), no benchmark
runners (separate `bench/` in the llama-core repo), no per-branch python test
suites (those live in llama-core's `suites/<branch>/`; this repo's CI only proves
"branch builds + is ABI-sound"). Keep the fork surgical — upstream rebases drag
anything extra through every diff.

### 3. The source of truth today (read from `structured-agents-v2`, never edit)

Everything below lives in the reference checkout at
`/home/andrew/Documents/Projects/structured-agents-v2/`:

- `.scratch/projects/17-llama-cpp-inference-lab/build-llamacpp.sh` — the build
  script to port (profiles, cmake flags, manifest emission).
- `.scratch/projects/17-llama-cpp-inference-lab/patches/` —
  `p2-mixed-batch-lora.patch` (267-line, 5 files) and `nanbeige-arch.patch`
  (conversion/ + gguf-py/ + `src/llama-arch.{h,cpp}` + `src/llama-context.cpp`).
- `.scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/` — existing
  cloned trees: `src` (anchor), `src-p2fork` (anchor + patch applied),
  `src-nanbeige` (nanbeige lineage), plus prior `out-*` lib sets. These are
  **starting material for history**, not to be copied wholesale.
- `benchmarks/project20/abi_smoke_gate.py` — the behavioral gate to port
  (surface probe + Ornith gen + tokenizer round-trip; fail-closed).
- `.scratch/projects/17-llama-cpp-inference-lab/llama_cffi_build.py`,
  `_llama_cffi.c`, `README-cffi-bindgen.md`, `cffi_smoke_trivial.py` — the cffi
  API-mode bindgen spike to promote to a first-class gate.
- `.scratch/projects/17-llama-cpp-inference-lab/ci/llama-cpp-bindgen.yml` — the
  draft workflow (BUILD → BINDGEN → SMOKE → BENCH stage graph; "BINDGEN failing
  == ABI drift" semantics; nightly canary `0 6 * * *`).
- `.scratch/projects/17-llama-cpp-inference-lab/06-LLAMACPP-BUILD-WORKFLOW.md` —
  the ABI-anchor rule, Mode A vs Mode B, gate definitions. **Read fully.**
- `.scratch/projects/17-llama-cpp-inference-lab/10-CUDA-BUILD-FINDINGS.md` and
  `11-BUILD-SPEED.md` — CUDA toolchain and ccache/Ninja facts.
- `devenv.nix` (repo root) — the GPU env plumbing to reference for any GPU smoke
  (`LD_LIBRARY_PATH=/run/opengl-driver/lib:$lib:$cuda_ld...` so the real libcuda
  wins; `LLAMA_TEST_MODEL`; `CUDA_VISIBLE_DEVICES=0`).
- `.scratch/projects/22-llama-cpp-fork-reorg/01-RECOMMENDATION.md` and
  `03-BENCHMARK-LIBRARY-RECOMMENDATION.md` — the carve-out plan and the bench
  decision. `01` §2-5 is the topology/update/merge/build spec you are executing.

### 4. The ABI-anchor rule (read first)

- Anchor: llama.cpp commit `c588c4f47` = build `b10103` (ggml 0.16.0) = the exact
  commit llama-cpp-python 0.3.34's ctypes bindings match. `anchor/b10103` is a
  **tagged commit**, and the anchor never moves silently.
- Bindings are not generated today; a struct/enum change is silent memory
  corruption. The **cffi bindgen gate** (compile failure == ABI drift) is the
  tripwire; promote it from spike to first-class CI stage on day one.
- Mode B (`LLAMA_CPP_LIB_PATH` lib swap) = fast iteration, same-ABI experiments
  only. Mode A (source rebuild of python-llama-cpp) = what ships. This repo
  produces the lib sets + headers that make both possible; it never ships a
  Mode-B-only artifact as a release.

### 5. Branch topology (from `01-RECOMMENDATION.md` §2-3)

```
upstream-track          # tracks upstream tags (rebases), never built
anchor/b10103           # c588c4f47 — tagged, the ABI anchor
feat/p2-mixed-batch-lora   # anchor + the 5-file P2 change as real commits
feat/nanbeige-arch         # anchor + nanbeige arch (conversion + C++ arch) as commits
integration                # merge-down target; what gets built and shipped
release-<anchor>[-<featureset>]  # tags, not branches: every released lib
```

Rules: every feature branch's base is the current anchor (or `integration` if it
needs another feature); a branch is mergeable into `integration` only when its
build + gates are green against current `integration`; bumping the anchor is a
deliberate act that triggers the full pipeline (§7).

### 6. Build facts (from `build-llamacpp.sh`, verify against the source)

- Profiles: `cpu-light` (GGML_CUDA=OFF), `cuda-3060` and `p2fork`
  (GGML_CUDA=ON, `-DCMAKE_CUDA_ARCHITECTURES=86`). `p2fork` bakes in the pinned
  ref + patch and is otherwise flag-identical to `cuda-3060` (ABI-compatible).
- Common flags: Release, Ninja, `BUILD_SHARED_LIBS=ON`, tests/examples/server
  OFF, `GGML_NATIVE=ON`, `GGML_CCACHE=ON`; build with `-j"$(nproc)"` targeting
  the `llama` target only.
- Output: `out-<profile>-<ref>/lib/` (libllama + libggml{,-base,-cpu} + libmtmd)
  **plus `out-<profile>-<ref>/include/`** (the exact headers) so cffi API-mode
  bindgen compiles against the same source that produced the `.so`s.
- Manifest `build-manifest.json` keys — **this is a cross-repo contract**:
  `llama_cpp_commit` / `commit` / `ref`, `build_id` / `profile`, `ggml_version`,
  `seq_adapter_routing`, `patch_sha256`. `llama-core`'s
  `src/structured_agents/llama_core/diagnostics.py` (see `_build_manifest` /
  `_text` / `_flag`) reads these names; keep them stable and additive only.
- ccache: `CCACHE_DIR=/home/andrew/.cache/llamacpp-ccache` exists; keep
  `GGML_CCACHE=ON` (11-BUILD-SPEED.md:4-25). The build dir gets `rm -rf`'d per
  run in the current script; decide whether to keep that (ccache is what makes it
  fast anyway) and document it.

### 7. Gates — what "green" means for a branch

1. **Build**: `build-llamacpp.sh --ref <branch> --profile cuda-3060` completes;
   lib set + headers + manifest emitted.
2. **ABI smoke gate** (port of `benchmarks/project20/abi_smoke_gate.py`):
   fork symbols resolve on `LLAMA_CPP_LIB_PATH` (P2 branch), Ornith GGUF loads and
   greedily generates ~32 tokens, tokenizer round-trips. GPU env per
   `devenv.nix` (libcuda-first `LD_LIBRARY_PATH`, GPU 0, model path).
3. **cffi bindgen** (port of `llama_cffi_build.py`): compile the API-mode binding
   against THIS build's `include/` + `lib/`. Compile failure == ABI drift == STOP.
4. Optional per-branch capability probe for the suite system in llama-core
   (`nm -D libllama.so | grep <fork symbol>`); emit the probe results in the
   manifest or a sidecar so llama-core's suite runner can decide run/skip.

### 8. CI/Act on the rig (build this repo's half)

- Workflows (GitHub-Actions format, executed locally by **Act** on the rig):
  `build.yml` (matrix ref × profile; cpu-light anywhere, cuda-3060 on the rig),
  `bindgen.yml` (BUILD → BINDGEN → SMOKE), `canary.yml` (nightly `0 6 * * *`:
  fetch upstream, build stock, run each branch's build+smoke against stock →
  per-branch pass/fail/skip). The python-suite stage (`suites.yml`) belongs to the
  llama-core repo — do not build it here.
- GPU legs: `runs-on: [self-hosted, cuda, rtx3060]`, **serialized/exclusive**
  (lock file or queue job; 2×3060s contend), GPU 0 pinned in-job.
- No CI exists at repo root today anywhere in the project — you are not
  migrating, you are building. There is no Act config; create `.actrc` +
  labels as part of the deliverables.

### 9. Environment

- Build inside the devenv/nix CUDA shell (gcc, cmake, ninja, CUDA toolkit,
  ccache). The reference repo's `devenv.nix` and the scratch `cuda-shell.nix`
  show the required pieces.
- llama-cpp-python 0.3.34 exists only in the reference spike venv
  (`.venv-spike`); the gates load the lib via `LLAMA_CPP_LIB_PATH`. Do not add
  llama-cpp-python to this repo's lockfile as a build dependency; it is a gate
  consumer, not a build input.

---

## Key references (read before answering)

- `.scratch/projects/22-llama-cpp-fork-reorg/01-RECOMMENDATION.md` (§2 topology,
  §3 update workflow, §5 merge-down + build pipeline, §8 phases)
- `.scratch/projects/17-llama-cpp-inference-lab/06-LLAMACPP-BUILD-WORKFLOW.md`
- `.scratch/projects/17-llama-cpp-inference-lab/19-P2-FORK-DESIGN.md`,
  `20-P2-MIXED-BATCH-GO.md`, `21-P2-THROUGHPUT.md`
- `.scratch/projects/19-moe-moa-reactive-inference/PORT-PLAN-NANBEIGE-P2.md`
- `build-llamacpp.sh`, `patches/*.patch`, `llama_cffi_build.py`,
  `README-cffi-bindgen.md`, `ci/llama-cpp-bindgen.yml`
- `benchmarks/project20/abi_smoke_gate.py`, `devenv.nix`
- `src/structured_agents/llama_core/diagnostics.py` (the manifest contract)

All paths above are relative to the reference checkout
`/home/andrew/Documents/Projects/structured-agents-v2/` unless absolute.

---

## Questions to answer, with evidence

### Q1 — Repo initialization and history

How exactly do you create the fork? Decide and document: clone `ggml-org/llama.cpp`
vs fork on GitHub first; the `upstream` remote policy; whether `anchor/b10103`
starts from the existing `.llamacpp-builds/src` tree (verify it is a clean
checkout of `c588c4f47` first) or a fresh clone; what `integration` starts as;
`.gitignore` policy (build dirs, `out-*`, `.llamacpp-builds/` must never be
committed). State which approach you take and why.

### Q2 — Patch → real commits, with provenance

For each patch: apply as **real commits on the feature branch**, not a single
squashed blob if the patch has natural layers (P2 = API surface vs context impl
vs graph impl; nanbeige = conversion + gguf-py vs C++ arch). Requirement:
**provenance survives** — the commit message records the source patch path and its
`sha256` (the current manifest records `patch_sha256`; keep that value
derivable). Show the exact commit list you intend and the evidence (patch
diffstat, `git apply --check` against the anchor).

### Q3 — Build tooling port

Port `build-llamacpp.sh` with the smallest delta: `--ref` must accept branch
names; the `p2fork` profile's hard-coded anchor/patch defaults become
repo-local defaults; manifest emission unchanged (contract, §6). What breaks if
`--ref` is a branch instead of a detached commit? Verify your port produces a
`build-manifest.json` byte-compatible with what `diagnostics.py` reads.

### Q4 — Gate port

Port `abi_smoke_gate.py` and the cffi bindgen into this repo (`ci/gates/`).
What is the minimal cdef surface for the bindgen gate (the spike's real minimal
llama.h cdef + the two fork symbols)? What must the smoke gate check for the
nanbeige branch, whose capability is an *arch*, not extra symbols? Keep the
gates fail-closed and GPU-0-only.

### Q5 — CI shape

Design `build.yml` / `bindgen.yml` / `canary.yml` for Act on the rig: matrix,
stage graph, path filters, GPU serialization mechanism, `release-<anchor>` tag
handling, and how the canary reports per-branch drift against fresh upstream.
The stage-graph semantics from the draft (`ci/llama-cpp-bindgen.yml`) are the
portable part — keep them.

### Q6 — Nanbeige verification scope (be honest)

The nanbeige branch can only be *built and arch-recognized* here: conversion
tools present, `llama-arch.cpp` recognizes `MODEL_ARCH.NANBEIGE`, build
succeeds. The **semantic** validation (looped/recurrent-in-depth correctness)
depends on the `build_lora_mm` loop-step threading spike (PORT-PLAN-NANBEIGE-P2
§1.2) and belongs to llama-core suites. State exactly what you will and will not
verify for this branch, and what evidence you need to call it "staged" vs
"shipped".

### Q7 — Boundary enforcement

This repo must not grow: python library code, benchmark runners, or per-branch
python suites. Enumerate the concrete things you deliberately left out and the
one-line justification each (e.g., "abi_smoke_gate stays here because it
validates the fork lib; run_library_throughput goes to bench/ because it measures
the python library").

---

## Deliverables

1. **Repo initialized** — remote configured, branches created from real history,
   pushed. `anchor/b10103` verified = `c588c4f47`.
2. **`feat/p2-mixed-batch-lora` green end-to-end** — build → smoke gate → cffi
   bindgen all pass on the rig, with the artifacts + manifest as evidence.
3. **`feat/nanbeige-arch` staged** — builds, arch recognized; semantic validation
   deferred per Q6, with the deferral documented.
4. **Build pipeline ported** — `ci/build/build-llamacpp.sh` (+ profiles),
   `ci/gates/abi_smoke_gate.py`, `ci/gates/cffi_bindgen.py`, manifest schema doc.
5. **CI workflows + Act config** — `ci/workflows/{build,bindgen,canary}.yml`,
   `.actrc`, runner labels, GPU serialization.
6. **README + architecture doc** — branch topology, ABI-anchor rule, Mode A/B,
   how the three repos interlock (fork → bindings fork → llama-core), how a
   future anchor bump is performed.
7. **Report** — what you found, decisions with evidence, deviations from this
   prompt and why, and the exact commands to reproduce every gate.

## Definition of done (checklist)

- [ ] `git log` on each feature branch shows real commits with patch provenance
- [ ] `anchor/b10103` is tagged and resolves to `c588c4f47`
- [ ] P2 branch: `build --ref feat/p2-mixed-batch-lora --profile p2fork` →
      smoke gate exit 0 → bindgen compile exit 0, on the rig, GPU 0
- [ ] `build-manifest.json` keys unchanged from the contract (§6)
- [ ] Nanbeige branch builds with the arch recognized; deferral documented
- [ ] Workflows listed by `act -l` on the rig; canary runs nightly
- [ ] No python library, benchmark, or suite code in this repo
- [ ] Commit messages: plain, zero AI attribution, per project rule

---

## Ground rules

- **ABI-anchor rule is load-bearing.** Never bump the anchor silently; a bump is
  the full pipeline. A cffi bindgen compile failure is a STOP, not a TODO.
- **Read-only against `structured-agents-v2`.** Cite file:line for every claim
  about existing code or behavior.
- **Never propose vLLM/SGLang.** The project owns the decode path in python on
  purpose.
- **No AI attribution in commit messages** — no `Co-Authored-By`, no tool
  mention, ever.
- Fail-closed everywhere: a missing capability is a skip or a loud fail, never a
  silent pass. For any token-exact check you do run: batch-1 ground truth +
  known-flip reporting (batched-greedy FP nondeterminism is proven in
  `21-P2-THROUGHPUT.md`).
- Mode A vs Mode B discipline: Mode B artifacts are iteration-only; `release-`
  tags imply Mode A provenance.
- If a decision in this prompt is wrong in the field, say so in the report with
  evidence — do not rubber-stamp it.

---

*Placeholders to fill before handing off: the `inferference` GitHub remote URL;
whether the rig's Act runner labels are already registered; the exact upstream
remote URL if you already created a GitHub fork.*
