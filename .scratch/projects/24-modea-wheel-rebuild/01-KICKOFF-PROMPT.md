# Kickoff prompt — Project 24: Mode A wheel rebuild (`24c5d3dbc`, CUDA-LoRA NaN fix) + release-grade wiring of the inferference consumer refactor

> Hand this document to the next agent session verbatim (it is self-contained).
> This is a **gated EXECUTION session**, not a planning session: the design,
> the patch, the plan, and the landings are all done — this session rebuilds
> the Mode A wheel from the nan-fix integration commit, swaps the reference's
> wiring to the rebuilt wheel, fixes the stale anchor docs, and records the
> release. Session root: **`/home/andrew/Documents/Projects/inferference`**
> (the wheel build lives there); the reference repo
> `/home/andrew/Documents/Projects/structured-agents-v2` is touched only for
> the wiring swap in P4. No AI attribution anywhere.

---

## 1. Context — what is already done (consume, do NOT re-derive)

The project23 consumer refactor is **landed and merged** (reference repo,
`main` = `2c898a8`, merge of `project23-inferference-consumer-refactor`
`e613ce4`): `src/structured_agents/llama_core/` deleted, 10 consumer flips →
`inferference.X`, lazy `__init__` surface, devenv repointed, `uv.lock` pinned.
The step-7/8 artifacts (read-only, do not regenerate):

- **Design report** — `inferference/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md`
  (§Q2.2 is the **wheel gap** this session closes; §Q5 has the verification
  commands; Q6.1.7 is the wheel-rebuild owner action).
- **Implementation plan** — `structured-agents-v2/.scratch/projects/23-inferference-consumer-refactor/01-IMPLEMENTATION-PLAN.md`
  (P6.1/P6.2 = the wheel-rebuild decision framing; P7 = the runnable sequence
  the refactor was executed with).
- **Patch** — `inferference/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch`
  (applied; kept as the record).

### 1.1 Current wiring (verified 2026-08-04 by the project23 execution session)

- **Reference devenv venv** (`.devenv/state/venv`, CPython 3.13.13): carries
  `inferference` (editable, `path = ../inferference`) + `llama-cpp-python
  0.3.34` from the **Mode A wheel**
  `inferference/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl`
  (sha256 `81a5dba8f044…`, built from integration `97ad953ef`, anchor
  `0ab9d6fed` = b10233) — **this wheel predates the CUDA-LoRA NaN fix
  (`24c5d3dbc`): the gap this session closes**.
- **Reference `pyproject.toml` `[tool.uv.sources]`**: `llama-cpp-python =
  { path = "/home/andrew/Documents/Projects/inferference/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl" }`
  — note: **uv rejects `url = "file://…"` sources** ("URL scheme is not
  allowed"); the canonical pin is a `path` source on the wheel file. The lock
  records the wheel file + sha256; zero index/sdist refs (verify with
  `grep -B3 -A8 '^name = "llama-cpp-python"' uv.lock`).
- **Reference `devenv.nix`**: project17 scripts default `$lib` to
  `$PWD/.devenv/state/venv/lib/python3.13/site-packages/llama_cpp/lib` — this
  path is **stable across wheel swaps** (the new wheel installs its bundled
  libs to the same site-packages location), so P4 needs **no devenv.nix lib
  path edit**, only a `uv sync` + re-verification. project20 defaults to the
  Mode B `$INF/ci/build/.llamacpp-builds/out-p2fork-24c5d3dbc/lib` (nan-fix
  already present there) — unchanged. `project23-gpu-contract` runs the
  inferference contract suite (verified 89/89, GPU 0).
- **Fork** — `/home/andrew/Documents/Projects/llama-infernal`, `integration`
  branch = `24c5d3dbc4ed8b72e42364a10d25a6a2f02c392a` (i.e. integration IS at
  the nan-fix commit; `97ad953ef` and `0ab9d6fed` are its ancestors).
- **Anchor identity** — `0ab9d6fed` = build **b10233**
  (`libllama.so.0.0.10233`, ggml 0.18.0, 2026-08-02); `anchor/b10233` branch
  created; pairing with llama-cpp-python 0.3.34 (cdef canary green). Source:
  `inferference/.scratch/projects/001-llama-cpp-fork-reorg/13-ANCHOR-BUMP-HATS-BENCH-REPORT.md`.
- **Mode A build machinery** — `inferference/ci/build/mode-a-prep.sh`
  (materializes `vendor/llama-cpp-python` @ v0.3.34 with nested
  `vendor/llama.cpp` re-pointed to `llama-infernal@integration`, `FORK_REF`
  env overrides the fork ref; emits `ci/build/vendor-manifest.json`) +
  `ci/build/mode-a-build.sh` (source rebuild `CMAKE_ARGS="-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=86" FORCE_CMAKE=1 uv pip install --no-binary
  llama-cpp-python`, installs into `ci/modea/.venv`, emits
  `ci/build/modea-version-tuple.json`). See `ci/build/mode-a.md`.
- **KNOWN BUG in `mode-a-build.sh` (~line 52)**: the tuple JSON hardcodes
  `"llama_cpp_anchor": "c588c4f47"` (stale b10103). The on-disk tuple on this
  machine was fixed post-hoc to `0ab9d6fed`. **This session must fix the
  script** (record the operative anchor `0ab9d6fed`/b10233 — or read it from
  the tree — not the stale hardcode).
- **Wheel artifact** — the pinned wheel file in `ci/modea/wheels/` is produced
  separately from the build (the build installs into the venv). Exact
  wheel-artifact command = **VERIFY-AT-EXECUTION** (shape:
  `uv build --wheel <vendor/llama-cpp-python>` or `uv pip wheel` into
  `ci/modea/wheels/`).
- **Expected validation counts** (the "78/17/6/1" record from the tuple's
  `wheel_validation`): library **78/78**, P2 **17/17**, hats **6/6**,
  nanbeige **pass** — achieved against wheel `81a5dba8…` WITHOUT lib swap
  (`13-ANCHOR-BUMP-HATS-BENCH-REPORT.md` line 18). The rebuilt wheel must
  re-achieve these plus the reference gates.
- **venv-clobber fact** (already proven, reuse): devenv does NOT run `uv
  sync` on entry (no `devenv:python:uv` task, no sync script, no
  `uv.sync.checksum`); the venv is rebuilt only if the python interpreter
  store path changes. A re-sync is safe because the wheel is pinned in
  `[tool.uv.sources]` — no drift risk.

### 1.2 The wheel gap (what this session fixes) — 02-REPORT §Q2.2

Wheel `81a5dba8…` (built `97ad953ef`) predates `24c5d3dbc` (the CUDA-LoRA NaN
fix: `ggml-cuda: store Q8_1 activation scales in fp32 to fix inf/NaN on the
LoRA path`, `04a8cc3aa`). Probe adapters are clean on Mode A; full-coverage
LoRA workloads regress (NaN) on Mode A only. Mode B `out-p2fork-24c5d3dbc`
covers the gap today. **Release-grade Mode A = the rebuilt wheel from
`24c5d3dbc`.**

---

## 2. Mission — ordered, gated steps (run top-to-bottom, fail-closed, rollback per step)

Session env: everything inside `NIXPKGS_ALLOW_UNFREE=1 devenv shell` (`bash
-c`, never `bash -lc`). GPU discipline: `CUDA_VISIBLE_DEVICES=0` + the
`ci/runner/gpu-serialized.sh` flock (GPU 1 often hosts a vLLM runner). The
runtime `LD_LIBRARY_PATH` for llama_cpp = `/run/opengl-driver/lib` first +
the reference spike's `.cuda_runtime_ld` contents
(`structured-agents-v2/.scratch/projects/17-llama-cpp-inference-lab/.cuda_runtime_ld`:
gcc-15.2.0-lib, zlib, cuda12.9-cudart, libcublas, graphics-drivers).

### P1 — baseline re-verify (fast, don't assume)

- `git -C /home/andrew/Documents/Projects/llama-infernal rev-parse
  integration` → must equal `24c5d3dbc4ed8b72e42364a10d25a6a2f02c392a`.
- `cat inferference/ci/build/modea-version-tuple.json` → wheel sha256
  `81a5dba8…`, built `97ad953ef`, anchor `0ab9d6fed`.
- Reference `git log --oneline -1` (main) = `2c898a8`.
- Reference venv gate: `.devenv/state/venv/bin/python -c "import llama_cpp,
  inferference, structured_agents"` with the runtime `LD_LIBRARY_PATH` →
  `llama_cpp.__version__ == 0.3.34`.
- **Gate**: all four facts hold, else stop and report (do not proceed).

### P2 — rebuild the Mode A wheel from `24c5d3dbc`

1. **Fix the stale anchor hardcode** in `ci/build/mode-a-build.sh` (currently
   writes `"llama_cpp_anchor": "c588c4f47"`). Record `0ab9d6fed` (b10233) —
   ideally read the anchor from the vendored tree/known anchor source, not a
   hardcode. **Gate**: the tuple's `llama_cpp_anchor` field writes
   `0ab9d6fed`.
2. `ci/build/mode-a-prep.sh` with `FORK_REF=24c5d3dbc` (or verify
   `FORK_REF=integration` resolves to `24c5d3dbc` first). **Gate**:
   `ci/build/vendor-manifest.json` records `llama_cpp.commit =
   24c5d3dbc…` and `llama_cpp_python.commit` = the v0.3.34 pin.
3. `ci/build/mode-a-build.sh` (inside devenv shell). **Gate**:
   `ci/build/modea-version-tuple.json` records `llama_cpp_built_commit =
   24c5d3dbc…`, anchor `0ab9d6fed`, and the wheel's bundled libs exist at
   `ci/modea/.venv/lib/python3.13/site-packages/llama_cpp/lib`.
4. **Produce the wheel artifact** into `ci/modea/wheels/`
   (`llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl`, replacing the
   `81a5dba8…` one — keep a backup of the old wheel first, e.g.
   `ci/modea/wheels/backup-81a5dba8/`). VERIFY-AT-EXECUTION: exact command
   shape (`uv build --wheel <vendor/llama-cpp-python>` — the pip-installed
   venv package is not the artifact; the artifact must come from the vendored
   tree so it carries the `vendor/llama.cpp@24c5d3dbc` build).
5. **Probe install** (throwaway venv in /tmp, reference env untouched):
   `uv venv /tmp/modea-probe -p 3.13 && uv pip install --python
   /tmp/modea-probe/bin/python --no-deps <new wheel>` then `import llama_cpp`
   with the runtime `LD_LIBRARY_PATH`. **Gate**: imports; new wheel sha256
   differs from `81a5dba8…`.

### P3 — verify the rebuilt wheel (Mode A provenance: the wheel's OWN bundled libs, no lib swap)

- `ci/gates/abi_smoke_gate.py --model <Ornith GGUF>` with
  `LLAMA_CPP_LIB_PATH` = the new wheel's bundled lib dir (per
  `ci/build/mode-a.md` §Verification).
- `ci/gates/cffi_bindgen.py --lib-dir <new wheel lib dir>` — cdef tripwire
  against the rebuilt lib (the anchor rule: `abi_drift: none`).
- `ci/gates/nanbeige_arch_gate.py` — nanbeige pass on the rebuilt wheel.
- **Contract suite 89/89**: `ci/runner/gpu-serialized.sh
  ci/library/.venv/bin/python -m pytest ci/tests/ -q` with
  `LLAMA_CPP_LIB_PATH` = the rebuilt wheel's bundled lib (so the two GPU
  tests run). NOTE: inferference's pyproject has `addopts = "-q"` — a
  second `-q` suppresses the summary line; assert via the progress dots
  (89 dots, no `s`/`F`/`x`) or drop the extra `-q`.
- **The 78/17/6/1-style counts** (library/P2/hats/nanbeige suites per
  `modea-version-tuple.json` `wheel_validation`): re-run against the rebuilt
  wheel, no lib swap. VERIFY-AT-EXECUTION: the exact suite commands (the
  `13-ANCHOR-BUMP-HATS-BENCH-REPORT.md` + `mode-a.md` record them).
- **Nan-fix assertion**: a full-coverage-LoRA probe (the path `24c5d3dbc`
  fixed) must be NaN-free on the rebuilt wheel. VERIFY-AT-EXECUTION: the
  exact probe command (report §Q2.2 names the workload class; the P2 fork
  tests in the contract suite exercise the LoRA path).
- **Gate**: all of the above green, recorded in the new
  `modea-version-tuple.json` `wheel_validation`.

### P4 — swap the reference wiring to the rebuilt wheel (reference repo, run from its root)

1. Backup + replace: `structured-agents-v2/pyproject.toml`
   `[tool.uv.sources]` `llama-cpp-python` `path` → the NEW wheel file; keep
   the ABI-anchor comment, drop/adjust the "wheel gap" note (now resolved).
2. `uv lock` (inside the reference devenv shell). **Gate**:
   `grep -B3 -A8 '^name = "llama-cpp-python"' uv.lock` → source = new wheel
   path, hash = new sha256; still zero index/sdist refs.
3. `UV_PROJECT_ENVIRONMENT="$PWD/.devenv/state/venv" uv sync --all-extras`.
   **Gate**: `.devenv/state/venv/bin/python -c "import llama_cpp"` →
   `0.3.34`, and `site-packages/llama_cpp/lib` now holds the rebuilt libs.
4. **Reference gate battery** (all must pass under the devenv venv python
   with the runtime `LD_LIBRARY_PATH`):
   - lazy loader: `import structured_agents` → `inferference`/`llama_cpp`/
     `numpy` NOT in `sys.modules`; `from structured_agents import
     MultiLoRARouter` resolves to the inferference module object; unknown
     names raise `AttributeError`.
   - 11 flipped files import-resolve (benchmarks/project17/*,
     benchmarks/project20/*, examples/*, tests/typecheck_prefix_cache.py).
   - AST assertion: `pytest tests/test_config.py::test_only_config_imports_importlib_in_package_source -q`.
   - framework suite (43 tests): the 13 kept test files, DBOS session
     fixture, `test_live` skipped unless `SAV_LIVE=1`.
   - `project23-gpu-contract` (the devenv script) → contract suite 89/89 with
     the reference's unit now being the rebuilt wheel.
   - ruff: `ruff check --select I` clean on the flipped files (62 pre-existing
     UP045-style errors are out of scope — do not fix them).
5. **devenv.nix**: no lib-path edit expected (site-packages path is stable) —
   VERIFY-AT-EXECUTION with `devenv shell -- project17-gpu-pytest
   --collect-only -q` (rc=0) and `devenv shell -- project20-gpu-pytest
   --collect-only -q` (rc=0).

### P5 — doc fixes + housekeeping (inferference side)

- **AGENTS.md stale anchor text**: replace `c588c4f47` (b10103) with
  `0ab9d6fed` (b10233) wherever it appears as the operative anchor.
- **`ci/build/mode-a-build.sh` anchor hardcode fix** (from P2.1) + any other
  stale `c588c4f47` references in ci/ docs.
- Mark the wheel-gap notes resolved: `02-REPORT.md §Q2.2` and the reference
  plan `01-IMPLEMENTATION-PLAN.md P6.1` (append a resolution note — the
  session may edit the plan doc's status or leave a `24-…` record; do not
  rewrite history).
- **Owner decision inputs** (state each with a recommendation, decide by
  default if no ruling):
  1. **Bench removal** (report §Q3.1 / plan P6.3): remove the 10 flipped
     entrypoints once benchkit ownership is confirmed. This session: verify
     benchkit ownership in inferference first; if confirmed, still default to
     a **follow-up session** (keep-thin is the conservative option) unless the
     owner rules otherwise.
  2. **Orphan pruning**: the reference's spike venv
     (`.scratch/projects/17-llama-cpp-inference-lab/.venv-spike`) and old
     builds (`out-cuda-3060-postfix2`, `out-p2fork-c588c4f47`) — house rule:
     state it, delete nothing. The new unit is now proven, so pruning is
     permitted — but it is the owner's call; this session only documents the
     orphan inventory.
  3. **Release tag**: follow the house convention from
     `13-ANCHOR-BUMP-HATS-BENCH-REPORT.md` (the prior release was tagged
     `release-b10233-p2-hats`). VERIFY-AT-EXECUTION: the exact tag naming
     convention; propose `release-b10233-p2-hats-nanfix` or similar and state
     it in the record — tagging/pushing is the owner's action.

### P6 — the record

- Update `ci/build/modea-version-tuple.json` (new sha256, built commit
  `24c5d3dbc`, anchor `0ab9d6fed`, wheel_validation counts, wheel_file path).
- Write `inferference/.scratch/projects/004-modea-wheel-rebuild-nanfix/01-REPORT.md`
  (or the project-dir convention this repo uses) with: commands run, every
  gate result, the two VERIFY-AT-EXECUTION resolutions, deviations, and the
  owner-decision inputs.
- Commit/push per repo rules: **inferference is gitman-managed (the owner
  commits via gitman)**; the reference is raw git (the owner commits). No AI
  attribution in any message.

---

## 3. Ground rules (load-bearing, inherited from the inferference AGENTS.md)

- **ABI-anchor rule**: llama-cpp-python 0.3.34 ↔ anchor `0ab9d6fed` (b10233).
  The anchor does NOT move; the cdef tripwire stays green; the rebuilt wheel
  must never bind against stock llama.cpp (Mode A = the vendored fork tree at
  `24c5d3dbc`).
- **Mode A vs Mode B**: Mode B (`LLAMA_CPP_LIB_PATH` lib swap) is iteration
  only. `release-` provenance implies Mode A (source rebuild). NEVER ship a
  Mode-B-only artifact as a release. The rebuilt wheel is Mode A.
- **Fail-closed everywhere**: every step has a verifiable gate. Anything not
  verifiable without mutating the env is marked VERIFY-AT-EXECUTION with the
  exact command — never silently assumed.
- **No vLLM/SGLang proposals. No AI attribution** in any message/commit text
  (`Co-Authored-By` and tool mentions are forbidden, forever).
- **GPU discipline**: `CUDA_VISIBLE_DEVICES=0` + the gpu-serialized flock;
  GPU 1 often hosts a vLLM runner.
- **Environment**: `NIXPKGS_ALLOW_UNFREE=1 devenv shell` (`bash -c`, never
  `bash -lc`). Do not activate venvs manually. Ephemeral /tmp scratch is
  allowed for probes; never mutate the repo venvs outside the prescribed
  `uv sync`.
- **The reference repo's untracked WIP is the owner's** (19, 21,
  `src/structured_agents/training/`) — leave alone, exclude from every diff.
- **The anchor/stale docs**: fix `c588c4f47` → `0ab9d6fed` references as
  directed (P5); do not touch any anchor/cdef code.
- Rollback per step: reference repo `git checkout -- .` (clean tree before
  starting); inferference changes are small doc/script edits — revertible.

---

## 4. Reference material (read-only; cite `file:line`)

- `inferference/ci/build/mode-a.md` — the Mode A recipe + verification
  commands (the §Verification block is the P3 command source).
- `inferference/ci/build/mode-a-prep.sh`, `ci/build/mode-a-build.sh` — the
  build scripts (note the anchor hardcode bug).
- `inferference/ci/build/modea-version-tuple.json` + `vendor-manifest.json` —
  the build records (the new tuple is the P6 deliverable).
- `inferference/ci/gates/abi_smoke_gate.py`, `cffi_bindgen.py`,
  `nanbeige_arch_gate.py` — the Mode A gates.
- `inferference/.scratch/projects/001-llama-cpp-fork-reorg/13-ANCHOR-BUMP-HATS-BENCH-REPORT.md`
  — anchor identity (b10233), the release tag convention, the 78/17/6/1
  counts, the cdef/ABI gate commands.
- `inferference/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md`
  — §Q2.2 wheel gap, §Q5 verification commands, §Q6.1.7 rebuild action.
- `inferference/ci/runner/` — `gpu-serialized.sh`, `rig-env.sh`
  (`/run/opengl-driver/lib` first; MODEL_* env; the `_build_ld` helper).
- `structured-agents-v2/.scratch/projects/23-inferference-consumer-refactor/01-IMPLEMENTATION-PLAN.md`
  — P6.1/P6.2 (rebuild framing), P7 (the sequence already executed), P1.2
  (venv-population proof).
- `structured-agents-v2/devenv.nix`, `pyproject.toml`, `uv.lock` — the
  reference wiring P4 touches.
- `inferference/AGENTS.md` — the load-bearing rules (§3).

---

## 5. Deliverables + definition of done

1. **The rebuilt Mode A wheel** from `24c5d3dbc` at
   `ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl`
   (old wheel backed up), verified Mode A (no lib swap): gates +
   contract suite 89/89 + the 78/17/6/1-style counts + the nan-fix probe.
2. **The reference wiring swap** (P4): pyproject `[tool.uv.sources]` + new
   `uv.lock` + re-synced devenv venv; the full reference gate battery green
   under the rebuilt wheel; `project23-gpu-contract` green.
3. **Doc fixes**: `mode-a-build.sh` anchor hardcode, AGENTS.md anchor text,
   the wheel-gap resolution notes; `modea-version-tuple.json` updated.
4. **The session record** (P6): report with commands, gates, deviations,
   VERIFY-AT-EXECUTION resolutions, owner-decision inputs (bench removal,
   orphan pruning, release tag).
5. Owner review: commit/push per repo rules (gitman in inferference, raw git
   in the reference), zero AI attribution.

Checklist:
- [ ] P1 baseline re-verified with commands, not assumed
- [ ] P2 wheel rebuilt from `24c5d3dbc` (anchor `0ab9d6fed` recorded; prep +
      build + artifact + /tmp probe gates green)
- [ ] P3 Mode A verification without lib swap (smoke/bindgen/nanbeige gates,
      89/89, 78/17/6/1-style counts, nan-fix probe)
- [ ] P4 reference swap + full gate battery (lazy, surface, 11 flipped files,
      AST, 43-test framework suite, project23-gpu-contract, ruff --select I)
- [ ] P5 doc fixes + owner-decision inputs stated (bench removal, orphan
      pruning, release tag)
- [ ] P6 record updated (tuple + session report); no AI attribution; out-of
      scope explicit (bench removal execution, orphan pruning, release tag
      push — owner actions)

---

## 6. Working style + suggested first moves

Study first, verify second, execute third. Suggested order:

```bash
# from the inferference root, inside NIXPKGS_ALLOW_UNFREE=1 devenv shell (bash -c)
INF=/home/andrew/Documents/Projects/inferference
REF=/home/andrew/Documents/Projects/structured-agents-v2

# 1. Read the design + build docs (before anything else):
#    $INF/ci/build/mode-a.md
#    $INF/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md   (§Q2.2, §Q5, §Q6.1.7)

# 2. P1 baseline re-verify:
git -C /home/andrew/Documents/Projects/llama-infernal rev-parse integration   # expect 24c5d3dbc…
cat "$INF/ci/build/modea-version-tuple.json"                                  # expect sha 81a5dba8…, built 97ad953ef
git -C "$REF" log --oneline -1                                                # expect 2c898a8
#    reference venv import gate (runtime LD_LIBRARY_PATH from the spike's .cuda_runtime_ld)

# 3. P2 rebuild:
#    fix the anchor hardcode in $INF/ci/build/mode-a-build.sh first
#    FORK_REF=24c5d3dbc $INF/ci/build/mode-a-prep.sh
#    NIXPKGS_ALLOW_UNFREE=1 devenv shell -- $INF/ci/build/mode-a-build.sh
#    produce the wheel artifact into ci/modea/wheels/ (VERIFY-AT-EXECUTION: uv build --wheel <vendor tree>)

# 4. P3 verify (Mode A, no swap): gates + contract suite + 78/17/6/1-style counts + nan-fix probe

# 5. P4 swap the reference (pyproject path → new wheel; uv lock; uv sync; full gate battery)

# 6. P5/P6 docs + record; hand off owner decisions (bench removal, orphan pruning, release tag)
```

Work the checklist top-to-bottom; every gate green before the next step; on
any failure, stop, record, and roll back (reference: `git checkout -- .`) —
do not paper over a gate.
