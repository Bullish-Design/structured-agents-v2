# Kickoff prompt — Step 8: implementation PLAN for the `structured-agents-v2` → `inferference` refactor (planning session, run from the reference root)

> Hand this document to the next agent session verbatim (it is self-contained).
> This is a **PLANNING session, not an execution session**: the deliverable is a
> detailed, gated, file-by-file **implementation plan** for applying the step-7
> refactor design to `structured-agents-v2`, grounded in THIS repo's actual
> environment (venv management, devenv scripts, ABI units on disk). The session
> runs with cwd = the **root of `/home/andrew/Documents/Projects/structured-agents-v2`**.
> It writes exactly one thing into the repo — the plan document under
> `.scratch/projects/23-inferference-consumer-refactor/` — and nothing else.
> No code is changed, nothing is installed into the repo's venv, nothing is
> committed, nothing is pushed. The refactor's execution is a later session (or
> the owner), consuming this plan.

---

## 1. Context — what is already done (consume, do NOT re-derive)

The step-7 study-and-explain session (in `inferference`) already produced the
design, the mechanical proof, and a prepared patch. It is all here, read-only
for you:

- **The design report** — `/home/andrew/Documents/Projects/inferference/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md`
  (Q1 17:17 module mapping with per-module symbol diffs; Q2 dependency/ABI
  design incl. the wheel gap; Q3 benchmarks/examples/tests plan; Q4 constraint
  verification; Q5 shim verification — 46/46 reference import statements and
  89/89 contract tests passed; Q6 execution plan + risk register; Q7 patch
  status; owner decision inputs). **Read it fully first — every claim about
  this repo's code is already cited `file:line` there.**
- **The prepared patch** — `/home/andrew/Documents/Projects/inferference/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch`
  (45 files: 10 consumer flips + `pyproject.toml` + `src/structured_agents/__init__.py`
  lazy-loader restructure + `tests/test_config.py` fix + `tests/typecheck_prefix_cache.py`
  flip; deletions: `src/structured_agents/llama_core/` (17 files) + 14 duplicate
  llama_core test files. `git apply --check` CLEAN against this repo's HEAD
  `b546ff2`). **Do not regenerate it.**
- The step-7 kickoff prompt — `…/003-reference-consumer-refactor/01-KICKOFF-PROMPT.md`
  (the decision framing the plan must carry forward).

Your job is **not** to re-do any of that. Your job is to turn it into the
**implementation plan**: the exact ordered, gated sequence of commands and
edits an execution session (or the owner) runs **in this repo**, resolving the
environment facts and owner decisions the design deliberately left open.

### The current state of THIS repo (verify each line yourself, cite `file:line`)

- Plain **git** repo at HEAD `b546ff2` — **no repoman/gitman/jj** (no
  `gitman.toml`, no `.jj`); the owner commits. Untracked WIP that is NOT yours:
  `.scratch/projects/19-moe-moa-reactive-inference/`,
  `.scratch/projects/21-arrow-adapter-routing/`, `src/structured_agents/training/`
  — do not touch, do not plan around them beyond "left alone".
- **No `AGENTS.md` / `.agents/` in this repo.** The load-bearing rules you
  inherit come from the inferference repo's `AGENTS.md` (see §3).
- `devenv.nix` (12 KB): `languages.python = { enable = true; version = "3.13";
  venv.enable = true; uv.enable = true; }` — the devenv venv currently holds
  torch/dbos/xgrammar (`pyproject.toml` deps) but **no `llama_cpp` binding**;
  the binding lives only in the spike venv
  `.scratch/projects/17-llama-cpp-inference-lab/.venv-spike` (Python 3.13.13),
  loaded Mode B against b10103-era builds (`out-cuda-3060-postfix2`,
  `out-p2fork-c588c4f47`, anchor `c588c4f47` — see `tests/test_seq_routing.py:33`).
- `pyproject.toml` declares NO `llama-cpp-python` today; `uv.lock` has no
  llama-cpp-python entry.
- `devenv.nix` scripts `project17-*` / `project20-*` hard-code the spike venv +
  old lib dirs (`LLAMA_CPP_LIB_PATH` at `devenv.nix:38,60,102,143,175` approx.;
  `.cuda_runtime_ld`; `PROJECT17_SPIKE_PY` / `PROJECT20_SPIKE_PY` defaults).

### The ABI units available on this machine (for the plan's wiring decision)

| Unit | Identity | Nan-fix? | Location |
| --- | --- | --- | --- |
| Mode A wheel | llama-cpp-python 0.3.34 built from integration `97ad953ef` (anchor `0ab9d6fed` = b10233), sha256 `81a5dba8…` | **no** (predates `24c5d3dbc`) | `/home/andrew/Documents/Projects/inferference/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl` (246 MB) |
| Mode B build | integration `24c5d3dbc` (b10233 + P2 + hats + **nan-fix**) | **yes** | `/home/andrew/Documents/Projects/inferference/ci/build/.llamacpp-builds/out-p2fork-24c5d3dbc/lib` |
| This repo's own | spike-venv llama-cpp-python + `out-cuda-3060-postfix2` (anchor `c588c4f47` = b10103) | no | `.scratch/projects/17-llama-cpp-inference-lab/…` (**obsolete — the flip replaces it**) |

---

## 2. Mission — the implementation plan the session must produce

Write `.scratch/projects/23-inferference-consumer-refactor/01-IMPLEMENTATION-PLAN.md`
answering, in order, with `file:line` evidence from BOTH repos and verified
environment facts (run the commands; do not assume):

### P1 — verified baseline (the plan's foundation)

- HEAD/status/untracked set of this repo (exact).
- **How this repo's devenv venv is populated**: does `NIXPKGS_ALLOW_UNFREE=1
  devenv shell` re-sync `pyproject.toml` deps into `.devenv/state/venv/` on
  entry (check devenv's uv module behavior / a lockfile timestamp / a
  `uv sync` run log)? **This decides whether a manually-installed Mode A wheel
  survives a re-entry** — the single most important environment fact for P2.
  Prove it with evidence (e.g. `devenv shell -- uv pip list` vs
  `site-packages/` contents before/after a re-enter; `ls -la
  .devenv/state/venv/lib/python3.13/site-packages/`).
- Spike venv contents (llama_cpp version, ctypes module) and the old lib
  builds actually on disk.
- `git apply --check` of the step-7 patch against this repo's real tree
  (non-destructive — allowed).

### P2 — the dependency/ABI wiring decision (the big one)

The design (02-REPORT §Q2) leaves the mechanism open. The plan must pick one
with evidence and state the exact commands:

- **(A) Declare `inferference` (path dep) + `llama-cpp-python==0.3.34` in
  `pyproject.toml`**, letting uv resolve/build the binding — requires the fork's
  `vendor/llama.cpp` swap (mode-a-prep pattern) + CUDA toolchain + `CMAKE_ARGS`
  or the binding would vendor STOCK llama.cpp (ABI-anchor violation). Heavy;
  verify what `uv.lock` would do (e.g. `uv lock --dry-run` if cheap/safe).
- **(B) Mirror the inferference `ci/library/.venv` pattern**: uv-managed deps
  (inferference path dep; xgrammar already pinned) + **install the Mode A wheel
  with `uv pip install --force-reinstall --no-deps`** into the devenv venv —
  binding + lib one unit, no source build. **Risk to prove: does a devenv
  re-entry clobber the manually-installed wheel?** (P1's answer decides this.)
- **(C) Keep the binding out of uv entirely** (today's pattern), installing
  wheel or lib into the devenv venv by hand.

State a recommendation and the fallback. Also decide: does the reference venv
keep torch/transformers (yes — xgrammar's import-time baggage is already
satisfied there; `pyproject.toml:15-16`).

### P3 — the refactor change set, ordered and gated

Consume the step-7 patch. Decide **apply-as-one** (patch file, pre/post gates)
vs **hand-applied in the 02-REPORT §Q6.1 order** (flip consumers → test
surgery → delete package → repoint env), and specify per step:
- exact files touched (from the patch; list the 10 consumer flips, the
  `__init__.py` restructure, `test_config.py`, `typecheck_prefix_cache.py`,
  `pyproject.toml`, the 17+14 deletions),
- the gate that proves the repo is still importable after the step
  (e.g. `devenv shell -- <venv python> -c "import structured_agents"` stays
  light; the 46 flipped import sites resolve; `test_config.py:89-110`
  assertion passes),
- the rollback for that step (git `checkout`/`restore`; the patch is
  revertible; a `git stash`/branch bookmark before starting).

### P4 — the devenv repoint (the design left it to the owner; the plan must specify it)

- The exact `devenv.nix` edits: every `project17-*`/`project20-*` script's
  `$py` (spike venv) and `$lib` (`out-cuda-3060-postfix2` /
  `out-p2fork-c588c4f47`) → the §P2 unit (Mode B `out-p2fork-24c5d3dbc` now,
  or the wheel's bundled lib dir), including `.cuda_runtime_ld` handling and
  the `PROJECT17_SPIKE_PY` / `PROJECT20_SPIKE_PY` override defaults.
- Whether this lands in the same change or a follow-up (recommend: same change
  for the GPU-gated test gate to be meaningful).
- What happens to the now-orphaned spike venv and old builds (left in place,
  or pruned per the housekeeping convention — state it, don't delete anything).

### P5 — the test gate (reference side + cross-repo)

- Reference-side: the framework tests (`tests/test_agent.py`,
  `test_approval.py`, `test_authority.py`, `test_config.py`, `test_constraint.py`,
  `test_engine.py`, `test_fornix.py`, `test_plane.py`, `test_grammar_soak.py`,
  `test_live.py`, `typecheck_constraint.py`) + the reference's own
  ruff/ty config (`pyproject.toml` `[tool.ruff]`/`[tool.ty]`) against the
  flipped imports and the pinned ABI unit. Note the DBOS session fixture
  (`tests/conftest.py:8-25`) needs the reference venv's dbos — plan for it.
- Cross-repo: the contract suite stays green in inferference (`testee verify`
  there — the source of truth); the GPU suites (`p2-mixed-batch-lora`,
  `nanbeige-hats`, `router`, `prefix-cache`, `nanbeige-arch`) unaffected.
- The GPU gate commands, GPU 0 discipline (`CUDA_VISIBLE_DEVICES=0`; GPU 1 may
  host a vLLM runner), the `gpu-serialized` flock pattern from the inferference
  `ci/runner/`.

### P6 — the owner decision inputs (state, with a recommended default each)

1. **Wheel rebuild: prerequisite or interim gap?** (02-REPORT §Q2.2: rebuild
   from `24c5d3dbc` = prerequisite for release-grade Mode A; probe workloads
   clean on `81a5dba8…`). Default if no ruling: **interim gap + Mode B
   `out-p2fork-24c5d3dbc`**; the wheel rebuild is a separate owner session
   (NEVER a Mode-B-only artifact as a release — AGENTS.md Mode rule).
2. **Mode B vs Mode A for the refactored reference** — default: Mode B for
   iteration now, Mode A (rebuilt wheel) at release.
3. **Benchmarks/examples: keep-thin vs remove** (02-REPORT §Q3.1 — every
   llama_core-dependent runner is ported to inferference's benchkit; the flip
   in the patch keeps them as delegating entrypoints). Default: **flip now,
   remove in a follow-up** once benchkit ownership is confirmed.
4. **Top-level core surface: keep or drop** (the 7 additive `_LAZY` entries in
   the patch — `from structured_agents import MultiLoRARouter`). Default: keep
   (additive; consumers import inferference directly anyway).
5. **Apply mechanics: patch-as-one vs hand-applied steps** — default:
   hand-applied in the §Q6.1 order (each step importable), the patch as the
   review reference.

### P7 — the runnable execution sequence + risk register

- A single copy-pasteable command sequence (with the gates inline) the owner or
  an execution session runs top-to-bottom in this repo, inside
  `NIXPKGS_ALLOW_UNFREE=1 devenv shell` (`bash -c`, never `bash -lc`).
- Risk register for the EXECUTION phase: venv clobber on re-enter (P1/P2),
  the wheel gap (P6.1), PEP 562 lazy-map drift (already proven — restate the
  proof command), `test_config.py` assertion, devenv script path drift, the
  untracked WIP (19/21/training) being outside the change set, rollback at
  each step.

---

## 3. Ground rules (load-bearing, inherited from the inferference AGENTS.md)

- **This session is planning-only.** The ONLY repo write is the plan document
  under `.scratch/projects/23-inferference-consumer-refactor/`. No code edits,
  no `pyproject`/`devenv.nix`/`uv.lock` changes, no venv installs, no commits
  (the owner commits; raw git — this repo is NOT repoman/gitman-managed), no
  pushes.
- **Ephemeral /tmp scratch is allowed** for pre-flight verification that must
  not touch the repo env: e.g. a throwaway venv in /tmp to prove the Mode A
  wheel installs cleanly (`uv venv /tmp/… && uv pip install --no-deps
  <wheel> && import llama_cpp`), import checks against the step-7 shim pattern,
  `git apply --check` (non-destructive). Never inside `.devenv/` or the repo
  venv.
- **ABI-anchor rule.** llama-cpp-python 0.3.34 ↔ anchor `0ab9d6fed` (b10233).
  The anchor does NOT move; the cdef tripwire stays green; the plan's wiring
  must never bind against stock llama.cpp (vendor-swap or wheel — never a bare
  uv source build of llama-cpp-python without the fork's vendor).
- **Fail-closed everywhere.** Every plan step has a verifiable gate. Anything
  you cannot verify in this session (e.g. a devenv re-enter clobber test that
  would mutate the repo venv) is marked "verify at execution" with the exact
  command — never silently assumed.
- **No vLLM/SGLang proposals. No AI attribution** in any message/commit text
  the plan contains (`Co-Authored-By` and tool mentions are forbidden in this
  family, forever).
- **GPU discipline** for any GPU step: `CUDA_VISIBLE_DEVICES=0`; GPU 1 often
  hosts a vLLM runner.
- **The reference repo's untracked WIP is the owner's.** (19, 21, training/)
  not yours — leave alone, exclude from every diff list.
- **Environment**: everything inside `NIXPKGS_ALLOW_UNFREE=1 devenv shell`
  (`bash -c`, never `bash -lc`). Do not activate the venv manually.

---

## 4. Reference material (read-only; cite `file:line`)

- `/home/andrew/Documents/Projects/inferference/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md` — **read fully first** (the design; Q6.1 is the migration order you operationalize; Q5 has the verification commands).
- `/home/andrew/Documents/Projects/inferference/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch` — the change set you plan around (verify `git apply --check`; never apply).
- `/home/andrew/Documents/Projects/inferference/AGENTS.md` — the load-bearing rules (§3).
- This repo: `devenv.nix` (all 6 project17/project20 scripts), `pyproject.toml`, `uv.lock`, `tests/conftest.py`, `tests/test_config.py:89-110`, `src/structured_agents/__init__.py` (the current PEP 562 loader), `.scratch/projects/17-llama-cpp-inference-lab/` (spike venv + old builds + `.cuda_runtime_ld`), `src/structured_agents/` (the framework layers).
- `/home/andrew/Documents/Projects/inferference/ci/build/modea-version-tuple.json` + `ci/modea/wheels/` + `ci/build/.llamacpp-builds/` — the ABI units (§1 table).
- The step-7 kickoff prompt (`…/003-reference-consumer-refactor/01-KICKOFF-PROMPT.md`) for the original decision framing.

---

## 5. Deliverables + definition of done

1. **The implementation plan** — `.scratch/projects/23-inferference-consumer-refactor/01-IMPLEMENTATION-PLAN.md` covering P1–P7, every step gated and rollback-able, every claim `file:line`-cited, with the runnable command sequence (§P7) front and center.
2. **The verified-baseline log** — embedded in the plan: what you actually ran (HEAD, venv population evidence, spike contents, `git apply --check` result, wheel-install-into-/tmp-venv proof if you did it) and what each proved.
3. **Owner decision inputs** — P6 table with recommended defaults, so the owner can rule before the execution session starts.
4. **Explicit out-of-scope list** — the untracked WIP, the wheel rebuild (owner session), any fork/upstream work, anything Mode-B-only-as-release.

Checklist:
- [ ] P1 baseline verified with commands run, not assumed (esp. the devenv venv population mechanism)
- [ ] P2 ABI/dependency wiring decided with evidence; venv-clobber question answered (or marked verify-at-execution with the exact command)
- [ ] P3 ordered gated steps, importable at every step, rollback per step
- [ ] P4 devenv repoint with exact `devenv.nix` edits
- [ ] P5 test gate (reference framework tests + cross-repo contract suite + GPU discipline)
- [ ] P6 owner decisions with defaults
- [ ] P7 runnable command sequence + risk register
- [ ] Out-of-scope list explicit; reference tree otherwise untouched; no commits/pushes; no venv mutation; `git apply --check` only
- [ ] No AI attribution; no vLLM/SGLang; anchor/cdef untouched; every unverifiable gate marked verify-at-execution

---

## 6. Working style + suggested first moves

Study first, verify second, plan third. Suggested order:

```bash
# from the repo root, inside NIXPKGS_ALLOW_UNFREE=1 devenv shell (bash -c)
INF=/home/andrew/Documents/Projects/inferference

# 1. Read the design (do this before anything else):
#    $INF/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md

# 2. Baseline facts:
git status --short && git log --oneline -1
ls .devenv/state/venv/lib/python3.13/site-packages/ | grep -iE "llama|torch|xgrammar|dbos" || true
.devenv/state/venv/bin/python --version
ls .scratch/projects/17-llama-cpp-inference-lab/.venv-spike/bin/python && \
  .scratch/projects/17-llama-cpp-inference-lab/.venv-spike/bin/python -c "import llama_cpp; print(llama_cpp.__version__)"

# 3. The patch is non-destructive to check:
git apply --check "$INF/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch" \
  && echo "patch applies cleanly against HEAD"

# 4. The devenv venv population question (P1/P2): inspect devenv's uv behavior,
#    then decide whether a manual wheel install survives a re-enter. If you
#    need to prove the wheel installs, use a THROWAWAY venv in /tmp:
#    uv venv /tmp/abi-probe && uv pip install --python /tmp/abi-probe/bin/python \
#      --no-deps "$INF/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl" \
#      && /tmp/abi-probe/bin/python -c "import llama_cpp; print(llama_cpp.__version__)"

# 5. Draft the plan, then walk it once more as a checklist against the patch
#    (every patch file must appear in exactly one step).
```
