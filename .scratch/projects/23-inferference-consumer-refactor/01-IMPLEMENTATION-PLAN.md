# Implementation plan — `structured-agents-v2` → `inferference` refactor (Step 8, planning session)

Session 2026-08-04. Executes
`.scratch/projects/23-inferference-consumer-refactor/03-IMPLEMENTATION-PLAN-KICKOFF-PROMPT.md`
verbatim — a **PLANNING session**: the deliverable is this gated, ordered,
file-by-file implementation plan for applying the step-7 refactor design
(`inferference/.scratch/projects/003-reference-consumer-refactor/02-REPORT.md`,
patch `reference-refactor.patch`) to this repo. This session changed exactly
one thing in the repo: this document. No code, no venv mutation, no commits,
no pushes.

**Every claim below was verified by running the command, not assumed.**
Unverifiable items are explicitly marked **VERIFY-AT-EXECUTION** with the
exact command. All paths relative to the reference root
`/home/andrew/Documents/Projects/structured-agents-v2` unless absolute.
`INF=/home/andrew/Documents/Projects/inferference`.

---

## P1 — verified baseline (the plan's foundation)

### P1.1 repo state

| Fact | Verified value |
| --- | --- |
| HEAD | `baae1356` ("wip: snapshot project19 moe-moa inference spikes, project21 concept, training token_monitor") — the step-7 patch was generated against `b546ff2`; this snapshot commit **added** files only (no overlap with any patch path) |
| `git status --short` | `?? .scratch/projects/23-inferference-consumer-refactor/` (this session's output dir) only; working tree otherwise clean |
| VC | plain git; **no repoman/gitman/jj** (no `gitman.toml`, no `.jj`); the owner commits |
| Patch applicability | `git apply --check "$INF/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch"` → **CLEAN against `baae135`** |

### P1.2 the devenv venv and how it is populated (the P1/P2 clobber question)

The venv is `.devenv/state/venv`, uv-created (`pyvenv.cfg`: `uv = 0.11.6`),
CPython **3.13.13**. Site-packages holds **dbos 2.23.0, torch 2.12.0,
xgrammar 0.2.1, pytest 9.0.3, hypothesis 6.156.9, ty 0.0.46, ruff 0.15.12** —
and **NO `llama_cpp`** (the kickoff's claim confirmed).

The population mechanism was verified from devenv 2.1.2's own artifacts
(not assumed):

1. **Task inventory** (`.devenv/state/tasks.db`): exactly three tasks —
   `devenv:files:cleanup`, `devenv:python:virtualenv`, `devenv:enterShell`.
   **There is no `devenv:python:uv` sync task.**
2. **The virtualenv task script**
   (`/nix/store/158xq8kpjll2v47017amf049crcfsv66-devenv-python-virtualenv`,
   the watched file in `task_run`): recreates the venv **only if the Python
   interpreter store path changed** (`$VENV_PATH/.devenv_interpreter` marker);
   `requirements=""` — the `uv pip install -r` branch is never reached; the
   only unconditional write is `devenv-profile.pth`.
3. **No `devenv-python-uv` script exists in the nix store for this repo**
   (the uv-sync script that other repos get — e.g. `pydantree`, which sets
   `languages.python.uv.sync.enable = true` — was never generated here).
4. **The current generated shell** (`.devenv/shell-c07b493f2d43fa1a.sh`,
   Jul 30) contains **no `uv sync` / `_devenv_uv_sync`** anywhere; it does
   export `UV_PROJECT_ENVIRONMENT=.devenv/state/venv` (`:159-160`) and
   `UV_PYTHON_PREFERENCE='only-system'` (`:259`).
5. **`uv.sync.checksum` does not exist** in the venv — `uv sync` has never
   run against the current config (the checksum file is how the uv module
   skips re-syncs once it has run).

**Conclusion (P1/P2): a manually-installed wheel in `.devenv/state/venv`
survives a `devenv shell` re-entry.** The venv is only rebuilt when the
python interpreter version changes, and no `uv sync` runs on entry. The venv
has already survived a full devenv re-evaluation: created 2026-07-24, it is
unchanged across the Jul 30 re-eval.

**Caveats — VERIFY-AT-EXECUTION (fail-closed, not assumed):**
- The `.devenv/profile` (Jul 30) predates the last `devenv.nix` change
  (Aug 1, commit `5161566` — the project17/20 scripts). The first `devenv
  shell` entry re-evaluates; the python config (`devenv.nix:189-194`) is
  unchanged, so the virtualenv task's watched-file hash should not change and
  the venv should be untouched. Verify with:
  `devenv shell -- bash -c 'ls .devenv/state/venv/lib/python3.13/site-packages/ | grep -i llama_cpp'` (must still list `llama_cpp` after the wheel install).
- If the owner ever adds `languages.python.uv.sync.enable = true` to
  `devenv.nix` (as pydantree did), or runs `uv sync` manually, uv **prunes**
  packages not in `uv.lock` — the manual wheel would be removed. The plan's
  P2 wiring makes this benign by pinning the wheel in `[tool.uv.sources]`.

### P1.3 the spike venv and old builds (what the flip replaces)

- `…/17-llama-cpp-inference-lab/.venv-spike/bin/python` — llama-cpp-python
  **0.3.34** (verified: imports with the spike's `.cuda_runtime_ld`
  `LD_LIBRARY_PATH`; fails without it — the binding needs the nix
  gcc/cudart/cublas libs).
- `.cuda_runtime_ld` (1 line): gcc-15.2.0-lib + zlib + cuda12.9-cudart +
  libcublas + graphics-drivers store paths.
- Old builds on disk: `out-cuda-3060-postfix2`, `out-p2fork-c588c4f47`,
  `out-cuda-3060-p2fork`, `build-p2fork-c588c4f47`, etc. — the **obsolete**
  unit the refactor replaces (its anchor `c588c4f47` = b10103 predates the
  b10233 unit inferference pins).

### P1.4 the framework layers (report §Q2.4 re-verified)

`grep -rn "llama_core\|llama_cpp" src/structured_agents --include="*.py"`
(excluding `llama_core/` and `__init__.py`) → only **string literals**:
`engine/llama_cpp.py:22` (`name = "llama_cpp"`), `engine/__init__.py:11,14`
(backend registry), `agent.py:45` (default engine string). **Zero framework
imports of `llama_core` or the binding.** The framework is unchanged code,
but its venv must carry the pinned binding because inferference's modules
import it at runtime.

### P1.5 the llama_core importer inventory (the flip surface)

Grep across `src benchmarks examples tests` (excluding `.scratch`,
`__pycache__`): **29 files** reference `llama_core` — the 17 package modules
(themselves), `src/structured_agents/__init__.py` (docstring only; the `_LAZY`
map at `__init__.py:15-41` contains **zero** llama_core entries — confirmed),
the **10 consumer files** the patch flips, `tests/typecheck_prefix_cache.py`
(flipped), and the **14 duplicate llama_core tests** (deleted by the patch).
`tests/test_xgrammar_api_contract.py`, `tests/test_project17_workload.py`
(corpus by path, `:7`), `tests/test_state_blob_model.py` do **not** import
llama_core — they stay untouched.

### P1.6 the ABI units on this machine (verified)

| Unit | Identity | Nan-fix? | Verified location |
| --- | --- | --- | --- |
| Mode A wheel | llama-cpp-python 0.3.34, built from integration `97ad953ef`, anchor `0ab9d6fed` (= b10233), sha256 `81a5dba8f044…` (`ci/build/modea-version-tuple.json`) | **no** (predates `24c5d3dbc`) | `$INF/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl` (246 MB) |
| Mode B build | integration `24c5d3dbc` (b10233 + P2 + hats + nan-fix) | **yes** | `$INF/ci/build/.llamacpp-builds/out-p2fork-24c5d3dbc/lib` (libggml-*/libllama present) |
| This repo's own | spike-venv binding + `out-cuda-3060-postfix2` / `out-p2fork-c588c4f47` | no | `.scratch/projects/17-llama-cpp-inference-lab/…` (**obsolete — replaced**) |

**Wheel-install proof (run in /tmp, reference env untouched):** a fresh
`uv venv -p python3.13` + `uv pip install <wheel>` (with the wheel's declared
deps: diskcache, jinja2, numpy, typing-extensions) → `import llama_cpp`
succeeds with the runtime `LD_LIBRARY_PATH` (gcc libstdc++/libgomp + the
`.cuda_runtime_ld` contents + `/run/opengl-driver/lib` first), and the wheel
bundles `libllama.so.0.0.18` + libggml-* in
`site-packages/llama_cpp/lib` — binding + lib as one unit, no source build.

### P1.7 cross-repo reference facts (verified)

- `$INF` pyproject: name `inferference`, 0.1.0, `requires-python >=3.13`;
  deps pydantic/numpy/xgrammar==0.2.1/llama-cpp-python==0.3.34 (same pins as
  the reference's, plus its own llama-cpp-python — no conflict).
- `$INF/ci/library/.venv` holds llama_cpp 0.3.34 + inferference (editable) +
  xgrammar 0.2.1; `testee.toml` → `python = "ci/library/.venv/bin/python"`,
  quick/detailed = ruff + ruff-format + ty + pytest.
- `$INF/ci/runner/`: `gpu-serialized.sh` (exclusive flock on
  `/tmp/llama-ci-gpu.lock`, exports `CUDA_VISIBLE_DEVICES=0`), `rig-env.sh`
  (`NIXPKGS_ALLOW_UNFREE=1`, `CCACHE_DIR`, `/run/opengl-driver/lib` first),
  `library-venv.sh`, `run-suite.sh`, `gate-venv.sh`.
- GPU suites unaffected by the refactor (live in `$INF/suites/`):
  `nanbeige-arch, nanbeige-hats, p2-mixed-batch-lora, prefix-cache, router`.
- **AGENTS.md anchor text is stale**: it states the anchor is `c588c4f47`
  (b10103), but `modea-version-tuple.json` and both kickoff prompts pin
  `0ab9d6fed` (b10233). The **operative** anchor for this refactor is
  `0ab9d6fed`; the stale AGENTS.md text is an inferference-side doc fix for
  the owner (out of scope here, flagged in P6.8).

---

## P2 — the dependency/ABI wiring decision (the big one)

### The trap that rules out option (A) as stated

Declaring `llama-cpp-python==0.3.34` in `pyproject.toml` and letting uv
resolve it is an **ABI-anchor violation by default**: PyPI's cp313 wheel for
0.3.34 is a CPU-only build against **stock** llama.cpp, and a source build
would vendor stock llama.cpp — both bind the 0.3.34 cdefs against the wrong
`llama.h` (the exact thing the anchor rule forbids). The kickoff's option (A)
is only viable with the fork's vendor swap + CUDA toolchain + `CMAKE_ARGS` —
heavy, and `uv lock` would embed a stock-build path.

### Recommended: option (B) — uv-managed code deps + wheel-pinned binding (hardened)

The patch's pyproject hunk already adds `inferference==0.1.0` +
`llama-cpp-python==0.3.34` + `[tool.uv.sources] inferference = { path =
"../inferference" }`. The plan **hardens** it with one additive line so that
**even `uv sync`/`uv lock` can never resolve the binding from stock**:

```toml
[tool.uv.sources]
inferference = { path = "../inferference" }
# ABI-anchor pair: the binding must come from the Mode A wheel (b10233, anchor
# 0ab9d6fed) — NEVER a PyPI cp313 wheel or a source build (stock llama.cpp).
llama-cpp-python = { url = "file:///home/andrew/Documents/Projects/inferference/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl" }
```

Then:

1. `uv lock` — rewrites `uv.lock` (adds inferference + the wheel URL + the
   wheel's transitive deps: diskcache/jinja2/markupsafe — verified installable
   in P1.6). **Gate:** `grep -c "llama-cpp-python" uv.lock` ≥ 1 and the lock
   entry's source is the wheel URL, never an sdist/PyPI index entry.
2. `UV_PROJECT_ENVIRONMENT=.devenv/state/venv uv sync --all-extras` (the
   current devenv shell already exports `UV_PROJECT_ENVIRONMENT`; setting it
   explicitly is belt-and-suspenders) — installs inferference (editable, path
   dep) + the wheel + keeps pytest/hypothesis/ty/ruff (dev extras, present
   today). **Gate:** `.devenv/state/venv/bin/python -c "import llama_cpp,
   inferference, structured_agents"` with the runtime `LD_LIBRARY_PATH`
   (P1.6's libs; `/run/opengl-driver/lib` first).

**Why this is safe against the clobber question (P1):** devenv does not run
`uv sync` on entry (no `devenv:python:uv` task, no sync script, no checksum
file, current shell has no sync invocation); the wheel install survives
re-entries. And if the owner later runs `uv sync` manually or enables
`uv.sync.enable`, the source override re-installs the **same** wheel — no
drift, no stock build. This makes the wiring reproducible from the lock,
which the kickoff's plain-B (manual `uv pip install --no-deps` per entry)
cannot guarantee.

### Fallback: option (B′) — manual install, binding out of uv entirely

If `uv lock`/`uv sync` proves troublesome at execution (e.g. index
unavailability for diskcache/jinja2), fall back to the report's pattern —
inferference editable without deps, then the wheel by hand:

```bash
uv pip install --python .devenv/state/venv/bin/python -e ../inferference --no-deps
uv pip install --python .devenv/state/venv/bin/python --force-reinstall --no-deps \
  "$INF/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl"
```

Consequence (state it in the execution log): `uv.lock` stays stale w.r.t.
pyproject — any later `uv sync` resolves the stock-build trap and prunes the
wheel; B′ requires re-running the two commands after any venv rebuild. That
is why B (hardened) is recommended.

### Rejected: option (C)

Keeping the binding out of uv entirely (today's pattern) is B′ without the
inferference path dep — but the flipped consumers need `import inferference`
in the reference venv, so the path dep must land anyway; C adds no benefit.

### Venv contents decision (report §Q2.4, kickoff P2)

The reference venv **keeps torch/transformers** — xgrammar 0.2.1's import-time
baggage is already satisfied there (`pyproject.toml:15-16`; torch 2.12.0 in
site-packages), and the framework needs them. No change. The wheel's new
transitive deps (diskcache/jinja2/markupsafe) are additive and harmless.

---

## P3 — the refactor change set, ordered and gated

### Decision: apply-as-one (patch file) with the gate battery in Q6.1 order

Rationale (evidence, not opinion): the patch `git apply --check`-es CLEAN
against the real tree (P1.1); the report §Q5.2 already proved the **applied**
tree passes every intermediate gate (46/46 imports, lazy-loader semantics,
`test_config.py` AST assertion, `llama_core` gone); and hand-editing 45 files
re-introduces drift the patch removes. **Rollback is total and cheap** (the
patch touches only tracked files; working tree is clean). The Q6.1 logical
order is preserved as the **gate sequence** run after the apply — each gate
still proves its layer before the next.

**Fallback:** if any gate fails after the apply, hand-apply in the report
§Q6.1 order (dependency flip → `__init__` → consumer flips → test surgery →
delete package) with the patch as the review reference, gates per step.

### The change set (exactly the 45 patch files)

| Group | Files | Type |
| --- | --- | --- |
| Consumer flips (10) | `benchmarks/project17/context_pool_router.py`, `run_json_workload.py`, `run_native_state_decompose.py`, `run_prefix_cache.py`, `benchmarks/project20/abi_smoke_gate.py`, `run_library_throughput.py`, `examples/benchmark_local.py`, `multi_lora_router.py`, `ornith_json_grammar.py`, `soak_grammar.py` | modify (`inferference.X` imports) |
| Typecheck flip (1) | `tests/typecheck_prefix_cache.py` | modify |
| Package surface (2) | `src/structured_agents/__init__.py`, `pyproject.toml` | modify |
| Test fix (1) | `tests/test_config.py` (`:89-110` assertion: importlib set drops to `[__init__.py, config.py]`) | modify |
| llama_core deletion (17) | `src/structured_agents/llama_core/__init__.py`<br>`src/structured_agents/llama_core/batching.py`<br>`src/structured_agents/llama_core/benchmark.py`<br>`src/structured_agents/llama_core/decode.py`<br>`src/structured_agents/llama_core/diagnostics.py`<br>`src/structured_agents/llama_core/fingerprint.py`<br>`src/structured_agents/llama_core/grammar.py`<br>`src/structured_agents/llama_core/lsp_tree.py`<br>`src/structured_agents/llama_core/middleware.py`<br>`src/structured_agents/llama_core/models.py`<br>`src/structured_agents/llama_core/node_blend_live.py`<br>`src/structured_agents/llama_core/node_delta_live.py`<br>`src/structured_agents/llama_core/node_delta.py`<br>`src/structured_agents/llama_core/prefix_cache_live.py`<br>`src/structured_agents/llama_core/prefix_cache.py`<br>`src/structured_agents/llama_core/router.py`<br>`src/structured_agents/llama_core/seq_routing.py` (the whole package) | delete |
| Duplicate test deletion (14) | `tests/test_llama_core_batching.py`, `tests/test_llama_core_benchmark.py`, `tests/test_llama_core_grammar.py`, `tests/test_llama_core_middleware.py`, `tests/test_llama_core_models.py`, `tests/test_llama_core_router.py`, `tests/test_lsp_tree.py`, `tests/test_node_delta.py`, `tests/test_owned_decode.py`, `tests/test_persistent_prefix_cache.py`, `tests/test_prefix_cache_contracts.py`, `tests/test_prefix_cache_live.py`, `tests/test_seq_routing.py`, `tests/test_seq_routing_gpu.py` | delete |

(76 insertions / 6278 deletions per the report §Q7.)

### Gate sequence (run inside `NIXPKGS_ALLOW_UNFREE=1 devenv shell -- bash -c`, never `bash -lc`)

1. **Package importable / lazy-loader semantics** (report §Q5.2.3, re-run):
   fresh process: `import structured_agents` → assert `inferference` /
   `llama_cpp` / `numpy` are NOT in `sys.modules`; then
   `from structured_agents import MultiLoRARouter` resolves to the
   inferference module objects; unknown names still raise `AttributeError`.
2. **`llama_core` gone**: `grep -rln "llama_core" src benchmarks examples tests`
   → only historical `.scratch` docs remain.
3. **46 flipped import sites resolve**: import each of the 11 flipped files
   under the devenv venv (loop in P7) — mechanical, same statements the
   report executed 46/46.
4. **AST assertion**: `pytest tests/test_config.py::test_only_config_imports_importlib_in_package_source -q`.
5. **Framework suite**: the kept tests (P5.1).
6. **Top-level surface**: `from structured_agents import AdapterSpec,
   EngineConfig, GenerationRequest, GenerationResult, MultiLoRARouter,
   RouteRequest, RouterConfig` all resolve (the 7 additive `_LAZY` entries).

Rollback at any point: `git checkout -- .` (all patch paths are tracked,
clean working tree before the apply) — or, if a branch bookmark was created
(P7 step 1), `git checkout <bookmark> -- .`.

---

## P4 — the devenv repoint (exact `devenv.nix` edits)

The spike venv and the old builds (`out-cuda-3060-postfix2`,
`out-p2fork-c588c4f47`) are **replaced** by the §P2 unit. All 5 project17/20
scripts reference them; every reference must move. The new defaults:

- **`$py`** (the interpreter carrying the binding): `.devenv/state/venv/bin/python`
  (the devenv venv — after P2 it holds llama_cpp + inferference).
- **`$lib`** (the `LLAMA_CPP_LIB_PATH` unit): the wheel's bundled lib dir
  `.devenv/state/venv/lib/python3.13/site-packages/llama_cpp/lib` (Mode A,
  self-contained in this repo), **overridable** to the Mode B build
  `$INF/ci/build/.llamacpp-builds/out-p2fork-24c5d3dbc/lib` (nan-fix,
  recommended for iteration per P6.2).
- **`.cuda_runtime_ld`**: unchanged — it is a plain file of nix store paths
  (gcc-15.2.0-lib, cudart, cublas, graphics-drivers) still valid; keep all
  five `cuda_ld="$(tr -d '\n' < "$spike/.cuda_runtime_ld")"` lines as-is.
- **Stale override-variable names** (`PROJECT17_SPIKE_PY`,
  `PROJECT20_SPIKE_PY`): keep the names (minimal churn) but document in the
  script comment that they now select the reference interpreter; the defaults
  change. Alternatively rename to `PROJECT17_PY`/`PROJECT20_PY` — owner's
  taste, default: keep.

Exact edits:

| Location (`devenv.nix`) | Current | New |
| --- | --- | --- |
| `:31` `project17_lib` | `…/17-llama-cpp-inference-lab/.llamacpp-builds/out-cuda-3060-postfix2/lib` | `"${PROJECT17_LIB_PATH:-$PWD/.devenv/state/venv/lib/python3.13/site-packages/llama_cpp/lib}"` |
| `:64,70` spike python (json-workload) | `…/.venv-spike/bin/python` | `.devenv/state/venv/bin/python` |
| `:90` `lib` (prefix-cache) | `…/out-cuda-3060-postfix2/lib` | same new default as `:31` |
| `:107,111` spike python (prefix-cache) | `…/.venv-spike/bin/python` | `.devenv/state/venv/bin/python` |
| `:135` `py` default (p17-gpu-pytest) | `"${PROJECT17_SPIKE_PY:-$spike/.venv-spike/bin/python}"` | `"${PROJECT17_SPIKE_PY:-$PWD/.devenv/state/venv/bin/python}"` |
| `:137` `lib` (p17-gpu-pytest) | `"$spike/.llamacpp-builds/out-cuda-3060-postfix2/lib"` | `"${PROJECT17_LIB_PATH:-$PWD/.devenv/state/venv/lib/python3.13/site-packages/llama_cpp/lib}"` |
| `:167` `py` default (p20-gpu-pytest) | `"${PROJECT20_SPIKE_PY:-$spike/.venv-spike/bin/python}"` | `"${PROJECT20_SPIKE_PY:-$PWD/.devenv/state/venv/bin/python}"` |
| `:169` `PROJECT20_FORK_LIB` default | `"${PROJECT20_FORK_LIB:-$spike/.llamacpp-builds/out-p2fork-c588c4f47/lib}"` | `"${PROJECT20_FORK_LIB:-$INF/ci/build/.llamacpp-builds/out-p2fork-24c5d3dbc/lib}"` (Mode B default for the p2 fork unit; the wheel-lib default for p17 keeps Mode A) |

**Critical addition — the deleted-test defaults:** `project17-gpu-pytest`'s
default pytest args (`tests/test_node_delta.py tests/test_prefix_cache_live.py
-k gpu`) and `project20-gpu-pytest`'s (`tests/test_seq_routing_gpu.py`) point
at **files the refactor deletes** — the scripts would fail-closed (exit 4,
no tests). After the refactor the reference owns **no GPU-gated tests**; the
GPU contract suite is cross-repo. Recommended new defaults:

- `project17-gpu-pytest` / `project20-gpu-pytest` no-arg default: run the
  reference's framework suite under the new unit env
  (`exec "$py" -m pytest -o addopts='-rs' tests/ -q`), and document that the
  GPU contract gate is cross-repo.
- Add one script `project23-gpu-contract` that runs the cross-repo contract
  suite with GPU discipline:
  `"$INF/ci/runner/gpu-serialized.sh" "$INF/ci/library/.venv/bin/python" -m pytest "$INF/tests/" -q`
  (flock serializes on GPU 0; this is the report's own reproduction recipe
  shape, §Reproduction).

**Same change, not a follow-up** (kickoff P4): the repoint must land in the
same change as the flip, or every GPU/benchmark devenv script is broken the
moment the patch applies. The `project17-pytest-zellij` script (plain
`pytest`, no spike references) needs **no edit** — it runs the devenv venv's
pytest against the framework tests.

**Orphans**: the spike venv and the old builds are **left in place** (house
rule: state it, delete nothing). The devenv scripts stop referencing them;
a future owner can prune `.scratch/projects/17-llama-cpp-inference-lab/`'s
`.venv-spike` + `.llamacpp-builds/out-*` once the new unit is proven.

---

## P5 — the test gate (reference side + cross-repo)

### P5.1 reference-side framework suite (unchanged tests, new unit)

Kept tests (13 + 2 typechecks): `test_agent`, `test_approval`,
`test_authority`, `test_config`, `test_constraint`, `test_engine`,
`test_fornix`, `test_plane`, `test_grammar_soak`, `test_live`,
`test_project17_workload`, `test_state_blob_model`, `test_xgrammar_api_contract`
+ `typecheck_constraint.py`, `typecheck_prefix_cache.py` (flipped). The DBOS
session fixture (`tests/conftest.py:8-25`) needs dbos in the venv — present
(2.23.0). `test_live` is skipped unless `SAV_LIVE=1` (marker at
`pyproject.toml` `[tool.pytest.ini_options]`).

Gate commands (P7): the framework suite + the AST assertion + the lazy-loader
checks, all under the devenv venv python with the §P2 unit env.

### P5.2 cross-repo (source of truth)

- inferference `testee verify` (contract suite, `ci/library/.venv`) stays the
  canonical gate — the 89 tests the report ran against the wheel (GPU 0,
  gpu-serialized) still pass; unaffected by anything the reference does.
- GPU suites `p2-mixed-batch-lora`, `nanbeige-hats`, `router`, `prefix-cache`,
  `nanbeige-arch` (inferference `suites/`) unaffected — the flip changes no
  inferference code.
- Reference-side GPU gate: none remain in-repo after the deletions; the new
  `project23-gpu-contract` (P4) runs the cross-repo suite. GPU discipline:
  `CUDA_VISIBLE_DEVICES=0` (GPU 1 may host a vLLM runner), the `gpu-serialized`
  flock pattern.

---

## P6 — owner decision inputs (state + recommended default each)

| # | Decision | Recommendation (default) | Notes |
| --- | --- | --- | --- |
| 1 | Wheel rebuild: prerequisite or interim gap? | **Interim gap + Mode B `out-p2fork-24c5d3dbc`**; rebuild is a separate owner session (never a Mode-B-only artifact as a release — AGENTS.md Mode rule) | Probe workloads clean on `81a5dba8…` (report §Q2.2); full-coverage LoRA workloads would regress on Mode A only — Mode B covers |
| 2 | Mode B vs Mode A for the refactored reference | **B for iteration now; A (rebuilt wheel) at release** | Same ABI unit either way; the wheel's bundled lib is the Mode A default in P4, overridable to Mode B |
| 3 | Benchmarks/examples: keep-thin vs remove | **Flip now (the patch); remove in a follow-up** once benchkit ownership is confirmed | Every runner has a benchkit port (report §Q3.1); keep-thin is the conservative option |
| 4 | Top-level core surface: keep or drop the 7 `_LAZY` entries | **Keep** (additive; consumers import inferference directly anyway) | Report §Q2.3 |
| 5 | Apply mechanics: patch-as-one vs hand-applied | **Apply-as-one** with the Q6.1 gate battery (P3) | Differs from the kickoff's default (hand-applied) — evidence: patch verified clean, applied-tree gates pre-proven (report §Q5.2), total rollback cheap |
| 6 | `uv.lock` machine-specificity (wheel URL + `../inferference` path) | **Accept for dev** (same class as the path dep); release installs the built inferference wheel | P2 |
| 7 | Deleted-test defaults in `project17/project20-gpu-pytest` | **Repoint to the framework suite; add `project23-gpu-contract`** | P4 — without this the scripts exit 4 after the flip |
| 8 | **Stale AGENTS.md anchor text** (inferference side: says `c588c4f47`/b10103; actual `0ab9d6fed`/b10233) | **Fix in inferference AGENTS.md** (owner doc action) | Informational; operative anchor for this refactor is `0ab9d6fed` |

---

## P7 — the runnable execution sequence + risk register

Single copy-pasteable sequence, top-to-bottom, inside
`NIXPKGS_ALLOW_UNFREE=1 devenv shell` (`bash -c`, never `bash -lc`), run in
the reference root. Every step is gated and rollback-able. `INF` and the
runtime `LD_LIBRARY_PATH` are set once at the top.

```bash
# ============ Step 0 — bookmark + env ============
set -euo pipefail
INF=/home/andrew/Documents/Projects/inferference
REF=/home/andrew/Documents/Projects/structured-agents-v2
cd "$REF"
git checkout -b project23-inferference-consumer-refactor   # rollback point; or: git stash
spike=.scratch/projects/17-llama-cpp-inference-lab
CUDA_LD="$(tr -d '\n' < "$spike/.cuda_runtime_ld")"
GCC_DIR=$(dirname "$(/nix/store/lvwga6ivl1d4lnw0zis9ajs0rqx9gp4i-gcc-15.2.0/bin/gcc -print-file-name=libstdc++.so)")
GOMP_DIR=$(dirname "$(/nix/store/lvwga6ivl1d4lnw0zis9ajs0rqx9gp4i-gcc-15.2.0/bin/gcc -print-file-name=libgomp.so)")
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$GCC_DIR:$GOMP_DIR:$CUDA_LD"
VENV_PY=.devenv/state/venv/bin/python

# ============ Step 1 — apply the step-7 patch (45 files) ============
git apply --check "$INF/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch"
git apply "$INF/.scratch/projects/003-reference-consumer-refactor/reference-refactor.patch"
git status --short | wc -l            # GATE: 45
grep -rln "llama_core" src benchmarks examples tests || echo "llama_core gone"   # GATE
# rollback: git checkout -- .   (or git checkout project22-llama-cpp-fork-reorg -- .)

# ============ Step 2 — P2 wiring (hardened option B) ============
# append to [tool.uv.sources] in pyproject.toml (already has inferference path):
#   llama-cpp-python = { url = "file://$INF/ci/modea/wheels/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl" }
uv lock                                                    # GATE: grep -c "llama-cpp-python" uv.lock >= 1
UV_PROJECT_ENVIRONMENT="$REF/.devenv/state/venv" uv sync --all-extras
"$VENV_PY" -c "import llama_cpp, inferference, structured_agents; print(llama_cpp.__version__)"  # GATE

# ============ Step 3 — lazy-loader + surface gates ============
"$VENV_PY" -c "
import sys
import structured_agents
assert 'inferference' not in sys.modules and 'llama_cpp' not in sys.modules, 'not lazy'
from structured_agents import MultiLoRARouter, AdapterSpec, EngineConfig, GenerationRequest, GenerationResult, RouteRequest, RouterConfig
print('lazy surface OK')"
"$VENV_PY" -c "import structured_agents, sys
try: structured_agents.__no_such_symbol__
except AttributeError: print('fail-closed OK')"

# ============ Step 4 — flipped consumers resolve ============
for f in benchmarks/project17/context_pool_router.py benchmarks/project17/run_json_workload.py \
         benchmarks/project17/run_native_state_decompose.py benchmarks/project17/run_prefix_cache.py \
         benchmarks/project20/abi_smoke_gate.py benchmarks/project20/run_library_throughput.py \
         examples/benchmark_local.py examples/multi_lora_router.py examples/ornith_json_grammar.py \
         examples/soak_grammar.py tests/typecheck_prefix_cache.py; do
  PYTHONPATH=src:.devenv/state/venv/lib/python3.13/site-packages "$VENV_PY" -c \
    "import importlib.util, sys; spec=importlib.util.spec_from_file_location('m','$f'); m=importlib.util.module_from_spec(spec); sys.modules['m']=m; spec.loader.exec_module(m)" || { echo "FLIP FAIL: $f"; exit 1; }
done; echo "46 import sites resolve"

# ============ Step 5 — AST assertion + framework suite ============
"$VENV_PY" -m pytest tests/test_config.py::test_only_config_imports_importlib_in_package_source -q
"$VENV_PY" -m pytest -q tests/test_agent.py tests/test_approval.py tests/test_authority.py \
  tests/test_config.py tests/test_constraint.py tests/test_engine.py tests/test_fornix.py \
  tests/test_plane.py tests/test_grammar_soak.py tests/test_live.py tests/test_project17_workload.py \
  tests/test_state_blob_model.py tests/test_xgrammar_api_contract.py   # GATE: all pass (live skipped unless SAV_LIVE=1)

# ============ Step 6 — P4 devenv.nix repoint (P4 table edits) ============
# ...apply the devenv.nix edits from P4 (lib/py defaults + deleted-test defaults + project23-gpu-contract)...
devenv shell -- bash -c 'echo devenv shell OK'                              # GATE: devenv.nix still evaluates (loud failure otherwise)
devenv shell -- project17-gpu-pytest --collect-only -q                    # GATE: no deleted-file refs
# VERIFY-AT-EXECUTION: wheel survives a re-enter (P1.2 caveat):
devenv shell -- bash -c 'ls .devenv/state/venv/lib/python3.13/site-packages/ | grep -i llama_cpp'

# ============ Step 7 — cross-repo gates ============
cd "$INF" && ci/runner/gpu-serialized.sh ci/library/.venv/bin/python -m pytest tests/ -q   # contract suite, GPU 0
cd "$REF" && "$INF/ci/runner/gpu-serialized.sh" "$INF/ci/library/.venv/bin/python" -m pytest "$INF/tests/" -q  # project23-gpu-contract body

# ============ Step 8 — owner review ============
# git diff --stat (14 modified, 31 deleted), commit message per house style,
# zero AI attribution, then decide P6 items 1/2/3/4.
```

**VERIFY-AT-EXECUTION items** (fail-closed — cannot be proven without mutating
the repo env, so they are marked, not assumed): (1) the wheel survives the
first post-Aug-1 `devenv shell` re-evaluation (P1.2; exact command above);
(2) `uv lock` reaches the index for diskcache/jinja2 (offline fallback =
option B′); (3) `project23-gpu-contract` under an actual GPU-0 lock (needs the
flock + a free GPU; the GPU suites are otherwise unaffected).

### Risk register (execution phase)

| Risk | Class | Mitigation / status |
| --- | --- | --- |
| venv clobber on re-enter | env | mechanism verified (P1.2): no uv sync task/script, interpreter-only rebuild; wheel survives; verify-at-execution command provided; B hardening makes even a future `uv sync` safe |
| Stock llama.cpp build trap (PyPI wheel / sdist resolution) | ABI | prevented by construction: `[tool.uv.sources]` url → Mode A wheel; B′ fallback uses `--no-deps`; never a bare `uv sync` without the source override |
| Wheel gap (nan-fix absent in `81a5dba8…`) | behavior on full-coverage LoRA GPU paths | documented (report §Q2.2); probe adapters unaffected; Mode B `out-p2fork-24c5d3dbc` covers the gap (P6.1/2) |
| PEP 562 lazy-map drift (missed symbol → silent `ImportError` at call time) | the classic lazy-loader trap | 7 entries verified against inferference modules (report §Q5.2.3); unknown names fail loudly (gate, step 3) |
| `test_config.py:89-110` AST assertion | test brittleness | pre-validated (report §Q5.2.4); gate, step 5 |
| Deleted-test defaults in the gpu-pytest scripts | env | P4 repoints them; gate `--collect-only` (step 6) |
| `devenv.nix` path drift (lib/py/cuda_ld) | env | exact edits in P4; `devenv shell` entrance gate (step 6) |
| The snapshot commit `baae135` (19/21/training WIP) | scope | outside the change set by construction; the patch touches none of its files (`git apply --check` clean proves it) |
| AGENTS.md stale anchor text (`c588c4f47` vs `0ab9d6fed`) | doc | flagged P6.8 (inferference-side owner fix); operative anchor for this refactor is `0ab9d6fed` |
| Rollback | process | branch bookmark (step 0); `git checkout -- .` restores all 45 patch paths; per-step gates localize any failure before it compounds |

---

## Out of scope (explicit)

- The untracked-WIP snapshot (19, 21, `src/structured_agents/training/` —
  committed as `baae135`, still outside the change set; excluded from every
  diff list).
- The wheel rebuild from `24c5d3dbc` (owner session; P6.1).
- Any fork/upstream work, any vLLM/SGLang proposal, any anchor/cdef change,
  any Mode-B-only-as-release artifact.
- The stale AGENTS.md anchor text (P6.8; inferference-side doc fix).

## Checklist

- [x] P1 baseline verified with commands run, not assumed (esp. the devenv
      venv population mechanism — tasks.db + virtualenv script + no uv-sync
      task + no checksum file + no sync in the current shell)
- [x] P2 wiring decided with evidence (B hardened, wheel-probe proof, B′
      fallback, option A trap documented); clobber question answered + the
      one residual marked verify-at-execution with the exact command
- [x] P3 ordered gated steps, importable at every gate, rollback per step
- [x] P4 devenv repoint with exact `devenv.nix` edits (5 scripts, line-numbered)
- [x] P5 test gate (framework suite + cross-repo contract suite + GPU 0 /
      gpu-serialized discipline)
- [x] P6 owner decisions with recommended defaults (8 items)
- [x] P7 runnable command sequence + risk register
- [x] Out-of-scope list explicit; reference tree otherwise untouched; no
      commits/pushes; no venv mutation (only /tmp scratch); `git apply
      --check` only
- [x] No AI attribution; no vLLM/SGLang; anchor/cdef untouched; every
      unverifiable gate marked verify-at-execution
