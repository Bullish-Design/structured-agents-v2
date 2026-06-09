# Phase 2 Kickoff — Dual-path runtime + runner (`DualPathRuntime` / `DualPathRunner`)

You are implementing **Phase 2** of the dual-path track: the **durable runtime** that wraps two
`StructuredAgent`s as `DBOSAgent`s, runs them concurrently (Architecture C), and persists a
`ComparisonRecord` via the Phase-1 data core. This doc is **self-contained**: read it fully, confirm
the green baseline, then implement. Everything it references is in the repo.

---

## 0. The starting prompt (this is what kicked you off)

> Implement Phase 2 of the structured-agents-v2 dual-path layer: `DualPathRuntime` (DBOS
> register→launch→shutdown lifecycle) and `DualPathRunner` (top-level `asyncio.gather` over two
> `DBOSAgent`s, sampling, validate/diff/persist). Build on the Phase-1 data core
> (`src/structured_agents_v2/dual_path/`, already on `main`) and the proven spike
> (`.scratch/projects/03-dual-path/spike/run_spike.py`). Read
> `.scratch/projects/04-dual-path-runtime/PHASE_2_KICKOFF.md` in full first, confirm green
> (`devenv shell -- uv run --extra dev --extra dual-path pytest -q`), then implement the runtime +
> runner + Postgres/DBOS-gated tests. Run everything inside `devenv shell --`. No AI-attribution
> trailers in commits.

## 1. How to operate (read first)

- **Run ALL commands inside the devenv shell:** `devenv shell -- <cmd>`. Gives Python 3.13.13 + uv.
  Never bare `uv run` or system python.
- Dual-path work needs **both** extras: `devenv shell -- uv run --extra dev --extra dual-path <cmd>`.
- Type check: `… ty check src` (checker is **ty**, Astral's — not mypy). Lint: `… ruff check src tests`,
  format `… ruff format src tests`. Tests: `… pytest -q`.
- **Baseline is GREEN on `main`:** 56 passed / 3 skipped, ty + ruff clean, dual_path subpackage at
  100% cov. **Confirm green before changing anything.**
- **Postgres must be running** (the dual-path tests are Postgres-gated). See §6 to start it.
- **Git:** commit to `main` when green (the repo keeps phases linear on `main`). **No
  AI-attribution trailers** (user global rule: no "Co-Authored-By", no "Generated with").
- If you need an interactive command (e.g. `psql`), ask the user to run it with `! <cmd>`.

## 2. Authoritative context (read these, in order)

1. **This file.**
2. `.scratch/projects/03-dual-path/CONCEPT.md` — the design. Focus **§4.1 `DualPathRuntime`**,
   **§4.2 `DualPathRunner`**, §4.3 `ComparisonRecord`, §7 layout, §8 acceptance, §9 open Qs.
3. `.scratch/projects/03-dual-path/spike/FINDINGS.md` — the verified DBOS facts (cite-backed).
4. `.scratch/projects/03-dual-path/spike/run_spike.py` — **the proven reference implementation.**
   Phase 2 is essentially this, refactored into `DualPathRuntime` + `DualPathRunner`. Also
   `runner.py` (spike helpers) and `schemas.py` there.
5. `src/structured_agents_v2/dual_path/` — the Phase-1 data core you build on (§3 below).
6. `src/structured_agents_v2/{backend,profile,agent,capture}.py` — the core (small; read it).

Auto-memory (`MEMORY.md` + `dual-path-research.md`) indexes the same facts.

## 3. What Phase 1 already gives you (on `main`, behind `[dual-path]`)

`src/structured_agents_v2/dual_path/` — **DBOS-free** data core, public API:

```python
from structured_agents_v2.dual_path import (
    ComparisonRecord, ModelIdentity, build_comparison_record,
    profile_version, schema_version, content_hash, lib_version,
    Comparator, ComparisonSignal, ExactFieldComparator,
    ComparisonStore, ComparisonExport, EvalSummary, GroupEval,
    DualPathError, DualPathConfigError,
)
```

- `ModelIdentity(kind="vllm"|"frontier", wire_model, base?, adapter?, adapter_rev?, vllm_tag?, provider?, model_id?)`
- `build_comparison_record(*, run_id, prompt, profile: AgentProfile, output_type: type[BaseModel],
  primary_model: ModelIdentity, reference_model: ModelIdentity|None, primary_output: BaseModel|None,
  reference_output: BaseModel|None, primary_error=None, reference_error=None, reference_skipped=False,
  primary_usage=None, reference_usage=None, primary_workflow_id=None, reference_workflow_id=None,
  comparator: Comparator|None=None) -> ComparisonRecord` — validates, diffs (signal only when **both**
  outputs valid), versions. **This is the assembler the runner calls.**
- `ComparisonStore(pg_url)` → `.init_schema()`, `.save(record) -> int`, `.query(*, profile_version=,
  schema_version=, agreement_exact=, limit=) -> list[ComparisonRecord]`. Table `comparison_records`
  (jsonb + GIN). `ComparisonExport(store)` → `.to_sft_jsonl(...)`, `.eval_view(...)`.

**Phase 2 must NOT change the data core's public behavior** (its tests are the contract).

## 4. What Phase 2 builds

Two new modules in `src/structured_agents_v2/dual_path/`, per CONCEPT §4.1–§4.2:

### `runtime.py`

```python
class DualPathConfig(BaseModel):
    app_name: str = "dual-path"
    pg_url: str                       # DBOS system DB + ComparisonStore (same Postgres)
    run_admin_server: bool = False
    default_sample_rate: float = 1.0  # fraction of runs that also fire the reference leg

class DualPathRuntime:
    def __init__(self, config: DualPathConfig) -> None: ...   # constructs DBOS(config=DBOSConfig(...))
    def register(self, name: str, *, primary: StructuredAgent, reference: StructuredAgent,
                 sample_rate: float | None = None, comparator: Comparator | None = None,
                 model_step_config: StepConfig | None = None) -> "DualPathRunner": ...
    def launch(self) -> None: ...     # DBOS.launch() — call AFTER all register() calls
    def shutdown(self) -> None: ...   # DBOS.destroy()
    def __enter__(self)/__exit__(...) # context-managed launch/shutdown
```

`register()` MUST:
- wrap `primary.agent` and `reference.agent` in `DBOSAgent(...)` with **unique** names
  (e.g. `f"{name}@primary"`, `f"{name}@reference"`) and the given `model_step_config` —
  **this happens before `launch()`** (DBOS registers workflows at `DBOSAgent.__init__`; see §5);
- raise `DualPathConfigError` if either profile's resolved decode mode is **not `json_schema`**
  (decision #4: only json_schema agents get a teacher) — check via
  `profile.resolve()` → `spec.mode == "json_schema"`;
- raise `DualPathConfigError` if called **after** `launch()`.

### `runner.py`

```python
class DualPathRunner:
    name: str
    async def run(self, prompt: str, *, force_reference: bool | None = None, **kw) -> ComparisonRecord
    @property
    def primary(self) -> DBOSAgent: ...      # escape hatch
    @property
    def reference(self) -> DBOSAgent: ...
```

`run()` flow (mirror `run_spike.py` gate 3):
1. **Sampling:** run the reference leg iff `force_reference` is True, or (when `None`)
   `random.random() < sample_rate`. Else primary-only.
2. Assign workflow ids: `pid, rid = f"primary-{uuid}", f"reference-{uuid}"`; wrap each
   `agent.run(prompt)` in `with SetWorkflowID(wid):`.
3. **Concurrency:** top-level `await asyncio.gather(primary_run, reference_run)` (no parent
   workflow). Primary-only path: just `await primary_run`.
4. Per leg, capture failures: if `agent.run` raises, record `*_error=str(exc)`, output `None`
   (don't fail the whole `run`).
5. Build `ModelIdentity` for each leg from its `StructuredAgent.profile` + `Backend`
   (primary: `kind="vllm"`, `wire_model = profile.adapter or backend.default_model`,
   `adapter=profile.adapter`; reference: `kind="frontier"`, `wire_model=model_id=ref backend.default_model`,
   `provider` best-effort from base_url). The runtime/runner will need a handle to each backend or
   the resolved identity — decide how to thread it (pass identities into `register`, or derive from
   the `StructuredAgent`; the `StructuredAgent` exposes `.profile` and `.agent`).
6. Extract usage with a small helper (`result.usage()` → `{input_tokens, output_tokens,
   total_tokens, requests}`), call `build_comparison_record(...)`, `store.save(record)`, return it.

`run()` **always returns a `ComparisonRecord`**; `record.primary_output` is the user-facing answer.

### Wire-up
- Update `dual_path/__init__.py` to export `DualPathRuntime`, `DualPathConfig`, `DualPathRunner`
  (and broaden the import guard to also cover `dbos`, not just `psycopg`).
- The `tool.ty.overrides` for `dual_path/**` (unresolved-import) already covers the `dbos` import.

## 5. Verified DBOS facts you depend on (from the spike — don't re-derive)

- `from pydantic_ai.durable_exec.dbos import DBOSAgent, StepConfig` ; `from dbos import DBOS,
  DBOSConfig, SetWorkflowID`.
- `DBOSAgent(structured_agent.agent, name="unique", model_step_config=StepConfig(max_attempts=…,
  retries_allowed=…))` — **preserves the Phase-2 wire shape byte-for-byte** (model=adapter,
  `response_format` json_schema, `extra_body`). The `RequestCapture` httpx hook still works
  (`Backend(capture=True)`; the body is on `structured_agent._capture.last.body`).
- **Ordering constraint (critical):** construct **all** `DBOSAgent`s *before* `DBOS.launch()`.
  Lifecycle: `DBOS(config=DBOSConfig(name=app_name, system_database_url=pg_url,
  run_admin_server=False))` → build agents → `DBOS.launch()` → run → `DBOS.destroy()`.
- **Architecture C:** top-level `asyncio.gather` over two `DBOSAgent.run()` coroutines (each its own
  workflow) runs two independent durable workflows — **no parent `@DBOS.workflow`**, so the
  `DBOSParallelExecutionMode` (tool-call ordering) caveat never applies.
- Assign/correlate workflow ids with `with SetWorkflowID(wid):` around each `agent.run`.
- Introspection is **async-only inside a loop**: `await DBOS.list_workflow_steps_async(wid)` (the
  sync variant raises). `DBOS.list_workflows`, `retrieve_workflow` also exist.
- DBOS is a **process-global singleton** — only one `DBOS(...)`/`launch()`/`destroy()` per process.
  In pytest use a module/session fixture; `destroy()` on teardown. Two test modules can't both hold a
  live DBOS — keep DBOS-using tests in one module (or serialize).

## 6. Infrastructure

- **Postgres** runs on `127.0.0.1:5433`, DB `dual_path` (cluster under `.devenv/state/pg-dualpath`,
  gitignored). Default URL: `postgresql://andrew@127.0.0.1:5433/dual_path` (env override
  `DUAL_PATH_TEST_PG_URL`). If it's not up:
  ```bash
  PGDATA=.devenv/state/pg-dualpath
  pg_ctl -D "$PGDATA" -l "$PGDATA/server.log" -o "-p 5433 -k $PGDATA/sock -c listen_addresses=127.0.0.1" start
  pg_isready -h 127.0.0.1 -p 5433        # expect: accepting connections
  ```
  (initdb already done. `createdb -h 127.0.0.1 -p 5433 -U andrew dual_path` if the DB is missing.)
- DBOS uses the same Postgres as its system DB and auto-creates its `dbos.*` schema on launch.
- The `[dual-path]` extra (already in `pyproject.toml`) provides `dbos` + `psycopg`.

## 7. Tests to write (`tests/test_dual_path_runtime.py`, Postgres+DBOS gated)

Reuse the spike's in-process ASGI mock (returns Command-JSON keyed by the wire `model`; copy the
`MockOpenAI` from `run_spike.py` or factor a shared fixture). Gate: `pytest.importorskip("dbos")` +
the `127.0.0.1:5433` socket probe (pattern in `tests/test_dual_path_store.py`). One DBOS lifecycle
per module (fixture: init → register agents → launch; teardown: destroy). Cover:

1. `register()` two agents + `launch()`; a non-`json_schema` profile → `DualPathConfigError`;
   `register()` after `launch()` → error.
2. `run()` returns a `ComparisonRecord` with a validated `primary_output`; the captured wire shape
   matches the Phase-2 contract (`model`=adapter, `response_format.type=="json_schema"`).
3. A dual run persists exactly one row (assert via `ComparisonStore.query`/row id); `*_workflow_id`
   set and correlatable.
4. Sampling: `sample_rate=0.0` (or `force_reference=False`) → `reference_skipped=True`,
   `reference_output is None`, record still saved; `force_reference=True` → reference runs.
5. Error path: a reference leg that raises (point the reference at a mock that returns invalid JSON,
   or a bad URL) → `reference_valid=False` + `reference_error` set, record still saved, primary OK.

Keep the lean-core suite green with `dbos`/`psycopg` **absent** (tests skip).

## 8. Acceptance for Phase 2

- `runtime.py` + `runner.py` implemented; `__init__` exports them; import guard covers `dbos`.
- All §7 tests green; full suite green; ty + ruff clean; dual_path subpackage coverage stays high.
- No change to Phase-1 data-core behavior (its tests still pass unmodified).
- Update `CONCEPT.md` §7 phase list (mark Phase 2 built) and the `dual-path-research` memory.

## 9. Decisions already made (don't relitigate)

Architecture **C**; two stores (DBOS execution / jsonb data); **json_schema-only** reference
(refuse others at `register`); **Postgres + jsonb**; **register-before-launch** lifecycle;
**synchronous gather** for MVP with **sampling** for spend control (fire-and-forget/async reference
is a later phase). Open (defer, note if hit): adapter registry for `adapter_rev`; strict-schema
rewrite (a *reference-path* concern, surfaces live in Phase 3, not here); cost/price table (Phase 4).

## 10. Commands cheat-sheet

```bash
devenv shell -- uv run --extra dev --extra dual-path pytest -q
devenv shell -- uv run --extra dev --extra dual-path ty check src
devenv shell -- uv run --extra dev --extra dual-path ruff check src tests
# run the proven spike (sanity that infra is up):
devenv shell -- uv run --extra dev --extra dual-path python .scratch/projects/03-dual-path/spike/run_spike.py
```

## 11. Suggested task order

1. Confirm green baseline + Postgres up; read §2 sources + skim `run_spike.py`.
2. `runtime.py`: `DualPathConfig`, `DualPathRuntime.__init__/register/launch/shutdown` (+ json_schema
   guard, register-after-launch guard). 3. `runner.py`: `DualPathRunner.run` (sampling, SetWorkflowID,
   gather, error capture, identity, usage, build+save). 4. `__init__` exports + guard. 5. Tests (§7).
   6. Green-up (ty/ruff/pytest), update CONCEPT + memory, commit to `main`.
```
