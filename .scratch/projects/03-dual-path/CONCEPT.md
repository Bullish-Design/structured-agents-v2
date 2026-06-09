# Dual-path inference + DBOS capture — Concept (draft)

## Document status

- **Status:** Implementable concept for the dual-path / eval-capture layer. Drafted **after** the
  spike, grounded in its verified facts.
- **Builds on:** `03-dual-path/spike/FINDINGS.md` (verified), `03-dual-path/spike/SPIKE_PLAN.md`
  (decisions), `02-library-wrapper/CONCEPT.md` (the core it sits on — Phase 1+2 on `main`).
- **Package:** new optional subpackage `src/structured_agents_v2/dual_path/` behind a `[dual-path]`
  extra. **The lean core never imports DBOS.**
- **Python:** 3.13 · **PydanticAI:** 1.87.0 · **DBOS:** 2.23.0 · **Postgres:** 17 · run via
  `devenv shell -- …`.
- **Core thesis:** Every *sampled* agent run executes against the local vLLM `Backend` **and** a
  frontier OpenAI-compatible API concurrently; both outputs are validated against the same
  `ConstrainedOutput`; each run is persisted as a **versioned, comparable `ComparisonRecord`** in
  Postgres `jsonb` — simultaneously SFT/preference data for the small local models and an ongoing
  local-vs-frontier eval. DBOS provides durability, retries, and run identity.

---

## 1. What we are building — and the facts it rests on (all verified in the spike)

This layer is a **capture + comparison harness**, not a new agent runtime. It wraps existing
`StructuredAgent`s; it adds durability, a reference leg, and a portable data record.

| Fact (verified in spike) | Consequence for this design |
|---|---|
| `DBOSAgent` preserves the Phase-2 wire shape byte-for-byte (`model`=adapter, `response_format` json_schema, `extra_body` intact) | We wrap `StructuredAgent.agent` directly; no change to Phase 1/2 |
| Top-level `asyncio.gather` over two `DBOSAgent`s → two independent durable workflows | **Architecture C**: no parent workflow, no nested-workflow determinism caveat |
| DBOS persists each call as a `{name}__model.request` step whose output is a pydantic-ai `ModelResponse` (version-coupled) | DBOS = *execution* store; our `ComparisonRecord` jsonb = *data* store (two stores) |
| `ComparisonRecord` round-trips through a Postgres `jsonb` column, queryable via `->`/`->>`, GIN-indexable | jsonb is the portable, query-stable training/eval SoT |
| All `DBOSAgent`s must be registered **before** `DBOS.launch()` | A `DualPathRuntime` owns an explicit **register → launch → shutdown** lifecycle |
| `DBOS.list_workflow_steps` is async-only inside a loop | Introspection/export uses the `_async` API |
| DBOS runs on SQLite *or* Postgres | We use **Postgres** (jsonb); SQLite stays a zero-dep test fallback |

### Non-goals (explicit)

- Not a serving/ensembling layer (MVP). The reference leg is for *data + evals*, not to pick the
  "better" answer at request time. (Fire-and-forget/async reference is a later option, §9.)
- No training/fine-tuning here — we *produce* SFT/eval data; training is out-of-band.
- No new model client or parser — we reuse `Backend` / `StructuredAgent` / PydanticAI.
- Bare-string modes (grammar/regex/choice) are **out of scope** — only `json_schema` agents get a
  teacher (the frontier path can't enforce non-json_schema constraints).
- Not woven into the core — DBOS + Postgres stay an opt-in extra.

---

## 2. Design tenets

1. **Separate, optional, opt-in.** `dual_path/` imports the core; the core never imports it. Behind
   `[dual-path]`; a clear error if DBOS is missing.
2. **Architecture C.** Two independent `DBOSAgent`s + a top-level `gather`. No parent workflow.
3. **Two stores, two jobs.** DBOS = durable execution/replay/retry. `ComparisonRecord` jsonb =
   portable, versioned data for SFT + evals. Never export training data from DBOS's tables.
4. **The reference path is just a `Backend`.** `caps(xgrammar=False, lora=False)` + a provider
   structured-output descriptor. No special client.
5. **json_schema only.** Reference legs are validated against the same Pydantic type; unenforceable
   modes are refused at registration with a clear error.
6. **Versioned and reproducible.** Every record carries content hashes of the profile and the
   resolved schema, plus structured model identities.
7. **Explicit lifecycle.** `register → launch → run → shutdown`, to satisfy DBOS's
   register-before-launch rule and to keep the global singleton contained.
8. **Escape hatches.** The underlying `DBOSAgent` and `pydantic_ai.Agent` stay reachable.

---

## 3. Architecture

```text
 Application: builds primary + reference Backends, registers dual-path agents
      │
      ▼
 DualPathRuntime  ── owns DBOS lifecycle (config, register → launch → shutdown)
      │ register(name, primary: StructuredAgent, reference: StructuredAgent, …)
      ▼
 DualPathRunner (per agent)
      │  await run(prompt):
      │     ┌─ asyncio.gather ────────────────────────────────┐   (top-level, no parent workflow)
      │     │  primary  DBOSAgent.run  → durable workflow+step │   model=adapter, response_format json_schema
      │     │  reference DBOSAgent.run → durable workflow+step │   frontier, caps(xgrammar=False,lora=False)
      │     └──────────────────────────────────────────────────┘
      │  validate both → Comparator → ComparisonRecord (versioned)
      ▼
 ComparisonStore (Postgres jsonb, GIN) ──▶ ComparisonExport ──▶ SFT JSONL · eval view
      ▲
 DBOS system schema (workflow_status, operation_outputs) — execution SoT, replay/retry
```

---

## 4. Core abstractions

### 4.1 `DualPathRuntime` — DBOS lifecycle owner (the register-before-launch entrypoint)

The spike proved DBOS requires every workflow registered before `launch()`. The runtime makes that
the API shape, and contains the process-global `DBOS` singleton in one place.

```python
class DualPathConfig(BaseModel):
    app_name: str = "dual-path"
    pg_url: str                      # DBOS system DB + ComparisonStore (same Postgres)
    run_admin_server: bool = False
    default_sample_rate: float = 1.0 # fraction of runs that also fire the reference leg

class DualPathRuntime:
    def __init__(self, config: DualPathConfig) -> None: ...   # constructs DBOS(config=…)
    def register(
        self,
        name: str,
        *,
        primary: StructuredAgent,
        reference: StructuredAgent,
        sample_rate: float | None = None,
        comparator: "Comparator | None" = None,
        model_step_config: StepConfig | None = None,
    ) -> "DualPathRunner": ...        # wraps both .agent in DBOSAgent (BEFORE launch); validates json_schema
    def launch(self) -> None: ...      # DBOS.launch() — after all registrations
    def shutdown(self) -> None: ...    # DBOS.destroy()
    def __enter__/__exit__            # context-managed launch/shutdown
```

`register()` raises `DualPathError` if either profile's decode mode is not `json_schema`, or if the
reference backend advertises `caps.xgrammar`/`caps.lora` it shouldn't (sanity), or if called after
`launch()`.

### 4.2 `DualPathRunner` — one dual-path agent

```python
class DualPathRunner:
    name: str
    async def run(self, prompt: str, *, force_reference: bool | None = None, **kw) -> ComparisonRecord:
        # 1. decide sampling: run reference iff rng < sample_rate (force_* overrides)
        # 2. assign workflow ids via SetWorkflowID(primary-<uuid> / reference-<uuid>)
        # 3. gather(primary.run, reference.run)  — or primary-only when sampled off
        # 4. validate both, Comparator → signals, build ComparisonRecord
        # 5. ComparisonStore.save(record);  return it
    @property
    def primary(self) -> DBOSAgent: ...     # escape hatch
    @property
    def reference(self) -> DBOSAgent: ...
```

`run()` always returns a `ComparisonRecord` (the primary output is `record.primary_output`). On a
sampled-off call, `reference_*` fields are `None` and `reference_skipped=True`.

### 4.3 `ComparisonRecord` — the unit of training/eval data (Pydantic, jsonb-serialized)

```python
class ModelIdentity(BaseModel):
    kind: Literal["vllm", "frontier"]
    wire_model: str                 # the OpenAI `model` field actually sent
    base: str | None = None         # base checkpoint (vllm)
    adapter: str | None = None      # LoRA name (vllm)
    adapter_rev: str | None = None  # TODO: needs an adapter registry (see §8)
    vllm_tag: str | None = None     # container/image tag
    provider: str | None = None     # frontier provider id
    model_id: str | None = None     # frontier model id

class ComparisonRecord(BaseModel):
    # identity / versioning
    run_id: str
    profile_version: str            # sha256 of serialized AgentProfile
    schema_version: str             # sha256 of resolved model_json_schema()
    primary_model: ModelIdentity
    reference_model: ModelIdentity | None
    decode_mode: Literal["json_schema"]
    lib_version: str
    primary_workflow_id: str | None
    reference_workflow_id: str | None
    created_at: str
    # payload
    prompt: str
    instructions: str
    primary_output: dict | None
    reference_output: dict | None
    primary_error: str | None
    reference_error: str | None
    raw_primary: dict | None        # escape hatch (optional)
    raw_reference: dict | None
    # signals
    primary_valid: bool
    reference_valid: bool
    reference_skipped: bool = False
    signal: "ComparisonSignal | None"   # agreement + diff (see 4.4)
    primary_usage: dict | None
    reference_usage: dict | None
    cost: dict | None               # derived from usage × price table (4.6)
```

`profile_version` and `schema_version` are *both* present because `output_type_ref` is a reference;
the resolved schema hash makes a record reproducible even if the ref later points elsewhere.

### 4.4 `Comparator` — pluggable agreement signal

```python
class ComparisonSignal(BaseModel):
    agreement_exact: bool
    field_diff: dict[str, list[Any]]       # field -> [primary, reference]
    score: float | None = None             # 0..1 partial credit (optional)

class Comparator(Protocol):
    def compare(self, primary: BaseModel, reference: BaseModel) -> ComparisonSignal: ...

class ExactFieldComparator:                 # MVP default (proven in the spike)
    def compare(self, primary, reference) -> ComparisonSignal: ...   # exact == + field-level diff
```

Per-output-type comparators (normalized paths, set-equality for lists, semantic for free text via a
later model call) register against the runner. MVP ships exact + field-level only.

### 4.5 `ComparisonStore` + `ComparisonExport` — Postgres jsonb is the data SoT

```python
class ComparisonStore:
    def __init__(self, pg_url: str) -> None: ...
    def init_schema(self) -> None: ...                 # table + GIN index on record jsonb
    def save(self, record: ComparisonRecord) -> int: ...
    def query(self, **filters) -> Iterable[ComparisonRecord]: ...   # by profile/schema version, agreement, model

class ComparisonExport:
    def to_sft_jsonl(self, path, *, require_reference_valid=True, only_agreement: bool | None=None): ...
    def eval_view(self, *, by="primary_model") -> "EvalSummary": ...  # validity %, agreement %, regression vs prior model
```

Table: promoted columns (`run_id`, `*_workflow_id`, `profile_version`, `schema_version`,
`agreement_exact`) for cheap filters + the full `record jsonb` (GIN) for everything else. Export
reads the store, never DBOS.

### 4.6 Reference path — frontier structured output + cost

```python
class ReferenceCaps(BaseModel):
    structured_outputs: Literal["json_schema", "json_object", "none"] = "json_schema"
    # fallback ladder applied when building the reference agent:
    #   json_schema  -> NativeOutput (OpenAI strict)            [MVP target]
    #   json_object  -> PromptedOutput/json mode + validate + retry
    #   none/text    -> text + Pydantic validate + retry (StepConfig.max_attempts)
```

The reference `Backend` is the existing one with `caps(xgrammar=False, lora=False)`; `ReferenceCaps`
selects the PydanticAI output strategy. **Cost** = `usage × price_table[model_id]` (a small
`PriceTable`, ours). OpenAI strict json_schema is the MVP path; DeepSeek/OpenRouter fallbacks are an
extension point (Phase 3), pending the live Gate-4 verification.

---

## 5. End-to-end usage sketch

```python
from structured_agents_v2 import Backend, BackendCaps, AgentProfile
from structured_agents_v2.dual_path import DualPathRuntime, DualPathConfig

primary_be   = Backend(base_url="http://vllm:8000/v1", default_model="base", caps=BackendCaps())
reference_be = Backend(base_url="https://api.openai.com/v1", api_key=KEY,
                       default_model="gpt-4o-mini", caps=BackendCaps(xgrammar=False, lora=False))

primary_profile   = AgentProfile(name="cmd", adapter="cmd-adapter",
                                 instructions="Emit one command.", output_type_ref="app.schemas:Command")
reference_profile = AgentProfile(name="cmd", adapter=None,
                                 instructions="Emit one command.", output_type_ref="app.schemas:Command")

rt = DualPathRuntime(DualPathConfig(pg_url=PG, default_sample_rate=0.2))
cmd = rt.register("cmd", primary=primary_be.build(primary_profile),
                  reference=reference_be.build(reference_profile),
                  model_step_config=StepConfig(max_attempts=3, retries_allowed=True))
rt.launch()                                  # after all registrations

record = await cmd.run("Create a file notes.txt")   # gather, validate, diff, persist
serve(record.primary_output)                  # the user-facing answer is the local model's
# ... later, out of band:
ComparisonExport(store).to_sft_jsonl("teacher.jsonl", require_reference_valid=True)
rt.shutdown()
```

---

## 6. Package layout

```text
src/structured_agents_v2/dual_path/
├── __init__.py        # public exports; raises a clear error if `dbos` missing (extra not installed)
├── runtime.py         # DualPathRuntime, DualPathConfig (DBOS register→launch→shutdown)
├── runner.py          # DualPathRunner (gather, validate, diff, persist, sampling)
├── record.py          # ComparisonRecord, ModelIdentity, ComparisonSignal, versioning hashes
├── comparator.py      # Comparator protocol, ExactFieldComparator
├── store.py           # ComparisonStore (Postgres jsonb) + ComparisonExport + PriceTable
├── reference.py       # ReferenceCaps + fallback-ladder output-strategy selection
└── errors.py          # DualPathError, DualPathConfigError
```

`pyproject.toml`: `[project.optional-dependencies] dual-path = ["dbos>=0.26"]` (already added for the
spike; `psycopg` ships with `dbos`). Tests live in `tests/` and skip without `dbos` + Postgres
(pattern proven by `tests/test_dual_path_spike.py`).

---

## 7. Build phases

1. **Data core (no DBOS) — ✅ BUILT (2026-06-09).** `src/structured_agents_v2/dual_path/`:
   `errors.py`, `comparator.py` (`ComparisonSignal`/`Comparator`/`ExactFieldComparator`), `record.py`
   (`ModelIdentity`/`ComparisonRecord`/versioning hashes/`build_comparison_record`), `store.py`
   (`ComparisonStore` jsonb + `ComparisonExport` + `EvalSummary`/`GroupEval`), `__init__.py` (extra
   guard). Tests: `tests/test_dual_path_record.py` (pure) + `tests/test_dual_path_store.py`
   (Postgres-gated). 100% cov on the subpackage, ty+ruff clean. `[dual-path]` extra + a scoped
   `tool.ty.overrides` for the subpackage added to `pyproject.toml`.
2. **Runtime + runner (DBOS) — ✅ BUILT (2026-06-09).** `runtime.py` (`DualPathConfig`,
   `DualPathRuntime` register→launch→shutdown + context manager; json_schema-only + register-after-launch
   guards; owns the process-global DBOS singleton and the shared `ComparisonStore`), `runner.py`
   (`DualPathRunner.run`: sampling via `force_reference`/`sample_rate`, `SetWorkflowID`, top-level
   `asyncio.gather` over two `DBOSAgent` legs, per-leg error capture, `ModelIdentity` derived off the
   wire model, usage, `build_comparison_record` + `store.save`; `primary`/`reference` escape hatches).
   `__init__.py` exports them + the import guard now covers `dbos`. Tests:
   `tests/test_dual_path_runtime.py` (Postgres+DBOS-gated, one lifecycle on a dedicated loop, ASGI mock):
   guards, wire-shape survival, one-row dual persist + correlatable `*_workflow_id`, sampling skip/force,
   reference-error capture. runner 100% / runtime 90% cov, ty+ruff clean, full suite green. *(The spike,
   productionized and tested.)*
3. **Reference path hardening:** `reference.py` capability ladder; live OpenAI json_schema (Gate 4);
   resolve the strict-schema rewrite (core CONCEPT §10 #1) so frontier strict mode doesn't 400;
   DeepSeek/OpenRouter fallbacks.
4. **Sampling, cost, export polish:** `sample_rate` wiring, `PriceTable`, SFT/preference exporters,
   `eval_view` (validity %, agreement %, regression vs previous `primary_model`).
5. **Later:** fire-and-forget/async reference via a DBOS queue (decouple primary latency);
   `services.postgres` in `devenv.nix` for one-command infra; Logfire spans alongside DBOS.

This track depends only on Phase 2; it runs **parallel to Fleet (Phase 3)** and does not block it.

---

## 8. MVP acceptance criteria

1. `DualPathRuntime` registers two agents and launches; registering a non-`json_schema` agent raises
   `DualPathError`; registering after `launch()` raises.
2. `DualPathRunner.run()` returns a `ComparisonRecord` with a validated `primary_output`; the wire
   shape through both `DBOSAgent`s matches the Phase-2 contract (re-asserted via capture).
3. A dual run persists one `jsonb` row; nested fields are queryable (`record->…`); a GIN index exists.
4. `ExactFieldComparator` yields `agreement_exact` + `field_diff`; a divergent field is captured.
5. Sampling: with `sample_rate<1`, some runs are primary-only (`reference_skipped=True`) and still
   persist a record.
6. DBOS persists one step per leg; `*_workflow_id` on the record correlates to the DBOS workflow.
7. `ComparisonExport.to_sft_jsonl` emits teacher rows gated on `reference_valid`; `eval_view` reports
   validity/agreement by `primary_model`.
8. A live OpenAI reference leg returns a schema-valid output (Gate 4) and records usage + cost.
9. The lean core test suite passes with `dbos` **not** installed (dual-path tests skip).

---

## 9. Resolved decisions & remaining open questions

**Resolved (spike + brainstorm)**
- Architecture **C**; two stores (DBOS execution / jsonb data); separate `[dual-path]` extra;
  json_schema-only reference; Postgres+jsonb; register-before-launch lifecycle. (All in §1 table.)

**Still open**
1. **Adapter identity/reproducibility.** `ModelIdentity.adapter_rev` needs an **adapter registry/
   manifest** (adapters are produced out-of-band today). Until then, `wire_model` + `vllm_tag` are
   best-effort. Prereq for fully reproducible SFT provenance.
2. **Strict-schema rewrite** (core CONCEPT §10 #1) — confirmed *pulled forward* by the reference
   path; resolve in Phase 3 (does real OpenAI strict 400 on our `NativeOutput` schema?).
3. **Sync vs fire-and-forget.** MVP is synchronous gather. If dual-path ever feeds serving, add an
   async/queued reference leg (Phase 5) so primary latency is unaffected.
4. **Comparator depth.** Exact + field-level for MVP; normalized/semantic per-type comparators (and
   whether semantic agreement justifies an extra model call) deferred.
5. **DeepSeek/OpenRouter** structured-output surface — verify live before relying on them (Phase 3).
6. **Retention/PII.** Records hold prompts + outputs; a retention/scrubbing policy is a deployment
   concern before this runs on real traffic.
```
