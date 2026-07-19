# Final Refactor Guide: `structured-agents-v2`

**Baseline:** `v0.2.0` (`70d97fa`)  
**Scope:** every remaining finding in `.scratch/projects/09-structured-agents-v2-code-review/CODE_REVIEW.md` (CR-01 through CR-15).  
**Outcome:** a fail-closed core library and a dual-path subsystem whose comparison records are schema-safe, durable, idempotent, asynchronous, and exercised against the PostgreSQL instance managed by `devenv`.

This is an implementation guide, not a request to preserve current APIs at all costs. Security and record-integrity fixes take precedence. Make the changes in the phase order below; each phase has clear acceptance tests. Do not reimplement the v0.1 defects already fixed in `v0.2.0`.

## 1. Guardrails and target architecture

### Invariants to preserve

- The core install remains free of DBOS and PostgreSQL dependencies. Database code stays behind the `dual-path` extra.
- Model generation never executes a side effect. `Executor` remains the explicit authority boundary.
- The closed backend remains intentionally narrow: loopback-only, one structured request path, no raw SDK/client escape hatch.
- A comparison record is only a valid SFT/eval datum when its primary and reference outputs satisfy the *same canonical output contract*.
- `devenv.nix` owns all local PostgreSQL lifecycle and test connection configuration. No test may use a developer name, a hard-coded port, or an independently started database as its default.

### End state

```text
AgentProfile / Backend config ──strict parsing──> built StructuredAgent[OutputT]
                                                     │
                               DualPathRuntime.register() verifies canonical schema equality
                                                     │
                                 DBOS leg workflows ─┴─> durable finalization workflow/step
                                                               │
                                      async pooled ComparisonStore (upsert by full run UUID)
                                                               │
                                             PostgreSQL supplied only by devenv.nix
```

`ComparisonRecord` becomes an immutable materialized artifact. A retry for the same run ID must return the same stored record rather than create another row or re-run completed model legs.

## 2. First change: make `devenv.nix` the database/DBOS source of truth

The current file comments out PostgreSQL, while runtime and store tests default to `postgresql://andrew@127.0.0.1:5433/dual_path`. Replace that arrangement before changing dual-path code.

### 2.1 Enable a project-owned PostgreSQL service

In `devenv.nix`, enable the project PostgreSQL service, declare a dedicated database and role, and export one URL for both DBOS and the comparison store. Keep the exact PostgreSQL package/options consistent with the installed devenv version; the intended structure is:

```nix
services.postgres = {
  enable = true;
  initialDatabases = [{
    name = "structured_agents";
  }];
  initialScript = ''
    create role structured_agents login;
    grant all privileges on database structured_agents to structured_agents;
  '';
};

# The service module supplies PGHOST/PGPORT for its managed instance. Construct the URL
# in enterShell so it is never committed with a machine-specific host, user, or port.
enterShell = ''
  export DUAL_PATH_PG_URL="postgresql://structured_agents@$PGHOST:$PGPORT/structured_agents"
  export DUAL_PATH_TEST_PG_URL="$DUAL_PATH_PG_URL"
'';
```

If the enabled devenv PostgreSQL module uses a Unix socket rather than `PGHOST`/`PGPORT`, construct the URL with that socket path, for example `postgresql:///structured_agents?host=$PGHOST`. Do **not** fall back to `127.0.0.1:5433`, an `andrew` role, `localhost`, or a global OS service. Verify the resulting URL with a `psql "$DUAL_PATH_PG_URL" -c 'select 1'` script.

Add scripts to make the supported workflows obvious:

```nix
scripts.db-check.exec = ''
  psql "$DUAL_PATH_PG_URL" -v ON_ERROR_STOP=1 -c 'select current_database(), current_user'
'';

scripts.test-core.exec = ''
  uv run --extra dev pytest -m 'not live and not dual_path'
'';

scripts.test-dual-path.exec = ''
  uv run --extra dev --extra dual-path pytest -m dual_path
'';
```

Use the project service for DBOS as well: `DualPathConfig.pg_url` must receive `DUAL_PATH_PG_URL`; no separate DBOS URL is introduced. Do not put a password in Git. For CI, provide the same variable from the CI database service secret/configuration, but retain the same URL variable name and test commands.

### 2.2 Test-fixture refactor

- Add a `dual_path` pytest marker in `pyproject.toml`.
- Replace every module-level `PG_URL` default and socket probe in `test_dual_path_runtime.py` and `test_dual_path_store.py` with a session fixture that reads `DUAL_PATH_TEST_PG_URL` and fails clearly if absent.
- Do not skip database tests just because the service is unavailable. Local developers enter `devenv`; CI provisions its service. Skips are reserved for explicitly opt-in live inference tests only.
- Use a per-test schema or a transaction/savepoint fixture. Avoid a shared `truncate comparison_records` table because it prevents parallel test execution and can collide with a running local application.
- DBOS is process-global. Keep DBOS tests in a serial pytest group or one worker, give each registration a UUID-suffixed test name, and always destroy it in `finally`.

Acceptance command:

```bash
devenv shell -- test-dual-path
```

must run, not skip, the database/DBOS test suite against the devenv service.

## 3. Phase 1 — strict configuration and credential safety

### 3.1 Create one strict configuration base (CR-01)

Add `src/structured_agents_v2/config.py`:

```python
from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    """Configuration sent across an authority or persistence boundary."""

    model_config = ConfigDict(extra="forbid")
```

Make the following declarative/configuration classes inherit from it:

- `BackendCaps`, `Backend`
- `AgentProfile`
- `DecoderSpec`
- `RoutingTable`
- `DualPathConfig`
- persisted record/config models where rejecting unknown database JSON is desired (`ModelIdentity`, `ComparisonRecord`, comparator config if added)

Do not blindly apply this rule to result/value models intended to mirror provider payloads. For arbitrary provider settings, keep a deliberately named field such as `provider_extra: dict[str, JsonValue] = Field(default_factory=dict)` instead of accepting unknown top-level keys.

Also replace mutable model defaults (`Backend.caps = BackendCaps()` and `AgentProfile.model_settings = {}`) with `Field(default_factory=...)` even though Pydantic copies them. It makes ownership explicit.

Tests:

- A misspelled `BackendCaps(xgramar=False)` raises `ValidationError`.
- Misspelled profile, decoder, route, and dual-path fields each raise.
- Intended nested `model_settings`/`provider_extra` data remains accepted only through its explicit extension field.

### 3.2 Keep API keys out of ordinary representation and dumps (CR-02)

In `Backend`, make `api_key` a `SecretStr` field with `repr=False` and exclude it by default from serialization. Prefer a private attribute if serializing/reconstructing `Backend` is not an intentional public feature. At provider construction, pass only `self.api_key.get_secret_value()`.

The required observable behavior is:

```python
backend = Backend(..., api_key="super-secret")
assert "super-secret" not in repr(backend)
assert "super-secret" not in str(backend)
assert "super-secret" not in backend.model_dump_json()
assert "api_key" not in backend.model_dump()
```

If callers need to persist a backend descriptor, offer an explicit redacted `public_config()` method; do not ask them to remember `exclude={"api_key"}`. Audit error messages and test fixtures so they never interpolate the actual key.

## 4. Phase 2 — make the core API transactional and typed

### 4.1 Atomic fleet rebuild (CR-08)

Refactor `AgentSet.build()` so it never mutates `self.agents` or `self.routing` until all work succeeds:

1. Validate duplicate profile names.
2. Build into `candidate_agents`.
3. Validate a proposed routing table against `candidate_agents`. Change `_validate_routing` and `_check_route_coverage` to accept the candidate mapping rather than reading `self.agents`.
4. Assign both instance attributes together at the end.
5. If candidate construction creates resources that must be closed, close those candidates on failure without closing the old backend client.

Test a fleet with working old agents/routing, call `build()` with a new invalid route, then prove both old agents and old routing still work.

### 4.2 Reject duplicate policies (CR-09)

Replace the policy dictionary comprehension in `BaseExecutor.__init__` with a one-pass registry builder that raises `PolicyError` on the second occurrence of a name. Test that neither first nor last policy is silently selected.

### 4.3 Carry output types through the public API (CR-12)

Make the wrapper generic:

```python
OutputT = TypeVar("OutputT")

class StructuredAgent(Generic[OutputT]):
    async def run(self, prompt: str, **kwargs: Any) -> AgentResult[OutputT]: ...
```

Thread that type through `Backend.build`, `AgentSet.__getitem__`, `BatchResult`, and `RoutedResult` where the type can genuinely be preserved. A heterogeneous `AgentSet` cannot promise one output type for arbitrary string lookups; use overloads for literal keys or retain `StructuredAgent[Any]` only at that dynamic boundary. Do not claim typing precision that runtime routing cannot provide.

Add a small static type fixture checked by `ty` proving an annotated `StructuredAgent[Plan]` returns `AgentResult[Plan]` and `.output` is `Plan`.

### 4.4 Treat `output_type_ref` as code-equivalent configuration (CR-13)

Document the trust rule in `AgentProfile` and README: profiles containing import references may only come from reviewed code unless an allowlist is supplied. Add an explicit resolver policy, for example `allowed_module_prefixes: tuple[str, ...] | None` supplied by the trusted application/bootstrap code, and reject modules outside it before importing. The library default may remain unrestricted only for in-process programmatic profiles; loading profiles from files, databases, or network sources must require an allowlist.

Test malformed paths, missing attributes, non-model targets, and an otherwise valid module blocked by the allowlist.

## 5. Phase 3 — establish the dual-path contract before running anything (CR-03)

The current runtime merely checks both legs use `json_schema`; that is insufficient. A Pydantic model from the reference leg can be structurally unrelated and is currently treated as valid.

### 5.1 Canonical contract identity

Add a small immutable `OutputContract` value, created during registration:

```python
@dataclass(frozen=True)
class OutputContract:
    output_type: type[BaseModel]
    canonical_schema: dict[str, Any]
    schema_hash: str
    strict: bool
```

Canonicalize the JSON schema before hashing: serialize with sorted keys and compact separators, and remove only fields that are explicitly non-semantic under the documented contract. Do not remove validation constraints, required fields, enum values, descriptions used by model behavior, or decoder strictness. A simple full sorted-schema SHA-256 is the safe initial implementation.

In `DualPathRuntime.register()`:

1. Resolve both profiles once.
2. Require each to be `json_schema` and to have a non-`None` model output type.
3. Build an `OutputContract` for each profile.
4. Require equal schema hashes and equal `strict` settings; otherwise raise `DualPathConfigError` identifying both type refs and hashes.
5. Store the primary contract in the runner and record both profile/schema identities if the API later supports intentional cross-contract comparisons.

Choose **schema equivalence**, not Python object identity, as the initial rule. It permits separately imported but equivalent model classes while protecting the wire/data contract. Document this decision.

### 5.2 Validate returned legs against that exact contract

Replace `_unpack(raw)` with `_unpack(raw, expected_type)`.

- Exceptions become a bounded, non-secret error category/message.
- A non-model output is invalid.
- A `BaseModel` of any other class must be revalidated with `expected_type.model_validate(output.model_dump(mode="json"))`.
- A validation error makes that leg invalid and records an error; it never becomes a teacher target.
- `build_comparison_record()` accepts expected output type or already-normalized expected instances and determines `*_valid` from that, not generic `isinstance(BaseModel)`.

Tests must register two different output schemas and fail immediately; simulate a reference result with a different Pydantic model and verify `reference_valid=False`, no SFT export, and a recorded validation error.

## 6. Phase 4 — durable, idempotent, nonblocking comparison persistence (CR-04, CR-05, CR-06)

### 6.1 Data model and migration

Use a full UUID string for `run_id`; allow an optional caller-supplied idempotency key and validate it as a UUID/opaque bounded identifier. Never truncate a UUID.

Replace the ad-hoc DDL string with versioned migrations owned by the application (for example a small `dual_path/migrations/` runner). The first migration must:

```sql
alter table comparison_records
  add constraint comparison_records_run_id_key unique (run_id);

create table if not exists comparison_runs (
  run_id text primary key,
  runner_name text not null,
  prompt jsonb not null,
  status text not null check (status in ('pending', 'finalized', 'failed')),
  primary_workflow_id text not null,
  reference_workflow_id text,
  created_at timestamptz not null default now(),
  finalized_at timestamptz
);
```

For an existing database, first deduplicate old rows deterministically (retain the earliest or most complete row), then add the unique constraint. Make this migration transactional and test the upgrade path.

Avoid storing raw prompts in an unbounded operational table if prompts can contain sensitive data. Apply the project retention/access policy; at minimum, document that dual-path capture is sensitive telemetry.

### 6.2 Async pooled store

Replace synchronous `psycopg.connect()` calls with an async connection pool (`psycopg_pool.AsyncConnectionPool`, added explicitly to the `dual-path` extra). Expose:

```python
class ComparisonStore:
    async def open(self) -> None: ...
    async def aclose(self) -> None: ...
    async def init_schema(self) -> None: ...
    async def save_idempotent(self, record: ComparisonRecord) -> ComparisonRecord: ...
    async def query(...) -> list[ComparisonRecord]: ...
```

Use parameterized `INSERT ... ON CONFLICT (run_id) DO ... RETURNING record`. The conflict branch must be deterministic: if the stored payload is equal, return it; if it differs, raise an idempotency-conflict error rather than overwriting evidence. `ComparisonExport` becomes async or is given an explicit synchronous boundary that calls it outside application event loops; do not hide a blocking connection in an async method.

As a temporary, narrowly scoped bridge only, `asyncio.to_thread` around a synchronous store is acceptable. It is not the final state because it still creates one connection per call and lacks backpressure/pooling.

Test with a deliberately slow database operation plus an independent coroutine; the coroutine must make progress before the save completes. Test concurrent saves for one run ID and assert exactly one stored record.

### 6.3 Put finalization in the durable workflow boundary

The present architecture protects only each model leg. Build an explicit DBOS durable finalization workflow/step that:

1. creates/loads the `comparison_runs` pending row using the supplied run ID;
2. invokes or joins the fixed-ID primary/reference leg workflows;
3. normalizes both outcomes against the stored `OutputContract`;
4. assembles the record;
5. performs the idempotent database upsert;
6. marks the run finalized in the same database transaction as the persisted record.

Use DBOS workflow IDs derived from the full run ID (`comparison:{run_id}`, `primary:{run_id}`, `reference:{run_id}`), not random abbreviated suffixes. Pass deterministic, serializable inputs to durable steps. Follow the DBOS version installed by the `dual-path` extra for the exact decorator/step API; do not call ordinary synchronous Psycopg work directly from an async DBOS workflow.

The recovery behavior must be explicit:

- Crash after a model leg succeeds but before finalization: resuming `comparison:{run_id}` reads the existing DBOS leg state and produces one record.
- Crash after insert but before workflow acknowledgement: retry returns the existing row via the unique key.
- Primary failure: record it regardless of reference sampling; a failed primary is not raised as a sampling-dependent public exception.
- Reference skipped: create a normal finalized record with `reference_skipped=True`.

If DBOS cannot safely serialize a value required for record assembly, make the model legs return a minimal serializable envelope and reconstruct/validate it in finalization. Do not weaken validation just to serialize an SDK result.

### 6.4 Consistent failure handling

Always collect the primary leg through the same `return_exceptions=True` normalization path, even when reference sampling is off. The runner may optionally re-raise *after* successful durable persistence through an explicit `raise_primary_error` option, but the default return path must have one consistent record-completeness contract.

Add failure-injection tests at: before either leg, after primary, after reference, before record insert, after record insert, and during retry. Every case must leave either a recoverable pending run or exactly one finalized record.

## 7. Phase 5 — dual-path lifecycle and query validation (CR-10, CR-11)

### 7.1 Runtime state machine

Use a private enum, e.g. `NEW`, `REGISTERING`, `LAUNCHED`, `CLOSED`, rather than a single `_launched` boolean.

- `register()` only works before launch.
- Validate `default_sample_rate` and override `sample_rate` using a finite `0 <= value <= 1` field validator. Reject NaN and infinity.
- Initialize/open the comparison store and run migrations before `DBOS.launch()` where possible. If DBOS must launch first, wrap later setup in `try/except`, call `DBOS.destroy()` on failure, close the pool, and transition to `CLOSED`.
- `shutdown()` is idempotent. It closes the store pool and destroys DBOS exactly once, including partially initialized instances.
- Do not construct multiple DBOS runtimes in one process without a documented ownership policy.

Tests: invalid rates, failed schema initialization, launch twice, shutdown twice, registration after launch, and cleanup after a partial launch failure.

### 7.2 Export/query API validation

Replace `ComparisonExport.eval_view(by: str)` with `by: Literal["primary_model", "profile_version"]`. Validate `ComparisonStore.query(limit=...)` as a bounded positive integer (for example `1..10_000`) before the query. Add tests for typos, zero, negative, and excessive limits.

## 8. Phase 6 — finish the closed boundary (CR-07)

In `ClosedBackend.__init__`, verify that `output_type` is a concrete `BaseModel` subclass. Build/cache its schema there after checking it is JSON serializable and bounded enough for the intended service.

In `run()`, wrap the HTTP request and response parsing in the same normalization boundary:

```python
try:
    response = await self._http_client.post(...)
    if response.status_code != 200:
        raise ClosedBackendError()
    # parse and validate
except ClosedBackendError:
    raise
except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
    raise ClosedBackendError() from None
```

Use `raise ... from None` to avoid exposing endpoint, TLS, connection, provider body, or schema details through exception chaining. Bound response consumption before JSON parsing (for example check `Content-Length` when present and use an explicit maximum body size); preserve the existing no-redirect and no-proxy policy.

Tests must cover connect failure, timeout, malformed JSON, malformed response shape, non-200 response, invalid output JSON/schema, oversized response, invalid output type, and confirm all transport failures expose only `ClosedBackendError`.

## 9. Phase 7 — packaging, tests, CI, and release hygiene (CR-14, CR-15)

### 9.1 Deterministic source distribution

The local worktree already has a deletion for `result` and an ignore rule. Commit that deletion after verifying it is the tracked absolute Nix symlink. Do not replace it with a relative or generated symlink.

Make Hatch sdist inputs rooted and allowlisted rather than broad directory patterns. Include only the project files deliberately shipped, e.g. `/src/structured_agents_v2/**`, `/tests/**`, `/README.md`, `/LICENSE`, `/pyproject.toml`, and any required migration package data. Explicitly exclude `/.scratch/**`, `/artifacts/**`, `/deploy/**`, `/result`, `/result-*`, environments, and generated coverage/build directories.

Add a packaging test that builds wheel and sdist in a temporary directory, lists their manifests, and compares them to an allowlist. It must assert no absolute symlink is followed and no deployment/scratch material appears. Run it in CI.

### 9.2 Test tiers

Define and document these commands:

| Tier | Command | Expected behavior |
| --- | --- | --- |
| Core | `devenv shell -- test-core` | Mandatory, hermetic; no Postgres or live model |
| Dual path | `devenv shell -- test-dual-path` | Mandatory, DBOS + devenv PostgreSQL; no live model |
| Quality | `devenv shell -- uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check src` | Mandatory |
| Packaging | `devenv shell -- uv build` plus manifest test | Mandatory |
| Live | `SAV_LIVE=1 devenv shell -- uv run --extra dev pytest -m live` | Explicitly opt-in, never required for normal CI |

Configure pytest to fail on unknown markers and unexpected skips in mandatory tiers. Do not use broad test skips to accommodate missing optional dependencies: the command selecting the dual-path tier installs its required extra.

### 9.3 CI

Add a top-level GitHub Actions workflow with separate core, dual-path/PostgreSQL, quality, and packaging jobs. The dual-path job must use the same `DUAL_PATH_TEST_PG_URL` variable contract as `devenv.nix`; it may use a CI PostgreSQL service because devenv processes are not the CI service manager, but it must not introduce different credentials, schemas, or fixtures in Python. Cache only safe dependency artifacts, not databases or secrets.

## 10. Suggested commit sequence and release gate

Keep commits reviewable:

1. `chore: enable devenv postgres and hermetic dual-path fixtures`
2. `fix: forbid unknown configuration and redact backend credentials`
3. `fix: make fleet and executor configuration transactional`
4. `feat: preserve structured-agent output typing`
5. `fix: enforce dual-path output-contract equivalence`
6. `feat: make comparison finalization durable and idempotent`
7. `fix: use async pooled dual-path store and validate runtime inputs`
8. `fix: normalize closed-backend transport failures`
9. `chore: lock down package manifest and add CI`

Before release, require all of the following:

- Unknown capability/configuration fields fail validation and no backend dump/repr leaks a secret.
- Re-registering mismatched primary/reference schemas fails before DBOS launch.
- Replaying one full run ID yields exactly one comparison row with a stable payload.
- Failure-injection tests prove a recoverable pending state or exactly one finalized record at every interruption point.
- Database I/O does not block the event loop and is served from a managed async pool.
- `devenv shell -- test-dual-path` has no infrastructure skips.
- Closed-backend transport failures reveal no HTTPX transport detail.
- A clean wheel/sdist manifest test passes, and no `result` symlink is tracked.
- Core, dual-path, quality, and packaging CI jobs pass; only explicitly selected live tests may skip.

At that point it is reasonable to describe dual-path output as a durable SFT/evaluation source. Until then, treat it as experimental telemetry rather than trusted training data.
