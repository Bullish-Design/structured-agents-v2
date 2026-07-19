# structured-agents-v2 — Deep Code Review

**Date:** 2026-07-17  
**Working checkout reviewed:** `7b4a053a9613770c1d93163255fdc637226f48ce` (`main`)  
**Upstream inspected:** `70d97fa0217575135b1e5ff50a0ff803b12956fd` (`origin/main`, tag `v0.2.0`)  
**Repository state at review time:** local `main` was 13 commits behind `origin/main`; the worktree was clean.

## 1. Executive summary

The library has a strong architectural core: constrained decoding, Pydantic validation,
routing, and side-effect authorization are separated into understandable layers. The wire
contracts are grounded in captured OpenAI-compatible request shapes, and the test suite makes
good use of an in-process ASGI server.

The checked-out `v0.1.0` code is not release-ready, however. It contains known runtime,
authority-boundary, request-capture, resource-lifecycle, test-infrastructure, and packaging
defects. Most of those core defects have already been fixed by the 13 commits culminating in
upstream `v0.2.0`.

The upstream hardening release does **not** resolve the deepest remaining risks:

1. Pydantic configuration objects silently ignore unknown fields, allowing security-relevant
   capability typos to fail open.
2. Backend API keys are exposed by ordinary Pydantic representation and serialization.
3. Dual-path registration does not prove that primary and reference agents share the same
   output contract, allowing incompatible teacher data to be recorded under the primary schema.
4. The comparison artifact is persisted outside DBOS, without idempotency or a unique run ID,
   so the system's central record is not actually durable.
5. The async dual-path runner performs synchronous connection-per-record PostgreSQL I/O on the
   event loop.
6. `ClosedBackend` promises detail-free transport failures but lets raw `httpx` exceptions escape.
7. Source distributions contain unrelated deployment material and host-specific files reached
   through a tracked absolute NixOS `result` symlink.
8. PostgreSQL-backed behavior remains non-hermetic and largely skipped in the normal suite.

### Overall assessment

- **Architecture:** good and coherent.
- **Core v0.1 implementation:** unsafe to release as checked out.
- **Core v0.2 implementation:** substantially improved.
- **Dual-path subsystem:** experimental; not yet production-grade or trustworthy as an SFT/eval
  source of truth.
- **Closed backend:** admirably narrow, but its error boundary is incomplete.
- **Packaging and release hygiene:** improved upstream, but the sdist remains polluted and
  host-dependent.

## 2. Scope and method

The review covered:

- Every module in [`src/structured_agents_v2`](../../../src/structured_agents_v2/).
- Every module in [`src/structured_agents_v2/dual_path`](../../../src/structured_agents_v2/dual_path/).
- The closed HTTP path in [`closed.py`](../../../src/structured_agents_v2/closed.py).
- All tests in [`tests`](../../../tests/).
- Packaging and tool configuration in [`pyproject.toml`](../../../pyproject.toml),
  [`uv.lock`](../../../uv.lock), and [`README.md`](../../../README.md).
- Git history and the 13-commit delta from the checkout to upstream `v0.2.0`.
- The downstream PydanticAI compatibility report in
  [`06-pydanticai-usage-compatibility/ISSUE.md`](../06-pydanticai-usage-compatibility/ISSUE.md).
- Built wheel and sdist contents.
- Focused runtime probes for concurrent capture attribution and closed-client connection errors.

Generated inference artifacts were treated as operational evidence rather than primary library
source. Deployment tooling was inspected for packaging and repository-hygiene effects, but it was
not subjected to the same line-by-line review as the Python library.

## 3. Architecture overview

The library is organized into four layers:

1. **Constraint declaration**
   - [`DecoderSpec`](../../../src/structured_agents_v2/decoder.py) maps `json_schema`, `grammar`,
     `regex`, and `choice` modes to PydanticAI output types and wire extensions.
   - [`ConstrainedOutput`](../../../src/structured_agents_v2/constrained.py) lets an output model
     carry its own decoding contract.
   - [`AgentProfile`](../../../src/structured_agents_v2/profile.py) serializes the binding between
     instructions, output type, adapter, decoder, policy, and model settings.

2. **Model execution**
   - [`Backend`](../../../src/structured_agents_v2/backend.py) builds the PydanticAI/OpenAI model,
     capability-gates it, and applies the resolved decoder.
   - [`StructuredAgent`](../../../src/structured_agents_v2/agent.py) wraps runs and exposes the
     validated output, usage, capture data, and underlying PydanticAI result.

3. **Composition and authority**
   - [`AgentSet`](../../../src/structured_agents_v2/fleet.py) provides concurrent batches and
     router-to-specialist dispatch.
   - [`Executor`](../../../src/structured_agents_v2/executor.py) separates validated intent from
     authorization and side effects.

4. **Optional operational paths**
   - [`dual_path`](../../../src/structured_agents_v2/dual_path/) runs local and reference agents,
     compares outputs, and persists training/evaluation records through DBOS and PostgreSQL.
   - [`ClosedBackend`](../../../src/structured_agents_v2/closed.py) provides a deliberately narrow,
     loopback-only, single-request JSON-schema path without PydanticAI escape hatches.

This dependency direction is one of the library's best properties. The most serious defects do
not come from confused layering; they come from insufficient validation and lifecycle guarantees
at the boundaries between those layers.

## 4. Findings still open in upstream v0.2.0

The following findings are present in the checked-out code and remained present when the
corresponding upstream `v0.2.0` files were inspected.

### CR-01 — Unknown configuration fields silently fail open

**Severity:** High  
**Area:** Configuration / capability enforcement  
**Files:** [`backend.py`](../../../src/structured_agents_v2/backend.py),
[`profile.py`](../../../src/structured_agents_v2/profile.py),
[`decoder.py`](../../../src/structured_agents_v2/decoder.py),
[`fleet.py`](../../../src/structured_agents_v2/fleet.py),
[`dual_path/runtime.py`](../../../src/structured_agents_v2/dual_path/runtime.py)

All declarative models inherit Pydantic's default `extra="ignore"` behavior. Unknown keys are
accepted and discarded instead of rejected.

An empirical probe demonstrated the security-relevant case:

```python
BackendCaps(xgramar=False)
# BackendCaps(xgrammar=True, lora=True, server_default_backend=True)
```

The caller intended to disable XGrammar, but a typo silently left it enabled. The same pattern can
hide misspelled routing defaults, decoder options, policies, sample rates, and backend fields.

This is especially dangerous because `BackendCaps` is used as a build-time authority statement:
the library trusts it when deciding whether a profile may be constructed.

**Recommendation:**

- Add `model_config = ConfigDict(extra="forbid")` to every configuration model.
- Use a shared strict configuration base class to keep the policy consistent.
- Add negative tests for misspelled capability, route, decoder, and dual-path fields.
- Retain an explicit extension map only where arbitrary provider settings are truly intended.

### CR-02 — Backend credentials leak through representation and serialization

**Severity:** High  
**Area:** Secrets / observability  
**File:** [`backend.py`](../../../src/structured_agents_v2/backend.py)

`Backend.api_key` is a normal `str` Pydantic field. A probe showed both common representations leak
the full value:

```python
backend.model_dump()
# {'base_url': 'http://x', 'api_key': 'super-secret', ...}

repr(backend)
# Backend(base_url='http://x', api_key='super-secret', ...)
```

Normal debugging, structured logging, exception context, or configuration snapshots can therefore
expose provider credentials.

**Recommendation:**

- Store the key as `SecretStr` or an excluded private attribute.
- Set `repr=False` and exclude it from default serialization.
- Pass only `get_secret_value()` to the provider at construction time.
- Add tests proving the raw key is absent from `repr`, `str`, `model_dump`, and JSON output.

### CR-03 — Dual-path agents are not required to share an output contract

**Severity:** High  
**Area:** Training-data integrity  
**Files:** [`dual_path/runtime.py`](../../../src/structured_agents_v2/dual_path/runtime.py),
[`dual_path/runner.py`](../../../src/structured_agents_v2/dual_path/runner.py),
[`dual_path/record.py`](../../../src/structured_agents_v2/dual_path/record.py)

`DualPathRuntime.register()` checks only that each leg resolves to `json_schema`. It does not compare:

- Resolved output types.
- Schema hashes.
- Instructions.
- Relevant model settings.
- Strictness or other decoder details.

The runner then resolves and stores only the **primary** output type. `_unpack()` treats any
`BaseModel` as a valid output, and `build_comparison_record()` also defines validity as
`isinstance(output, BaseModel)` rather than `isinstance(output, expected_output_type)`.

Consequently, a reference agent can validate against a different model and still have its output:

- Compared against the primary output.
- Marked `reference_valid=True`.
- Stored under the primary schema version.
- Exported as the teacher target for the primary contract.

This is silent SFT/evaluation data corruption.

**Recommendation:**

- Resolve both profiles at registration.
- Require equal canonical JSON-schema hashes at minimum.
- Decide and document whether identical Python types are required or schema equivalence is enough.
- Validate each returned object against the expected primary type before marking it valid.
- Record both profile/schema identities if intentional cross-contract comparisons are supported.
- Add tests registering two different `json_schema` models and assert a `DualPathConfigError`.

### CR-04 — The central comparison record is not durably or idempotently persisted

**Severity:** High  
**Area:** Durability / data integrity  
**Files:** [`dual_path/runner.py`](../../../src/structured_agents_v2/dual_path/runner.py),
[`dual_path/store.py`](../../../src/structured_agents_v2/dual_path/store.py)

Each model leg is wrapped as a DBOS workflow, but comparison assembly and `store.save(record)` occur
after those workflows, outside a durable workflow or step. A crash after the model calls finish but
before the insert completes loses the comparison record permanently.

Other idempotency weaknesses compound this:

- `run_id` uses only the first 12 hex characters of a UUID.
- The table has no unique constraint on `run_id`.
- Inserts are unconditional, so a retry can create duplicate training rows.
- There is no durable status linking completed leg workflows to a pending comparison insert.

DBOS therefore protects the expensive calls but not the artifact the subsystem exists to create.

**Recommendation:**

- Make comparison assembly and persistence a durable, retryable step.
- Use a full UUID or caller-provided idempotency key.
- Add a unique constraint for `run_id` and use `INSERT ... ON CONFLICT`.
- Store enough state to recover comparisons whose legs completed before a process crash.
- Add failure-injection tests around the boundary between leg completion and record insertion.

### CR-05 — Synchronous PostgreSQL I/O blocks the async runner

**Severity:** High  
**Area:** Concurrency / performance  
**Files:** [`dual_path/runner.py`](../../../src/structured_agents_v2/dual_path/runner.py),
[`dual_path/store.py`](../../../src/structured_agents_v2/dual_path/store.py)

`DualPathRunner.run()` is async, but it calls the synchronous `ComparisonStore.save()` directly.
Every save opens a new synchronous Psycopg connection, inserts one row, commits, and closes it.

At capture volume this:

- Blocks the event loop during DNS/socket/authentication/query/commit work.
- Serializes unrelated async tasks behind database latency.
- Repeatedly pays connection setup cost.
- Creates avoidable pressure on PostgreSQL connection limits.

**Recommendation:**

- Prefer `psycopg.AsyncConnectionPool`/`psycopg_pool` and async queries.
- As an interim fix, move synchronous store operations to `asyncio.to_thread`.
- Reuse pooled connections for schema initialization, save, and query operations.
- Add a concurrency test proving a slow store does not stall unrelated tasks.

### CR-06 — Primary failure semantics depend on reference sampling

**Severity:** High  
**Area:** Error handling / record completeness  
**File:** [`dual_path/runner.py`](../../../src/structured_agents_v2/dual_path/runner.py)

When the reference runs, both legs are executed with `asyncio.gather(..., return_exceptions=True)`,
and failures become record data. When the reference is skipped, the primary leg is directly awaited.
A primary failure then escapes, no comparison record is written, and the documented "failing leg is
captured" behavior is violated.

The same primary error therefore has two public behaviors based solely on random sampling.

**Recommendation:** run the primary through the same exception-capturing path regardless of whether
the reference is sampled, then always build and persist a record when possible.

### CR-07 — `ClosedBackendError` does not cover transport failures

**Severity:** High  
**Area:** Privacy boundary / error normalization  
**File:** [`closed.py`](../../../src/structured_agents_v2/closed.py)

`ClosedBackendError` describes a deliberately detail-free transport or output-validation failure,
but the call to `AsyncClient.post()` occurs before the `try` block. Connection, timeout, protocol,
and TLS exceptions escape as detailed `httpx` exceptions.

A focused connection probe returned:

```text
httpx.ConnectError All connection attempts failed
```

That violates the module's narrow abstraction and can leak endpoint/transport details into a caller
that intentionally logs only fixed closed-backend errors.

**Recommendation:**

- Wrap the request itself in the normalization boundary.
- Catch `httpx.HTTPError` and raise a detail-free `ClosedBackendError`.
- Keep programmer/configuration errors distinct and fail during construction.
- Add tests for connect failure, timeout, malformed JSON, invalid schema output, and oversized bodies.

### CR-08 — Fleet rebuild is not atomic

**Severity:** Medium  
**Area:** State consistency  
**File:** [`fleet.py`](../../../src/structured_agents_v2/fleet.py)

`AgentSet.build()` assigns the newly built agent dictionary before validating the proposed routing
table. If routing validation raises, the fleet keeps the new agents while retaining its prior
routing state. That can produce a half-rebuilt fleet whose routing names no longer correspond to
the agent set.

**Recommendation:** build agents into a local dictionary, validate the proposed routing against
that candidate state, and assign both `agents` and `routing` only after every check passes.

### CR-09 — Duplicate policy names are silently overwritten

**Severity:** Medium  
**Area:** Authority configuration  
**File:** [`executor.py`](../../../src/structured_agents_v2/executor.py)

`BaseExecutor` constructs its registry with a dictionary comprehension. Repeated policy names are
silently resolved in favor of the last entry. In an authority registry this is dangerous: an
unexpected later policy can replace the reviewed allow rule and action without a configuration
error.

**Recommendation:** detect duplicates before constructing the registry and raise `PolicyError`.

### CR-10 — Dual-path sample rates and lifecycle transitions are insufficiently validated

**Severity:** Medium  
**Area:** Configuration / lifecycle  
**File:** [`dual_path/runtime.py`](../../../src/structured_agents_v2/dual_path/runtime.py)

`default_sample_rate` and per-runner `sample_rate` accept negative values, values greater than one,
NaN, and infinity. Their resulting behavior is accidental rather than explicit.

`launch()` also calls `DBOS.launch()` before initializing the comparison schema and marks the runtime
launched only after both operations succeed. If schema initialization fails, the process-global DBOS
singleton can remain active while `_launched` is false.

**Recommendation:**

- Use a finite bounded float constrained to `[0, 1]`.
- Define cleanup/rollback if any launch step fails.
- Make shutdown idempotent and track constructed/launched/destroyed states explicitly.
- Add tests for failed schema initialization and repeated shutdown.

### CR-11 — Export/query APIs silently accept invalid options

**Severity:** Medium  
**Area:** Evaluation correctness  
**File:** [`dual_path/store.py`](../../../src/structured_agents_v2/dual_path/store.py)

`ComparisonExport.eval_view(by=...)` treats every value other than `"primary_model"` as
`profile_version`. A typo silently changes the grouping dimension. `ComparisonStore.query(limit=...)`
also accepts negative or arbitrarily large values and delegates failures/resource use to PostgreSQL.

**Recommendation:** use a `Literal` or enum for grouping and validate a bounded positive query limit.

### CR-12 — The public typing story collapses to `Any`

**Severity:** Medium  
**Area:** Static API quality  
**File:** [`agent.py`](../../../src/structured_agents_v2/agent.py)

`AgentResult` is generic, but `StructuredAgent` is not, and both `run()` methods return
`AgentResult[Any]`. The package ships `py.typed`, yet downstream callers lose the output type at the
main wrapper boundary.

**Recommendation:** make `StructuredAgent` generic over its output type and preserve that type
through `Backend.build`, fleet lookup, routed results, and batch results where feasible.

### CR-13 — `output_type_ref` is executable configuration

**Severity:** Medium  
**Area:** Trust boundary  
**File:** [`profile.py`](../../../src/structured_agents_v2/profile.py)

Resolving a profile executes `importlib.import_module()` on the configured module path. This is
reasonable for trusted, in-tree Python configuration but unsafe if profiles are later loaded from
untrusted YAML, JSON, a database, or a remote registry.

**Recommendation:** document profiles as code-equivalent configuration and require an allowlist of
module prefixes before accepting externally sourced references.

### CR-14 — Source distributions include unrelated and host-specific content

**Severity:** Medium  
**Area:** Packaging / reproducibility  
**Files:** [`pyproject.toml`](../../../pyproject.toml), [`result`](../../../result)

The wheel was correct and small, but a real sdist build produced a 74 KB archive containing 70 files,
including:

- 25 files under `deploy/`, including vendored plugin tests.
- A benchmark artifact README.
- Four documentation files dereferenced through the tracked absolute `result` symlink into a local
  NixOS system closure.

The symlink points to a host-specific `/nix/store/...-nixos-system-server-...` path whose target is
approximately 3 GB. The current sdist happened to collect only four matching README files from it,
but build contents can vary by host and target availability.

The apparent cause is that Hatch's sdist include patterns are not rooted/restricted enough, while
the absolute `result` symlink is tracked in Git.

**Recommendation:**

- Remove `result` from version control and ignore it.
- Use rooted sdist patterns such as `/src/**`, `/tests/**`, `/README.md`, `/LICENSE`, and
  `/pyproject.toml`.
- Explicitly exclude `.scratch`, `artifacts`, `deploy`, `result`, and local environments.
- Add a CI packaging test that asserts an allowlisted archive manifest.

### CR-15 — Integration tests are non-hermetic and there is no top-level CI

**Severity:** Medium  
**Area:** Verification / regression prevention  
**Files:** [`test_dual_path_runtime.py`](../../../tests/test_dual_path_runtime.py),
[`test_dual_path_store.py`](../../../tests/test_dual_path_store.py),
[`pyproject.toml`](../../../pyproject.toml)

Dual-path integration tests require a pre-existing PostgreSQL server at `127.0.0.1:5433` and default
to a developer-specific `andrew` database user. Without that service the actual runtime and store
tests skip. No repository-level GitHub Actions or equivalent CI definition enforces the suite.

This allowed the new `ClosedBackend` async tests to land without their required pytest plugin.

**Recommendation:**

- Provision PostgreSQL as a test service/container with ephemeral credentials.
- Run core, dual-path, packaging, lint, and type-check jobs separately.
- Fail on unexpected skips or unknown pytest marks.
- Keep live model tests explicitly gated, but make all mock/database tests mandatory.

## 5. Defects in the checked-out v0.1 code already fixed upstream

These findings remain defects in the working checkout. Upstream `v0.2.0` contains explicit fixes,
which were inspected but not checked out into the working tree during this review.

### V1-01 — PydanticAI `usage` API incompatibility

**Severity:** High  
**Checkout behavior:** [`agent.py`](../../../src/structured_agents_v2/agent.py) calls `raw.usage()`.
The dependency allows every PydanticAI version greater than or equal to 1.87.0. A downstream
environment resolved a version where `usage` is a property, so a successful validated run raised
`TypeError: 'RunUsage' object is not callable` while constructing `AgentResult`.

The dual-path helper swallowed the analogous error and silently recorded no usage.

**Upstream status:** fixed by `97a9976`; PydanticAI is upgraded/bounded and usage is read as a
property with regression coverage.

### V1-02 — `AllowlistExecutor.execute()` bypasses the allowlist

**Severity:** High  
**Checkout behavior:** direct callers can invoke `execute()` and run the policy action without an
authorization check. Raising allow rules also crash rather than deny, synchronous actions block the
event loop, and `ExecResult.ok` is never false.

**Upstream status:** fixed by `f042acd`; `execute()` reauthorizes, rule errors fail closed, actions
are moved off the event loop in the routed path, and action failures become `ok=False` data.

### V1-03 — Bare-string constraints are trusted but not verified client-side

**Severity:** High  
**Checkout behavior:** regex/choice modes rely completely on backend enforcement. A mock,
misconfigured backend, or provider that ignores `extra_body` can return arbitrary text that then
flows toward an executor.

**Upstream status:** fixed for regex and choice by `f042acd` via `ConstraintViolationError` and a
client-side guard. Grammar remains server-trusted because local validation requires XGrammar.

### V1-04 — Eager JSON-schema compilation checks the parent's empty schema

**Severity:** High  
**Checkout behavior:** `SAV_GRAMMAR_CHECK=1` calls `model_json_schema()` in `__init_subclass__`, before
Pydantic has collected the subclass fields. The check compiles the inherited empty
`ConstrainedOutput` schema and produces false confidence.

**Upstream status:** fixed by `07bd719` using `__pydantic_init_subclass__`, with a regression test
asserting that the subclass schema reaches XGrammar.

### V1-05 — Concurrent request capture is misattributed and unbounded

**Severity:** Medium  
**Checkout behavior:** `StructuredAgent._result()` reads the shared capture's `.last` record when a
run completes. Same-agent concurrent calls can receive each other's request body. The record list
also retains full prompts forever.

A focused probe reproduced the wrong attribution:

```text
server got 'second'
server got 'first'
first first
second first
```

The second output was associated with the first request body.

**Upstream status:** fixed by `07bd719` using a per-run context-variable sink and a bounded deque.

### V1-06 — Decoder `extra_body` clobbers caller settings

**Severity:** Medium  
**Checkout behavior:** grammar/regex/choice construction replaces an existing profile
`model_settings["extra_body"]` rather than merging the structured-output constraint into it.

**Upstream status:** fixed by `07bd719`; caller keys are preserved and decoder keys win conflicts.

### V1-07 — HTTP clients are created per agent and never closed

**Severity:** Medium  
**Checkout behavior:** each captured/transport-backed build creates a new `httpx.AsyncClient`. There
is no close method, so a fleet creates multiple unclosed connection pools against one backend.

**Upstream status:** fixed by `2c31aad`; clients are shared per backend and `aclose()` is exposed.

### V1-08 — Batch failures discard sibling results

**Severity:** Medium  
**Checkout behavior:** `run_batch()` uses bare `asyncio.gather`, so one exception raises the batch
and loses successful sibling results.

**Upstream status:** fixed by `2c31aad` through `BatchResult` and `return_exceptions=True`.

### V1-09 — Explicit decoder conflicts are silently ignored

**Severity:** Medium  
**Checkout behavior:** if `output_type_ref` resolves to `ConstrainedOutput`, an explicit profile
decoder is silently discarded.

**Upstream status:** fixed by `2c31aad`; conflicting sources of truth raise `ConfigError`.

### V1-10 — Closed-backend async tests cannot run in the declared dev environment

**Severity:** High  
**Checkout behavior:** `tests/test_closed.py` uses `pytest.mark.asyncio`, but `pytest-asyncio` is absent
from the dev extra. The clean suite reports six failures and unknown-mark warnings.

**Upstream status:** fixed by `3c57554`/`8bd596d`; the plugin is declared.

### V1-11 — Dependency and documentation hygiene

**Severity:** Medium  
**Checkout behavior:**

- `grail` is a hard Git dependency despite zero imports in source or tests.
- Psycopg is imported directly by dual-path but arrives only transitively through DBOS.
- PydanticAI is unbounded despite a known compatibility break.
- `uv.lock` still records Grail with `rev=main` while `pyproject.toml` specifies a SHA.
- The README and package description are template stubs.

**Upstream status:** fixed by `8bd596d`, `97a9976`, and the `v0.2.0` release commit. Grail is removed,
Psycopg is explicit, PydanticAI is bounded, metadata is improved, and the README is substantive.

## 6. Verification results

### 6.1 Clean declared environment

Command:

```bash
devenv shell -- uv run --extra dev pytest
```

Result:

```text
77 passed, 6 failed, 7 skipped
```

All six failures came from async `ClosedBackend` tests because `pytest-asyncio` was not declared.
Coverage was 58%, largely because the optional dual-path package was not installed/executed.

### 6.2 Strongest locally runnable suite

Command shape:

```bash
devenv shell -- uv run --extra dev --extra dual-path --with pytest-asyncio pytest
```

Result:

```text
90 passed, 18 skipped
82% total coverage
```

The 18 skips include PostgreSQL-gated dual-path runtime/store tests and live inference tests. Relevant
coverage remained low in the most operationally risky modules:

- `dual_path/runner.py`: 29%
- `dual_path/runtime.py`: 42%
- `dual_path/store.py`: 36%

### 6.3 Static checks

```text
ruff check src tests: passed
ruff format --check src tests: passed
ty check src: 2 warnings
```

Both type-check warnings are unused suppression comments in `dual_path/store.py`.

Running Ruff against the entire repository, rather than the configured library/test scope, reports
issues in scratch and deployment scripts. Those do not affect the package check but demonstrate that
the repository has no single clean all-files lint target.

### 6.4 Packaging

The package built successfully into `/tmp`:

```text
structured_agents_v2-0.1.0-py3-none-any.whl  31 KB
structured_agents_v2-0.1.0.tar.gz            74 KB
```

The wheel manifest was correct. The sdist manifest was not; see CR-14.

### 6.5 Focused probes

Two targeted probes supplemented the test suite:

1. **Concurrent capture probe:** demonstrated that two overlapping runs on the same agent can return
   the wrong request body in `AgentResult`.
2. **Closed transport probe:** demonstrated that a connection failure escapes as raw
   `httpx.ConnectError` rather than `ClosedBackendError`.

## 7. Test-quality assessment

### What the tests do well

- Use an in-process ASGI implementation instead of network mocks at the wrong abstraction level.
- Assert actual OpenAI-compatible wire fields for JSON schema and bare-string constraints.
- Cover routing-table existence and `Literal` coverage checks.
- Prove concurrent dispatch rather than merely asserting that `asyncio.gather` was called.
- Test the executor's happy-path and denial behavior.
- Keep live inference explicitly gated.

### Important blind spots

- No strict-config typo tests.
- No credential representation/serialization tests.
- No mismatched primary/reference schema test.
- No crash/retry/idempotency test around comparison persistence.
- No unsampled-primary failure test.
- No closed transport-exception test.
- No fleet rebuild rollback test.
- No duplicate-policy test.
- No sdist manifest assertion.
- PostgreSQL tests depend on developer-local infrastructure and skip by default.
- No top-level CI detects missing plugins, unexpected skips, packaging drift, or type/lint failures.

## 8. What is genuinely strong

1. **The conceptual separation is correct.** Decoder constraints guarantee syntax, Pydantic validates
   structure, routing selects behavior, and executors decide authority. Those are distinct guarantees
   and the code mostly keeps them distinct.

2. **The implementation is wire-grounded.** `DecoderSpec.apply()` reflects captured PydanticAI/OpenAI
   request behavior, including the important distinction between native JSON-schema response format
   and `extra_body`-based grammar/regex/choice constraints.

3. **Effects remain explicit.** Model generation does not implicitly execute commands. The caller
   must invoke an executor path deliberately.

4. **Errors are usually actionable.** Configuration and routing failures generally identify the
   offending profile and explain the required correction.

5. **The codebase is compact.** The core can be understood end-to-end without hidden framework
   layers, which makes the remaining defects tractable.

6. **`ClosedBackend` demonstrates good scope discipline.** It avoids tools, capture, raw result
   exposure, SDK escape hatches, redirects, proxy trust, remote hosts, and retries. Its incomplete
   transport normalization is a localized defect rather than an architectural failure.

7. **Upstream responded well to prior defects.** The v0.2.0 commits are focused, tested, and generally
   move the implementation closer to its stated safety thesis.

## 9. Prioritized remediation plan

### Phase 0 — Establish the correct baseline

1. Fast-forward or rebase the working checkout to upstream tag `v0.2.0`.
2. Re-run the full core and dual-path suites in a hermetic environment.
3. Do not independently reimplement the v0.1 findings already fixed upstream.

### Phase 1 — Close configuration and secret leaks

1. Introduce a shared strict Pydantic configuration base (`extra="forbid"`).
2. Convert `Backend.api_key` to protected secret storage.
3. Add negative configuration and secret-redaction tests.

### Phase 2 — Make dual-path records trustworthy

1. Require primary/reference schema equivalence at registration.
2. Validate outputs against the expected type, not generic `BaseModel`.
3. Use a full idempotency key and a unique database constraint.
4. Move record assembly/persistence into a durable retryable step.
5. Make primary failure behavior independent of reference sampling.

### Phase 3 — Repair async/database behavior

1. Add pooled async Psycopg access.
2. Ensure database work does not block the event loop.
3. Define launch rollback and idempotent shutdown behavior.
4. Bound and validate sample rates and query limits.

### Phase 4 — Complete the closed boundary

1. Normalize `httpx` transport exceptions into detail-free `ClosedBackendError`.
2. Validate `output_type` at construction.
3. Bound response consumption and add transport/timeout tests.

### Phase 5 — Packaging and continuous verification

1. Remove and ignore the tracked absolute `result` symlink.
2. Restrict the sdist manifest with rooted allowlists and explicit excludes.
3. Add PostgreSQL-backed CI with ephemeral credentials.
4. Fail CI on unknown marks and unexpected skips.
5. Assert wheel and sdist manifests.

### Phase 6 — API refinement

1. Make fleet rebuilds transactional.
2. Reject duplicate policy names.
3. Validate export grouping options.
4. Preserve output types through generic `StructuredAgent`/fleet APIs.
5. Formalize the trust model for `output_type_ref` imports.

## 10. Release gate

Before describing dual-path output as a durable SFT/evaluation source of truth, require all of the
following:

- Primary and reference schemas are proven equivalent.
- Every run has a full, unique, idempotent identifier.
- Completed model legs cannot exist without a recoverable pending/saved comparison record.
- Database operations are pooled and nonblocking.
- Primary/reference failures are captured consistently.
- PostgreSQL integration tests run on every change.
- Crash/retry/duplicate-insert behavior is tested.
- Exported reference targets are revalidated against the recorded schema.

Before releasing the core library, require:

- Strict configuration models.
- Secret-safe backend representation.
- Detail-free closed transport failures.
- Clean default tests with no unexpected skips or unknown marks.
- Deterministic, allowlisted wheel and sdist manifests.

## 11. Final assessment

`structured-agents-v2` is not a failed design; it is a promising design whose operational claims
currently run ahead of its boundary validation and durability guarantees. The upstream v0.2.0
release addresses the most obvious core defects and should become the immediate baseline. The next
engineering effort should concentrate less on adding features and more on making configuration,
credentials, dual-path identity, persistence, and packaging fail closed and behave deterministically.

Once CR-01 through CR-07 and the hermetic integration-test gap are resolved, the library will have a
credible path from "well-designed experimental wrapper" to a dependable foundation for constrained
agent systems.
