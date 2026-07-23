# `structured-agents` 0.3.0 — Detailed Code Review

**Review date:** 2026-07-21  
**Commit:** `90725a5` (`main`, clean before deliverables)  
**Package version:** 0.3.0  
**Scope:** all package source, all first-party tests, public README and packaging configuration, relevant deployment documentation, recent design/research records, built artifacts, and installed DBOS/PydanticAI behavior.

## 1. Executive verdict

The library has a good architecture and a small, legible implementation. It separates output constraints, backend dialects, model execution, authorization, durable effects, human approval, and operational control more cleanly than many much larger agent frameworks.

The reviewed commit is **not production-ready as an authority or durable-agent library**. There are three release-blocking defects:

1. Approval parsing is fail-open: a truthy non-boolean value such as the string `"false"` is accepted as approval.
2. The schema-module allowlist is stored in a process-global stack and leaks across concurrent threads; one config request can resolve under another request's allowlist.
3. The public queue cannot submit a real library `Agent`; the test uses a structurally different fake and therefore proves only DBOS queue behavior, not the library integration.

There are also serious qualification gaps: an empty `all_of()` authorizes, approval messages lack per-request correlation, the external-effect "exactly once" language is too strong, two of three backend dialects are shipped as supported while explicitly unverified, and the source distribution includes unrelated deployment/vendor material.

### Release recommendation

Do not release 0.3.0 as production-ready. Fix CR-01 through CR-07, add regression tests using real public objects, lock down the source manifest, and publish an accurate lifecycle/security guide. The direct vLLM constrained-generation path can reasonably be called a tested beta after those changes; SGLang and llama.cpp constraint support should remain experimental until live contract tests pass.

## 2. Scorecard

| Area | Assessment | Notes |
|---|---|---|
| Architecture | Strong | Clear dependency direction and narrow abstractions |
| Readability | Strong | Roughly 1,000 package lines; small modules and useful names |
| Type design | Good | PEP 695 generics are effective on the direct agent path |
| Runtime correctness | Weak | Real-agent queue is broken; several casts hide mismatches |
| Security/authority | Unsafe | Two confirmed fail-open/cross-request defects |
| Durability semantics | Mixed | DBOS use is sound in several paths, but guarantees are overstated |
| Backend portability | Experimental | vLLM is the reference; SGLang/llama.cpp are doc-derived |
| Tests | Good breadth, key blind spots | 32 pass, but fakes miss the most important integration defect |
| Documentation | Inadequate | README is stale and omits lifecycle/security essentials |
| Packaging | Weak | sdist captures unrelated deployment and vendored plugin files |
| Release engineering | Incomplete | No first-party CI workflow; format check fails |

## 3. Method and reproducibility

### 3.1 Material reviewed

- Every file under [`src/structured_agents`](../../../src/structured_agents/).
- Every first-party file under [`tests`](../../../tests/).
- [`README.md`](../../../README.md), [`pyproject.toml`](../../../pyproject.toml), and lock/environment files.
- Backend deployment READMEs and the current backend-abstraction design review.
- The prior code-review/refactor documents, while accounting for the fact that they target a superseded package generation.
- Installed dependency source where the wrapper contract mattered, especially PydanticAI's `DBOSAgent.run` and DBOS workflow handles.
- Git history through `90725a5`.

Generated inference artifacts were treated as supporting operational evidence, not line-reviewed as library source. The large vendored GGUF plugin was inspected for packaging impact but not code-reviewed as part of this library.

### 3.2 Environment

| Dependency | Installed version |
|---|---:|
| `structured-agents` | 0.3.0 |
| DBOS | 2.23.0 |
| `pydantic-ai-slim` | 2.11.0 |
| Pydantic | 2.13.3 |
| HTTPX | 0.28.1 |
| XGrammar | not installed |

### 3.3 Verification results

| Command | Result |
|---|---|
| `devenv shell -- pytest` | 32 passed, 1 skipped in 11.27 s |
| `devenv shell -- ruff check src tests` | Passed |
| `devenv shell -- ty check src tests/typecheck_constraint.py` | Passed |
| `devenv shell -- ruff format --check src tests` | Failed: 8 files would be reformatted |
| `devenv shell -- uv build --out-dir /tmp/...` | Wheel and sdist built |
| Live model tests | Not run |
| Crash-recovery live test | Not run |

The ordinary skip is the module-gated live inference suite.

### 3.4 Focused probes

Four targeted probes were used to test contracts not covered by the suite:

1. Approval received `{"allowed": "false"}` and returned `Decision(allowed=True)`.
2. `all_of().decide(...)` returned `Decision(allowed=True)`.
3. Two concurrent config builds with different module allowlists both observed `module_b`.
4. Submitting a real built `Agent` through `Queue.submit()` raised `DBOSWorkflowFunctionNotFoundError` because `agent.raw.run` is not itself a registered workflow function.

The package build also showed unrelated deployment and vendored-plugin test files in the sdist.

## 4. Finding summary

| ID | Severity | Finding | Status |
|---|---|---|---|
| CR-01 | Critical | Approval truthiness allows malformed data to approve | Confirmed by probe |
| CR-02 | Critical | Global config allowlist crosses concurrent requests | Confirmed by probe |
| CR-03 | High | `Queue.submit` cannot enqueue a real `Agent` | Confirmed by real-object probe |
| CR-04 | High | Empty `all_of()` is an allow-all policy | Confirmed by probe |
| CR-05 | High | Approval messages are not correlated to an approval request | Present by construction |
| CR-06 | High | "Exactly once" overstates arbitrary external-effect guarantees | Design/contract defect |
| CR-07 | High | Unverified backend paths are advertised as supported | Explicit in source |
| CR-08 | High | Source distribution includes unrelated deployment/vendor content | Confirmed by built manifest |
| CR-09 | Medium | Serialized settings and config shapes are weakly validated | Confirmed by probe/inspection |
| CR-10 | Medium | Schema parsing is a cast and custom engines can violate the type contract | Present by construction |
| CR-11 | Medium | Global registries have collision and discovery races | Present by inspection |
| CR-12 | Medium | Lifecycle and HTTP-client ownership are underspecified | Present by inspection/probe |
| CR-13 | Medium | Subprocess/Fornix boundaries lack limits and strict result validation | Present by inspection |
| CR-14 | Medium | Dependency and private-API policy is fragile | Present by manifest/source |
| CR-15 | Medium | Tests, CI, formatting, and README do not meet release claims | Confirmed |

## 5. Detailed findings

### CR-01 — Malformed approval data can approve an operation

**Severity:** Critical  
**Area:** Authorization / human approval  
**File:** [`approval.py:44`](../../../src/structured_agents/approval.py#L44)

The approval parser does this:

```python
allowed = bool(message.get("allowed", False))
```

Python truthiness is not schema validation. All of these approve:

```python
{"allowed": "false"}
{"allowed": "no"}
{"allowed": 1}
{"allowed": [False]}
```

The focused probe returned:

```text
approval_string_false Decision(allowed=True, reason='')
```

The built-in `ApprovalClient` sends real booleans, but DBOS messaging is an external application boundary. A bot, UI, migration, manually written sender, or hostile client can provide a different shape. This is a fail-open authority defect.

**Required fix:** Require `type(allowed) is bool`, or validate the message with a strict Pydantic model such as `ApprovalDecision(allowed: StrictBool, reason: str = "")`. Treat every validation failure as `Decision(False, "invalid approval decision")`.

**Required tests:** Strings, integers, lists, missing fields, non-string reasons, extra fields under the chosen policy, and a valid strict boolean pair.

### CR-02 — The schema allowlist leaks across threads

**Severity:** Critical  
**Area:** Config trust boundary / concurrency  
**File:** [`config.py:56`](../../../src/structured_agents/config.py#L56), [`config.py:86`](../../../src/structured_agents/config.py#L86), [`config.py:122`](../../../src/structured_agents/config.py#L122)

`constraint_from_config()` pushes its allowlist onto one module-global list. Built-in schema factories recover the active allowlist by reading the last item.

That works only when every call is serialized on one thread. With overlapping threads:

```text
thread A: append allowlist A
thread B: append allowlist B
thread A: read last item -> B
thread B: read last item -> B
```

A barrier-controlled probe reproduced exactly that result:

```text
observed ['module_b', 'module_b']
```

This defeats the security purpose of the explicit `allow_modules` argument. A schema forbidden to one request may be imported if another request's broader allowlist is concurrently active. Concurrent `finally: pop()` operations can also disrupt nested state.

**Required fix:** Do not hide the allowlist in ambient mutable state. The strongest design changes `ConstraintFactory` to accept an explicit context, for example `factory(data, ConfigContext(allow_modules=...))`. If compatibility absolutely requires ambient state, use a `ContextVar`, not a process-global list; note that `ContextVar` addresses tasks and threads but explicit data flow remains clearer.

Entry-point factories need a documented context contract. A factory API that cannot receive the trust policy safely should not resolve schemas.

**Required tests:** Deterministic two-thread isolation, nested factory calls, concurrent entry-point discovery, exceptions during a factory, and independent asyncio tasks.

### CR-03 — The public queue does not work with the library's real agent

**Severity:** High; release blocker  
**Area:** Runtime correctness / durability / typing  
**File:** [`plane.py:56`](../../../src/structured_agents/plane.py#L56)  
**Masking test:** [`test_plane.py:43`](../../../tests/test_plane.py#L43)

`Queue.submit()` enqueues `agent.raw.run`:

```python
await self._queue.enqueue_async(agent.raw.run, prompt)
```

PydanticAI's `DBOSAgent.run` is a normal wrapper method that calls an internally registered workflow (`dbos_wrapped_run_workflow`). It is not the workflow object DBOS expects `enqueue_async()` to receive.

The real-object probe failed with:

```text
DBOSWorkflowFunctionNotFoundError:
Could not execute workflow <NONE>: run is not a registered workflow function
```

The test suite misses this because `QueueAgent.raw.run` is itself a hand-decorated `@DBOS.workflow`. That fake does not preserve the contract of the real object it replaces.

Even if the implementation enqueued `dbos_wrapped_run_workflow`, another mismatch remains: that workflow returns `AgentRunResult[T]`, while `Queue.submit()` casts its handle to `WorkflowHandleAsync[T]`. It would also bypass the library's `Constraint.parse()` post-check when the result is retrieved.

**Required fix:** Register a queueable workflow owned by the library `Agent` that returns the final parsed `T`, and enqueue that exact registered workflow. Alternatively, return a documented handle wrapper that delegates workflow status/ID and transforms `AgentRunResult.output` through `Constraint.parse()` in `get_result()`. The dedicated-workflow option better matches the advertised durable typed result.

**Required tests:** Build a real `Backend` with `MockTransport`, construct a real `Agent`, submit it through a real `Queue`, assert the handle result is `T`, test regex/choice post-validation, reuse a key, and recover it in a fresh DBOS process.

### CR-04 — `all_of()` with no policies authorizes everything

**Severity:** High  
**Area:** Authorization composition  
**File:** [`authority.py:60`](../../../src/structured_agents/authority.py#L60)

The `_AllOf.decide()` loop returns `Decision(True)` after iterating. With no authorizers, it therefore allows:

```text
empty_all_of Decision(allowed=True, reason='')
```

This is the mathematical identity of conjunction, but it is the wrong default for an API whose documented posture is default-deny. An empty collection commonly arises from a configuration mistake or filtered plugin list. That mistake becomes allow-all.

**Required fix:** Reject zero authorizers in `all_of()` with `AuthorityError`, or return a denial such as `Decision(False, "no authorizers configured")`. Apply the same explicit arity policy to `any_of()` for clarity, even though its current empty result denies.

**Required tests:** Empty, one-item, and multi-item composition; exceptions; and dynamically built empty lists.

### CR-05 — Approval messages identify a workflow, not a particular request

**Severity:** High  
**Area:** Approval integrity / replay  
**File:** [`approval.py:13`](../../../src/structured_agents/approval.py#L13), [`approval.py:27`](../../../src/structured_agents/approval.py#L27), [`approval.py:57`](../../../src/structured_agents/approval.py#L57)

An approval request is correlated only by workflow ID and a reusable topic. There is no request ID, nonce, command hash, sequence, or expected approver in the response.

Consequences:

- A message queued before the request can be consumed later.
- Two sequential approvals in one workflow share the same topic and overwrite the same `pending_command`/`pending_to` events.
- A delayed response to the first request can satisfy the second request.
- `ApprovalClient.pending()` can report stale approval events when a workflow later becomes pending for another reason.
- The `to` recipient is presentation data only; it is not checked against the sender.

DBOS makes messages durable, which makes correct correlation more important, not less.

**Required fix:** Generate or deterministically derive an `approval_id`; publish an immutable approval envelope containing ID, recipient, command digest, creation time, and state; receive on a request-specific topic or validate the ID in the message; mark the request resolved. Require the approving application to enforce actor-to-recipient authorization.

**Required tests:** Pre-sent messages, two sequential approvals, delayed first response, duplicate decisions, stale pending events, wrong request ID, and wrong recipient/actor.

### CR-06 — The "exactly once" claim is broader than the implementation can guarantee

**Severity:** High  
**Area:** Durability contract  
**Files:** [`README.md:3`](../../../README.md#L3), [`authority.py:140`](../../../src/structured_agents/authority.py#L140)

The implementation correctly uses a stable DBOS workflow ID to return a previously recorded result for a repeated business key. The tests prove that in one process after a successful recorded step.

That does not make an arbitrary external effect transactionally exactly-once. For `Subprocess` and `FornixEffector`, there is an unavoidable boundary between the effect completing and DBOS durably recording the step result. A crash in that window can cause replay. Remote systems have the same problem unless they accept an idempotency key or participate in a transaction.

Authorization is also outside the keyed effect workflow. Repeating the same key re-evaluates current policy before looking up the durable outcome. A policy change can therefore return denial for an operation whose effect previously succeeded, or allow a previously denied call because denials are not recorded under the key.

**Required fix:** Tighten documentation to "keyed durable execution with replay of recorded completions." Pass the business key into effectors and require idempotency-aware implementations for non-idempotent effects. If the authorization decision must be stable and auditable per operation, move decision recording and effect execution into one durable orchestration workflow with a versioned policy identity.

**Required tests:** Crash injection around the external side effect, policy changes for a reused key, concurrent duplicate keys, and idempotency conflicts where the same key is reused with a different command.

### CR-07 — Two backend implementations are advertised before contract verification

**Severity:** High  
**Area:** Backend correctness / capability claims  
**Files:** [`engine/sglang.py:1`](../../../src/structured_agents/engine/sglang.py#L1), [`engine/llama_cpp.py:1`](../../../src/structured_agents/engine/llama_cpp.py#L1)

Both modules explicitly say their constrained request shapes are unverified. Nevertheless, `supports` advertises the corresponding features and `Backend.build()` accepts them without warning.

The llama.cpp grammar path is particularly problematic:

- The public abstraction accepts EBNF.
- The optional `Grammar.check()` compiles it as XGrammar EBNF.
- llama.cpp expects GBNF.
- The engine passes the text through unchanged.

A successful build therefore does not mean the selected engine accepts the grammar dialect. SGLang choice lowering also assumes the server's regex matching semantics make the alternation equivalent to a finite choice.

**Required fix:** Mark unverified capabilities experimental in the public API, or fail unless the caller opts into them. Split `Grammar` by dialect or add a canonical grammar representation with proven lowering. Add live contract tests for every advertised constraint/backend pair and pin supported server versions.

The static render tests should remain, but they prove only dictionary shape, not server acceptance or constrained output.

### CR-08 — The source distribution contains unrelated repository content

**Severity:** High for release hygiene  
**Area:** Packaging / supply chain  
**File:** [`pyproject.toml:31`](../../../pyproject.toml#L31)

The Hatch sdist include list is not rooted tightly enough. A built 0.3.0 sdist contained, among other unrelated files:

- Deployment READMEs.
- A head-to-head artifact README.
- `deploy/sglang/native/pyproject.toml`.
- `deploy/vllm/native/vendor/vllm-gguf-plugin/pyproject.toml`.
- The vendored plugin's test suite.

The wheel was small, but the sdist is the input from which downstream build systems create wheels. Accidental inclusion creates nondeterminism and expands the audited supply-chain surface. Nested `pyproject.toml` files are especially undesirable in a source archive for another project.

**Required fix:** Use rooted Hatch patterns such as `/src/structured_agents/**`, `/tests/**`, `/README.md`, `/LICENSE`, and `/pyproject.toml`, plus explicit exclusions for `/.scratch/**`, `/artifacts/**`, `/deploy/**`, environments, and build outputs.

**Required test:** Build both artifacts in a clean temporary directory and compare their manifests to allowlists. Install the wheel and a wheel built from the sdist into isolated environments and run import/API smoke tests.

### CR-09 — Serialized config is shape-checked but not type-strict

**Severity:** Medium  
**Area:** Configuration correctness  
**Files:** [`agent.py:20`](../../../src/structured_agents/agent.py#L20), [`config.py:133`](../../../src/structured_agents/config.py#L133)

`Settings` is a plain dataclass. Runtime construction does not enforce annotations or useful ranges. The probe accepted:

```text
Settings(temperature='cold', max_tokens=-1)
```

`spec_from_config()` catches only constructor `TypeError`, so these values cross the supposed serialized-data boundary. Unknown top-level agent and constraint keys are silently ignored by the manual parsers, allowing misspellings to disappear. Direct `AgentSpec` construction also accepts empty names and nonsensical field values until a dependency fails later.

**Required fix:** Validate serialized mappings with strict Pydantic models or explicit validators before constructing dataclasses. Forbid unknown fields. Define finite/range constraints for temperature and `top_p`, positive bounds for `max_tokens`, strict integers excluding booleans for seeds, and a JSON-compatible policy for `extra_body`.

Keep the programmatic API ergonomic, but fail at `Backend.build()` at the latest.

### CR-10 — Schema parsing does not enforce the schema contract locally

**Severity:** Medium  
**Area:** Type soundness / extension boundary  
**File:** [`constraint.py:41`](../../../src/structured_agents/constraint.py#L41), [`agent.py:63`](../../../src/structured_agents/agent.py#L63)

`_Schema.parse()` is only:

```python
return cast(M, raw)
```

The built-in PydanticAI `NativeOutput` path normally returns the correct model, so this works on the reference path. But `Backend` publicly accepts `engine: Engine` and `model: Any`. A custom engine can claim `schema` support and render `str`; a custom model can return a different type. The final library boundary then asserts `M` without validation.

`Grammar.parse()` similarly validates only that output is a string. Its grammar acceptance is entirely trusted to the server.

**Required fix:** For schemas, return the instance only if it is already the expected class; otherwise call `model.model_validate(raw)` or validate a model dump. For custom engines, validate the `WireSpec` and its relationship to the constraint kind. Document which checks are server-enforced versus client-enforced.

**Required tests:** A lying custom engine, a custom model returning the wrong model class, dict input, invalid dict input, subclass behavior, and invalid grammar output where a local checker is available.

### CR-11 — Factory registration and discovery are unsynchronized global mutation

**Severity:** Medium  
**Area:** Extensibility / determinism  
**File:** [`config.py:49`](../../../src/structured_agents/config.py#L49)

The constraint factory dict and discovery flag are global and unlocked.

- `register_constraint()` silently overwrites an existing kind, including a built-in.
- Entry points use `setdefault`, so behavior depends on whether direct registration happened first.
- `_entry_points_discovered` becomes `True` before entry points finish loading; another thread can skip discovery while the registry is incomplete.
- A discovery failure leaves the flag true, preventing a retry after a transient/import-order problem.
- Tests mutate these globals and depend on process isolation/order discipline.

**Required fix:** Make registration initialization explicit and one-shot, reject duplicate names unless an explicit override API is used, and guard discovery with a lock plus a local candidate map committed only after successful loading. Prefer immutable registry snapshots after application startup.

### CR-12 — Lifecycle and client ownership are not a complete public contract

**Severity:** Medium  
**Area:** Resource management / usability  
**Files:** [`agent.py:41`](../../../src/structured_agents/agent.py#L41), [`plane.py:19`](../../../src/structured_agents/plane.py#L19), [`README.md`](../../../README.md)

DBOS requires an important order:

```text
configure DBOS -> construct all named agents/workflows -> launch -> run -> shutdown
```

Building an agent after launch produced a DBOS warning and could not recover/queue the workflow registration correctly. `configure()` mentions registration in its docstring, but the README gives users no runnable lifecycle example.

HTTP ownership is also ambiguous:

- If a caller supplies `http_client`, `Backend.aclose()` closes it, even though borrowed clients are commonly caller-owned and may be shared.
- If no client is supplied, `Backend` does not retain an explicitly closeable client of its own.
- `Backend` is not an async context manager.
- `shutdown()` is declared async but calls synchronous `DBOS.destroy()` without awaiting anything.

**Required fix:** Document and enforce a state machine. Reject agent registration after launch through the library boundary where detectable. Define client ownership explicitly; normally, close only clients the backend created unless `take_ownership=True`. Add `async with Backend(...)` support or a clear application shutdown recipe.

### CR-13 — Process effectors have no resource limits and Fornix decoding is loose

**Severity:** Medium  
**Area:** Reliability / resource exhaustion  
**Files:** [`authority.py:125`](../../../src/structured_agents/authority.py#L125), [`integrations/fornix.py:17`](../../../src/structured_agents/integrations/fornix.py#L17)

Avoiding a shell and strictly validating `argv` are excellent choices. However, both process effectors can run indefinitely and capture unbounded stdout/stderr into memory. There is no timeout, output limit, working-directory policy, environment policy, cancellation strategy, or process-tree termination.

`FornixEffector` also coerces returned values:

```python
int(result.get("returncode", ...))
str(result.get("stdout", ...))
```

Malformed values can become plausible values (`True -> 1`, `None -> "None"`) or leak a raw `ValueError` instead of the integration's error vocabulary.

**Required fix:** Add explicit execution limits and strict result-envelope validation. Decide whether environment and working directory are inherited, cleared, or allowlisted. On timeout/cancellation, terminate the process group and return or raise a stable typed error.

### CR-14 — Compatibility policy conflicts with broad dependency ranges and private imports

**Severity:** Medium  
**Area:** Maintenance / packaging  
**Files:** [`pyproject.toml:10`](../../../pyproject.toml#L10), [`plane.py:14`](../../../src/structured_agents/plane.py#L14)

The package constrains PydanticAI to `<3`, but leaves DBOS as `>=2.23` and Pydantic/HTTPX with no upper bounds. At the same time, `plane.py` imports `QueueRateLimit` from private `dbos._queue`.

That combination means a downstream resolver can install a future DBOS minor/major that moves the private type or changes workflow registration semantics. The lock protects this checkout, not consumers installing the library.

**Required fix:** Avoid the private import by defining the small typed mapping locally or using a public DBOS export. Declare and test a compatibility range. Add a dependency matrix for the minimum and newest supported versions, and use automated dependency updates with live contract tests.

### CR-15 — Release tests and documentation lag the implementation

**Severity:** Medium  
**Area:** Quality engineering / documentation  
**Files:** [`README.md`](../../../README.md), [`tests/test_plane.py`](../../../tests/test_plane.py), [`pyproject.toml`](../../../pyproject.toml)

The README ends by saying constraint codecs, agents, authority, approval, and plane services will land in later phases; all are already present. It contains no installation example, public API walkthrough, DBOS lifecycle, backend support matrix, approval trust model, idempotency guidance, serialized-config warning, or operational caveats.

The test suite has meaningful breadth, but important gaps include:

- No real `Agent` through `Queue`.
- No malformed truthy approval values.
- No approval request correlation/replay tests.
- No concurrent config allowlist test.
- No empty policy-combinator test.
- No package-manifest test.
- No invalid `Settings` tests.
- No custom engine/model contract tests.
- No SGLang or llama.cpp constrained live tests.
- No XGrammar compiler test in the normal quality gate.

There is no first-party top-level CI workflow. Ruff lint passes, but the format check reports eight files.

**Required fix:** Replace the README, add CI jobs for unit/type/lint/format/package manifest, separate opt-in live backend matrices, and ensure fakes conform to the actual public/dependency protocols they replace.

## 6. Additional observations

These are lower-severity issues or design constraints worth recording.

### 6.1 `ApprovalClient.pending()` is an unbounded N+1 query

It lists every pending workflow, then retrieves all events for each one. There is no pagination, recipient filter, topic identity in `PendingApproval`, or batch event API. At scale this will be slow and can expose commands across tenants unless the surrounding service applies access control.

### 6.2 Batch submission is not transactional

`submit_batch()` gathers independent submissions. If one enqueue call fails, siblings may already be durable even though the caller receives an exception and may lose their handles. This may be acceptable, but the API should state partial-success behavior and preferably return per-item submission outcomes.

### 6.3 Comparison failure behavior is asymmetric at the API level

The two legs are durably independent, but `asyncio.gather` raises if one fails and the public call returns no `Comparison` or surviving handle. Operators can recover the leg IDs only if they know the derived naming convention. A richer comparison outcome could preserve both statuses.

### 6.4 Business keys share one global namespace

Effect keys, queue keys, comparison-derived keys, and arbitrary application workflow IDs all enter DBOS's workflow-ID namespace. The library does not namespace or validate them. Applications need a documented convention and command/key conflict detection.

### 6.5 Local constraint checking is intentionally uneven

Regex and choice are post-validated; schema relies on PydanticAI; grammar relies on the inference server. That is defensible, but it should be explicit in the public contract so callers know which guarantees survive a misconfigured backend.

### 6.6 Local model injection is an escape hatch

`Backend(model=Any)` is useful for testing and advanced integration, but it bypasses the normal OpenAI provider construction while retaining engine capability claims. It should be typed as `Model`, documented as advanced/testing-only, and covered by contract tests.

### 6.7 Error normalization is incomplete

The custom error vocabulary is a good start. Dependency exceptions, HTTP errors, DBOS registration failures, Pydantic validation errors, `ValueError` from Fornix coercion, and subprocess OS errors can still escape directly. Decide deliberately which are public and normalize the rest with preserved causal context that does not leak secrets.

## 7. What the code does well

An intense review should preserve the parts that are already right.

### 7.1 The dependency direction is clean

Constraints do not import engines; engine modules translate constraints; the backend assembles dependencies; applications compose authority and workflows. There is little circular conceptual coupling.

### 7.2 The public surface is compact

The library resists recreating all of PydanticAI or DBOS. `Agent.run(prompt) -> T` is easy to understand. Engine selection is one backend constructor option.

### 7.3 The vLLM wire path has a strong regression guard

Golden tests assert the exact `extra_body` representation for regex, choice, and grammar, and `NativeOutput` for schemas. Engine-generated constraint fields override arbitrary user `extra_body`, which prevents callers from accidentally replacing the enforced constraint at the same key.

### 7.4 Denial precedes effects

`execute()` checks the authorizer before entering the effect workflow. `Denied` is ordinary data, making denial a normal business outcome rather than an exception that might trigger retries.

### 7.5 Subprocess invocation avoids shell parsing

The command must be a Pydantic model with a strict, non-empty list of strings. `subprocess.run` receives an argv tuple and never uses `shell=True`. This removes a major class of command-injection mistakes.

### 7.6 DBOS features are exercised, not merely wrapped

The suite demonstrates keyed replay, nested agent workflows, pending receives, resume, scheduling, queue concurrency, status, cancellation, and fork behavior against real DBOS 2.23 on SQLite. The optional crash worker goes further than most small libraries by modeling separate-process recovery.

### 7.7 The code is typed and readable

Generic types preserve schema and choice results through much of the API. Frozen dataclasses communicate value semantics. Modules are short enough to audit directly, and the error classes are domain-oriented.

### 7.8 Backend limitations are at least acknowledged in source

The SGLang and llama.cpp modules do not falsely claim live verification in their comments. The problem is that the public capability behavior does not yet reflect that honesty; the underlying engineering record is candid.

## 8. Recommended remediation order

### Phase 0 — Stop authority fail-open behavior

1. Strictly validate approval decision envelopes.
2. Replace the global allowlist stack with explicit context flow.
3. Make empty authorizer composition an error/denial.
4. Add request-level approval correlation.

These changes should land before any production use.

### Phase 1 — Repair the public durable runtime

1. Make a real `Agent` queueable and return `T`, not `AgentRunResult[T]` hidden by a cast.
2. Add a real-object queue integration and fresh-process recovery test.
3. Define the DBOS registration/lifecycle state machine.
4. Define business-key namespaces and idempotency-conflict behavior.

### Phase 2 — Make guarantees precise

1. Replace universal exactly-once wording with precise durable replay semantics.
2. Carry idempotency keys into effectors.
3. Version and durably record authorization decisions where stable authorization is required.
4. Add process timeouts, output bounds, and cancellation behavior.

### Phase 3 — Qualify backend support

1. Keep vLLM as the supported reference path.
2. Gate SGLang and llama.cpp constrained modes as experimental.
3. Resolve EBNF/GBNF semantics rather than passing ambiguous text through.
4. Add version-pinned live contract matrices for every advertised capability.

### Phase 4 — Harden configuration and extension points

1. Strictly validate all serialized fields and forbid unknowns.
2. Make factory registration deterministic and collision-safe.
3. Validate schema output locally.
4. Type custom engines/models and test dishonest implementations.

### Phase 5 — Release hygiene

1. Root and allowlist the sdist manifest.
2. Add package-manifest and clean-install tests.
3. Remove private DBOS imports or pin a supported range.
4. Replace the README with the conceptual and lifecycle material applications need.
5. Add CI and make format checking mandatory.

## 9. Proposed release gates

A production-oriented release should require all of the following:

- Malformed approval messages always deny.
- Approval responses are bound to a specific pending request.
- Concurrent config loads cannot observe another request's allowlist.
- Empty policy composition cannot authorize.
- A real public `Agent` can be queued and returns its declared `T`.
- Reusing an idempotency key with a different command is detected.
- External-effect semantics and required idempotency are accurately documented.
- Every advertised backend/constraint pair passes a live test against pinned server versions.
- XGrammar compilation tests run in a dedicated required job for supported grammars.
- The wheel and sdist match explicit manifests and contain no deployment/vendor artifacts.
- Unit, concurrency, type, lint, format, packaging, and minimum/newest-dependency jobs pass in CI.
- The README includes installation, lifecycle, backend matrix, authority model, approval security, idempotency, and shutdown examples.

## 10. Final assessment

The project has selected the right conceptual seams. It is much easier to harden this implementation than to repair a framework that confuses model output with authority or backend syntax with domain semantics.

The current danger is the contrast between architectural confidence and boundary implementation. The code looks clean, the normal suite is green, and DBOS terminology suggests strong guarantees; yet malformed approval data authorizes, concurrent allowlists cross, and the queue's only test does not use the actual agent it promises to queue.

Treat 0.3.0 as a well-designed experimental core. Fix the boundary defects, test with real public objects, narrow the claims, and qualify backend behavior. After that work, the library can become a strong foundation for durable constrained-agent applications without becoming a sprawling agent framework.
