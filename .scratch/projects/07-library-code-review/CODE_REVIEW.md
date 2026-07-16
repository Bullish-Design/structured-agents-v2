# structured-agents-v2 — Deep Code Review

> **RESOLVED IN v0.2.0 (2026-07-16).** All findings below were addressed per
> `REVIEW_REFACTOR_GUIDE.md`, tag `v0.2.0`. Status by ID:
> - **A1** usage()→property: fixed (pydantic-ai 2.11.0 upgrade) + regression tests (capture on/off).
> - **A2** grammar check saw parent schema: fixed via `__pydantic_init_subclass__` + subclass-schema test.
> - **A3** test_closed.py silently skipped: fixed (pytest-asyncio added; tests execute) + closed.py nits.
> - **A4** extra_body clobber: fixed (merge, decoder keys win) + test.
> - **A5** capture misattribution/unbounded: fixed (per-run contextvar sink + bounded deque) + concurrency test.
> - **B1–B5** executor: execute() re-checks authority, allow-rule exceptions fail closed, bare-string
>   guard (`ConstraintViolationError`), actions off the event loop, `ExecResult.ok` wired, docstring
>   example fixed + unknown-policy table.
> - **C (section)** shipped: BatchResult results-as-data, shared HTTP client + `aclose()`, decoder-conflict
>   ConfigError, model_settings key warning, dead cap removed, `set_routing()`, `_KNOWN_KEYS` expanded,
>   import-surface note, dual_path `DualPathDecodeMode` rename + runtime-shape docs. **Deferred (recorded):**
>   typed-generic `StructuredAgent`, dual_path durability/pooling/DBOS-ID verification.
> - **D1–D4** packaging: grail removed, psycopg declared, pydantic-ai bounded, README + description real.
> - **F** grail: removed. **G** fornix: integration deferred (design settled — in-tree stdlib module, not a dep).

**Date:** 2026-07-16
**Reviewed commits:**
- Working checkout: `05fa879` (detached HEAD — *behind main*)
- Latest `main`: `7b4a053` (adds `closed.py`, `tests/test_closed.py`, `LICENSE`, grail SHA pin)

**Scope:** every source file in `src/structured_agents_v2/` (core + `dual_path/` + `closed.py` on main), full test suite, `pyproject.toml`, the `.scratch` planning/research docs, and the downstream Lodestar issue (`../06-pydanticai-usage-compatibility/ISSUE.md`). Two sibling repos were studied for the dependency questions: `~/Documents/Projects/grail` and `~/Documents/Projects/fornix`.

**Verification performed (not just read):**
- Test suite re-run in devenv: **98 passed / 3 skipped, 98% coverage** (matches claims in memory/CONCEPT).
- Empirical probe of the `SAV_GRAMMAR_CHECK` bug (A2 below) — fake xgrammar module + env flag; the wrong schema was captured and is reproduced verbatim below.
- Git history inspection to establish the checkout-vs-main delta and locate `closed.py`/`LICENSE`.

**Verdict:** the architecture is genuinely good — disciplined layering, wire-grounded design (every decode-mode mapping traces to a captured request in `../02-library-wrapper/VERIFICATION.md`), honest module docstrings, and mostly-meaningful test coverage. But there are **five confirmed bugs** (section A), and the **executor — the module whose entire purpose is safety — has the sharpest edges** (section B). The `grail` dependency is dead weight and should be removed (section F); `fornix` is a good *containment* complement but not an executor replacement (section G).

---

## Related documents

| Doc | Relevance |
|---|---|
| `../02-library-wrapper/CONCEPT.md` (rev 2) | The authoritative design doc; §4.2 promises a bare-string guard that was never implemented (finding B4) |
| `../02-library-wrapper/VERIFICATION.md` | The captured wire shapes grounding `decoder.py`'s mode table |
| `../01-xgrammar-concept/STRUCTURED_AGENT_CONCEPT.md` | Thesis doc (constrained small agents + per-agent LoRA on vLLM) |
| `../06-pydanticai-usage-compatibility/ISSUE.md` | Downstream (Lodestar) report of finding A1, dated 2026-07-14, with recommended fix + acceptance criteria |
| `../03-dual-path/CONCEPT.md`, `../04-dual-path-runtime/PHASE_2_KICKOFF.md` | Dual-path design context for section C findings |

---

## Repo-state note (read first)

The working copy is a **detached HEAD at `05fa879`** while `main` is at `7b4a053`. Main additionally contains:

- `src/structured_agents_v2/closed.py` + `tests/test_closed.py` (commit `065a197`, "feat: add closed structured-output backend") — a deliberately narrow, loopback-only, JSON-schema-only client built for Lodestar's privacy/authority requirements. It is *not* exported from `__init__.py` (deliberate, per its docstring); downstream imports `structured_agents_v2.closed` directly.
- `LICENSE` (MIT).
- `pyproject.toml`: grail pinned from floating `rev = "main"` to SHA `f804a52332f66fbafefa5bdfbde664a17328ee01`.

This matters for the downstream "Phase A.4" item: *re-pinning Lodestar to a reviewed newer SHA picks up `closed.py` and `LICENSE` — but finding A1 is still unfixed on main*, so the re-pin should wait for that fix.

All `file:line` references below are against the working checkout (`05fa879`) unless marked *(main)*.

---

## A. Confirmed bugs

### A1 — `raw.usage()` breaks valid outputs on newer PydanticAI (the Phase A.4 blocker) — **HIGH**

`agent.py:51`:

```python
return AgentResult(output=raw.output, usage=raw.usage(), request_body=request_body, raw=raw)
```

In the locked PydanticAI (1.87.0) `usage` is a method, so the in-repo suite is green. In the newer PydanticAI that Lodestar resolved, `usage` is a `RunUsage` **property** — calling it raises `TypeError: 'RunUsage' object is not callable` *after* the output has already been validated. A fully successful model call fails while constructing `AgentResult`. This is exactly the downstream report in `../06-pydanticai-usage-compatibility/ISSUE.md` (observed 2026-07-14 against commit `5ca6142`), and **it is still present on main (`7b4a053`, `agent.py:51`)**.

The same incompatibility hides more quietly in `dual_path/runner.py:44-50`: `_usage()` wraps `result.usage()` in a broad `except Exception → None`, so on newer PydanticAI the dual-path layer **silently records no usage data** instead of crashing — a data-quality regression that would go unnoticed in the training/eval records.

Root cause: unbounded `pydantic-ai>=1.87.0` in `pyproject.toml:15` plus a version-sensitive call.

**Fix (per the ISSUE.md recommendation, which is correct):**
1. Compat shim in `_result()` (and in `dual_path/runner._usage()`):
   ```python
   usage = raw.usage
   if callable(usage):
       usage = usage()
   ```
2. Regression test: valid JSON-schema completion over the in-process ASGI transport → `AgentResult.output` returned without raising, tested with `capture=False` **and** `capture=True`.
3. Constrain the pydantic-ai version range in package metadata to the tested range.

### A2 — `SAV_GRAMMAR_CHECK=1` validates the parent's empty schema — **HIGH** (probe-confirmed)

`constrained.py:48-52` runs `check_compilable()` from `__init_subclass__`. Under Pydantic v2, `__init_subclass__` fires during `type.__new__`, **before the metaclass collects the subclass's fields and builds its core schema**. `cls.model_json_schema()` therefore resolves through the *inherited* (parent) schema.

Empirical probe (fake `xgrammar` module recording what it is handed; json_schema-mode subclass with `action`/`reason` fields; `SAV_GRAMMAR_CHECK=1`):

```
CLASS DEFINED OK
calls: ['json_schema']
schema handed to xgrammar: {"description": "A Pydantic model that carries its own
constrained-decoding contract.", "properties": {}, "title": "ConstrainedOutput", "type": "object"}
```

The eager check compiles an **empty object schema** — it always passes, giving false confidence in the exact scenario the feature exists for (failing fast on un-compilable schemas at class-definition time, CONCEPT §5).

Why the suite never caught it: `tests/test_constrained.py:124-144` only exercises **regex** mode at class-definition time. Regex/grammar/choice read plain class attributes (`__regex__` etc.), which *do* exist that early — only `model_json_schema()` is stale. The explicit post-hoc `Model.check_compilable()` call path works fine.

**Fix:** move the eager hook from `__init_subclass__` to Pydantic's `__pydantic_init_subclass__` (a classmethod fired after model completion). `_validate_mode_fields` can stay in `__init_subclass__` (class attrs are available there). Add a json_schema-mode class-definition test asserting the *subclass's* schema (with its fields) reaches xgrammar.

### A3 — `tests/test_closed.py` never executes — **HIGH** *(main only)*

`tests/test_closed.py` *(main)* uses `@pytest.mark.asyncio`, but:
- `pytest-asyncio` is **not** in the dev dependencies (`pyproject.toml` dev extra: pytest, pytest-cov, ty, ruff — verified identical on main), and
- no `asyncio_mode` is configured in `[tool.pytest.ini_options]`.

Without a plugin, pytest collects `async def` tests, emits *"async def functions are not natively supported"*, and **skips them**. (anyio's plugin, present transitively via httpx, only activates for `@pytest.mark.anyio`.) The "regression proofs" for the newest, security-sensitive module on main are silently not running.

**Fix:** either add `pytest-asyncio` + `asyncio_mode = "auto"`, or rewrite in the repo's established sync style (`asyncio.run(...)` inside sync tests — every other test file does this). Check CI logs for the skip warning that should have flagged this.

### A4 — `Backend.build` silently clobbers user-supplied `extra_body` — **MEDIUM**

`backend.py:93-95`:

```python
settings: dict[str, Any] = dict(profile.model_settings)
if app.extra_body:
    settings["extra_body"] = app.extra_body
```

For grammar/regex/choice agents, any `extra_body` the profile carried in `model_settings` (e.g. vLLM sampling extensions) is **silently overwritten** by the decoder's structured-outputs body. json_schema mode is unaffected (its `app.extra_body` is empty).

**Fix:** merge, with the decoder's keys winning on conflict:
```python
settings["extra_body"] = {**settings.get("extra_body", {}), **app.extra_body}
```

### A5 — Capture misattributes request bodies under concurrency; unbounded growth — **MEDIUM**

Each built agent gets one `RequestCapture` (`backend.py:89`), and `agent.py:50` reads `self._capture.last` when a run **completes**:

```python
request_body = self._capture.last.body if (self._capture and self._capture.records) else None
```

`run_batch` explicitly supports the same agent appearing multiple times concurrently — `tests/test_fleet.py:69` runs `("git_ops","a"), ("file_edit","b"), ("git_ops","c")`. Two overlapping runs of one agent race: whichever request was recorded last is attributed to **both** results. Since capture exists precisely to introspect "what did *this* run put on the wire," misattribution defeats the feature.

Separately, `RequestCapture.records` (`capture.py:60`) is append-only forever — a slow memory leak for `capture=True` on a long-lived fleet.

**Fix:** correlate per-run (snapshot `len(records)` before the awaited run and take the slice after; or tag requests via `httpx` request extensions), and cap/ring-buffer `records`.

---

## B. The executor — right concept, sharp edges

`executor.py` is the library's safety thesis ("XGrammar guarantees *syntax*; the executor guarantees *authority*"), so it is held to the highest bar.

### B1 — `AllowlistExecutor.execute()` never checks the allowlist

`executor.py:149-154`: `execute()` looks up the policy and runs its `action` **unconditionally**. Authorization only happens if the caller remembers to call `authorize()` first. The `Executor` protocol (`executor.py:75-80`) actively encourages the two-step split, so any caller who goes straight to `execute()` bypasses the "default-deny safety net" entirely. `route_and_execute` (`fleet.py:137-138`) does the dance correctly, but a safety boundary should be safe against misuse, not just correct usage.

**Fix (defense-in-depth):** re-check `allow` inside `execute()` (cheap — it's a local callable), or make `run()` the only public entry point and demote `execute` to a protected hook.

### B2 — A raising `allow` rule crashes instead of failing closed

`executor.py:127,146` call `p.allow(command)` bare. If the rule raises, the exception propagates out of `authorize()` — the boundary neither allows nor denies; it crashes the pipeline. The module's own worked example is vulnerable: `allow=lambda c: c.value.split()[1] in {"status","diff","log"}` (`executor.py:27`) raises `IndexError` on the command `"git"`.

**Fix:** wrap the rule call; convert exceptions to `Decision(False, f"allow rule raised: {exc!r}")`. An authority boundary fails **closed**.

### B3 — The module docstring example cannot work as written

`executor.py:22-33`: `run_git(cmd)` reads `cmd.value` from a regex-mode `GitCommandLine`. But regex mode produces a **bare `str`** (see B4), not the model — `c.value` would raise `AttributeError`. The example only holds for json_schema-mode commands. Fix the example alongside B4.

### B4 — The promised bare-string guard was never implemented

`../02-library-wrapper/CONCEPT.md` §4.2 states: *"For the bare-string modes the library validates the returned string against `regex`/`choices` as a guard, then (for `choice`) coerces to the declared `Literal` so the caller still gets a typed value."*

`decoder.py:59-72` does neither — it returns `output_type=str` with no client-side check. Consequences:

1. **Fields on regex/choice-mode `ConstrainedOutput` subclasses are dead.** `GitCommandLine.value: str` (in `constrained.py`'s docstring, `test_constrained.py:26-29`, and the CONCEPT examples) is never populated; callers get a plain `str`. The class is only a spec carrier, which the docs do not say.
2. **No safety net when the server doesn't enforce.** Against any backend that ignores `extra_body` (mis-declared `caps.xgrammar`, a frontier API, the test mock), **completely unconstrained text flows into the executor with zero validation**. XGrammar server-side enforcement is the design, but the library currently *trusts* rather than *verifies* — at odds with the authority story.

**Fix:** implement the guard in a small wrapper around the run result (re.fullmatch for regex; membership for choice; optionally coerce choice back to a declared `Literal` field). ~15 lines; then correct the docstrings/CONCEPT or the behavior to agree.

### B5 — Smaller executor items

- `route_and_execute` runs sync `Policy.action`s (the example is `subprocess.run`) directly on the event loop (`fleet.py:138`), blocking the concurrency `run_batch` is built to deliver. Consider `asyncio.to_thread` for actions, or document the constraint.
- `ExecResult.ok` is never `False` on any code path — dead signal. Either populate it (catch action exceptions → `ok=False`) or drop it.
- Asymmetry to document loudly: `DryRunExecutor` with an **unknown** policy allows by default (`executor.py:124-126`), while `AllowlistExecutor` raises `PolicyError`. Both are defensible; the contrast deserves a table in the docstring.

---

## C. Design & robustness (medium/low)

### Core

- **`run_batch` failure semantics** (`fleet.py:102`): bare `asyncio.gather` — one failed call raises, all sibling results are lost, and still-running tasks' exceptions go unretrieved (asyncio "Task exception was never retrieved" noise). For the batch-throughput API, offer results-as-data (`return_exceptions=True` surfaced per-item) or document the all-or-nothing contract.
- **HTTP client lifecycle** (`backend.py:57-63`): every `build()` with capture or a transport creates an `httpx.AsyncClient` that is **never closed**; there is no `aclose()`/`close()` anywhere in the library. N agents = N connection pools against one vLLM server. Consider one shared client per `Backend` + explicit close (also improves connection reuse for `run_batch`).
- **Explicit `decoder` silently ignored** (`profile.py:78-81`): when `output_type_ref` resolves to a `ConstrainedOutput`, a user-supplied `decoder` is discarded without a word. Conflicting explicit config should raise `ConfigError`.
- **Typing story ends at the wrapper** (`agent.py:21,53`): `AgentResult[OutputT]` is declared generic, but `StructuredAgent` isn't generic and `run()` returns `AgentResult[Any]`. A library whose pitch is typed outputs hands downstream code `Any`. Making `StructuredAgent` generic over the profile's output type (even via `cast` at build) restores the static story.
- **`model_settings` typos pass silently** (`backend.py:96`): `OpenAIChatModelSettings(**settings)` — TypedDicts don't validate at runtime; a misspelled setting name is silently dropped by pydantic-ai. A key-check against `OpenAIChatModelSettings.__annotations__` at build time would catch it.
- **Dead capability knob**: `BackendCaps.server_default_backend` (`backend.py:38`) is declared, documented, and never read. Wire it (per-request `guided_decoding_backend` when false — CONCEPT open question #4) or remove it.
- **`fleet.routing` mutable post-build**: validated only through `build()`; `tests/test_fleet.py:229` itself reassigns it to bypass validation. A `set_routing()` that validates would close the hole.
- **`capture.py:_KNOWN_KEYS`** (`capture.py:15-27`) omits `temperature`, `top_p`, `seed`, `stop`, `n`, penalties, etc. — `extra_body_keys` misreports ordinary sampling params as extra-body keys, making the diagnostic misleading.
- **`output_type_ref` is `importlib` on data** (`profile.py:54`): once profiles load from YAML/JSON (phase 5 plan), a profile file is an arbitrary-import execution vector. Acceptable for a personal library; document it, or gate behind an allowlist of importable module prefixes.

### dual_path

- **Context manager unusable for the main workflow** (`dual_path/runtime.py:111-117`): `__enter__` calls `launch()`, but `register()` refuses after launch — so the natural `with DualPathRuntime(cfg) as rt: rt.register(...)` always raises. Either make `__enter__` not launch (launch lazily on first run) or document the required shape (`rt = DualPathRuntime(cfg); rt.register(...); with rt: ...`).
- **The comparison record itself is not durable** (`dual_path/runner.py:143`): `store.save(record)` runs *outside* any DBOS workflow. A crash after both legs complete but before `save()` loses the record — the durability guarantee covers the legs, not the artifact the whole system exists to produce. Consider making assembly+save a DBOS step keyed by `run_id` (idempotent insert).
- **Connection-per-record** (`dual_path/store.py:55`): `psycopg.connect` per `save()` — expensive at capture volume. A pooled connection (psycopg_pool) or a shared connection with retry is the obvious upgrade.
- **`SetWorkflowID` concurrency caveat** (`dual_path/runner.py:105-114`): two legs run concurrently on one event loop, each wrapping its `await` in `with SetWorkflowID(wid)`. If DBOS implements this with contextvars, task isolation makes it safe; if thread-locals, the IDs can cross legs. **Verify against the pinned DBOS version** before trusting workflow-ID attribution.
- **Name shadowing** (`dual_path/record.py:23`): `DecodeMode = Literal["json_schema"]` redefines and narrows the core `decoder.DecodeMode`. Rename (e.g. `DualPathDecodeMode`) to avoid import confusion.
- Minor: `ComparisonExport.eval_view(by=...)` treats any value other than `"primary_model"` as `profile_version` silently (`store.py:178`) — validate the parameter.

### closed.py *(main)*

Well built for its purpose: loopback-only URL validation, bounded inputs (model-name regex, 4 KB instructions, 16 KB prompt, ≤600 s timeout), strict `response_format: json_schema`, no capture/raw/result escape hatches, detail-free `ClosedBackendError` (deliberate — Lodestar must not persist provider data), and it owns **and closes** its HTTP client (`aclose()`), which the core `Backend` doesn't. It also sidesteps A1 entirely by not using PydanticAI — which is presumably why it exists. Beyond A3 (its tests don't run), two nits:

- `finally: del response` in `run()` is a no-op ritual — scope exit already drops the local. Remove it or state the actual intent.
- `_validated_loopback_url` accepts only literal `127.0.0.1`/`::1` hostnames, rejecting `localhost` — presumably deliberate (DNS-rebinding avoidance); deserves a one-line comment so it isn't "fixed" later.

---

## D. Packaging & docs

1. **`grail` is a dead hard dependency** — zero imports in `src/` or `tests/` (grep-verified). See section F. Remove from `dependencies` and `[tool.uv.sources]`, re-lock.
2. **`[dual-path]` extra omits `psycopg`** (`pyproject.toml:32-34` lists only `dbos>=0.26`), but `dual_path/store.py:16-18` imports psycopg directly and `dual_path/__init__.py:18` requires it. It currently arrives transitively via dbos — one dbos refactor away from breaking. Declare it explicitly.
3. **`pydantic-ai>=1.87.0` unbounded** — already bitten (A1). Constrain to the tested range and document the supported window.
4. **README is a 3-line stub** and the pyproject `description` is template boilerplate ("A Python project for structured agents"). The module docstrings are excellent design prose — a real README could be assembled from them in an hour.
5. **Cut releases.** Version is `0.1.0`, untagged, with a real downstream consumer pinning raw SHAs. Tag via `gitman release` so "re-pin Lodestar to a reviewed newer SHA" becomes "pin v0.2.0". (Note: `gitman` was not on PATH in this repo's devenv when checked — worth fixing too.)
6. `LICENSE` (MIT) now exists on main — matches `license = { text = "MIT" }`; resolved.

---

## E. What's genuinely good

- **Layering discipline is real.** `backend.py` is the sole importer of `pydantic_ai.models.openai` (as designed); cross-module imports are `TYPE_CHECKING`-gated; the dependency direction is clean throughout.
- **Wire-grounded design.** The decode-mode table in `decoder.py:12-17` traces to captured request bodies in VERIFICATION.md, not guesses. The NativeOutput-vs-tool-path insight (PydanticAI's default is the function-calling tool, not `response_format`) is exactly the kind of verified fact a wrapper library should be built on.
- **Coherent error hierarchy.** Every error message names the offending agent/profile and states the remedy. `ConstraintConfigError` at class definition is the right failure point.
- **Explicit-effects philosophy carried through.** Nothing executes implicitly; denials are data in the batch path (`route_and_execute`) and exceptions in the imperative path (`BaseExecutor.run`) — a defensible split, consistently implemented.
- **The test suite mostly earns its 98%.** In-process ASGI mock, wire-shape assertions, an actual in-flight concurrency proof (`test_fleet.py:81-120`), fail-closed executor tests. The two blind spots found (A2's mode gap, A3's silent skip) are the exceptions, not the rule.
- **`closed.py` is a model of scope discipline** — it does one thing, refuses everything else, and validates all its inputs.

---

## F. Grail: remove it

**Question asked:** do we even need grail?
**Answer: no.** Findings from studying `~/Documents/Projects/grail` (v3.0.0):

- **What it is:** a Pydantic-native wrapper around **Monty** — a restricted Python interpreter written in Rust (PyPI `pydantic-monty`, hard-pinned at `==0.0.6`) — for executing `.pym` scripts (a restricted Python dialect with `@external`/`Input()` declarations) under resource limits (`Limits`: max_memory/max_duration/max_recursion, presets STRICT/DEFAULT/PERMISSIVE). The sandbox itself is entirely Monty; grail is the parse→check→codegen→execute→error-mapping tooling around it.
- **Usage here:** **zero imports** anywhere in `src/` or `tests/` (grep-verified). It is a leftover from the archived 01/02 "Grail toolset library" direction, which `../02-library-wrapper/CONCEPT.md` explicitly lists as a non-goal ("No Grail/Monty toolset plane").
- **Cost of keeping it:** a git dependency (breaks pure-PyPI installs and reproducibility guarantees), which transitively pins a **pre-release (0.0.6) native Rust interpreter** — the security-critical component is early-stage, and grail's own history shows pin churn around it. Also drags a Python ≥3.13 floor (moot here, but relevant for consumers).
- **Note on the downstream item:** main already pinned grail to SHA `f804a52` — that satisfies "verify the Grail pin" literally, but the better resolution is deletion.

**Action:** remove `grail` from `dependencies` and `[tool.uv.sources]`, `uv lock`, done. If a future feature wants in-process sandboxed *Python* execution (running model-generated code, not model-generated *commands*), re-add it then as an optional extra — its capability profile (limits + host-provided externals) genuinely fits that use case; nothing today uses it.

---

## G. Fornix: containment layer, yes — executor replacement, no

Findings from studying `~/Documents/Projects/fornix` (v0.4.1, active through 2026-06-18):

**What it is:** a CLI orchestrator that runs a command — especially an agent — in a **disposable btrfs-snapshot copy of a git repo inside a bubblewrap (srt / `anthropic-experimental/sandbox-runtime`) sandbox with cgroup-v2 limits**, captures exactly what changed, and promotes changes back under a lock via jj. Isolation profile: allow-only filesystem writes (sandbox + results dirs), read denylist over sensitive paths (`~/.ssh`, `~/.aws`, repo-main, sibling sandboxes, …), default-deny egress with optional per-domain allowlists, and `MemoryMax`/`CPUQuota`/`RuntimeMaxSec` via `systemd-run --user --scope`.

**Three facts that shape the integration:**

1. **It's a CLI, not a library.** The package exports only `__version__`; the seam is `fornix box [--check CMD] -- <argv>` emitting one JSON `Result` line on stdout, and `fornix apply` as the only writer back to the real repo. (Internal-but-importable: `fornix.box.run_box(settings, item, ...) -> Result`, all Pydantic models.)
2. **It has no authorization concept.** Fornix trusts its input and bounds the blast radius. The Executor answers "*may* this command run?" (allowlist policy); fornix answers "run it where it can't hurt anything, and show me the diff." **They compose; neither replaces the other.**
3. **It is explicitly *not* a hostile-code boundary** (its own README: cooperating-agent threat model — prevent accidental overreach, not adversarial escape) and it is **effectively NixOS-only** with a provisioned substrate (btrfs `/cortex/fornix` volume, srt on PATH, cgroup delegation, jj; `doctor` hard-fails without them; no fallbacks). Also: fornix's in-repo `CODE_REVIEW.md` is **stale** — it reviews a "Phase 2" codebase that was since demolished and rewritten; its headline findings (H1–H3) appear addressed in the current code. Don't inherit its conclusions.

**Recommendation:**

- Keep the core library's Executor exactly as it is *philosophically* (allowlist = authorization), fix the B-section edges, and keep the core free of platform-locked dependencies.
- Add fornix as an **optional, app-layer integration** — e.g. a `FornixExecutor(BaseExecutor)` in an `examples/` or `contrib/` module whose `Policy.action` serializes the validated command model to argv (never a shell string — fornix's `Item.cmd` is validated argv, a good match for the validated-command philosophy) and shells out to `fornix box --check … -- <argv>`, parsing the JSON `Result` into `ExecResult`. That yields the full pipeline: **XGrammar constrains syntax → Pydantic validates the command → allowlist authorizes → fornix contains the execution and captures the diff → `apply` promotes under a lock.**
- The mapping is natural for repo-mutating commands (file edits, git ops — the library's own worked examples). For non-repo commands, fornix's fork/diff/apply machinery is dead weight; plain allowlist execution remains the right path there.
- Net effect on this repo's dependencies: **zero** — grail exits, fornix never enters `pyproject.toml`; the integration is a subprocess boundary owned by the application.

---

## H. Prioritized action list

| # | Action | Why first |
|---|---|---|
| 1 | **Fix A1** — usage compat shim in `agent.py` + `dual_path/runner.py`, regression test per ISSUE.md acceptance criteria, bound pydantic-ai | The downstream Phase A.4 blocker; unblocks the Lodestar re-pin |
| 2 | **Fix A3** — make `test_closed.py` actually run (pytest-asyncio or sync rewrite) | The closed path is trusted downstream but currently unproven |
| 3 | **Drop grail; add psycopg to `[dual-path]`; re-lock; tag a release** | Cheap, removes a git dep + a latent import break; gives Lodestar a version to pin |
| 4 | **Fix A2** — `__pydantic_init_subclass__` + a json_schema class-definition test | The grammar-check feature is currently a no-op for its primary mode |
| 5 | **Harden the executor** — B1 (re-check in execute), B2 (fail-closed allow), B4 (bare-string guard), B3 (fix docstring example) | The safety module should be safe against misuse, not just correct usage |
| 6 | **Fix A4/A5** — extra_body merge; per-run capture correlation + record cap | Correctness of config and of the introspection feature |
| 7 | Work down section C (run_batch semantics, client lifecycle, decoder-conflict error, typing, dual_path durability) | Robustness; none are release blockers individually |
| 8 | README + description + docs sweep (align CONCEPT §4.2 with implementation after item 5) | Docs currently promise behavior the code doesn't have |

Items 1–3 form a small, high-leverage first lane; after landing them, re-pin Lodestar to the tagged release (which also picks up `closed.py` and `LICENSE`, completing the downstream Phase A.4 item).
