# TESTS — the test architecture for constric (structured-agents v3)

The test surface *is* part of the design: it pins both halves of the codec, the `Outcome` algebra,
the authority fail-closed guarantees, the wire shapes, and — new in v3 — the **static type story**
as an executable regression. `ty` 0.0.46 is the sole checker (no pyright/mypy in devenv, SPIKES.md),
so `ty`-level assertions run in CI via `ty check`.

Bar per phase: `devenv shell -- pytest` green, `devenv shell -- ty check src` clean, `devenv shell
-- ruff check src tests` clean.

Test technique carried verbatim from v2 (SALVAGE.md): **in-process `httpx.ASGITransport` mock** + a
tiny OpenAI-shaped app; **wire-shape assertions** against captured bodies; a real **in-flight
concurrency proof**; the **`live` marker gated on `SAV_LIVE=1`**.

---

## T1 — Codec round-trip property tests (the single most valuable new surface)

For each `Constraint`, the codec's two halves are pinned in one place:
- **Valid round-trip:** for valid `x`, `constraint.parse(model_emits(x)) == x`.
  - `Schema(M)`: `parse(valid_M_instance) is that instance` (identity — pydantic-ai already validated).
  - `Regex(p)`: `parse(s) == s` for `s` matching `p`.
  - `Choice(*o)`: `parse(o_i) == o_i` and the **static** type is the literal (see T7).
  - `Grammar(e)`: `parse(s) == s` (passthrough).
- **Out-of-constraint rejection:** text violating the constraint → `parse` raises `_ParseRejected`,
  which `Agent.run` maps to `Violated`. (`Regex`: non-matching string; `Choice`: a non-member.)
- **Property form:** Hypothesis strategies over sample schemas / patterns / option-sets; assert the
  round-trip and the rejection law. This is where the "parse, don't validate" thesis is proven.

Files: `tests/test_constraint_roundtrip.py`.

---

## T2 — Wire-shape assertions (carried from v2 `test_wire_shapes.py`, VERIFICATION.md)

`constraint.wire()` produces **exactly** the empirically-captured bodies:

| constraint | asserted `wire()` |
|---|---|
| `Schema(M)` | `output_type` is `NativeOutput(M, strict=True)`; `extra_body == {}` |
| `Regex(p)` | `output_type is str`; `extra_body == {"structured_outputs": {"regex": p}}` |
| `Choice(*o)` | `output_type is str`; `extra_body == {"structured_outputs": {"choice": list(o)}}` |
| `Grammar(e)` | `output_type is str`; `extra_body == {"structured_outputs": {"grammar": e}}` |

Plus the assembled `/chat/completions` body over the ASGI mock matches the captured shape (the
`response_format` json_schema body for `Schema`; top-level `structured_outputs` for string modes —
the "extra_body lands verbatim at top level" fact). Files: `tests/test_wire_shapes.py`.

---

## T3 — `Outcome` algebra

- `then` short-circuits: `Denied/Violated/Failed .then(f)` returns self unchanged; `Ok(v).then(f) ==
  f(v)`. (Monad left-identity / short-circuit laws.)
- `map`: `Ok(v).map(f) == Ok(f(v))`; non-Ok passes through.
- `unwrap`: `Ok(v).unwrap() == v`; each non-Ok maps to the right raised error (`Failed.error`
  re-raised; `Denied`/`Violated` → `RuntimeError` carrying the reason).
- `value_or`: `Ok(v).value_or(d) == v`; non-Ok → `d`.
- **`run_batch` surfaces per-item failures as `Failed` with no lost siblings** (v2 BatchResult fix,
  generalized): a batch with one failing call returns a `list[Outcome]` where that slot is `Failed`
  and every sibling is its own `Ok`/outcome. No exception escapes; no "Task exception never
  retrieved" noise.

Files: `tests/test_outcome.py`, `tests/test_fleet.py` (batch).

---

## T4 — Authority fail-closed (the safety module, highest bar)

Every v2 executor finding becomes an *assertion that the structural fix holds*:
- **B1 impossible:** there is no `execute` entrypoint that skips `decide` — the `_Executor`
  composition binds `decide` before `run`. Test: a command the authorizer denies produces `Denied`
  and the effector's side effect **never fired** (spy effector asserts zero calls).
- **B2 fail-closed:** an `Allowlist` rule that raises → `Decision(allowed=False)` (not a crash). Test
  the module's own historical footgun (`c.argv[1]` on a one-element argv).
- **B5 meaningful `Effect.ok`:** a `Subprocess`/effector that raises → `Effect(ok=False)` → `Failed`;
  a clean effect → `Effect(ok=True)` → `Ok`.
- **Default-deny:** unknown/unmatched policy → `Denied`.
- **`authorize(a) >> Null()`** performs no effect (dry-run) but authorizes.
- **Composition:** `all_of`/`any_of` truth tables.
- **Fornix:** `FornixEffector` serializes to argv (never a shell string), parses a mocked JSON
  `Result` → `Effect`; absent fornix → `BackendCapabilityError`.

Files: `tests/test_authority.py`, `tests/test_fornix.py`.

---

## T5 — Context axis

- `Context.of("…")` → one `PREFIX` system segment (migration-compatible with a bare instruction str).
- `LinearPrefixProvider.assemble`: messages in segment order; `query` appended as trailing user;
  `NONE` segments dropped from cache but present in messages.
- **Capability degradation is correctness-invariant:** a `CHUNK` segment assembled on
  `caps.chunk_cache=False` yields the **same messages** as `PREFIX` (only cache hints differ);
  `Fidelity.EXACT` forbids blend regardless of annotations.
- **Cache namespace (invariant #10):** `cache_salt` folds `base_model + adapter`; two agents with
  different adapters over identical content get **different** salts (no wrong KV sharing). One agent,
  two contents → different salts. Content alone never determines the salt.

Files: `tests/test_context.py`.

---

## T6 — The `closed` guarantees (runnable — fixes v2 A3, Layer 0+1 only, no pydantic-ai)

Ported from v2 `test_closed.py` but **made to actually execute** (A3: v2's async tests silently
skipped — add `pytest-asyncio` + `asyncio_mode="auto"`, or the repo's sync `asyncio.run` style; the
latter matches every other v2 test file and needs no new dep). Assertions:
- **Loopback rejection:** `https://…`, `http://localhost/…`, off-box host, and any URL with
  credentials/query/fragment → `ValueError` at construction. (`localhost` **must** be rejected — the
  DNS-rebinding guard; keep the "do not add localhost" comment and a test that asserts its rejection.)
- **Bounded-input rejection (exact limits):** `model` outside `[A-Za-z0-9._:-]{1,128}`; instructions
  empty or >4096 bytes; prompt non-str or >16_384 bytes; timeout ≤0 or >600 → `ValueError`.
- **One request, no retry:** invalid JSON output, an `extra="forbid"` violation, and each of
  `408/429/500` → `ClosedBackendError` and **exactly one** request on the mock (`len(app.requests)==1`).
- **Strict json_schema body only:** request has `response_format.type=="json_schema"`,
  `json_schema.strict is True`, name `"closed_output"`, `stream is False`, and **none** of
  `tools/tool_choice/store/user/logprobs/temperature/extra_body`.
- **No retention:** the shared `wire/client.call` under `Retention.NONE` yields `usage is None` and
  `raw is None`; `ClosedBackend.run` returns only a validated `BaseModel`.
- **No escape hatches:** `ClosedBackend` has no `agent`/`run_sync`/`build`/`attach_transport`.
- **Detail-free error:** `str(ClosedBackendError())` leaks no status/body/validation detail;
  `__cause__` retains the original.
- **Runs without pydantic-ai:** the whole `test_closed.py` module imports neither `pydantic_ai` nor
  `structured_agents.agent` (a module-level import assertion / the layering test T8).
- **v2-signature shim:** the compatibility `ClosedBackend(**v2_kwargs)` constructs and runs
  identically (Lodestar-migration proof).

Files: `tests/test_closed.py`.

---

## T7 — Type-level regressions (executable static-story protection)

A `ty`-checked test module (`tests/test_types.py`) using `typing.assert_type`, run by `ty check`:
- `assert_type(Choice("keep","skip"), Constraint[Literal["keep","skip"]])` (S1).
- `assert_type(Schema(FileEditPlan).parse(obj), FileEditPlan)`.
- `assert_type(Regex(r"…"), Constraint[str])`; `assert_type(Grammar(e), Constraint[str])`.
- `agent = backend.build(AgentSpec(constraint=Schema(FileEditPlan), …))`;
  `assert_type(await agent.run(p), Outcome[FileEditPlan])`;
  `assert_type((await agent.run(p)).unwrap(), FileEditPlan)`;
  `assert_type((await agent.run(p)).map(lambda x: x.paths), Outcome[list[str]])` (S2 — the method path).
- `assert_type(fleet.typed("git_ops", GitCommand), Agent[GitCommand])` (DECISION C).
- **Negative:** a `Settings(temperatrue=…)` typo is a `ty` error (typed `Settings`, not a dict) — a
  `# ty: expect-error` marker asserts it *is* rejected (kills v2's silent-typo finding).

> **Documented limitation (RISKS.md R1):** the `match oc: case Ok(value=v)` narrowing is `@Todo` under
> ty 0.0.46 (S2a). T7 therefore asserts the **method-combinator** path (`.unwrap`/`.map`/`.value_or`),
> which *is* typed, and marks the `match` path as runtime-tested (T3) with a static-narrowing xfail to
> flip green when ty matures. The typed promise is delivered via methods; `match` remains a
> runtime-correct convenience.

---

## T8 — Import-layering enforcement (the one-way dependency rule made executable)

A test that parses each module's imports (AST) and asserts the layering:
- **`pydantic_ai.models.openai` is imported by `agent.py` and NO other module** (the single-importer
  invariant — the load-bearing discipline).
- `constraint.py` imports **only** `pydantic_ai.output.NativeOutput` of the pydantic-ai surface (the
  deliberate marker-only concession, DESIGN §constraint / RISKS R2) — and nothing from `models`.
- `wire/`, `constraint.py`, `outcome.py`, `authority.py`, `context.py`, `closed.py` import **no**
  `pydantic_ai.models` at all.
- `closed.py` imports **no** `pydantic_ai` and **no** `agent`/`fleet` (pydantic-ai-free path,
  DECISION I.2).
- No module imports a strictly-higher layer (Layer N never imports Layer >N).
- `config.py` is the **only** module doing `importlib`/`import_module` on data (grep/AST test — the
  localized import vector, DECISION K).

Files: `tests/test_layering.py`.

---

## T9 — Observe (behind `[observe]`)

- An `Observer` receives each pipeline stage's `Outcome`; the `DualPathObserver` records a
  `(primary Ok[T], reference Ok[T])` pair.
- **Idempotency:** persisting the same `run_id` twice inserts once (the DBOS-step-keyed-on-run_id fix
  for v2's "record itself not durable" — §C).
- Absent the extra → a clean `ImportError` naming `[observe]`.
- `SetWorkflowID` isolation verified against the pinned DBOS (RISKS.md R5) before trusting
  workflow-id attribution.

Files: `tests/observe/…`.

---

## v2 → v3 test-map (salvage ledger for tests; see SALVAGE.md for the full ledger)

| v2 test file | v3 home | disposition |
|---|---|---|
| `test_constrained.py` | `test_constraint_roundtrip.py` + `test_types.py` | **rewritten** — no `ConstrainedOutput` subclass; codec round-trip + static asserts instead |
| `test_decoder.py` | folded into `test_wire_shapes.py` | **merged** — `DecoderSpec` is gone; the mode→wire table is now `Constraint.wire()` |
| `test_wire_shapes.py` | `test_wire_shapes.py` | **carried** — same captured bodies (VERIFICATION.md) |
| `test_capture.py` | `test_agent.py` (capture-in-`Ok.wire`) | **rewritten** — delivery moves from ContextVar to `Ok.wire` (DECISION N); concurrency proof kept |
| `test_backend.py` | `test_agent.py` | **rewritten** — `Backend.build` + gating + merge + typed `Settings` |
| `test_profile.py` | `test_agent.py` + `test_config.py` | **split** — typed `AgentSpec` (code) vs `spec_from_config` (edge) |
| `test_agent.py` | `test_agent.py` | **rewritten** — `run -> Outcome[T]`; A1 property access (S3) |
| `test_executor.py` | `test_authority.py` | **rewritten** — Authorizer×Effector; B1/B2/B5 as structural assertions |
| `test_fleet.py` | `test_fleet.py` | **carried-in-spirit** — `Outcome` list, `Router` coverage, in-flight concurrency proof (the concurrency test is a verbatim-worthy technique) |
| `test_closed.py` | `test_closed.py` | **carried + made-to-run** (A3 fix) |
| `test_dual_path_*.py` | `tests/observe/…` | **rewritten** — observers over the spine |
| `test_live.py` | `test_live.py` | **carried** — `SAV_LIVE=1` gated; points at tower |
| `conftest.py` (ASGI mock) | `conftest.py` | **carried verbatim** — the in-process OpenAI-shaped app |
