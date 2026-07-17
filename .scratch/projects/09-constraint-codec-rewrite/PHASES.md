# PHASES — the build plan for constric (structured-agents v3)

Sequencing is ordered by **value-per-coherence** (concept's closing note), and every phase must
leave a **green, self-contained, demonstrable** state: `pytest` green, `ty check src` clean, `ruff
check` clean, plus a runnable demonstration. Difficulty does not gate design (kickoff), but each
phase is a place you could *stop* and still have a coherent artifact.

The spine insight (concept): **Phase 1 alone delivers most of the elegance**, because the
constraint-as-codec is what the whole library is secretly organized around.

Legend per phase: **Scope · Modules landed · Acceptance (tests + ty + demo).**

---

## Phase 0 — Repo genesis (DECISION H)

**Scope.** New repository `constric` (or keep `structured-agents` — *pending user confirm*, DECISION
H). `copyroom new` from `template-py`; wire `repoman`; devenv (Python 3.13, hatchling+uv); plain git;
re-index the fleet. Package `structured_agents/`. Extras skeleton per DECISION I/I.2 (`[agent]`,
`[grammar-check]`, `[observe]`, `[fornix]`); lean core = `pydantic + httpx` only.

**Modules landed.** Empty package + `errors.py` (the layer-less tree).

**Acceptance.**
- `devenv shell -- pytest` (0 tests) green; `ty check src` + `ruff` clean.
- `pip install .` pulls **only** `pydantic + httpx` (no pydantic-ai) — proves the `[agent]`-extra
  split (DECISION I.2). `pip install .[agent]` pulls pydantic-ai-slim[openai] pinned `>=2.11,<3`.
- Demo: `python -c "import structured_agents; from structured_agents.errors import ConstricError"`.

---

## Phase 1 — The codec spine: `wire/` + `constraint.py` (+ `outcome.py` minimal)

**Scope.** The linchpin. Land the pydantic-ai-free `wire/` primitives and the `Constraint[T]` codec
with all four constructors and the verified wire/parse table. Land `Outcome` *enough* to be the parse
result (`Ok`/`Violated` + the base-class combinators), but not the full pipeline yet. **This is the
phase that pays for the rewrite.**

**Modules landed.** `wire/{transport,request,client,errors}.py`, `constraint.py`, `outcome.py`.

**Acceptance.**
- **Codec round-trip property tests** (TESTS.md T1) — for each constraint, `parse(model_output(x)) ==
  x` for valid `x`; out-of-constraint text raises → `Violated`. The single most valuable surface.
- **Wire-shape assertions** (T2) — `Schema/Regex/Choice/Grammar .wire()` produce **exactly** the
  captured `response_format`/`extra_body structured_outputs` bodies from VERIFICATION.md.
- **ty regressions** (T7) — `assert_type(Choice("a","b"), Constraint[Literal["a","b"]])` (S1);
  `assert_type(Schema(M).parse(x), M)`; the `Outcome` method-combinator types (S2).
- `ty check src` + `ruff` clean; the whole layer imports **no** pydantic-ai (import-layering test T8
  partial).
- **Demo:** a script that builds each constraint, prints `.wire()`, and round-trips `.parse()` on
  sample text — no server, no pydantic-ai.

---

## Phase 2 — Authority: `authority.py` + `integrations/fornix.py`

**Scope.** `Authorizer × Effector`, `Allowlist` (default-deny, structural fail-closed), `all_of`/
`any_of`, effectors (`Null`, `Subprocess`), the `authorize(a) >> effector` composition, and the
`FornixEffector` plugin. Depends only on `outcome`+`errors`, so it lands before the agent layer and is
demonstrable standalone.

**Modules landed.** `authority.py`, `integrations/fornix.py`.

**Acceptance.**
- **Authority tests** (T4): `Allowlist.decide` is total (a raising rule → `Decision(False)` — B2);
  `authorize(a) >> Null()` performs no effect; `Effect.ok` false iff the effector raised (B5); there
  is no `execute` that can skip `decide` (B1 — the composition binds them).
- **Fornix**: `FornixEffector` serializes a validated command to argv and parses a JSON `Result`
  (mock the subprocess); absent fornix → `BackendCapabilityError`.
- `ty` + `ruff` clean.
- **Demo:** `authorize(Allowlist({"git_safe": …})) >> Null()` on a sample command → `Outcome[Effect]`,
  showing an allowed dry-run and a denied command both as **data** (no exceptions).

---

## Phase 3 — Config/code split: `config.py`

**Scope.** The serialization edge (DECISION K): `constraint_from_config`/`spec_from_config` with the
`allow_modules` allowlist, the per-seam registry, and entry-point discovery. Round-trip `Constraint`
serde (`to_config`/`from_config`).

**Modules landed.** `config.py`, `constraint.to_config` completion.

**Acceptance.**
- **Serde round-trip**: every constraint `from_config(c.to_config()) == c` (behaviorally).
- **Allowlist**: a `ref` outside `allow_modules` → `ConfigError`; no `importlib` on data anywhere else
  (grep test).
- `ty` + `ruff` clean.
- **Demo:** load an `AgentSpec` from a dict `{"constraint":{"kind":"schema","ref":"demo:Plan"}, …}`
  with an explicit allowlist; show a disallowed module rejected.

---

## Phase 4 — The `closed` preset over shared wire (Lodestar path)

**Scope.** `closed_backend()` + `ClosedBackend` as a thin preset over `wire/`+`constraint` (DECISION
L), preserving every guarantee **byte-for-byte** (invariant #8), plus the v2-signature compatibility
shim (DECISION H). **No pydantic-ai** (DECISION I.2 — closed installs without `[agent]`).

**Modules landed.** `closed.py`.

**Acceptance.**
- **Closed guarantees** (T6, ported from v2 `test_closed.py` but *made to run* — A3 fix, sync style or
  pytest-asyncio): loopback rejection (`https`/`localhost`/off-box → `ValueError`); bounded-input
  rejection (exact limits); one-request-no-retry (invalid output & 408/429/500 each → one request,
  `ClosedBackendError`); strict json_schema body with **no** tools/tool_choice/store/user/logprobs;
  no-retention (`WireResult.usage/raw is None`); no escape hatches (`ClosedBackend` lacks
  `agent`/`run_sync`/`build`/`attach_transport`).
- **Import-layering** (T8): `closed.py` imports neither pydantic-ai nor `agent`/`fleet`.
- `pip install .` (no `[agent]`) then `import structured_agents.closed` **works** — proves the
  pydantic-ai-free install.
- **Demo:** `closed_backend(...)` against the in-process ASGI mock → one request, validated model,
  detail-free error on a forced 500. **Lodestar-migration demo:** the v2-signature shim constructs and
  runs identically.

> After Phase 4, the plan has re-delivered v2's most safety-critical surface on the shared
> primitives, with the A3 (tests don't run) and duplication smells gone. A natural release point.

---

## Phase 5 — Context axis: `context.py`

**Scope.** The neutral per-segment `Context`/`Segment`/`Reuse`/`Fidelity` model + `LinearPrefixProvider`
(DECISION G). `CHUNK` machinery latent (id slot, enum variant, cap flag) but only `PREFIX`/`NONE`
assembled. Cache-namespace bookkeeping (`hash+base_model+adapter`) implemented even though the default
provider is prefix-only (so the invariant is enforced from day one).

**Modules landed.** `context.py`.

**Acceptance.**
- **Context tests** (T5): `Context.of("…")` → one `PREFIX` system segment; `LinearPrefixProvider`
  emits messages in order, appends `query` as trailing user; a `CHUNK` segment on a `chunk_cache=False`
  backend **degrades** to `PREFIX`/`NONE` (correctness invariant — same output modulo cache hints);
  `cache_salt` folds `base_model+adapter` (two adapters → different salts — invariant #10).
- `ty` + `ruff` clean.
- **Demo:** assemble a mixed `Context` (system PREFIX + few-shot PREFIX + query NONE) and print the
  wire messages + (empty, on plain vLLM) cache hints.

---

## Phase 6 — The agent layer: `agent.py` (+ `fleet.py`)

**Scope.** `AgentSpec[T]`, `Backend` (sole pydantic-ai importer, one shared client, capability gating,
`AdapterProvider`), `Agent[T]` (`run -> Outcome[T]`, capture in `Ok.wire`), then `Fleet`/`Router` with
`execute` as an `Outcome.then` chain. This is where pydantic-ai enters and the full `Outcome` spine
(all four variants) is exercised end to end.

**Modules landed.** `agent.py`, `fleet.py`.

**Acceptance.**
- **Type flow** (T7): `assert_type((await backend.build(AgentSpec(constraint=Schema(FileEditPlan),
  …)).run(p)), Outcome[FileEditPlan])` — no cast (S2).
- **A1 regression**: valid json_schema completion over ASGI returns `Ok.value` **without raising**,
  with `capture=False` **and** `capture=True`; `raw.usage` accessed as a property (S3).
- **A4**: user `extra_body` merged, decoder keys win. **A5**: capture correlation — two overlapping
  runs of one agent in `run_batch` each get **their own** `Ok.wire` (concurrency proof, ported from v2
  `test_fleet.py`). **Settings typo**: a bad field is a *type* error (typed `Settings`), demonstrated
  in a `ty`-checked test.
- **Fleet**: `run_batch -> list[Outcome[T]]` surfaces per-item `Failed` with no lost siblings (R);
  `Router` `Literal`-coverage validated at build; `execute` = route∘generate∘authorize∘effect as one
  bind chain returning `Outcome[Effect]`; `aclose()` closes the one shared client.
- **Import-layering** (T8): `agent.py` is the **sole** importer of `pydantic_ai.models.openai`.
- `ty check src` + `ruff` clean; **full suite green**.
- **Demo:** the concept §15 worked example end-to-end against the ASGI mock — single typed agent
  (`fleet.typed("git_ops", GitCommand)`) and the full autonomous pipeline
  (`fleet.execute(msg, authorize(Allowlist(...)) >> FornixEffector())`), all outcomes as data.

> After Phase 6, the entire core thesis (three axes composing, authority pipeline, typed end-to-end)
> is live and green. **The primary release** (v3.0.0).

---

## Phase 7 — Live cutover verification (tower vLLM)

**Scope.** Point a `Backend` at `http://tower:8000/v1` and run `deploy/vllm/verify.sh` semantics
against v3: json_schema, xgrammar regex/choice/grammar, per-agent LoRA. No new library code — this
proves the wire cooperation is unchanged from v2's verified cutover.

**Modules landed.** none (deploy scripts carried from v2, SALVAGE.md).

**Acceptance.**
- `SAV_LIVE=1`-gated live tests pass on tower (health → models → json_schema → xgrammar → lora),
  mirroring v2. Each constraint's `.wire()` is accepted and enforced server-side; a `Regex`/`Choice`
  round-trips through real XGrammar.
- **Demo:** `deploy/vllm/verify.sh` (adapted) green against tower.

---

## Phase 8 — Observe axis: `observe/` ([observe] extra)

**Scope.** Pipeline `Observer`s; `DualPathObserver` fanning to a reference `Agent[T]` and persisting
the `(Ok[T], Ok[T])` pair in a DBOS step keyed on `run_id` (idempotent — fixes v2 durability gap).
Carry the jsonb store + SFT export. Behind `[observe]` (dbos + **psycopg declared**, D2).

**Modules landed.** `observe/{__init__,runtime,store,record,compare}.py`.

**Acceptance.**
- **Observe tests**: an `Observer` receives each stage's `Outcome`; the dual-path observer records a
  pair; re-running the same `run_id` **does not double-insert** (idempotency). `SetWorkflowID`
  contextvar-vs-threadlocal behavior verified against the pinned DBOS (RISKS.md R5) before trusting
  workflow-id attribution.
- Import guarded: absent `[observe]` → a clean `ImportError` naming the extra.
- `ty` + `ruff` clean; full suite green.
- **Demo:** run a message through an `Agent[T]` with a `DualPathObserver` attached; show the persisted
  comparison record and an SFT-export line.

---

## Phase sequencing rationale (why this order)

1. **Codec first (P1)** — the concept the library is organized around; everything else takes a
   `Constraint[T]` as input. Delivering it green first *is* the elegance win.
2. **Authority (P2)** before agent — it depends only on `outcome`, is safety-critical, and is fully
   demonstrable without a model. Landing it early keeps the "decisions are data" spine honest before
   the pipeline exists to tempt shortcuts.
3. **Config (P3)** before closed/agent — both consume specs; the allowlist must exist before any
   string→code path ships.
4. **Closed (P4)** before the agent layer — it is the Lodestar-critical, pydantic-ai-free path; landing
   it on the shared `wire/` primitives *proves* the extraction (DECISION L) before the rich path adds
   pydantic-ai. Also the earliest point Lodestar could migrate.
5. **Context (P5)** before agent — `AgentSpec` carries a `Context`; the assembler must exist for
   `build`.
6. **Agent+Fleet (P6)** — the top of the stack; introduces pydantic-ai last, so every lower layer is
   proven framework-free first. The full `Outcome` spine and all v2 findings' structural fixes are
   exercised here.
7. **Live (P7)** — cooperation proof, no code.
8. **Observe (P8)** — optional, additive, behind an extra; last because it observes a finished spine.

Each of P1–P6 is independently green and demonstrable; P4 and P6 are natural release points (closed-
only, then full).
