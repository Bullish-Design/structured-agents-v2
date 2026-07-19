# PHASES — the build plan for the durable agent plane

Supersedes the pre-reframe `PHASES.md`. Sequenced by **value-per-coherence**; every phase leaves a
**green, self-contained, demonstrable** state:

- `devenv shell -- pytest` green **on SQLite** (no Postgres — spike 1),
- `devenv shell -- ty check src` clean,
- `devenv shell -- ruff check src tests` clean,
- a runnable demo.

Each phase is a place you could *stop* with a coherent artifact. Grounding: all foundational
assumptions are spiked & confirmed (`DURABLE_PLANE.md` §Foundational spikes; files in
`REVIEW_SPIKES/`). Test seams carried from v2 (SALVAGE.md): in-process `httpx` mock for wire-shape
assertions, `TestModel` for deterministic no-network generation, the `live` marker gated on
`SAV_LIVE=1`.

**Standing conventions (locked by the spikes):**
- **Async-first.** DBOS sync entrypoints raise inside a running loop → the whole public surface is
  `async def`. Tests use `pytest-asyncio` (`asyncio_mode="auto"`) — and it must *actually run* (v2 A3
  lesson: assert the async tests execute, don't let them silently skip).
- **DBOS singleton lifecycle.** One `launch()` per test process; workflow/step decorators register at
  import *before* launch; `destroy()` on teardown. The `conftest` owns this (Phase 2).
- **Determinism rule.** Workflow bodies are deterministic between steps; LLM/effect calls are steps.

---

## Phase 0 — Repo genesis + `errors.py`

**Scope.** New repo (**name HELD**, DECISION H — Phase 0 blocked until the user picks it; import
package is `structured_agents` regardless). `copyroom new` from `template-py`; wire `repoman`; devenv
(Python 3.13, hatchling+uv); plain git; re-index the fleet. Core deps: `pydantic`,
`pydantic-ai-slim[openai] >=2.11,<3`, `dbos >=2.23`, `httpx`. Dev: `pytest`, `pytest-asyncio`, `ty`,
`ruff`. Extras: `[grammar-check]` (xgrammar), `[fornix]` (no dep — names the seam). **No `[observe]`
extra** — observation is core now (durable store). Production Postgres is a runtime URL, not a dep.

**Modules landed.** Empty package + `errors.py` (the layer-less exception tree: `ConstricError`,
`ConfigError`, `ConstraintConfigError`, `ConstraintCompileError`, `BackendCapabilityError`,
`ConstraintViolation`, `AuthorityError`; error-message discipline carried from v2).

**Acceptance.** `pytest` (0 tests) green; `ty`+`ruff` clean; `pip install .` resolves the four core
deps; demo: `python -c "import structured_agents; from structured_agents.errors import ConstricError"`.

---

## Phase 1 — The constraint codec (pure; the phase that pays)

**Scope.** The linchpin — DBOS-free, server-free. `constraint.py`: the `Constraint[T]` Protocol +
`Schema/Regex/Choice/Grammar` constructors + `WireSpec` + `parse` + `check`. Wire bodies are the
**verified-verbatim** table (`REVIEW_SPIKES/bodies.json` — captured on 2.11, no drift). `Schema.parse`
= identity (rich path is the only path — honest now); string modes guard in `parse` and raise
`ConstraintViolation`. `Choice[S: str]` infers the literal (spike S1).

**Modules landed.** `constraint.py`.

**Test seam.** Pure Python — no network, no DBOS.

**Acceptance.**
- **T1 codec round-trip** (Hypothesis): `parse(model_emits(x)) == x` for valid `x`; out-of-constraint
  text raises `ConstraintViolation`. Per mode: `Schema` identity on a validated instance; `Regex`
  `fullmatch`; `Choice` membership → the literal; `Grammar` passthrough.
- **T2 wire-shape** (verbatim vs `bodies.json`): `Schema.wire().output_type is NativeOutput(M)`,
  `extra_body=={}`; string modes → `output_type is str`, `extra_body=={"structured_outputs":{…}}`.
- **T7 ty regressions**: `assert_type(Choice("a","b"), Constraint[Literal["a","b"]])`;
  `assert_type(Schema(M).parse(obj), M)`; `Regex`/`Grammar` → `Constraint[str]`.
- `ty`+`ruff` clean; imports no DBOS.
- **Demo:** build each constraint, print `.wire()`, round-trip `.parse()` on sample text — pure, no
  server, no DB.

> Phase 1 delivers most of the typed-decoding elegance with zero infrastructure. Natural checkpoint.

---

## Phase 2 — The durable agent + the DBOS/SQLite test harness

**Scope.** Where durability and pydantic-ai enter. `agent.py`: `Settings`, `AgentSpec[T]`, `Backend`
(sole pydantic-ai importer; one shared httpx client; `_gate` capability check → `BackendCapabilityError`;
`extra_body` merge, decoder keys win), `Agent[T]` wrapping a `DBOSAgent` — `run(prompt) -> T`, durable
by construction (spike 2). Establish the **`conftest` DBOS harness**: launch on a temp SQLite system db
(`use_listen_notify=False`), session-scoped, `destroy()` on teardown; the singleton/register-before-launch
discipline.

**Modules landed.** `agent.py`, `tests/conftest.py` (DBOS/SQLite harness + `TestModel` + httpx mock).

**Test seam.** `TestModel` for deterministic generation (no network); a mock-transport `OpenAIChatModel`
for wire-shape-over-the-request assertions; DBOS on SQLite for durability.

**Acceptance.**
- **Type flow (T7):** `assert_type(await backend.build(AgentSpec(constraint=Schema(FileEditPlan),…)).run(p), FileEditPlan)` — no cast.
- **Durability (spike 2):** a top-level `Agent.run` **auto-creates a root workflow** (assert a
  `<name>.run` SUCCESS row via `list_workflows_async`); the same `Agent.run` **nested** in a user
  `@DBOS.workflow` appears as a **child** (`<parent>-N`).
- **A1 regression:** a valid completion returns `T` without raising; `raw.usage` accessed as a
  property (S3). **A4:** user `extra_body` merged, decoder keys win. **Settings typo** is a `ty` error
  (`# ty: expect-error`).
- **Wire-shape over the request** matches `bodies.json` for all four modes through the real
  `Backend`→pydantic-ai path (not just `constraint.wire()` in isolation).
- **Import-layering (T8):** `agent.py` is the **sole** importer of `pydantic_ai.models.openai`.
- `ty`+`ruff` clean; full suite green **on SQLite**.
- **Demo:** `backend.build(spec).run(prompt)` with a `TestModel` → typed value, and its durable
  workflow row shown via the observability read (previewing Phase 5).

---

## Phase 3 — Authority + exactly-once effects (+ fornix)

**Scope.** `authority.py`: `Decision` (data), `Authorizer[C]` / `Allowlist` (default-deny, wraps each
rule → `Decision(False)` once), `Effector[C]` implemented as a **`@DBOS.step`** (exactly-once + retry),
`Subprocess`, `execute(authorizer, effector, command, *, key=None)` (`key`→`SetWorkflowID` = business
exactly-once, spike 5), `Denied`. `integrations/fornix.py`: `FornixEffector` (validated argv →
`fornix box --check … -- <argv>`, JSON `Result` → effect; guarded import → `BackendCapabilityError`).

**Modules landed.** `authority.py`, `integrations/fornix.py`.

**Acceptance.**
- **B1 structural:** no path runs an effect without a decision — `execute` binds `decide` before the
  step; a denied command → `Denied`, spy effector asserts **zero** calls. (Document the residual
  hazard: `Effector.run` is directly callable; blessed path is `execute`.)
- **B2 fail-closed:** a raising allow-rule → `Decision(allowed=False)`, not a crash (the historical
  `argv[1]` footgun).
- **Exactly-once (spike 5):** `execute(…, key="order-42")` run twice fires the side-effecting step
  **exactly once** (counter==1); a different key fires independently.
- **B5:** an `Effector` step that raises → the workflow surfaces the failure (DBOS retry policy
  exhausted → error), a clean one → the effect output; the success signal is meaningful (no dead
  `ok` bool — `Effect` dropped).
- **Fornix:** serializes to argv (never a shell string), parses a mocked JSON `Result`; absent fornix
  → `BackendCapabilityError`.
- `ty`+`ruff` clean; suite green on SQLite. **Demo:** a user-authored `@DBOS.workflow` doing
  generate→`execute(allowlist, Subprocess())`, showing an allowed exactly-once effect and a denied
  command (as `Denied` data), both durable.

---

## Phase 4 — Human-in-the-loop durable approval  · **release point**

**Scope.** `approval.py`: `Approval.request(command, *, to, timeout) -> Decision` (durable pause via
`recv_async`, `set_event_async` to publish the pending command; timeout → `Decision(False,"timeout")`),
`ApprovalClient.approve/deny/pending` (out-of-process approver via `send_async`/status query).

**Modules landed.** `approval.py`.

**Acceptance (spike 4).**
- A workflow that calls `approvals.request` is durably **PENDING** until a message arrives (assert
  status via `get_status`); an `ApprovalClient.approve` resumes it → SUCCESS; `deny` → `Denied`;
  timeout → denied.
- The pending command is inspectable via `get_event` (`ApprovalClient.pending()` lists what awaits
  approval).
- Restart-survival documented: the `recv` state persists in the system db (recovery re-enters).
- `ty`+`ruff` clean; suite green. **Demo:** a gated pipeline (`generate → approvals.request(to="ops")
  → effect`) that blocks, is approved out-of-band, and completes exactly-once.

> After Phase 4 the safety-critical core is whole: autonomous *and* human-gated durable pipelines,
> exactly-once, all composable by the user. **First release (v1.0.0-rc).**

---

## Phase 5 — Plane services: queues · schedules · observability · eval  · **release point**

**Scope.** `plane.py`: `launch(database_url=None)` (SQLite default, Postgres URL for prod); `Queue`
(durable concurrency / batch / rate-limit over `Agent.run`, `submit(…, key=None)`); `schedule(cron)`
(durable cron over a user workflow); observability (`workflows/status/fork/cancel`, thin async wrappers
over DBOS); `compare(primary, reference, prompt)` (dual-path eval fan-out — both legs durable, the
store *is* the record; subsumes v2 `dual_path/`).

**Modules landed.** `plane.py`.

**Acceptance.**
- **Queue:** N submissions run under a concurrency cap; batch returns handles; per-item failure is
  isolated (no lost siblings). Confirm the 2.23 rate-limit kwarg while wiring.
- **Schedule:** a `@schedule("* * * * *")` workflow fires on the durable timer (fast-poll in test via
  `scheduler_polling_interval_sec`). Confirm the 2.23 scheduled-workflow decorator surface while wiring.
- **Observability:** `workflows(status="PENDING")` lists a blocked approval; `fork(wid, from_step=k)`
  replays from a step; `cancel(wid)` cancels.
- **compare:** runs both legs durably and records the `(primary, reference)` pair; re-running the same
  `key` does not double-insert (idempotent — the durability guarantee, not a bespoke store).
- `ty`+`ruff` clean; suite green on SQLite. **Demo:** schedule a rate-limited batch sweep; list/fork a
  run; a dual-path `compare` producing an SFT-export line.

> After Phase 5 the **full durable plane** is live: the toolkit the user composes arbitrary durable
> agent workflows from. **v1.0.0.**

---

## Phase 6 — Config edge (`config.py`)

**Scope.** The serialization boundary (DECISION K): `constraint_from_config`/`spec_from_config` with the
`allow_modules` allowlist + per-seam registry + entry-point discovery; `Constraint.to_config` round-trip.
The *one* place strings become code, gated.

**Modules landed.** `config.py`.

**Acceptance.** Serde round-trip (`from_config(c.to_config())` ≡ `c`); a `ref` outside `allow_modules`
→ `ConfigError`; `config.py` is the **only** module doing `importlib` on data (AST/grep test);
`ty`+`ruff` clean. **Demo:** load an `AgentSpec` from a dict with an explicit allowlist; show a
disallowed module rejected.

> Optional/deferrable: if every early consumer is code-first (`Schema(FileEditPlan)` in Python), Phase
> 6 can slip after live cutover without blocking anything.

---

## Phase 7 — Live cutover verification (tower vLLM)

**Scope.** Point a `Backend` at `http://tower:8000/v1`; run `deploy/vllm/verify.sh` semantics against
the plane: json_schema, xgrammar regex/choice/grammar, per-agent LoRA — now inside a **durable
workflow** end-to-end. No new library code (deploy scripts carried from v2, SALVAGE.md).

**Acceptance.** `SAV_LIVE=1`-gated tests pass on tower (health→models→json_schema→xgrammar→lora); each
constraint's `.wire()` is accepted and enforced server-side; a durable pipeline (generate→authorize→
effect) round-trips through real vLLM with exactly-once. **Demo:** `deploy/vllm/verify.sh` (adapted)
green against tower; a real durable agent run recovers after a mid-run kill (crash-recovery proof).

---

## Sequencing rationale

1. **Codec first (P1)** — pure, infra-free, the concept everything takes as input; delivering it green
   *is* the elegance win.
2. **Durable agent + harness (P2)** — DBOS + pydantic-ai enter together behind the SQLite harness;
   proves always-on durability at the smallest unit before anything composes on it.
3. **Authority (P3)** — the exactly-once effect is the plane's payoff; land it right after the agent so
   "decisions are data, effects are exactly-once" is real before pipelines tempt shortcuts.
4. **Approval (P4)** — the HITL gate completes the safety story; natural release point.
5. **Plane services (P5)** — queues/schedules/observability/eval turn the primitives into the full
   plane; the second release point.
6. **Config (P6)** — the string→code edge, after every code-first path is proven; deferrable.
7. **Live (P7)** — cooperation + crash-recovery proof against real vLLM; no new code.

Optional, not core: a `Router`/`Fleet` helper is a *user-authored pattern* (a `Choice` agent
dispatching to specialists) in the toolkit model — add a thin helper later if it earns its place;
it is not a required phase.

## What's not here (subsumed or dropped vs the old plan)
- `wire/` layer — **gone** (existed to share with `closed`; no `closed`). Rich `Backend` uses
  pydantic-ai + httpx directly; the capture hook lives in `agent.py`.
- `outcome.py` — **gone** (`run -> T` + exceptions).
- `context.py` — **deferred** (`instructions: str`; add the cache axis with a chunk-cache consumer).
- `closed.py` — **dropped** (downstream's concern).
- `observe/` as a subsystem — **subsumed** by the durable store + `compare()` (Phase 5).
