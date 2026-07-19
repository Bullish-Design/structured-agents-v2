# Build Kickoff — structured-agents v3, the durable agent plane

**Use this to start a clean *build* session.** The design is finished and empirically grounded; your
job is to **build it**, phase by phase, each phase left green and demonstrable. This document is
self-contained enough to start from, but the authoritative spec lives in the docs it points to — read
them first.

---

## Mission

Rebuild `structured-agents` (this repo) from scratch as a **durable-primitives toolkit for building
crash-recoverable, exactly-once, observable constrained-agent workflows on DBOS + vLLM/XGrammar.**

The organizing unit is a **durable DBOS workflow**; constrained generation, authorization, and effects
are durable *steps*. The library ships the durable *primitives*; **the user authors the workflows**
(toolkit, not framework). Every design decision below is settled and spike-verified — you are
executing, not re-planning. (You *may* overturn a decision, but only with hard new evidence, and say so
loudly.)

---

## Read first (in this order — they are the spec)

All in `../09-constraint-codec-rewrite/`:

1. **`DURABLE_PLANE.md`** — the north-star concept: thesis, the plane, the toolkit, the module map,
   dependency/infra implications, and the "Foundational spikes — ALL CONFIRMED" section (the ground
   truth your build rests on).
2. **`DESIGN_DURABLE.md`** — the module-by-module spec with **real, ty-checkable signatures**,
   responsibilities, one-way layering, invariants, and impl notes. **This is what you implement.**
3. **`PHASES_DURABLE.md`** — the 8-phase build order (P0–P7), each with scope, modules landed, test
   seam, and falsifiable acceptance criteria. **This is your task list.**
4. **`REVIEW_SPIKES/`** — the from-scratch spike reproductions that ground every claim (`spike{1..5}_*.py`
   for DBOS; `capture.py`/`bodies.json` for the 2.11 wire shapes; `s1_*`/`s2*`/`s3_*` for the type
   surface). Re-run any you want to re-verify; do not re-derive them from doubt.

Supporting/provenance (skim as needed, not authoritative for the *durable* design — they predate the
reframe): `00-PLAN.md`, `DECISIONS.md`, `DESIGN.md`, `PHASES.md`, `SIMPLIFIED.md`, `REVIEW.md`, and the
v2 review at `../07-library-code-review/CODE_REVIEW.md`. Wire-shape provenance:
`../02-library-wrapper/VERIFICATION.md`.

---

## What you're building (the shape, in one screen)

```
DBOS durable plane  (Postgres in prod; SQLite for dev/test — spike 1)
  Agent[T].run(prompt) -> T        constrained generation via pydantic-ai DBOSAgent (durable steps)
  Authorizer.decide -> Decision    policy check, returns DATA (never raises for a denial)
  Effector.run -> R                the side effect: a @DBOS.step, EXACTLY-ONCE + retry
  Approval.request -> Decision     durable pause for a human (DBOS recv); survives restart
  Queue.submit / schedule / observability / compare    the plane services
  → the user composes these inside their own @DBOS.workflow functions
```

Modules (target): `errors.py`, `constraint.py`, `agent.py`, `authority.py`, `approval.py`, `plane.py`,
`config.py`, `integrations/fornix.py`. Layering (one-way, AST-enforced by test T8):
`errors/constraint (L0) → agent, authority (L1) → approval, plane, config, fornix (L2)`. `authority` is
deliberately independent of `agent`/`constraint`.

---

## Decisions already made — do NOT relitigate (unless hard evidence)

- **Result model:** `run -> T`, **raise** on failure. **No `Outcome` sum-type spine, no combinators.**
  `Denied` is the one runtime decision-as-data (returned by `execute`, not raised). `ConstraintViolation`
  is raised by `Agent.run` when the backend didn't enforce.
- **Durability is always-on.** Every `Agent.run` is durable (its own root workflow at top level; a child
  workflow when nested). DBOS is a **core dependency**, not an extra.
- **Toolkit, not framework.** The library owns **no** canonical pipeline; users author `@DBOS.workflow`s.
- **Input is `instructions: str`.** The `Context`/`Reuse`/cache axis is **deferred** (no consumer yet).
- **`closed` is dropped** (downstream's concern). No `wire/` layer, no `outcome.py`, no `context.py`,
  no `closed.py`. `observe`/`dual_path` as a subsystem is **subsumed** by the durable store + `compare()`.
- **Full DBOS surface is first-class in v1:** queues, scheduled agents, messaging + human-in-the-loop
  approval, observability (query/status/fork/replay).
- **Authority is core** (the exactly-once effect is the payoff).
- **Name kept: `structured-agents`; package `structured_agents`; repo rebuilt IN PLACE.** The `v0.2.0`
  git tag preserves the frozen artifact, so clearing `src/` is safe.
- **The public API is async-first.** No sync mirror in v1 (DBOS sync entrypoints raise inside a loop).

---

## Empirically confirmed — build on these, don't re-derive (see REVIEW_SPIKES/)

- **SQLite is first-class for dev/test.** `DBOSConfig(name=…, system_database_url="sqlite:///<abs>",
  use_listen_notify=False)` — single file, migrations auto-applied, **no Postgres for `pytest`**.
- **`DBOSAgent`** (`from pydantic_ai.durable_exec.dbos import DBOSAgent`) wraps a pydantic-ai `Agent`;
  its `.run` is durable — top-level auto-creates a root workflow (`<name>.run`), nested becomes a child
  (`<parent>-N`). No `SetWorkflowID` needed at top level.
- **`SetWorkflowID` is contextvar-scoped → async-safe** under `asyncio.gather` (retires the old R5).
- **Human-in-the-loop:** `recv_async(topic, timeout_seconds)` blocks durably (status PENDING, survives
  restart); external `send_async(workflow_id, msg, topic)` resumes; timeout → `None` → denied.
  `set_event_async`/`get_event_async` publish the pending command for inspection.
- **Exactly-once = `SetWorkflowID(key)` as the business id.** No separate idempotency API; the workflow
  id *is* the dedup key. `execute(..., key=)` and `Queue.submit(..., key=)` map to it.
- **Wire bodies verbatim on 2.11** (`bodies.json`): `NativeOutput(M)` → clean `response_format:
  json_schema` (no tool leak); the three string modes → `structured_outputs.{regex,choice,grammar}` at
  the request top level; `extra_body` passed via `model_settings={"extra_body":{…}}`. A *plain* model
  (no `NativeOutput`) still rides the function-tool path → the library MUST keep applying `NativeOutput`
  for schema mode.
- **Type surface (S1/S2/S3):** `Choice[S: str] → Constraint[Literal[…]]`; `Schema(M).parse` types as
  `M`; `raw.usage` is a property (no `usage()` call). A bare `Outcome` union does NOT narrow under `ty`
  — which is *why* we dropped it.

**DBOS lifecycle (locked):** `configure(...)` creates the singleton → **build all agents/effectors/
workflows (this is when DBOS decorators register)** → `launch()` → run → `shutdown()`/`destroy()`.
Registration must precede `launch()`. Inside an event loop, use the `*_async` DBOS entrypoints
(`list_workflows_async`, `start_workflow_async`, `send_async`, `recv_async`, `set_event_async`,
`get_event_async`) — the sync ones raise.

---

## Your first moves

1. **Read the three plan docs + skim the spike files** (above). Confirm the design in your own head
   before touching code.
2. **Git:** the plan docs are committed on branch `plan/constraint-codec-v3`. Start a **fresh build
   branch** (plain git — this repo uses plain git, NOT gitman/jj). Confirm with the user how they want
   the branch/merge structured if unsure.
3. **Phase 0 — genesis (see PHASES_DURABLE.md P0).** This **clears `src/` and scaffolds the fresh
   package.** ⚠️ **This is destructive to the working tree — CONFIRM with the user immediately before
   running the clear.** The `v0.2.0` tag preserves the old artifact, so it's recoverable, but do not
   wipe `src/` unprompted. Then:
   - New package `src/structured_agents/`, `errors.py` (the tree in DESIGN_DURABLE §errors).
   - `pyproject.toml`: core deps `pydantic`, `pydantic-ai-slim[openai]>=2.11,<3`, `dbos>=2.23`, `httpx`;
     dev `pytest`, `pytest-asyncio`, `ty`, `ruff`; extras `[grammar-check]` (xgrammar), `[fornix]` (no dep).
   - devenv wired (Python 3.13); `pytest`/`ty`/`ruff` green (0 tests).
4. **Then proceed P1 → P7** in order, each phase left green + demonstrable before moving on.

---

## Ground rules (repo)

- **Run everything in devenv:** `devenv shell -- <cmd>` from the repo root. **Never** bare
  `uv`/`python`/`pytest`/`ty`/`ruff` (the shell here is fresh and lacks the env). `cd`-ing elsewhere
  breaks devenv resolution.
- **Version control is plain git** in this repo (NOT gitman/jj — see the project memory). Commit
  regularly as you land each green sub-step. **No AI-authorship trailers/attributions** in commits,
  PRs, code comments, or docs.
- **Quality bar, every phase:** `devenv shell -- pytest` green **on SQLite** (no Postgres), `devenv
  shell -- ty check src` clean, `devenv shell -- ruff check src tests` clean, plus a runnable demo.
- **Tests are async-first:** `pytest-asyncio` (`asyncio_mode="auto"`), and **assert they actually
  execute** — v2's finding A3 was async tests silently skipping. The `conftest` owns the DBOS singleton
  lifecycle (one launch per test process; register decorators before launch; `destroy()` on teardown;
  a fresh temp SQLite file per session).
- **Verify, don't trust** (the discipline that got us here): when you wire a DBOS feature whose exact
  2.23 surface the plan flagged as "confirm when wiring" — the **`Queue` rate-limit kwarg** and the
  **scheduled-workflow decorator** — verify it against the installed lib before building on it. Never
  guess a request shape; the wire table is captured in `bodies.json` and must be asserted (T2).

---

## Environment

- Python **3.13.13**, `ty` **0.0.46** (sole checker — no pyright/mypy), `pydantic` **2.13.3**,
  `pydantic-ai(-slim)` **2.11.0**, `dbos` **2.23.0**, `xgrammar` (behind `[grammar-check]`, pulls
  torch/CUDA — dev/CI only).
- Live vLLM for Phase 7: **tower** at `http://tower:8000/v1` (Qwen3-4B-AWQ, XGrammar json_schema/regex/
  choice verified; per-agent LoRA). Gated by `SAV_LIVE=1`. Deploy scripts salvage from v2 `deploy/`.
- Project memory (loaded each session) records the environment, the review→reframe history, and the
  confirmed spike facts — trust it, but re-verify anything it names before depending on it.

---

## Definition of done (per phase, and overall)

- **Per phase:** the PHASES_DURABLE.md acceptance criteria pass — falsifiably (a real assertion, not a
  smoke test) — with `pytest`/`ty`/`ruff` green on SQLite and a demo you can run. P4 (approval) and P5
  (plane services) are the two release points (v1.0.0-rc, then v1.0.0).
- **Overall:** the toolkit lets a user author a durable `@DBOS.workflow` composing `Agent.run` →
  `execute`/`approval` → effect, with exactly-once effects, human-in-the-loop pause, queues/schedules,
  and observability — green on SQLite, proven live on tower (P7), with a crash-recovery demo.

---

## What NOT to do

- **Don't rebuild the dropped pieces:** no `wire/` layer, no `Outcome`/combinators, no `context.py`
  cache axis, no `closed.py`. If you feel the pull, re-read the "Decisions already made" section — each
  was removed deliberately, with evidence in `REVIEW.md`/`SIMPLIFIED.md`.
- **Don't add a sync API mirror**, a canonical pipeline object, or a `Fleet`/`Router` as core — routing
  is a user-authored pattern in the toolkit model (add a thin helper later only if it earns its place).
- **Don't trust the pre-reframe docs** (`DESIGN.md`/`PHASES.md`/`00-PLAN.md`) for the durable design —
  they describe the superseded shape. `*_DURABLE.md` win.
- **Don't run the destructive `src/` clear (P0) without confirming with the user first.**

---

## Suggested kickoff message for the new session

> Study `.scratch/projects/10-durable-agent-plane-build/BUILD_KICKOFF.md` in detail (and the plan docs
> it points to in `../09-constraint-codec-rewrite/`), then begin Phase 0 — but confirm with me before
> clearing `src/`.
