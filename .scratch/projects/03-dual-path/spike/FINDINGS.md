# FINDINGS — Dual-path inference (Architecture C) spike

Verified empirically on 2026-06-09. Environment: `dbos==2.23.0`, `pydantic-ai==1.87.0`,
Python 3.13.13, PostgreSQL 17.9 on `127.0.0.1:5433`. Spike: `run_spike.py` (gates 1-3, GPU-free)
+ `tests/test_dual_path_spike.py` (gated). **Go/no-go: GO** — DBOS is a viable substrate.

## TL;DR

Architecture C works as designed. Two independent `DBOSAgent`s, joined by a **top-level**
`asyncio.gather` (no parent workflow), each ran as its own durable DBOS workflow; the Phase-2
constrained wire shape survived the wrapper byte-for-byte; a versioned `ComparisonRecord` was
assembled, diffed, and persisted to Postgres as `jsonb`; and DBOS independently persisted each LLM
call as a step. All gates green with no GPU and no frontier call.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| 1 — one DBOSAgent durable run vs mock | ✅ | `Command(action='create', …)` validated; DBOS lifecycle init→launch→destroy clean under pytest |
| 2 — wire shape survives the wrapper | ✅ | wrapped body **==** unwrapped body; `model=cmd-adapter`, `response_format.type=json_schema`, no stray `extra_body` |
| 3 — dual gather + ComparisonRecord | ✅ | two workflows concurrent; `agreement_exact=False`, `field_diff={'reason': [...]}`; row saved to `jsonb`; DBOS persisted 1 step each |
| 4 — live frontier reference leg | ⏸ deferred | gated test written; needs `SAV_LIVE=1` + `REF_API_KEY` |

## Verified facts (with cites)

1. **DBOS does NOT require Postgres for dev/test — but we chose Postgres anyway.** SQLite system
   DB works lifecycle-clean (`dbos/_sys_db.py:483`, `_sys_db_sqlite.py`). We use Postgres
   (`DBOSConfig(system_database_url="postgresql://…")`) so records persist as `jsonb`. DBOS created
   its `dbos.*` system schema automatically (`workflow_status`, `operation_outputs`, `queues`, …).

2. **The wrapper preserves the constrained wire shape.** DBOS wraps *above* httpx, so the Phase-2
   `RequestCapture` httpx hook is untouched. Captured wrapped request:
   `model=cmd-adapter` (adapter as the wire model — LoRA path intact), top-level keys
   `[messages, model, response_format, stream]`, `response_format` = full json_schema with
   `name="Command"` + `additionalProperties` present. Identical to the unwrapped run. → **Phase 2's
   `NativeOutput`/`extra_body`/adapter contract is safe through `DBOSAgent`.**

3. **What DBOS persists per LLM call** = a step named `f"{name}__model.request"` (confirmed
   `cmd@primary__model.request`) whose **output is a pydantic-ai `ModelResponse`** repr
   (`parts=[TextPart(...)], usage=RequestUsage(input_tokens=11, output_tokens=7),
   model_name='cmd-adapter'`), with `function_id`, `started_/completed_at_epoch_ms`. This is
   pydantic-ai's **normalized** shape, **not** the raw OpenAI JSON body. Recoverable, but
   version-coupled to pydantic-ai internals.

4. **Two stores, by design (answers §8 Q4).** DBOS's step store is the *execution* record
   (durable, replayable, pydantic-ai-shaped); our **`ComparisonRecord` in a `jsonb` column** is the
   *data* record (portable, query-stable, indexable). The spike queried nested jsonb
   (`record->'field_diff'->'reason'`, `record->'primary_output'->>'action'`) directly — the
   performance/indexability win. A GIN index on `record` is created by the store.

5. **Top-level gather over two workflows is fine (answers §8 Q1/the real parallel caveat).** Using
   `SetWorkflowID` to assign `primary-…`/`reference-…` IDs, `asyncio.gather` over the two
   `DBOSAgent.run()` coroutines produced two independent, concurrently-completing workflows
   (overlapping `started/completed` epochs). No parent workflow, so the
   `DBOSParallelExecutionMode` (tool-call ordering) caveat never applies. **Architecture C
   sidesteps nested-child-workflow determinism entirely.**

6. **Lifecycle ordering constraint (real, must document).** `DBOSAgent.__init__` performs the
   `@DBOS.workflow` registration, and DBOS requires **all workflows registered before
   `DBOS.launch()`**. So in the spike: `DBOS(config)` → build all `DBOSAgent`s → `DBOS.launch()` →
   run → `DBOS.destroy()`. The eventual library must respect this (agents constructed before
   launch), which is in mild tension with the lazy `Backend.build(profile)` factory.

7. **Introspection API is async-only inside a loop.** `DBOS.list_workflow_steps(...)` raises if an
   event loop is running; use `await DBOS.list_workflow_steps_async(...)`. `list_workflows`,
   `retrieve_workflow`, `SetWorkflowID` all confirmed present.

8. **Usage extraction works** — `result.usage()` → tokens flow into the record
   (`input_tokens`/`output_tokens`). Cost mapping (price-per-token table) remains ours.

## Open questions — status after the spike

- **§5 Postgres infra** → RESOLVED for dev/test: a self-managed cluster under `.devenv/state/`
  (gitignored), or `services.postgres` in `devenv.nix` for reproducibility. Prod placement (remora
  box vs sidecar) still a deployment decision.
- **§8 Q2 wire shape** → RESOLVED: preserved (fact 2).
- **§8 Q4 one store or two** → RESOLVED: two (fact 4); `ComparisonRecord` jsonb is the data SoT.
- **§8 Q1 minimal durable run / GPU-free test** → RESOLVED: yes (gates 1-3, no GPU).
- **§5 frontier structured-output per provider** → DEFERRED to Gate 4 (live). OpenAI strict
  `json_schema` 400-on-noncompliant is the thing to watch; pulls CONCEPT §10 open-#1 (strict schema
  rewrite) forward as a reference-path blocker. Test scaffolding is in place.
- **§8 Q6 sampling / sync vs fire-and-forget** → not exercised; Architecture C runs both
  synchronously at the runner. Async/sampled reference leg is a later layer (a DBOS queue), not
  needed to prove the path.
- **§8 Q7 comparator** → exact + field-level diff implemented and exercised (`reason` diverged).
  Semantic/normalized comparators deferred.

## Risks / sharp edges to carry into CONCEPT

- **DBOS process-global lifecycle** vs. a "thin binding" library — biggest design tension. Keep
  dual-path a **separate optional subpackage + `[dual-path]` extra**; the lean core never imports
  DBOS. The library will need an explicit "register agents, then launch" entry point.
- **Persisted step shape is pydantic-ai-version-coupled** — do not treat DBOS tables as the
  training-data store; export from `ComparisonRecord`.
- **Strict-schema compliance** matters more on the frontier path than on vLLM/XGrammar.

## Repro

```bash
# Postgres (one-time): initdb + pg_ctl on :5433, createdb dual_path   (see SPIKE_PLAN.md)
devenv shell -- uv run --extra dev --extra dual-path python .scratch/projects/03-dual-path/spike/run_spike.py
devenv shell -- uv run --extra dev --extra dual-path pytest tests/test_dual_path_spike.py -q
# live gate 4:  SAV_LIVE=1 REF_API_KEY=… REF_MODEL=… REF_BASE_URL=…  pytest -k live
```
