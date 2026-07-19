# Phase 3 DBOS test-lifecycle note — 2026-07-18

## Evidence

- Environment: DBOS 2.23.0, Python 3.13.13, SQLite system database, pytest-asyncio 1.4.0.
- First focused authority run (before the lifecycle adjustment):
  `devenv shell -- pytest tests/test_authority.py tests/test_fornix.py`
  passed the first async tests, then failed later workflow invocations with
  `RuntimeError: cannot schedule new futures after shutdown` from
  `concurrent.futures.thread.ThreadPoolExecutor.submit`.
- The same B5 test alone passed and surfaced its intended `RuntimeError("effect failed")`.
  This ruled out an effect-retry or `SetWorkflowID` semantic failure.
- Cause: DBOS is process-global and launched once in the session fixture, while pytest-asyncio
  had been creating a fresh event loop per async test. Its worker executor was associated with a
  loop that pytest subsequently closed.

## Minimal fix

Set `asyncio_default_test_loop_scope = "session"` in `pyproject.toml`, matching the existing
session-scoped DBOS singleton lifecycle. No DBOS, pydantic-ai, model, or test behavior was changed.

## Reverification

- `devenv shell -- pytest` → 12 passed
- `devenv shell -- ty check src tests` → all checks passed
- `devenv shell -- ruff check src tests` → all checks passed

## Scope

This proves the SQLite Phase 3 authorization/effect boundary and keyed replay behavior. It does not
prove crash recovery across a separate process; that remains the DBOS contract exercised in later
durability work.

---

# Phase 4 approval messaging evidence — 2026-07-18

## Evidence

- Environment: DBOS 2.23.0, Python 3.13.13, pytest-asyncio 1.4.0, and the
  session-global SQLite system database from `tests/conftest.py`.
- Installed DBOS signatures confirm the async APIs used by this phase:
  `set_event_async(key, value)`, `recv_async(topic, timeout_seconds)`,
  `send_async(destination_id, message, topic)`, and
  `list_workflows_async(status="PENDING")`. `get_all_events_async(workflow_id)`
  exposes an existing workflow's events without waiting for a missing key.
  [DBOS workflow messaging documentation](https://docs.dbos.dev/python/tutorials/workflow-messaging)
  describes `send`/`recv` as durable workflow messaging.
- Focused runtime command: `devenv shell -- pytest tests/test_approval.py -q`.
  Result: 4 passed. A workflow blocked in `recv_async` was observed as `PENDING`;
  its raw `pending_command` event was retrieved; an external async send resumed
  it; an explicit denial returned `Denied` data and made zero effect calls; and
  a receive timeout returned `Decision(False, "timeout")`.

## Minimal implementation

- `Approval.request` publishes the raw command as `pending_command` and the
  recipient as `pending_to`, then receives only on its configured topic.
- `ApprovalClient.pending` filters `PENDING` workflows by these two published
  events, preventing unrelated pending DBOS workflows from being reported as
  approvals.
- The public timeout remains optional. When it is absent the implementation
  calls `recv_async` without `timeout_seconds`, preserving DBOS's own default;
  DBOS's typed API does not accept `None` for that parameter.

## Scope

This proves the in-process SQLite durable-message transition from PENDING to
SUCCESS and the data-denial composition boundary. It does not independently
prove recovery after a separate-process crash/restart; that remains the DBOS
durability contract and Phase 7's recovery verification scope.

---

# Phase 5 plane-service evidence — 2026-07-18

## Installed DBOS 2.23 contract

- `Queue(name, concurrency=None, limiter=None, ...)` takes its rate limit as
  a `QueueRateLimit` mapping with `{"limit": int, "period": float}`; durable
  submission is `await queue.enqueue_async(workflow, *args)`. The public
  `Queue` wrapper translates its `(limit, period_seconds)` API to that mapping.
- `@DBOS.scheduled(cron)` is the 2.23 scheduling decorator. Its scheduler
  invokes the workflow with two UTC `datetime` arguments: the scheduled time
  and the actual dispatch time. The installed implementation parses cron with
  `second_at_beginning=True`, so the SQLite test uses `*/2 * * * * *` and
  `scheduler_polling_interval_sec=0.05` rather than sleeping through a minute.
- Async observability APIs are `list_workflows_async`,
  `get_workflow_status_async`, `fork_workflow_async(workflow_id, start_step)`,
  and `cancel_workflow_async`. There is no `get_workflow_handle_async` in this
  release.

## Focused runtime evidence

- `devenv shell -- pytest tests/test_plane.py::test_queue_caps_concurrency_and_isolates_batch_failures tests/test_plane.py::test_compare_is_keyed_and_durable -q`
  passed. A DBOS queue with concurrency 2 kept peak concurrent runs at or below
  2; a failed item surfaced only through its own handle while its three siblings
  completed. A pair of real `Agent` legs used two keyed durable runs; repeating
  the same comparison key made exactly two model requests total, not four.
- `devenv shell -- pytest tests/test_plane.py::test_scheduled_workflow_fires -vv`
  passed in 4.15s. The scheduled workflow fired and appeared as SUCCESS in the
  SQLite workflow store.
- The observability focused test passed: a blocked `Approval` was listed as
  PENDING, status lookup returned PENDING, a workflow fork from step 0 replayed
  its meaningful step, and a blocked receive transitioned to CANCELLED.

## First failures and minimal corrections

- The first collection attempt assumed a private `dbos._workflow` module for
  handle types. DBOS 2.23 exports `WorkflowHandleAsync` and `WorkflowStatus`
  from `dbos`; the wrapper now imports those public names.
- The first scheduler run failed with
  `TypeError: scheduled_workflow() takes 0 positional arguments but 2 were given`.
  Installed source confirms the two-datetime callback contract, so the test
  now accepts those arguments. No DBOS lifecycle, model, or durable semantics
  were changed to hide the failure.

## Scope

This proves the Phase 5 SQLite operational surface in-process: queue limits and
failure isolation, cron dispatch, workflow-store observability, fork/cancel,
and keyed dual-agent idempotency. It does not independently prove recovery
across a separate process, which remains Phase 7 scope.
