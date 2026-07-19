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
