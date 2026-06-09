# Spike Plan — Dual-path inference (Architecture C) with DBOS

> Kickoff for the **build** phase of the dual-path track. This is a throwaway spike under
> `.scratch/`, not library code. It proves the path empirically; FINDINGS.md records what we
> learn; CONCEPT.md (later) turns it into a design.

## Decisions locked (2026-06-09 brainstorm)

1. **Architecture C** — two **independent** `DBOSAgent`s, joined by a **top-level**
   `asyncio.gather`, then a thin comparator. **No parent `@DBOS.workflow`** wrapping the pair,
   so we sidestep the concurrent-child-workflow determinism question entirely.
2. **Separate optional subpackage + `[dual-path]` extra** — DBOS stays opt-in; the lean core
   never imports it. (Spike lives in `.scratch/`; the eventual home is `dual_path/`.)
3. First spike proves **C**, GPU-free, against the in-process ASGI mock.
4. **Bare-string modes (grammar/regex/choice) skip dual-path** — only `json_schema` agents get
   a teacher. The frontier path can't enforce them.
5. **Storage = PostgreSQL** (user directive, 2026-06-09). DBOS uses it as its system DB, and we
   store each `ComparisonRecord` (+ the validated Pydantic outputs) in a **`jsonb`** column for
   query performance and indexability. SQLite is kept only as a documented zero-dependency
   fallback for the bare lifecycle test — the real substrate is Postgres.

## Facts already established before building (verified on disk)

- `dbos==2.23.0` installed via the `[dual-path]` extra. Pulls psycopg + sqlalchemy.
- DBOS runs against **PostgreSQL** as its system DB (`DBOSConfig(system_database_url=…)`); it also
  supports SQLite (`dbos/_sys_db.py:483`) which we verified works lifecycle-clean — kept only as a
  zero-dep fallback. **We use Postgres** so `ComparisonRecord` + validated outputs persist as
  `jsonb`. Postgres provisioned via `services.postgres` in `devenv.nix`.
- DBOS wraps **above** httpx, so the Phase-2 `RequestCapture` httpx hook is untouched (see Gate 2).
- `DBOSModel` persists each model call as a `@DBOS.step`; its input is pydantic-ai's normalized
  `(messages, ModelSettings, ModelRequestParameters)` and output a `ModelResponse` — **not** the
  raw OpenAI HTTP body. Two different layers; informs the "one store or two" question (Gate 3).

## The gates (each is a go/no-go; stop and report on a surprise)

| Gate | Proves | Pass criterion |
|---|---|---|
| **1** | One `DBOSAgent` wrapping `StructuredAgent.agent` runs durably vs the mock; DBOS global lifecycle survives pytest | A json_schema output validates; `DBOS.launch()/destroy()` clean across tests |
| **2** | Wire shape survives the wrapper | Captured body through `DBOSAgent` is byte-identical to the unwrapped path (`response_format` json_schema / `extra_body` / `model`=adapter) |
| **3** | Top-level `gather` over two `DBOSAgent`s yields two validated outputs + a `ComparisonRecord`; inspect DBOS-persisted steps vs our record | Two independent workflows complete concurrently; record assembles; we can read back persisted steps |
| **4** (live) | Real OpenAI reference leg: `NativeOutput` strict round-trip + usage extraction | Schema-valid output; note whether strict mode 400s. Skipped unless creds + `SAV_LIVE=1` |

## Ordering constraint discovered

DBOS workflows must be **registered before `DBOS.launch()`**. `DBOSAgent.__init__` performs the
`@DBOS.workflow` decoration, so **all `DBOSAgent`s must be constructed before launch**. The spike
constructs agents, then launches. (Whether post-launch construction works is itself a Gate-1
check to record.)

## File layout

```
.scratch/projects/03-dual-path/spike/
├── SPIKE_PLAN.md          # this file
├── schemas.py             # spike output types (json_schema command models)
├── runner.py              # DBOS lifecycle helper, DBOSAgent builder, ComparisonRecord + comparator
├── run_spike.py           # runnable: gates 1–3 vs mock; writes artifacts + a report
├── artifacts/             # captured_request.json, comparison_record.json, dbos_steps.json
└── FINDINGS.md            # verified facts + §5/§8 answers + go/no-go  (written after running)
tests/test_dual_path_spike.py   # pytest gates 1–3 (importorskip dbos); gate 4 gated on SAV_LIVE
```

## How to run

```bash
devenv shell -- uv run --extra dev --extra dual-path python \
  .scratch/projects/03-dual-path/spike/run_spike.py
devenv shell -- uv run --extra dev --extra dual-path pytest tests/test_dual_path_spike.py -q
# live (gate 4): REF_BASE_URL/REF_API_KEY/REF_MODEL + SAV_LIVE=1
```

## Acceptance for the spike

Gates 1–3 green GPU-free; artifacts written; FINDINGS.md answers: Postgres-or-SQLite, wire-shape
survival, persisted-step shape vs `ComparisonRecord` (one store or two), DBOS lifecycle ergonomics
in a library, and a go/no-go on DBOS as the substrate. Gate 4 run opportunistically when creds exist.
```
