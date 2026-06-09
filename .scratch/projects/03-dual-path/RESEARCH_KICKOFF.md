# Research Kickoff — Dual-path inference (local vLLM ‖ frontier API) with DBOS

You are starting a **research/investigation phase**, not a build phase. Goal: figure out
how to run every agent call **twice** — once against our local vLLM server (constrained,
the thing we ship) and once against a frontier API (OpenAI / DeepSeek / OpenRouter, the
"teacher") — so we can **compare**, **persist for fine-tuning + evals**, with **versioned
tracking and logging**, built on **PydanticAI's DBOS durable-execution integration**.

This doc is self-contained. Read it fully, confirm the facts, then produce a **spike +
FINDINGS.md** (mirroring `01-xgrammar-concept/spike/`) *before* writing a design or code.

---

## 0. The starting prompt (paste this to begin the new session)

> Investigate dual-path inference for structured-agents-v2: every agent run executes
> against our local vLLM `Backend` AND, concurrently, against a frontier OpenAI-compatible
> API (OpenAI/DeepSeek/OpenRouter), validating both outputs against the same PydanticAI
> output type. Build the investigation on PydanticAI's built-in **DBOS** durable-execution
> integration (`pydantic_ai.durable_exec.dbos`) so model calls become durable, persisted,
> retriable steps — our substrate for versioned logging, fine-tuning data capture, and
> ongoing local-vs-frontier evals. Read
> `.scratch/projects/03-dual-path/RESEARCH_KICKOFF.md` in full first, confirm the green
> baseline (`devenv shell -- uv run --extra dev pytest -q`), then deliver a spike under
> `.scratch/projects/03-dual-path/spike/` + a `FINDINGS.md`, NOT production code. Run
> everything inside `devenv shell --`. No AI-attribution trailers in commits.

## 1. How to operate (read first)

- **Run ALL commands inside the devenv shell:** `devenv shell -- <cmd>` (gives Python
  3.13.13 + uv). Never bare `uv run` or system python. Env vars `LLM_BASE_URL`,
  `LLM_API_KEY`, `LLM_MODEL` are set by `devenv.nix` (today → the llama.cpp box).
- Type check: `devenv shell -- uv run --extra dev ty check src` (checker is **ty**, not mypy).
- Lint/format: `... ruff check src tests` / `... ruff format src tests`. Tests: `... pytest -q`.
- **Baseline is GREEN on `main`** (Phase 1 + 2: 41 passed, 2 skipped `live`, 95% cov, ty +
  ruff clean). Confirm green before touching anything.
- **This is a research phase.** First artifact is a throwaway **spike** + a **FINDINGS.md**
  of verified facts (the repo's pattern — see `01-xgrammar-concept/spike/FINDINGS.md`).
  Only after findings do we write a `CONCEPT.md` and, later, library code.
- **Git:** branch off `main`; for the research phase, committing the spike + FINDINGS +
  CONCEPT to `.scratch/` is enough. **No AI-attribution trailers** (no "Co-Authored-By",
  no "Generated with").
- If you need an interactive command (e.g. `psql`, a cloud login), ask the user to run it
  with `! <cmd>`.

## 2. Authoritative context (read these, in order)

1. **This file.** Then the verified facts in §4–§6 below — re-confirm them, don't trust blindly.
2. `.scratch/projects/02-library-wrapper/CONCEPT.md` (rev 2) — the library design the dual
   path plugs into: `ConstrainedOutput`, `DecoderSpec`, `Backend(+caps)`, `AgentProfile`,
   `StructuredAgent`, `AgentResult`. **Phase 1 + 2 are built and on `main`.**
3. `src/structured_agents_v2/` — the actual code (`backend.py`, `profile.py`, `agent.py`,
   `decoder.py`, `constrained.py`, `capture.py`, `errors.py`). Small; read it.
4. `.scratch/projects/01-xgrammar-concept/spike/{run_spike.py,FINDINGS.md}` — the spike
   pattern to imitate, and the verified request path (response_format / extra_body / model).
5. `deploy/vllm/README.md` — the local vLLM target (same `:8000/v1` contract; XGrammar +
   LoRA). The frontier APIs are *also* OpenAI-compatible, so the same `Backend` represents
   them — just with `caps(xgrammar=False, lora=False)`.

Auto-memory (`~/.claude/projects/.../memory/MEMORY.md`) indexes the same facts.

## 3. The goal in one paragraph

For each prompt, run the **primary** path (our vLLM `StructuredAgent`: constrained decoding
+ per-agent LoRA, emits a validated command object) and one-or-more **reference** paths (a
frontier model behind the same OpenAI contract) **concurrently**. Validate every output
against the *same* `ConstrainedOutput`/Pydantic type via PydanticAI (the reference path
can't use XGrammar, so validation + retries are how we keep reference data clean). Persist
each run as a **versioned, comparable record** — prompt, instructions, schema version,
model identities (vLLM tag + adapter vs. frontier model id), both validated outputs, a
diff/agreement signal, usage/cost, timings. That store is simultaneously: (a) **SFT/eval
training data** for improving the small local models, and (b) an **ongoing eval** of "how
good is local vs. frontier." DBOS provides the durability, persistence, retries, and
observability so we don't hand-roll a logging/queue/state layer.

## 4. VERIFIED — PydanticAI DBOS integration (pydantic-ai 1.87.0, confirmed on disk)

Package path: `.devenv/state/venv/lib/python3.13/site-packages/pydantic_ai/durable_exec/`.
Three integrations ship: **`dbos`**, `temporal`, `prefect`. For DBOS:

```python
from pydantic_ai.durable_exec.dbos import (
    DBOSAgent, DBOSModel, StepConfig, DBOSParallelExecutionMode, DBOSMCPServer,
)
```

- **`DBOSAgent(wrapped: Agent, *, name=..., model_step_config: StepConfig | None = None,
  mcp_step_config=None, event_stream_handler=None, parallel_execution_mode=...)`** —
  wraps a `pydantic_ai.Agent`. **Requires a unique `name`** (used to name its workflows +
  steps; falls back to `wrapped.name`, errors if neither). It wraps `.run`/`.run_sync` in
  `@DBOS.workflow(name=f"{name}.run")`. *After wrapping, the original agent still works
  normally outside a DBOS workflow.* (`dbos/_agent.py`)
- **`DBOSModel(wrapped_model, *, step_name_prefix, step_config, event_stream_handler)`** —
  turns each model `request` / `request_stream` into a **`@DBOS.step`** named
  `f"{prefix}__model.request"`. **This is the key:** every LLM call becomes a durable,
  checkpointed step — its inputs and outputs are persisted by DBOS automatically, which is
  most of our "logging + training-data capture" for free. (`dbos/_model.py`)
- **`StepConfig`** (TypedDict, all optional): `retries_allowed: bool`,
  `interval_seconds: float`, `max_attempts: int`, `backoff_rate: float`. Per-step retry
  policy — relevant for flaky frontier APIs. (`dbos/_utils.py`)
- **`DBOSParallelExecutionMode`** — a *subset* of PydanticAI's `ParallelExecutionMode`
  ("parallel" is excluded because it can't guarantee deterministic ordering inside a
  durable workflow). Note this when we run multiple steps concurrently.

> Re-verify all of the above by reading those files; cite exact signatures in FINDINGS.md.

## 5. NOT YET CONFIRMED — investigate and record in FINDINGS.md

- **`dbos` is NOT installed.** It's an *optional extra* (`pydantic-ai-slim` exposes a
  `dbos` extra; confirmed). DBOS itself needs to be added (likely a new optional extra in
  `pyproject.toml`, e.g. `[dual-path]` or `[dbos]`) and **requires PostgreSQL** for its
  workflow/step store. Decide where Postgres runs (a `deploy/` sidecar? the remora box?)
  and how `devenv.nix` provides it for tests (devenv can run a Postgres service).
- **DBOS lifecycle/config:** how `DBOS(...)` is configured (db URL/config), `DBOS.launch()`,
  and how a workflow is invoked. Confirm against the installed `dbos` once added and the
  PydanticAI DBOS docs. Establish the minimum to run one durable agent run end-to-end.
- **What DBOS persists, and in what shape** — confirm step inputs/outputs are stored such
  that we can reconstruct (prompt, model, settings) → (raw completion, validated output,
  usage). If the persisted shape is awkward for training-data export, decide whether we
  *also* write our own record (see §7) or query DBOS's tables / Conductor.
- **Frontier structured-output support per provider** (no XGrammar there):
  - OpenAI — native Structured Outputs (`response_format: json_schema`, strict). Our
    `NativeOutput` json_schema path should map directly.
    Reference `Backend(caps=BackendCaps(xgrammar=False, lora=False))`.
  - DeepSeek — JSON mode; confirm json_schema vs. json-object only.
  - OpenRouter — varies by underlying model; may need the tool/function path or
    text+validation+retry fallback. Confirm what the OpenAI-compatible surface accepts.
  Record which of our `DecodeMode`s are enforceable on each, and what the fallback is.
- **Bare-string modes (grammar/regex/choice) on the reference path** can't be enforced.
  Decide: skip dual-path for those, or run reference "unconstrained + Pydantic-validate +
  retry" and flag lower-confidence labels. (Most command models are json_schema anyway.)
- **Logfire?** Pydantic's native tracing (`logfire`) is the obvious complement for spans /
  comparison dashboards. DBOS gives durability + its own observability/Conductor. Decide
  the division of labor (DBOS = durable state + step store; Logfire = traces/metrics) —
  but the user asked specifically to center this on **DBOS**, so lead with DBOS.

## 6. How it plugs into the existing library

The current path: `Backend.build(profile) -> StructuredAgent` wraps a `pydantic_ai.Agent`
exposed via `StructuredAgent.agent` (escape hatch). `AgentResult` already carries
`output / usage / request_body / raw`. The dual-path layer is **new modules on top** — it
does NOT change Phase 1/2 code and does NOT touch `backend.py`'s "sole importer of
`pydantic_ai.models.openai`" rule (the DBOS import is from `pydantic_ai.durable_exec.dbos`).

Sketch to evaluate in the spike (names provisional):

```python
# A reference (frontier) backend is just a Backend with no XGrammar/LoRA:
reference = Backend(base_url="https://api.openai.com/v1", api_key=..., default_model="gpt-...",
                    caps=BackendCaps(xgrammar=False, lora=False))

# Wrap each built agent's underlying pydantic_ai.Agent in a DBOSAgent (needs a unique name):
primary_agent   = DBOSAgent(local.build(profile).agent,     name=f"{profile.name}@vllm",
                            model_step_config=StepConfig(max_attempts=2))
reference_agent = DBOSAgent(reference.build(ref_profile).agent, name=f"{profile.name}@openai",
                            model_step_config=StepConfig(max_attempts=3, retries_allowed=True))

@DBOS.workflow(name="dual_path.run")
async def dual_run(prompt: str) -> ComparisonRecord:
    primary, ref = await asyncio.gather(primary_agent.run(prompt),
                                        reference_agent.run(prompt))  # each call = durable step
    return ComparisonRecord.build(profile_version, primary, ref)     # validated, diffed, persisted
```

Key questions the spike answers: does wrapping our agent in `DBOSAgent` preserve the
constrained wire shape (response_format / extra_body / model) we verified in Phase 2? Does
`asyncio.gather` over two DBOSAgents inside one workflow persist two clean steps? What does
the persisted step data look like for export?

## 7. Versioning, logging, and the data model (design target)

Define a **`ComparisonRecord`** (Pydantic, serializable) — the unit of training/eval data:

- **identity/versioning:** `profile_version` (content-hash of the serialized `AgentProfile`
  incl. instructions + output schema), `schema_version` (hash of `model_json_schema()`),
  `primary_model` (vLLM `VLLM_TAG` + base + adapter name/rev), `reference_model` (provider +
  model id), `decode_mode`, `lib_version`, `run_id`/`workflow_id` (from DBOS), timestamp.
- **payload:** `prompt`, resolved `instructions`, `primary_output` (validated),
  `reference_output` (validated or `None` + error), raw completions (escape hatch).
- **signals:** `agreement` (exact / field-level / semantic — define a comparator per type),
  `primary_valid` / `reference_valid` (Pydantic validation passed?), `usage` + derived
  `cost` per path, latencies.
- **export:** to JSONL (SFT/preference pairs: reference output as candidate teacher label,
  gated on `reference_valid`) and to an evals view (validity rate, agreement over a held
  set, regression vs. previous `primary_model`).

Decide: is the source-of-truth store **DBOS's persisted steps** (query/export later), a
**separate table/JSONL we write inside the workflow**, or both? Lean: write an explicit
`ComparisonRecord` (portable, schema-stable for training) *and* rely on DBOS for durability/
replay of the runs that produced it. Versioning must make every record reproducible.

## 8. Open questions to resolve (the investigation's job)

1. Minimal end-to-end: one `DBOSAgent` run, durable, against the mock — what infra (Postgres
   via devenv?) and config does it take? Can the GPU-free test suite exercise it?
2. Does the DBOS wrapper change/keep our constrained wire shape (verify with `RequestCapture`)?
3. Reference-path structured-output support + fallback per provider (§5).
4. Persisted-step shape vs. our `ComparisonRecord` — one store or two (§7)?
5. Cost/usage extraction per provider (PydanticAI `usage()` covers tokens; cost mapping is ours).
6. Sampling policy: always dual vs. sample N% (frontier-spend control); sync vs. fire-and-forget
   reference path (don't block the primary on a slow/expensive teacher).
7. Comparator strategy per output type (exact, field-level, normalized, semantic).
8. Where this lands in the phase plan — a new "dual-path / eval-capture" track parallel to
   Fleet (Phase 3) / Executor (Phase 4), or after vLLM cutover (Phase 5)?
9. DBOS vs. Temporal vs. Prefect — the user chose DBOS; record *why* it fits (lightweight,
   Postgres-backed, no separate worker cluster) so the choice is documented.

## 9. Research deliverables (acceptance for THIS phase)

1. `.scratch/projects/03-dual-path/spike/` — a runnable spike proving: (a) a `DBOSAgent`
   wrapping our `StructuredAgent.agent` runs durably; (b) the constrained wire shape is
   preserved; (c) a dual `asyncio.gather` run produces two validated outputs + a
   `ComparisonRecord`; (d) one frontier provider returns a schema-valid output via the
   reference `Backend`. GPU-free where possible (mock for local; a gated `live` for frontier).
2. `.scratch/projects/03-dual-path/FINDINGS.md` — verified facts (with file paths + exact
   signatures), each §5/§8 question answered or explicitly deferred, infra requirements
   (Postgres, extras, devenv), and a go/no-go on DBOS as the substrate.
3. `.scratch/projects/03-dual-path/CONCEPT.md` (draft) — the proposed abstractions
   (`ComparisonRecord`, reference `Backend`, the dual-path runner, versioning + export), a
   package-layout proposal, and a phased build plan — written ONLY after the spike.
4. `pyproject.toml` extra proposal for `dbos` (+ logfire?) — documented, not necessarily merged.
5. Memory + CONCEPT cross-links updated; `MEMORY.md` index points at this track.

Do **not** ship production library code in this phase. Prove the path, write it down.

## 10. Suggested task order

1. Confirm green baseline; read §2 sources + re-verify §4 against the files.
2. Add `dbos` (and stand up Postgres via devenv) in a throwaway way; get one `DBOSAgent`
   durable run working against the existing mock server.
3. Capture the wire shape through `DBOSAgent` (reuse `RequestCapture`) — confirm Phase 2
   shapes survive.
4. Stand up a reference `Backend` against one frontier provider (gated `live`, real key);
   validate a `ConstrainedOutput` round-trips.
5. Dual `asyncio.gather` inside one `@DBOS.workflow`; assemble a `ComparisonRecord`; inspect
   what DBOS persisted vs. what we wrote.
6. Write FINDINGS.md; then draft CONCEPT.md + the phase plan + pyproject extra.

## 11. Commands cheat-sheet

```bash
devenv shell -- uv run --extra dev pytest -q
devenv shell -- uv run --extra dev ty check src
devenv shell -- uv run --extra dev ruff check src tests
# Inspect the DBOS integration that ships with pydantic-ai 1.87.0:
devenv shell -- python -c "import pydantic_ai.durable_exec.dbos as d; print(d.__all__)"
ls .devenv/state/venv/lib/python3.13/site-packages/pydantic_ai/durable_exec/dbos/
# Frontier env (set per provider when running the gated live spike):
#   REF_BASE_URL / REF_API_KEY / REF_MODEL  (OpenAI | DeepSeek | OpenRouter)
```
