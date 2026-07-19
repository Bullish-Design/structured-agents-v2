# v3 — the durable agent plane (north-star concept)

Supersedes the *scope framing* of `00-PLAN.md`/`SIMPLIFIED.md`. The internal simplifications from the
2026-07-17 pass **carry** (`run -> T` + raise; no `Outcome` spine; `instructions: str`; no `closed`);
what changed is the **center of gravity**: the library is now organized around a **durable workflow
plane (DBOS)**, not a single value.

## Thesis

> **A durable-primitives toolkit for building crash-recoverable, exactly-once, observable
> constrained-agent workflows on DBOS + vLLM/XGrammar. The library is the plane and the primitives;
> the user authors the workflows.**

Three constraints from the user set the shape:
- **Toolkit, not framework.** The library ships durable *building blocks*; users write their own
  `@DBOS.workflow` functions composing them. The library owns **no** canonical pipeline.
- **Always-on durability.** Every `agent.run` / effect / approval is durable by construction — one
  uniform plane, Postgres always on the path.
- **Full DBOS surface, first-class in v1:** durable exec + crash-recovery + exactly-once effects
  (spine) · queues · scheduled agents · messaging + human-in-the-loop approval · observability
  (query/status/fork/replay).

## Why this is the strongest version of the thesis

Constrained decoding guarantees *syntax*; validation guarantees *shape*; authority guarantees
*permission* — but none of that stops an autonomous agent from **double-executing** a side effect after
a crash/retry. The durable plane closes that gap: an agent proposes a command, it's authorized, and the
effect runs **exactly once, survives restarts, and is queryable/replayable**. That is the property that
makes autonomous local agents *operable*, and it's why the plane — not the codec — is the spine.

**What we get for near-free:** pydantic-ai already ships `pydantic_ai.durable_exec.dbos.DBOSAgent`
(the model loop as durable DBOS workflow/steps); DBOS is 2.23.0 (async, queues, schedules, messaging,
fork/replay); this repo's `dual_path/` already drives `DBOSAgent` + `SetWorkflowID`. Durable
*generation* is a wrap, not a build.

---

## The plane

```
DBOS durable plane  (Postgres; SQLite for dev/test)
  ── every primitive below is durable by construction ──
  Agent[T].run(prompt)  → constrained generation      (DBOSAgent: model loop = durable steps) → T
  Authorizer.decide     → Decision                     (data; pure or a step)
  Effector.run          → side effect                  (@DBOS.step: EXACTLY-ONCE + retry policy)
  Approval.request      → Decision                     (DBOS recv/events: durably PAUSE for a human)
  Queue.submit          → durable concurrency/batch/rate-limit over agent runs
  schedule(cron)        → run a user workflow on a durable timer
  observability         → list/status/fork/replay/cancel over the workflow store
  compare(a, b)         → dual-path eval fan-out (both legs durable; store = the record)

  the user composes these inside their own @DBOS.workflow functions
```

`Constraint[T]` stays the linchpin **of the generate primitive**; it is no longer the whole thesis —
it's one durable primitive among several.

---

## The toolkit (real signatures)

### Generation — `constraint.py` (unchanged) + `agent.py`

```python
# constraint.py — the codec, exactly as pared (Schema/Regex/Choice/Grammar → Constraint[T])
def Schema[M: BaseModel](model: type[M], *, strict: bool = True) -> Constraint[M]: ...
def Regex(pattern: str) -> Constraint[str]: ...
def Choice[S: str](*options: S) -> Constraint[S]: ...
def Grammar(ebnf: str) -> Constraint[str]: ...

@dataclass(frozen=True)
class AgentSpec[T]:
    name: str
    constraint: Constraint[T]
    instructions: str
    adapter: str | None = None
    settings: Settings = Settings()

class Backend:                              # sole pydantic-ai importer; one shared httpx client
    def build[T](self, spec: AgentSpec[T]) -> Agent[T]: ...   # wraps a DBOSAgent
    async def aclose(self) -> None: ...

class Agent[T]:
    async def run(self, prompt: str) -> T: ...
        # DURABLE: its own workflow at top level; a nested workflow/steps inside a user @DBOS.workflow.
        # raises ConstraintViolation / model errors (DBOS applies the step retry policy).
    @property
    def raw(self) -> DBOSAgent: ...         # escape hatch
```

### Authority — `authority.py` (durable effects, exactly-once)

```python
@dataclass(frozen=True)
class Decision: allowed: bool; reason: str = ""

class Authorizer[C](Protocol):
    def decide(self, command: C) -> Decision: ...      # total; data; never raises for a domain reason

class Effector[C](Protocol):
    async def run(self, command: C) -> Any: ...        # implemented as @DBOS.step → exactly-once + retry

class Allowlist[C](Authorizer[C]): ...                 # default-deny; wraps each rule → Decision(False)
class Subprocess(Effector[BaseModel]): ...             # @DBOS.step; argv off the loop
class FornixEffector(Effector[BaseModel]): ...         # @DBOS.step; integrations/fornix.py

@dataclass(frozen=True)
class Denied: reason: str; command: Any                # the one decision-as-data

async def execute[C](authorizer: Authorizer[C], effector: Effector[C], command: C,
                     *, key: str | None = None) -> Denied | Any:
    # key → SetWorkflowID(key): the natural business id makes the effect exactly-once per command
    # (spike 5). Default: a fresh uuid (exactly-once per invocation).
    d = authorizer.decide(command)                     # decide before run
    if not d.allowed: return Denied(d.reason, command) # denial is data (not retried)
    return await effector.run(command)                 # exactly-once effect; failure raises → DBOS retry
```

### Human-in-the-loop — `approval.py` (durable pause for a human)

```python
class Approval[C]:
    async def request(self, command: C, *, to: str, timeout: float | None = None) -> Decision: ...
        # durably blocks on DBOS recv; survives process restarts; timeout → Decision(False, "timeout")
        # → a durable Authorizer you can drop into execute(): approve then effect, exactly-once.

class ApprovalClient:                                  # out-of-process approver (a CLI/UI/bot)
    def approve(self, workflow_id: str, *, reason: str = "") -> None: ...   # DBOS.send/set_event
    def deny(self, workflow_id: str, *, reason: str) -> None: ...
    def pending(self) -> list[WorkflowStatus]: ...     # what's awaiting approval right now
```

### Concurrency, scheduling, observability, eval — `plane.py`

```python
def launch(*, database_url: str | None = None) -> None: ...   # DBOS() + DBOS.launch(); SQLite if url is None

class Queue:                                           # durable concurrency / batch / rate-limit
    def __init__(self, name: str, *, concurrency: int | None = None,
                 rate_limit: tuple[int, float] | None = None) -> None: ...
    async def submit[T](self, agent: Agent[T], prompt: str, *, key: str | None = None) -> WorkflowHandle[T]: ...
        # key → SetWorkflowID(key): dedup/exactly-once per business id (spike 5)

def schedule(cron: str) -> Callable[[WF], WF]: ...     # decorator: run a user workflow on a durable timer

# observability over the workflow store (thin wrappers on DBOS):
def workflows(*, status: str | None = None, name: str | None = None) -> list[WorkflowStatus]: ...
def status(workflow_id: str) -> WorkflowStatus: ...
def fork(workflow_id: str, *, from_step: int) -> WorkflowHandle[Any]: ...
def cancel(workflow_id: str) -> None: ...

# eval / dual-path as a composable primitive, not a subsystem:
async def compare[T](primary: Agent[T], reference: Agent[T], prompt: str) -> Comparison[T]: ...
```

---

## What the user writes (the toolkit model in action)

```python
# 1) an autonomous, crash-safe, exactly-once fix workflow — user-authored
@DBOS.workflow()
async def triage_and_fix(msg: str) -> Denied | EditResult:
    plan = await file_edit_agent.run(msg)          # durable generate → FileEditPlan (typed)
    return await execute(allowlist, fornix_effector, plan)   # authorize + exactly-once contained effect

# 2) same, but gated on a human — durably pauses, survives restarts
@DBOS.workflow()
async def guarded_deploy(msg: str) -> Denied | DeployResult:
    cmd = await deploy_agent.run(msg)              # durable generate → DeployCommand
    decision = await approvals.request(cmd, to="ops")   # durable pause until approve/deny
    if not decision.allowed: return Denied(decision.reason, cmd)
    return await subprocess_effector.run(cmd)      # exactly-once

# 3) a scheduled, batched, rate-limited sweep
@schedule("0 * * * *")                             # every hour, durable
@DBOS.workflow()
async def hourly_sweep() -> None:
    handles = [await q.submit(triage_agent, item) for item in await backlog()]
    await DBOS.asyncio_wait(handles)               # durable fan-out
```

The library never wrote a pipeline; it handed the user durable `run` / `execute` / `request` / `submit`
/ `schedule` and got out of the way. That's "plane, not framework."

---

## Module & dependency map

```
errors.py       exception tree (layer-less)
constraint.py   Constraint[T] + Schema/Regex/Choice/Grammar          (pure; linchpin of generation)
agent.py        AgentSpec[T], Backend, Agent[T] (wraps DBOSAgent)     ── sole pydantic-ai importer
authority.py    Decision, Authorizer, Effector (@DBOS.step), Allowlist, execute()
approval.py     Approval, ApprovalClient (DBOS recv/events)
plane.py        launch(), Queue, schedule(), observability, compare()
config.py       spec_from_config + allow_modules (serialization edge; still useful)
integrations/fornix.py   FornixEffector
```

- **Core deps:** `pydantic`, `pydantic-ai-slim[openai] >=2.11,<3` (with `durable_exec.dbos`),
  `dbos >=2.23`, `httpx`. **Postgres** is the production datastore; **DBOS supports SQLite for
  local/dev/test** → no Postgres needed to `pytest` (verify).
- `observe`/`dual_path` as a *separate subsystem* is **gone** — subsumed by the durable store +
  `compare()`. `[observe]` extra collapses into core (or a tiny `psycopg` note for Postgres).
- Identity note: the center moved from the constraint codec to the durable plane. A codec-centric name
  (`constric`) under-sells it; consider a plane/durability-evoking name. (Still your call — DECISION H.)

---

## Foundational spikes — ALL CONFIRMED (2026-07-18, dbos 2.23.0 · pydantic-ai 2.11.0)

Every load-bearing assumption was verified against the real runtime. Spike files in
`REVIEW_SPIKES/spike{1..5}_*.py` + `capture.py`/`bodies.json`. The design stands.

1. **SQLite is first-class for dev/test — no Postgres needed.** ✓
   `DBOSConfig(name=…, system_database_url="sqlite:///<abs path>", use_listen_notify=False)` — DBOS has
   a native SQLite system-db backend (single file, migrations 1–34 applied on launch; it logs "SQLite
   … for development and testing; PostgreSQL recommended for production"). **The whole test story is
   SQLite-only** — no container, no fixture. `launch(database_url=None)` defaults to SQLite; production
   passes a Postgres URL.
2. **`DBOSAgent` top-level *and* nested both work, zero ceremony.** ✓
   Top-level `.run` (no surrounding workflow, no `SetWorkflowID`) **auto-creates its own root
   durable workflow** (`<name>.run`, SUCCESS). Nested inside a user `@DBOS.workflow` it becomes a
   **deterministic child workflow** (`<parent-id>-N`), so it survives parent replay. So `Agent.run`
   is durable in both contexts with no extra work — exactly the "always-on" model.
3. **`SetWorkflowID` is contextvar-scoped — async-safe (retires old RISKS R5).** ✓
   5 concurrent legs via `asyncio.gather`, each `with SetWorkflowID(f"wid-{i}")`, with interleaved
   `sleep`s: every leg's `DBOS.workflow_id` stayed pinned to its own id, no crossing. Safe for the
   library's asyncio execution model.
4. **Human-in-the-loop durable approval works — `recv` is the primitive.** ✓
   Workflow blocks on `recv_async(topic, timeout_seconds)` → status is durably **PENDING** (persisted;
   survives restart, recovery re-enters the `recv`); external `send_async(workflow_id, {...}, topic)`
   resumes it → SUCCESS; timeout → `recv` returns **None** → clean denied path. Pair with
   `set_event_async`/`get_event_async` (workflow→outside) to publish the pending command for a human/UI
   to inspect. `Approval.request` = `recv` + a `set_event`; `ApprovalClient.approve/deny` = `send`.
5. **Exactly-once = `SetWorkflowID` as the business key.** ✓
   A side-effecting `@DBOS.step` run twice under the same `SetWorkflowID("order-42")` fired **exactly
   once** (second was a cached replay); an independent key fired independently. **There is no separate
   idempotency API — the workflow id *is* the dedup key.** So `Effector`/queue ergonomics = "pass the
   logical command id as the workflow id." (Design implication: `execute`/`Queue.submit` take an
   optional `key: str` → `SetWorkflowID(key)`; default is a fresh uuid.)

**BLOCKER 3 (wire re-capture) — RETIRED.** ✓ All four `.wire()` bodies reproduce on 2.11 with no
drift: `NativeOutput` → clean `response_format: json_schema`, no tool leakage (pydantic-ai owns the
body: class-name `name`, OpenAI strict-transform adds `additionalProperties:false` + strips `title`,
`$defs`/`$ref` preserved); the three string modes → `structured_outputs.{regex,choice,grammar}` at the
request top level verbatim; `extra_body` via `model_settings={"extra_body":{…}}` hoists top-level.
**decoder.py's mode→body table carries verbatim.** (`REVIEW_SPIKES/capture.py`, `bodies.json`.) Note:
a *plain* model (no `NativeOutput`) still rides the function-tool path, so the library must keep
applying `NativeOutput` for schema mode (it already does).

## Design facts these spikes lock in

- **The public API is async-first.** DBOS's sync entrypoints (`list_workflows`, `start_workflow`,
  `send`, `recv`, `set_event`, `get_event`) **raise inside a running event loop** — only the `*_async`
  variants work there. The library's whole surface (`run`, `execute`, `Queue.submit`, `approvals.*`,
  observability) is `async def`; no sync mirror in v1.
- **DBOS is a process-global singleton.** One `launch()` per process; register all workflow/step
  decorators *before* launch; `destroy()` to tear down. Isolated tests run as separate processes (or
  `destroy()` between) — matches how v2's `dual_path/runtime.py` already structures it.
- **Determinism rule (document it):** workflow bodies must be deterministic between steps — LLM/effect
  calls are steps (fine); forbid ambient `time`/random/IO in the workflow body itself.

## Remaining (smaller) items, not blockers

- Queue rate-limit shape on 2.23 (`Queue(concurrency=…, limiter=…)`) — confirm the exact kwarg when
  wiring `Queue`.
- Scheduled-workflow decorator surface (`@DBOS.scheduled(cron)` vs the `ScheduleInput` API) — confirm
  when wiring `schedule()`.
- Name/identity (DECISION H, still user's): the codec-centric `constric` undersells a durable plane.
