# DESIGN — module-by-module spec for the durable agent plane (structured-agents v3)

The implementable heart of the durable-plane plan. Supersedes `DESIGN.md`. For each module:
**responsibility · public surface (real signatures) · dependencies (one-way) · invariants · impl
notes (grounded in the spikes).** Signatures written to typecheck under `ty` 0.0.46; PEP 695 generics
(`class Foo[T]`, `def f[T]`), `from __future__ import annotations` in every module.

- **Package:** `structured_agents/` (name kept; repo rebuilt in place — the `v0.2.0` tag preserves the
  frozen artifact).
- **Core deps:** `pydantic`, `pydantic-ai-slim[openai] >=2.11,<3`, `dbos >=2.23`, `httpx`.
- **Test datastore:** SQLite (spike 1) — no Postgres for `pytest`. Production passes a Postgres URL.

Every empirical claim traces to `REVIEW_SPIKES/` (S1 Choice, S2 encoding, S3 pydantic-ai surface,
`capture.py`/`bodies.json` wire shapes, `spike{1..5}` DBOS). Nothing here is assumed.

---

## Layer stack & dependency rule

```
Layer 2  approval.py   Approval, ApprovalClient          (→ authority)
         plane.py      configure/launch, Queue, schedule, observability, compare   (→ agent, authority)
         config.py     spec_from_config, allow_modules   (→ constraint, agent)
         integrations/fornix.py   FornixEffector          (→ authority)
Layer 1  agent.py      AgentSpec[T], Backend, Agent[T]    ── SOLE pydantic_ai.models.openai importer (→ constraint)
         authority.py  Decision, Authorizer, Effector, Allowlist, execute   (generic over C; → errors only)
Layer 0  constraint.py Constraint[T], Schema/Regex/Choice/Grammar, WireSpec  (→ errors, NativeOutput marker)
         errors.py     exception tree (layer-less)
```

A module imports only from lower/equal layers (+ `errors`). Enforced by an AST test (T8). `authority`
is deliberately **independent of `agent`/`constraint`** — it is command-shape-agnostic (`C` is any
validated value), which is what lets a durable effect pipeline be reasoned about on its own.

**Lifecycle invariant (from spike 2 + the DBOS singleton reality):**
```
configure(...)        # creates the DBOS singleton (SQLite default); decorators may now register
build agents/effectors/workflows   # DBOSAgent construction + @DBOS.step/@DBOS.workflow register HERE
launch()              # DBOS.launch() — after everything is registered
... run durable workflows ...
shutdown()            # DBOS.destroy()
```
Registration (agent/effector/workflow construction) **must precede `launch()`**. The `conftest`
enforces this ordering for tests; the public docs state it for users.

---

## `errors.py` — programmer/config error tree (layer-less)

**Responsibility.** Exceptions for **bugs and misconfiguration only**. Runtime domain signals are
*return values*, not exceptions (`Decision`/`Denied`); the exception here that *is* a runtime signal is
`ConstraintViolation` (a backend that didn't enforce → `Agent.run` raises it — a genuine failure, not a
decision).

```python
class StructuredAgentsError(Exception): ...             # base
class ConfigError(StructuredAgentsError): ...           # invalid spec/config
class ConstraintConfigError(ConfigError): ...           # a Constraint built inconsistently
class ConstraintCompileError(StructuredAgentsError): ...# check() failed (grammar-check extra)
class BackendCapabilityError(StructuredAgentsError): ...# agent needs a cap the backend lacks
class AuthorityError(StructuredAgentsError): ...         # misconfigured authority (not a denial)
class ConstraintViolation(StructuredAgentsError):        # parse() rejected raw output; raised by Agent.run
    def __init__(self, message: str, *, raw: str) -> None: ...
    raw: str
```

**Invariants.** (i) Imports nothing else in the package. (ii) Every message names the offending
agent/spec and states the remedy (v2 strength, kept).

---

## Layer 0 — `constraint.py` (the codec; pure)

**Responsibility.** The bidirectional codec: one value that shapes the wire out (`wire()`) and turns
raw output into typed `T` (`parse()`). The wire bodies are the **verified-verbatim** table
(`bodies.json`, captured on 2.11 — no drift).

```python
@dataclass(frozen=True)
class WireSpec:
    output_type: Any                                  # what pydantic_ai Agent(output_type=) receives
    extra_body: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Constraint[T](Protocol):
    def wire(self) -> WireSpec: ...
    def parse(self, raw: Any) -> T: ...               # guard/coerce → T; raise ConstraintViolation on a miss
    def check(self) -> None: ...                      # optional xgrammar compile-check; default no-op
    def to_config(self) -> dict[str, Any]: ...        # {"kind": ..., ...}

def Schema[M: BaseModel](model: type[M], *, strict: bool = True) -> Constraint[M]: ...
def Regex(pattern: str) -> Constraint[str]: ...
def Choice[S: str](*options: S) -> Constraint[S]: ...      # ty infers Constraint[Literal[...]] (S1)
def Grammar(ebnf: str) -> Constraint[str]: ...
```

**Wire/parse table (bodies verbatim from `bodies.json`):**

| ctor | `wire().output_type` | `wire().extra_body` | `parse(raw)` |
|---|---|---|---|
| `Schema(M)` | `NativeOutput(M, strict=strict)` | `{}` | **identity** — pydantic-ai already returned a validated `M` |
| `Regex(p)` | `str` | `{"structured_outputs":{"regex":p}}` | `re.fullmatch(p, raw)` or raise `ConstraintViolation` → `str` |
| `Choice(*o)` | `str` | `{"structured_outputs":{"choice":[*o]}}` | membership or raise → the `Literal` |
| `Grammar(e)` | `str` | `{"structured_outputs":{"grammar":e}}` | passthrough → `str` |

**Impl notes.** `Schema.parse` = identity is honest (rich path is the only path — no `closed`).
`check()`: `Schema` → `xgr.Grammar.from_json_schema(...)`; `Regex`/`Grammar` compile pattern/EBNF;
`Choice` no-op; all guarded by a try-import of `xgrammar` (absent → no-op). Because `Schema` holds a
**fully-formed** model class, `model.model_json_schema()` is complete — v2's A2 (parent empty schema)
is structurally impossible (no subclass).

**Dependencies.** `errors`; `pydantic`; `pydantic_ai.output.NativeOutput` (an inert declarative marker —
**not** `models.openai`). `NativeOutput` may be imported lazily inside `Schema.wire()` to keep merely
constructing a `Schema` dependency-light. **No DBOS. No `models.openai`.**

**Invariants.** (i) Pure — no DBOS, no transport, no side effects. (ii) `parse` always runs at
`Agent.run`; for `Schema` it is identity (the enforcement is pydantic-ai's `NativeOutput` validation),
for string modes it is the client-side guard. (iii) The wire table matches `bodies.json` byte-for-shape
(T2).

---

## Layer 1 — `agent.py` (AgentSpec[T], Backend, Agent[T]) — SOLE `pydantic_ai.models.openai` importer

**Responsibility.** Bind a typed spec to a **durable** pydantic-ai agent (`DBOSAgent`), run it, meet
`parse`. `Agent.run(prompt) -> T` (raise on failure); durable by construction (spike 2).

```python
@dataclass(frozen=True)
class Settings:                        # typed sampling params (a typo is a ty error)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentSpec[T]:
    name: str
    constraint: Constraint[T]          # carries T
    instructions: str                  # plain string (context axis deferred)
    adapter: str | None = None         # served LoRA/model name on the wire `model` field
    settings: Settings = Settings()

class BackendCaps(BaseModel):
    xgrammar: bool = True
    lora: bool = True

@runtime_checkable
class AdapterProvider(Protocol):
    async def resolve(self, adapter: str) -> str: ...    # logical → served name; ensure loaded

@dataclass(frozen=True)
class RequestRecord:                   # exact on-wire bytes (capture escape hatch)
    body: dict[str, Any]

class Backend:
    """Server + caps + ONE shared httpx client. Sole importer of pydantic_ai.models.openai.
    Building an Agent registers a DBOSAgent — do it BEFORE plane.launch()."""
    def __init__(self, *, base_url: str, api_key: str = "sk-none", default_model: str,
                 caps: BackendCaps = BackendCaps(), capture: bool = False,
                 adapters: AdapterProvider | None = None,
                 http_client: httpx.AsyncClient | None = None) -> None: ...
    def build[T](self, spec: AgentSpec[T]) -> Agent[T]: ...
    async def aclose(self) -> None: ...                  # closes the shared client

class Agent[T]:
    async def run(self, prompt: str) -> T: ...
        # DURABLE: top-level → its own root workflow; nested in a user @DBOS.workflow → child (<parent>-N)
        # raises ConstraintViolation / model+transport errors (DBOS applies the step retry policy)
    async def run_with_record(self, prompt: str) -> tuple[T, RequestRecord]: ...  # requires capture=True
    @property
    def raw(self) -> DBOSAgent: ...                       # escape hatch
```

**`Backend.build` (the wiring):**
```python
def build[T](self, spec: AgentSpec[T]) -> Agent[T]:
    self._gate(spec)                                  # BackendCapabilityError (Regex/Choice/Grammar vs xgrammar; adapter vs lora)
    spec.constraint.check()                           # compile-check if grammar-check present
    w = spec.constraint.wire()                        # WireSpec: output_type + extra_body
    settings = self._merge(spec.settings, w.extra_body)  # extra_body MERGE, decoder keys win (A4)
    pa = PydanticAgent(self._model(spec.adapter),     # from pydantic_ai import Agent as PydanticAgent
                       output_type=w.output_type,
                       model_settings=settings, instructions=spec.instructions, name=spec.name)
    dbos_agent = DBOSAgent(pa, name=spec.name)        # pydantic_ai.durable_exec.dbos — registers durable steps
    return Agent(spec, dbos_agent, capture=self.capture)
```

**`Agent.run`:**
```python
async def run(self, prompt: str) -> T:
    raw = await self._dbos_agent.run(prompt)          # durable model loop; may raise UnexpectedModelBehavior
    return self._spec.constraint.parse(raw.output)    # identity (schema) / guard (string modes); may raise ConstraintViolation
```
(`raw.usage` is a property (S3); `raw.output` is the generic `OutputDataT` field — for `Schema` it is
`M`, for string modes `str`. Type flows: `AgentSpec[FileEditPlan] → build → Agent[FileEditPlan] → run
→ FileEditPlan`, no cast.)

**Capture (optional, `capture=True`).** An httpx `event_hooks={"request": …}` on the shared client
records the exact bytes; `run_with_record` returns `(T, RequestRecord)` — per-run attribution is
structural (the record is returned by the same call; no `ContextVar`, no A5 race). Off by default.

**Dependencies.** `constraint`, `errors`, `httpx`, `pydantic_ai` (+`.models.openai`,
+`.durable_exec.dbos.DBOSAgent`, +`.output.NativeOutput`), `dbos` (indirectly via DBOSAgent).

**Invariants.** (i) Only this module imports `pydantic_ai.models.openai` (T8). (ii) `raw.usage`
accessed as a property (S3 — A1 impossible at the pin). (iii) `extra_body` merged, decoder keys win
(A4). (iv) `Settings` is a typed struct — a setting typo is a `ty` error. (v) One shared httpx client
per `Backend`, closed by `aclose`. (vi) `build` must run before `launch()` (registers the DBOSAgent).

---

## Layer 1 — `authority.py` (decision is data; failure is an exception; effects are exactly-once)

**Responsibility.** Split deciding from doing; the effect is a durable, exactly-once `@DBOS.step`.
Generic over the command type `C` and the effect result `R` — knows nothing about agents.

```python
@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""

@dataclass(frozen=True)
class Denied:                          # the one runtime decision-as-data
    reason: str
    command: Any

@runtime_checkable
class Authorizer[C](Protocol):
    def decide(self, command: C) -> Decision: ...        # TOTAL; never raises for a domain reason

@runtime_checkable
class Effector[C, R](Protocol):
    async def run(self, command: C) -> R: ...            # a @DBOS.step → exactly-once + retry; raises on real failure

class Allowlist[C](Authorizer[C]):
    """Default-deny. Wraps each rule in try/except → Decision(False) ONCE, at the boundary (B2)."""
    def __init__(self, rules: dict[str, Callable[[C], bool]]) -> None: ...
    def decide(self, command: C) -> Decision: ...        # unknown/unmatched/raising rule → Decision(False)

def all_of[C](*a: Authorizer[C]) -> Authorizer[C]: ...
def any_of[C](*a: Authorizer[C]) -> Authorizer[C]: ...

# built-in effectors (each decorates run with @DBOS.step):
class Null[C](Effector[C, None]):        async def run(self, command: C) -> None: ...        # dry-run: records intent, no effect
@dataclass(frozen=True)
class ProcessResult: returncode: int; stdout: str; stderr: str
class Subprocess(Effector[BaseModel, ProcessResult]):
    async def run(self, command: BaseModel) -> ProcessResult: ...   # validated argv off the loop, as a step

async def execute[C, R](authorizer: Authorizer[C], effector: Effector[C, R], command: C,
                        *, key: str | None = None) -> Denied | R:
    d = authorizer.decide(command)                       # decide BEFORE run (B1)
    if not d.allowed:
        return Denied(d.reason, command)                 # denial is data (never retried)
    return await effector.run(command)                   # exactly-once step; real failure raises
```

**Impl notes (grounded in spike 5).** Exactly-once comes from a `@DBOS.step` running inside a durable
workflow keyed by the business id. Two usages:
- **Inside a user `@DBOS.workflow`** (the common case): `execute` runs the effector step as part of that
  workflow; its replay after a crash does not re-fire the step.
- **Standalone with `key`**: `execute` wraps the effect in a workflow whose id is `key`
  (`SetWorkflowID(key)`), so a second `execute` with the same logical command id dedups to one effect.
The exact wrapping is finalized in Phase 3; the **contract** is "exactly-once per `key`."

**Invariants.** (i) `Authorizer.decide` is total (never raises for a domain reason). (ii) Effectors are
`async` `@DBOS.step`s (off the loop; exactly-once). (iii) Denials are `Denied` data; only *bugs* and
*real effect failures* raise. (iv) Default-deny (`Allowlist`). (v) Residual hazard, documented:
`Effector.run` is directly callable without a `Decision` — the blessed path is `execute`.

**Dependencies.** `errors`, `dbos`. **No `agent`, no `constraint`, no pydantic-ai.**

### `integrations/fornix.py` — `FornixEffector` (optional, `[fornix]` seam; stdlib subprocess)

```python
class FornixEffector(Effector[BaseModel, ProcessResult]):
    async def run(self, command: BaseModel) -> ProcessResult: ...
        # @DBOS.step: serialize validated command → argv (never a shell string), shell
        # `fornix box --check … -- <argv>`, parse the one-line JSON Result. Absent fornix → BackendCapabilityError.
```
Net dependency delta: **zero** (subprocess boundary). Compose as `execute(Allowlist(...),
FornixEffector(), cmd, key=...)`.

---

## Layer 2 — `approval.py` (human-in-the-loop durable pause)

**Responsibility.** A durable Authorizer: pause a workflow until a human approves/denies (spike 4).

```python
@dataclass(frozen=True)
class PendingApproval:
    workflow_id: str
    command: Any                       # from the workflow's published "pending_command" event
    to: str

class Approval[C]:
    def __init__(self, *, topic: str = "approval") -> None: ...
    async def request(self, command: C, *, to: str, timeout: float | None = None) -> Decision: ...
        # MUST be called inside a @DBOS.workflow. Publishes the command (set_event_async "pending_command"),
        # then blocks on recv_async(topic, timeout_seconds). Message → Decision; timeout/None → Decision(False,"timeout").

class ApprovalClient:                  # out-of-process approver (a CLI/UI/bot)
    def __init__(self, *, topic: str = "approval") -> None: ...
    async def approve(self, workflow_id: str, *, reason: str = "") -> None: ...   # send_async {allowed:True}
    async def deny(self, workflow_id: str, *, reason: str) -> None: ...            # send_async {allowed:False}
    async def pending(self) -> list[PendingApproval]: ...   # PENDING workflows + their published command
```

**Impl notes (spike 4).** `recv_async` is the durable blocking primitive — the PENDING state persists
in the system db and survives restart (recovery re-enters the `recv`). `set_event_async`/`get_event_async`
carry the command *out* for inspection. Compose:
```python
@DBOS.workflow()
async def guarded(msg: str) -> Denied | R:
    cmd = await agent.run(msg)
    d = await approvals.request(cmd, to="ops", timeout=3600)
    if not d.allowed: return Denied(d.reason, cmd)
    return await effector.run(cmd)
```

**Invariants.** (i) `Approval.request` runs inside a workflow (uses `recv`). (ii) Timeout → denied
(data). (iii) Uses only `*_async` DBOS entrypoints (sync ones raise inside a loop).

**Dependencies.** `authority` (`Decision`), `errors`, `dbos`.

---

## Layer 2 — `plane.py` (lifecycle · queues · schedules · observability · eval)

**Responsibility.** The durable-plane services the toolkit exposes over DBOS.

```python
def configure(*, database_url: str | None = None, app_name: str = "structured_agents",
              **dbos_config: Any) -> None: ...   # creates the DBOS singleton; SQLite if database_url is None
def launch() -> None: ...                        # DBOS.launch() — after all agents/effectors/workflows registered
async def shutdown() -> None: ...                # DBOS.destroy()

class Queue:                                     # durable concurrency / batch / rate-limit over Agent.run
    def __init__(self, name: str, *, concurrency: int | None = None,
                 rate_limit: tuple[int, float] | None = None) -> None: ...   # (limit, period_seconds)
    async def submit[T](self, agent: Agent[T], prompt: str, *, key: str | None = None) -> WorkflowHandle[T]: ...

def schedule(cron: str) -> Callable[[WF], WF]: ...   # decorator: run a user workflow on a durable timer

# observability (thin async wrappers over DBOS; sync variants raise inside a loop):
async def workflows(*, status: str | None = None, name: str | None = None) -> list[WorkflowStatus]: ...
async def status(workflow_id: str) -> WorkflowStatus: ...
async def fork(workflow_id: str, *, from_step: int) -> WorkflowHandle[Any]: ...
async def cancel(workflow_id: str) -> None: ...

# eval / dual-path as a composable primitive (subsumes v2 dual_path/):
@dataclass(frozen=True)
class Comparison[T]:
    prompt: str
    primary: T
    reference: T
    primary_workflow_id: str
    reference_workflow_id: str
async def compare[T](primary: Agent[T], reference: Agent[T], prompt: str,
                     *, key: str | None = None) -> Comparison[T]: ...   # both legs durable; store IS the record
```

**Impl notes.** `configure`/`launch`/`shutdown` encode the singleton lifecycle (spike 1/2). `Queue`
wraps `dbos.Queue` (confirm the 2.23 rate-limit kwarg name while wiring). `schedule` wraps DBOS
scheduled workflows (confirm the 2.23 decorator surface while wiring). `compare` runs both legs under
their own `SetWorkflowID` (spike 3 proved isolation) and returns the pair; re-running the same `key`
does not double-insert (durable idempotency — no bespoke store).

**Invariants.** (i) `launch()` after all registration. (ii) All read/query methods are `async` (sync
raise inside a loop). (iii) `compare` legs are isolated by `SetWorkflowID` (spike 3).

**Dependencies.** `agent`, `authority`, `errors`, `dbos`.

---

## Layer 2 — `config.py` (the serialization edge)

**Responsibility.** The *one* place strings become code, gated by an allowlist (DECISION K).

```python
def constraint_from_config(d: dict[str, Any], *, allow_modules: frozenset[str]) -> Constraint[Any]: ...
def spec_from_config(d: dict[str, Any], *, allow_modules: frozenset[str]) -> AgentSpec[Any]: ...
def register_constraint(kind: str, from_config: Callable[[dict[str, Any]], Constraint[Any]]) -> None: ...
# + entry-point group "structured_agents.constraints", discovered lazily.
```
Canonical forms: `{"kind":"schema","ref":"pkg:M"}`, `{"kind":"regex","pattern":…}`,
`{"kind":"choice","options":[…]}`, `{"kind":"grammar","ebnf":…}`. Round-trips with `to_config`.

**Invariants.** (i) Refuses any `ref` whose module prefix ∉ `allow_modules` → `ConfigError`. (ii) The
**only** module doing `importlib` on data (T8). (iii) In-code plugins need no registration (structural
typing).

**Dependencies.** `constraint`, `agent`, `errors`.

---

## Package surface (`__init__.py`)

Public vocabulary (locked): `Constraint`, `Schema`, `Regex`, `Choice`, `Grammar`, `WireSpec`;
`AgentSpec`, `Backend`, `BackendCaps`, `Agent`, `Settings`, `AdapterProvider`; `Decision`, `Denied`,
`Authorizer`, `Effector`, `Allowlist`, `all_of`, `any_of`, `Null`, `Subprocess`, `execute`;
`Approval`, `ApprovalClient`; `configure`, `launch`, `shutdown`, `Queue`, `schedule`, `workflows`,
`status`, `fork`, `cancel`, `compare`, `Comparison`; the `errors` tree. `pydantic_ai.Agent` collision
resolved by import discipline (`from pydantic_ai import Agent as PydanticAgent` inside `agent.py` only).

---

## End-to-end (the toolkit in action, real signatures)

```python
import structured_agents as sa
from structured_agents import Schema, AgentSpec, Backend, Allowlist, Subprocess, execute, Approval, Denied
from dbos import DBOS

sa.configure(database_url=None)                          # SQLite (dev); a Postgres URL in prod
backend = Backend(base_url="http://tower:8000/v1", default_model="qwen3-4b")
deploy_agent = backend.build(AgentSpec("deploy", Schema(DeployCommand),
                                       instructions="Emit one deploy command."))
approvals = Approval[DeployCommand]()
policy = Allowlist[DeployCommand]({"safe": lambda c: c.target in {"staging"}})

@DBOS.workflow()                                         # the USER authors the workflow
async def guarded_deploy(msg: str) -> Denied | ProcessResult:
    cmd = await deploy_agent.run(msg)                    # durable generate → typed DeployCommand
    d = await approvals.request(cmd, to="ops", timeout=3600)   # durable pause; survives restart
    if not d.allowed:
        return Denied(d.reason, cmd)
    return await execute(policy, Subprocess(), cmd, key=cmd.id)  # exactly-once contained effect

sa.launch()                                              # after all registration
# await guarded_deploy("ship v2 to staging")            # runs durably; approve out-of-band via ApprovalClient
```

---

## Traceability to the spikes (nothing assumed)

| Design element | Spike |
|---|---|
| `Choice[S: str] → Constraint[Literal]`, `Schema.parse` typing | S1, S2 |
| `raw.usage` property, `output_type=str`/`NativeOutput` accepted | S3 |
| wire bodies verbatim (all four modes, 2.11) | `capture.py`/`bodies.json` |
| SQLite test datastore, idempotent replay | spike 1 |
| `Agent.run` durable top-level + nested child | spike 2 |
| `SetWorkflowID` async-safe (compare legs, queues) | spike 3 |
| `Approval.request`/`ApprovalClient` durable pause | spike 4 |
| `execute(key=…)` exactly-once per business id | spike 5 |
| async-first surface, singleton lifecycle | spike cross-cutting notes |
