# structured-agents v3 — The Constraint-Codec Architecture

**Status:** concept / design target (greenfield rewrite)
**Date:** 2026-07-17
**Supersedes (conceptually):** `../02-library-wrapper/CONCEPT.md` (the v2 design shipped as v0.2.0)
**Relationship to v2:** a ground-up re-derivation, not a refactor. v2 is a well-built set of
modules; v3 is the same *domain* re-expressed around a single central abstraction so the
seams that v2 keeps working around disappear by construction.

> **Mandate for this document:** describe the cleanest, most elegant architecture possible for
> this domain, unconstrained by the v2 code. Correctness, layering, and type-honesty come first;
> backward compatibility is explicitly *not* a goal (the one shipped consumer, Lodestar, uses the
> `closed` path, which v3 preserves as a first-class preset — see §10).

---

## Table of contents

1. [The one-sentence thesis](#1-the-one-sentence-thesis)
2. [What went right in v2, and the four tensions that remain](#2-what-went-right-in-v2-and-the-four-tensions-that-remain)
3. [Design principles](#3-design-principles)
4. [The linchpin: `Constraint[T]` as a bidirectional codec](#4-the-linchpin-constraintt-as-a-bidirectional-codec)
5. [The `Outcome[T]` spine: decisions-as-data, uniformly](#5-the-outcomet-spine-decisions-as-data-uniformly)
6. [Authority as `Authorizer × Effector`](#6-authority-as-authorizer--effector)
7. [The layer stack and dependency direction](#7-the-layer-stack-and-dependency-direction)
8. [The wire layer (Layer 0) and `closed` as a preset](#8-the-wire-layer-layer-0-and-closed-as-a-preset)
9. [The agent layer (Layer 2): `AgentSpec[T]`, `Backend`, `Agent[T]`](#9-the-agent-layer-layer-2-agentspect-backend-agentt)
10. [The fleet + pipeline layer (Layer 3)](#10-the-fleet--pipeline-layer-layer-3)
11. [Config vs code: the serialization edge](#11-config-vs-code-the-serialization-edge)
12. [Capture, made explicit](#12-capture-made-explicit)
13. [Observation / dual-path reborn](#13-observation--dual-path-reborn)
14. [Module layout](#14-module-layout)
15. [End-to-end worked example + type-flow](#15-end-to-end-worked-example--type-flow)
16. [What v3 deliberately keeps from v2](#16-what-v3-deliberately-keeps-from-v2)
17. [Testing strategy](#17-testing-strategy)
18. [Open questions](#18-open-questions)
19. [Migration map (v2 → v3)](#19-migration-map-v2--v3)
20. [The three axes: constraint · adapter · context](#20-the-three-axes-constraint--adapter--context)
21. [The adapter axis & capability plugins](#21-the-adapter-axis--capability-plugins)
22. [The context axis: cache-cooperative input](#22-the-context-axis-cache-cooperative-input)
23. [Extensibility & plugin seams](#23-extensibility--plugin-seams)
24. [Naming glossary](#24-naming-glossary)

---

## 1. The one-sentence thesis

> **Constrain a small local model's output at three orthogonal layers — *syntax* (grammar),
> *shape* (schema), and *authority* (policy) — and make each layer an explicit, typed,
> composable value that flows through one uniform pipeline.**

Every design decision below is derived from taking "explicit, typed, composable value" literally
for all three layers. v2 honored this for authority (the `Executor`) but not for the constraint,
which it smeared across four places. v3 fixes that first, and the rest falls out.

---

## 2. What went right in v2, and the four tensions that remain

v2's architecture is genuinely good — disciplined layering, a wire-grounded decode-mode table,
a real authority boundary, honest docstrings, 98% test coverage. v3 keeps all of that (§16). But
five findings from the v0.2.0 review and one deferred item all trace back to **the same root
cause: the library's central concept — a *constraint* — is not a first-class value.** It is
currently expressed by the interaction of four things:

| Where the "constraint" lives in v2 | What it does | The smell |
|---|---|---|
| `ConstrainedOutput` dunder ClassVars (`__decode_mode__`, `__regex__`, …) | declares mode + params on a model subclass | stringly-magic; for regex/choice the subclass has **dead fields** and is only a spec carrier |
| `DecoderSpec` (`decoder.py`) | serializable mode + params | a second, parallel encoding of the same thing |
| `DecoderSpec.apply()` | maps spec → `(output_type, extra_body)` | the *outbound* half of the codec, isolated from… |
| `StructuredAgent._guard()` (`agent.py`) | validates bare-string output client-side | …the *inbound* half, which lives in a different module |

Because the constraint isn't one value, four otherwise-unrelated problems appear:

- **`Any` at the final seam.** `AgentResult[OutputT]` is generic but nothing fills `OutputT`;
  `run()` returns `AgentResult[Any]`. A "typed outputs" library hands callers `Any`. Making
  `StructuredAgent` generic requires a `cast` that is an *unverified assertion* (the string
  `output_type_ref` can't be checked against a passed-in type) and is a *lie* for the three
  bare-string modes (which return `str`, not the subclass).
- **Spec-carrier confusion.** You subclass a `BaseModel` and give it a regex, but its fields are
  never populated. The docstring has to explain that the class you defined is not the type you get.
- **Raise-vs-data inconsistency.** Denials are exceptions in `BaseExecutor.run` but data in
  `route_and_execute`; constraint violations raise; batch failures were (pre-fix) lost. There is no
  single answer to "how does a stage decline?"
- **Where does fornix go?** Containment is an *effect*, authorization is a *decision*, but the v2
  `Executor` fuses both into one object, so fornix has nowhere natural to attach and became a
  deferred "new executor subclass."

v3 dissolves all four by making the constraint a value (§4), unifying results (§5), and splitting
authority into decision × effect (§6).

---

## 3. Design principles

1. **One concept, one value.** If the domain has a noun (constraint, decision, effect, outcome),
   it is a first-class typed value with a single home — never an emergent property of four modules.
2. **Parse, don't validate.** A constraint is a *codec*: the same value that shapes the wire going
   out is the one that turns raw text into a typed Python value coming back. Guarding is parsing.
3. **Make illegal states unrepresentable.** A fail-closed authorizer *returns* a `Decision`; it is
   structurally incapable of leaking. A `Constraint[str]` truthfully yields `str`. No casts that
   assert what the checker can't verify.
4. **Decisions are data; exceptions are for bugs.** Every domain outcome (denied, violated, model
   failure) is a typed variant of `Outcome`. Exceptions are reserved for programmer/configuration
   error. One rule, everywhere.
5. **Effects are explicit and only at the edge.** Nothing runs implicitly. Generation never
   triggers a side effect; the wire is touched only inside `run`/`execute` calls the caller makes.
6. **Dependencies flow one way through named layers.** `wire → constraint → agent → fleet`. The
   privacy-critical `closed` path depends only on the bottom two and never imports pydantic-ai.
7. **Config is an edge, not the hot path.** Strings (dotted refs) live only where serialization
   demands them; the in-code API is fully typed. The import-execution vector is localized to one
   function with an allowlist.
8. **Preserve v2's proven facts.** The wire-grounded mode table, the single-importer discipline,
   the explicit-effects philosophy, and the test technique (in-process ASGI + wire-shape assertions)
   are load-bearing and carry over verbatim.
9. **Three orthogonal axes, never conflated.** The library governs exactly three things about a
   request — the **output** (constraint), the **weights** (adapter), and the **input** (context/cache)
   — each a separate value that *cooperates with* a server capability (XGrammar / LoRA / KV-reuse)
   it never reimplements. Keeping them separate is what keeps each clean; see §20. Cooperate with
   backend capabilities via request-shaping; model the *capability*, not any specific algorithm, so
   the library stays stable as those fields evolve.

---

## 4. The linchpin: `Constraint[T]` as a bidirectional codec

This is the change that pays for the whole rewrite. A constraint is not "a mode string plus a model
plus a guard." It is a single value that knows **how to shape the wire going out** and **how to turn
the raw output into a typed value `T` coming back**:

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class Constraint[T](Protocol):
    """A constrained-decoding contract AND the codec for its output type.

    Outbound: `wire()` says how to constrain the request.
    Inbound:  `parse()` turns the model's raw output into the typed value T (guarding/coercing).
    """

    def wire(self) -> WireSpec: ...
    def parse(self, raw: Any) -> T: ...
    def check(self) -> None: ...           # optional compile-check; default no-op
```

`WireSpec` is a tiny frozen value describing the two mutually-exclusive OpenAI-compatible
mechanisms (this is v2's `DecoderApplication`, kept — it is the wire-grounded crown jewel):

```python
@dataclass(frozen=True)
class WireSpec:
    output_type: Any                       # what pydantic-ai's Agent(output_type=) receives
    extra_body: dict[str, Any] = field(default_factory=dict)   # vLLM structured_outputs, etc.
```

### 4.1 The four constructors, each generic over its true output type

```python
def Schema[M: BaseModel](model: type[M]) -> Constraint[M]: ...
def Regex(pattern: str) -> Constraint[str]: ...
def Choice[*Opts](*options: *Opts) -> Constraint[Literal[*Opts]]: ...   # yields the Literal
def Grammar(ebnf: str) -> Constraint[str]: ...
```

Concretely:

| constructor | `wire().output_type` | `wire().extra_body` | `parse(raw)` |
|---|---|---|---|
| `Schema(FileEditPlan)` | `NativeOutput(FileEditPlan)` | `{}` | identity — pydantic-ai already returned a validated `FileEditPlan` |
| `Regex(r"git .*")` | `str` | `{"structured_outputs": {"regex": …}}` | `re.fullmatch`-or-raise → `str` |
| `Choice("keep","skip")` | `str` | `{"structured_outputs": {"choice": […]}}` | membership-or-raise → the `Literal` |
| `Grammar(ebnf)` | `str` | `{"structured_outputs": {"grammar": …}}` | passthrough (server-trusted) → `str` |

### 4.2 Why `parse(raw)` is uniform *and* honest

The subtlety that makes this elegant rather than a rename: **`parse` normalizes whatever pydantic-ai
hands back into `T`, and the work it does is exactly the guarantee that mode can enforce
client-side.**

- **Schema mode** keeps pydantic-ai's `NativeOutput`, so the model loop *already* enforces
  `response_format` and validates+retries; `parse` receives a validated `FileEditPlan` and returns
  it. The "backend didn't enforce" case is caught by pydantic-ai's own validation (it raises →
  `Failed`, §5). `parse` is honest identity here.
- **String modes** set `output_type=str`, so pydantic-ai returns raw text; `parse` performs the
  `fullmatch`/membership/coercion guard. This is v2's `_guard` (finding B4) — but now it lives *with
  the constraint*, not scattered in the agent, and it is the *same method* that schema mode uses.

So one signature, mode-appropriate internals, and — crucially — **the static type is true for every
mode**: `Regex` is `Constraint[str]` and really yields `str`; `Choice` is `Constraint[Literal[…]]`
and really yields the literal (v2's deferred coercion, open question #2, closes here for free).

### 4.3 What this single abstraction collapses

- **The typed-generic problem evaporates, without a cast.** Because the constraint carries `T`,
  `Agent[T]` and `Outcome[T]` flow from it mechanically (§9). No `cast`, no unverified assertion,
  no lie for string modes. The entire pros/cons debate about generics becomes moot — the type
  *is* the constraint.
- **`ConstrainedOutput` disappears.** For schema mode the model class alone is the constraint
  (`Schema(FileEditPlan)`); there is nothing to subclass and no dunder ClassVars. For string modes
  there was never a real model — you write `Regex(r"…")`, not a `BaseModel` with a dead `value` field.
- **`DecoderSpec`, `apply()`, and `_guard()` become one thing.** `wire()` is the outbound half,
  `parse()` the inbound half, on the same value.
- **The wire table stays authoritative** — it is the body of `wire()`, still traceable to captured
  requests (`VERIFICATION.md`).

### 4.4 Composability

Constraints compose where it is meaningful:

```python
Nullable(Schema(FileEditPlan))          # Constraint[FileEditPlan | None]
OneOf(Schema(EditPlan), Schema(Refusal))# Constraint[EditPlan | Refusal]  (tagged union at the wire)
```

These are ordinary functions returning `Constraint[…]`; the codec view makes them expressible where
v2's mode-enum could not.

---

## 5. The `Outcome[T]` spine: decisions-as-data, uniformly

v2 has four result types (`AgentResult`, `BatchResult`, `RoutedResult`, `RoutedExecution`) and five
entrypoints (`run`, `run_batch`, `route`, `route_and_run`, `route_and_execute`), and it declines in
two different ways (raise vs data). v3 recognizes that there is **one pipeline**:

```
message → [route] → generate+parse → [authorize] → [effect]
```

and models it with **one result type**, a tagged union that every stage produces:

```python
@dataclass(frozen=True)
class Ok[T]:
    value: T
    usage: Usage
    wire: RequestRecord | None = None      # present iff capture requested (§12)

@dataclass(frozen=True)
class Denied:
    reason: str
    command: Any

@dataclass(frozen=True)
class Violated:                            # parse() rejected the raw output
    reason: str
    raw: str

@dataclass(frozen=True)
class Failed:                              # model/transport error (already retried)
    error: Exception

type Outcome[T] = Ok[T] | Denied | Violated | Failed
```

### 5.1 Stages are functions returning `Outcome`, composed by bind

Each pipeline stage has the shape `A -> Outcome[B]`; the non-`Ok` variants short-circuit:

```python
class Outcome[T]:
    def then[U](self, f: Callable[[T], Outcome[U]]) -> Outcome[U]:
        return f(self.value) if isinstance(self, Ok) else self   # Denied/Violated/Failed pass through
```

So the whole pipeline is honest composition:

```python
outcome = (await agent.run(msg))          # Outcome[Command]
    .then(lambda cmd: authorizer.decide(cmd).as_outcome(cmd))   # Outcome[Command] | Denied
    .then(lambda cmd: effector.run(cmd))   # Outcome[Effect]
```

`route`, `authorize`, and `effect` are all just stages. There is no `RoutedResult` vs
`RoutedExecution` vs `AgentResult` — there is `Outcome[T]` with the stage's `T`. `run_batch`
returns `list[Outcome[T]]` (v2's `BatchResult` fix, generalized: failure is *always* data).

### 5.2 Ergonomics: `match`, plus an escape hatch for imperative callers

```python
match outcome:
    case Ok(value=plan):     apply(plan)
    case Denied(reason=r):   log.info("refused: %s", r)
    case Violated(reason=r): alert("backend not enforcing: %s", r)
    case Failed(error=e):    raise e

# For callers who *want* exceptions at a boundary:
plan = outcome.unwrap()      # returns T, or raises the appropriate typed error
```

**One rule** — decisions are data — replaces v2's raise-here/data-there split. Exceptions remain for
programmer/config error only (`ConfigError`, `BackendCapabilityError`; §9).

> **Note (open question, §18):** the full sum-type spine is the most invasive move and the least
> idiomatic-Python of the set. It is presented as the ideal; a pragmatic variant keeps `Ok`/`Failed`
> for `run` and reserves the union for the executed pipeline. The document commits to the ideal and
> flags the tradeoff.

---

## 6. Authority as `Authorizer × Effector`

The v2 `Executor` fuses *deciding* and *doing* into one object holding optional callables — the root
of findings B1 (execute skipped authorize), B2 (raising rule crashed), and B5 (`ok` never false),
and of "where does fornix go." v3 splits the two responsibilities:

```python
class Authorizer[C](Protocol):
    def decide(self, command: C) -> Decision: ...     # total; fail-closed by construction

class Effector[C](Protocol):
    def run(self, command: C) -> Effect: ...          # the side effect

@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    def as_outcome[C](self, command: C) -> Outcome[C]:
        return Ok(command, ...) if self.allowed else Denied(self.reason, command)

@dataclass(frozen=True)
class Effect:
    ok: bool
    output: Any = None
    detail: str = ""
```

- **`Allowlist(rules)` is an `Authorizer`.** It wraps each rule in try/except → `Decision(False)`
  once, at the boundary — so "fail closed" (B2) is *structural*, not a thing every rule must
  remember. Authorizers compose: `all_of(a, b)`, `any_of(a, b)`.
- **Effectors are the doing side:** `Null` (dry run), `Subprocess`, `Fornix`, `DbosStep`.
- **An executor is a composition**, not a subclass: `authorize(a) >> effect(e)` runs the effect iff
  the decision allows. `DryRun` = `a >> Null` — no special class. **Fornix** = `Allowlist >> FornixEffector`
  — the deferred Phase 6 integration becomes a one-line composition and a single new `Effector`, not
  a new executor hierarchy. The review's "authorization and containment compose, neither replaces the
  other" becomes literally the type signature.

This is single-responsibility, makes B1 impossible (there is no `execute` that can skip `decide` —
the pipeline binds `decide` before `run`), and makes `Effect.ok` meaningful because the `Effector`
is the only thing that can fail.

---

## 7. The layer stack and dependency direction

```
┌─ Layer 4  observe/      dual-path & eval capture as pipeline observers (optional, [observe] extra)
│
├─ Layer 3  fleet.py      Fleet, Router, pipeline combinators, Outcome bind
│            authority.py Authorizer × Effector, Allowlist, effectors
│
├─ Layer 2  agent.py      AgentSpec[T], Backend, Agent[T]  ── SOLE importer of pydantic_ai
│            outcome.py    Outcome[T] sum + bind
│
├─ Layer 1  constraint.py Constraint[T] codec, Schema/Regex/Choice/Grammar, WireSpec   (pure)
│
└─ Layer 0  wire/         OpenAI-compatible transport, request building, loopback/bounded guards
                          (httpx + pydantic only; NO pydantic_ai)
```

Dependencies flow strictly downward. Two consequences:

- **`closed` mode = Layer 0 + Layer 1 only** (§8) — it never imports pydantic-ai, preserving its
  attack-surface/dependency-minimization goal, but now *sharing* the wire primitives instead of
  duplicating them.
- **pydantic-ai is confined to Layer 2.** Everything above and below is framework-agnostic, so the
  library could later support a second agent runtime (or none) without touching the constraint,
  wire, authority, or fleet layers.

---

## 8. The wire layer (Layer 0) and `closed` as a preset

v2's `closed.py` re-implements loopback-URL validation, bounded-input guards, the `response_format`
builder, one-shot POST, client lifecycle, and detail-free errors — because it (correctly) refuses to
route privacy-critical data through pydantic-ai. The *dependency instinct is right*; the *duplicated
validation logic* is the smell. v3 extracts a small, pydantic-ai-free `wire` package that **both** the
rich agent path and the direct path share:

```python
# wire/transport.py
class Transport(Protocol):
    async def post(self, path: str, body: dict, headers: dict) -> Response: ...

class NetworkTransport(Transport):  ...          # ordinary httpx
class LoopbackTransport(Transport): ...          # rejects non-loopback hosts (DNS-rebinding safe)
class ASGITransport(Transport):     ...          # in-process test mock

# wire/request.py
def response_format(model: type[BaseModel], *, strict: bool = True) -> dict: ...
Model  = BoundedStr(max_bytes=128,  pattern=_MODEL_NAME)   # value objects, not ad-hoc if-guards
Prompt = BoundedStr(max_bytes=16_384)

# wire/client.py
class Retention(Enum): FULL, USAGE_ONLY, NONE     # what of the raw response is kept
async def call(transport, request, *, retention: Retention) -> WireResult: ...
```

Now the privacy story is a **composable knob**, not a bespoke class. `closed` becomes a thin preset:

```python
def closed_backend(base_url, api_key, model, output_type, instructions, timeout) -> ClosedAgent:
    return ClosedAgent(
        transport = LoopbackTransport(base_url),   # shared loopback guard
        retention = Retention.NONE,                # drop raw/usage — the privacy guarantee
        constraint = Schema(output_type),          # Layer-1 codec, shared
        model = Model(model), instructions = Prompt(instructions), timeout = timeout,
    )
```

Same guarantees Lodestar depends on (loopback-only, one request, json_schema-only, no capture, no
raw retention, detail-free errors), zero duplicated validation, and it rides the *same*
`Constraint[T]` codec as everything else. Lodestar's re-pin target is `closed_backend(...)`.

---

## 9. The agent layer (Layer 2): `AgentSpec[T]`, `Backend`, `Agent[T]`

```python
@dataclass(frozen=True)
class AgentSpec[T]:
    """A declarative, *typed* agent definition. This is v2's AgentProfile, reborn: it carries the
    Constraint[T] itself (not a string ref), so T flows statically from here to the Outcome."""
    name: str
    constraint: Constraint[T]
    instructions: str
    adapter: str | None = None          # per-agent LoRA; rides the wire `model` field
    authority: str | None = None        # policy name resolved by an Authorizer at execute time
    settings: Settings = Settings()     # validated sampling params (§ typo-warning folded in)

class Backend:                          # the SOLE importer of pydantic_ai.models.openai
    """Server + capabilities + one shared transport. build(spec) -> Agent[T]."""
    def build[T](self, spec: AgentSpec[T]) -> Agent[T]: ...
    async def aclose(self) -> None: ...          # closes the one shared httpx client

class Agent[T]:
    """The runnable. run() returns a fully-typed Outcome[T]."""
    async def run(self, prompt: str) -> Outcome[T]: ...
    @property
    def raw(self) -> pydantic_ai.Agent: ...      # escape hatch, unchanged
```

`Backend.build` wires the constraint to a pydantic-ai `Agent`:

```python
def build[T](self, spec: AgentSpec[T]) -> Agent[T]:
    self._gate(spec)                             # capability precondition (xgrammar/lora)
    w = spec.constraint.wire()                   # WireSpec: output_type + extra_body
    settings = self._settings(spec, w.extra_body)# merge extra_body (decoder wins), warn on typos
    pa = pydantic_ai.Agent(self._model(spec.adapter),
                           output_type=w.output_type, model_settings=settings,
                           instructions=spec.instructions)
    return Agent(spec, pa, self._capture)        # Agent[T] — T from spec.constraint
```

`Agent.run` is where `parse` and `Outcome` meet:

```python
async def run(self, prompt: str) -> Outcome[T]:
    try:
        raw = await self._pa.run(prompt)         # pydantic-ai: model loop + retries
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        return Failed(e)
    try:
        value: T = self._spec.constraint.parse(raw.output)   # guard/coerce → typed T
    except ConstraintViolation as v:
        return Violated(str(v), raw=str(raw.output))
    return Ok(value, usage=raw.usage, wire=self._captured())
```

**Type flow:** `AgentSpec[FileEditPlan]` → `build` → `Agent[FileEditPlan]` → `run` →
`Outcome[FileEditPlan]` → `Ok.value : FileEditPlan`. No cast anywhere; `ty` sees `FileEditPlan` end
to end because `Constraint[T]` carried it the whole way. This is the "typed outputs" promise
delivered *structurally*.

Capability gating (v2 `BackendCapabilityError`) stays a runtime precondition in `_gate` — a mis-capped
backend asked for `Grammar` raises a *config* error (a programmer mistake), not an `Outcome` variant.

---

## 10. The fleet + pipeline layer (Layer 3)

```python
class Fleet:
    """A set of Agent[Any] built from one Backend, plus an optional validated Router."""
    def build(self, specs: list[AgentSpec[Any]], router: Router | None = None) -> None: ...
    async def run_batch(self, calls: list[tuple[str, str]]) -> list[Outcome[Any]]: ...
    async def route(self, msg: str) -> str: ...
    async def run(self, msg: str) -> Outcome[Any]: ...                 # route ∘ generate
    async def execute(self, msg: str, auth: Authority) -> Outcome[Effect]: ...  # route ∘ generate ∘ authorize ∘ effect
    async def aclose(self) -> None: ...
```

The fleet is a *heterogeneous* collection (`Agent[Any]` — different specialists, different `T`), and
that is honest: there is no single `OutputT` for a fleet, so it is `Any` *by nature*, not by
oversight (this is the principled resolution of the "generics die at the fleet boundary" tension —
you regain concreteness the moment you pull a specific `Agent[T]` out via `fleet[name]`). Everything
composes through `Outcome.then`, so `execute` is literally `route` bound to `generate` bound to
`authorize` bound to `effect` — one expression, no bespoke `RoutedExecution` type.

`Router` (v2 `RoutingTable`) is unchanged in spirit: serializable router→specialist map, validated at
`build` with `Literal`-coverage checking. Routing stays *data*; the library never owns a hidden loop.

---

## 11. Config vs code: the serialization edge

v2's `output_type_ref="mod:Name"` string is simultaneously the typing dead-end *and* an
arbitrary-import vector. v3 splits `AgentSpec`'s two roles:

- **In-code, typed path is primary** — you pass a real `Constraint[T]`:
  ```python
  AgentSpec(name="file_edit", constraint=Schema(FileEditPlan), instructions="…", adapter="fe")
  ```
- **Config is an edge adapter** — strings are resolved to constraints in exactly one place:
  ```python
  # config.py
  def spec_from_config(d: dict, *, allow_modules: frozenset[str]) -> AgentSpec[Any]:
      constraint = constraint_from_config(d["constraint"], allow_modules=allow_modules)
      ...
  # {"constraint": {"kind": "schema", "ref": "myapp.schemas:FileEditPlan"}}
  # {"constraint": {"kind": "regex",  "pattern": "git .*"}}
  ```

`Constraint` has a canonical serializable form (a discriminated union keyed on `kind`), so config
round-trips. The import-execution concern is now **localized to `constraint_from_config`** and gated
by an explicit `allow_modules` allowlist — it is no longer latent in the hot path. Code stays in
types; strings live only at the boundary.

---

## 12. Capture, made explicit

v2 captures the wire via an httpx event hook writing to a `ContextVar` sink set around each run.
Clever, but ambient global state is the opposite of the library's otherwise-explicit character. v3
threads it through the return value: capture is opt-in per `Backend`, and when on, the request record
is carried *in the `Ok` outcome* (`Ok.wire`). No `ContextVar`, no reasoning about task-context
propagation, no shared mutable `records` buffer to bound. The wire record is data on the result, like
everything else.

(The httpx-hook *technique* for grabbing the exact bytes on the wire is kept — it is how the
wire-grounded design stays grounded. Only the *delivery* changes from ambient to explicit.)

---

## 13. Observation / dual-path reborn

v2's `dual_path/` is a well-separated DBOS subsystem for capturing local‖frontier comparisons for
fine-tuning and evals. In v3 it becomes a set of **observers over the pipeline** rather than a
parallel runner: an `Observer` receives each stage's `Outcome` and may persist it durably. The
dual-path "run both legs and compare" is one observer that fans a message to a second (reference)
`Agent[T]` and records the `(Ok[T], Ok[T])` pair. This:

- reuses the same `Agent[T]`/`Outcome[T]` spine instead of re-implementing runs;
- keeps DBOS + Postgres behind the `[observe]` extra (lean core unchanged);
- fixes v2's deferred durability items structurally — the record is an `Outcome`, and persisting an
  `Outcome` in a DBOS step keyed on `run_id` is idempotent by construction.

Details (connection pooling, `SetWorkflowID` contextvar verification, idempotent insert) are carried
over from the v2 review's section-C dual-path list as implementation notes, not re-litigated here.

---

## 14. Module layout

```
structured_agents/
  wire/                     # Layer 0 — pydantic-ai-free OpenAI-compatible transport
    __init__.py
    transport.py            #   Transport protocol; Network/Loopback/ASGI transports
    request.py              #   response_format builder; BoundedStr value objects (Model, Prompt)
    client.py               #   one-shot call; Retention policy (FULL/USAGE_ONLY/NONE)
    errors.py               #   detail-free wire errors
  constraint.py             # Layer 1 — Constraint[T], Schema/Regex/Choice/Grammar, WireSpec, serde
  outcome.py                # Outcome[T] sum (Ok/Denied/Violated/Failed) + then/unwrap
  authority.py              # Authorizer × Effector, Allowlist, Null/Subprocess effectors, Decision, Effect
  agent.py                  # Layer 2 — AgentSpec[T], Backend, Agent[T]  (SOLE pydantic_ai importer)
  fleet.py                  # Layer 3 — Fleet, Router, pipeline combinators
  config.py                 # serialization edge (spec_from_config, constraint_from_config, allowlist)
  closed.py                 # preset over wire+constraint (Layer 0+1), pydantic-ai-free
  errors.py                 # programmer/config error hierarchy
  integrations/
    fornix.py               # FornixEffector (Layer 3 plug-in; stdlib subprocess only)
  observe/                  # Layer 4 — DBOS observers ([observe] extra)
    __init__.py  runtime.py  store.py  record.py  compare.py
```

Compare to v2: `constrained.py` + `decoder.py` + `agent._guard` → `constraint.py`;
`executor.py` → `authority.py` (decomposed); `backend.py` + `agent.py` → `agent.py` (+ `wire/`);
`closed.py` reduced to a preset; `fleet.py` slimmed by the `Outcome` spine; the four result types →
`outcome.py`.

---

## 15. End-to-end worked example + type-flow

```python
from structured_agents import Backend, AgentSpec, Fleet, Router, Schema, Choice
from structured_agents.authority import Allowlist, Subprocess, authorize
from myapp.schemas import FileEditPlan, GitCommand      # plain pydantic BaseModels

backend = Backend(base_url="http://tower:8000/v1", api_key="…", default_model="base", capture=True)

router   = AgentSpec(name="router",   constraint=Choice("file_edit", "git_ops"),
                     instructions="Route to one specialist.", adapter="router")
file_ed  = AgentSpec(name="file_edit", constraint=Schema(FileEditPlan),
                     instructions="Produce file-edit plans only.", adapter="fe")
git_ops  = AgentSpec(name="git_ops",   constraint=Schema(GitCommand),
                     instructions="One safe git command.", adapter="go", authority="git_safe_v1")

fleet = Fleet(backend)
fleet.build([router, file_ed, git_ops],
            router=Router(router="router", routes={"file_edit": "file_edit", "git_ops": "git_ops"}))

# Single typed agent, class-in-hand — full static types, no cast:
agent  = fleet["git_ops"]                     # Agent[GitCommand]
oc     = await agent.run("show me the log")   # Outcome[GitCommand]
match oc:
    case Ok(value=cmd):                        # cmd: GitCommand   ← ty knows this
        print(cmd.argv)                        # autocompleted, field-checked
    case Violated(reason=r):
        alert(f"backend not enforcing: {r}")

# Full autonomous pipeline — route → generate → authorize → contain (fornix) → data outcome:
from structured_agents.integrations.fornix import FornixEffector
authority = authorize(
    Allowlist({"git_safe_v1": lambda c: c.argv[:1] == ["git"] and c.argv[1:2] and c.argv[1] in SAFE}),
) >> FornixEffector()                          # decision × containment, composed
result: Outcome[Effect] = await fleet.execute("show me the log", authority)
match result:
    case Ok(value=Effect(ok=True, output=diff)): promote(diff)
    case Denied(reason=r):                        log.info("refused: %s", r)
```

`reveal_type` at each step is concrete (`Agent[GitCommand]`, `Outcome[GitCommand]`, `GitCommand`) —
the "typed outputs" pitch, delivered without a single `cast`, honest across all four modes.

---

## 16. What v3 deliberately keeps from v2

- **Single-importer discipline** — `agent.py` (Backend) is the only module importing
  `pydantic_ai.models.openai`. Non-negotiable; enforced by the layer stack.
- **Wire-grounded design** — every mode traces to a captured request; it is the body of
  `Constraint.wire()`. `VERIFICATION.md`'s captured shapes carry over verbatim.
- **Explicit-effects philosophy** — nothing runs implicitly; effects only at `execute`.
- **Default-deny authority** — `Allowlist` denies unknown/unmatched; fail-closed is now structural.
- **The error hierarchy** for programmer/config errors (`ConfigError`, `BackendCapabilityError`).
- **The test technique** — in-process ASGI mock + wire-shape assertions + real in-flight concurrency
  proofs; `live` marker gated on `SAV_LIVE=1`.
- **The `closed` guarantees** — loopback-only, one request, json_schema-only, no retention, detail-free
  errors — preserved exactly, now as a preset over shared primitives.

---

## 17. Testing strategy

- **Codec round-trip (property tests):** for each `Constraint`, `parse(model_output(x)) == x` for
  valid `x`, and `parse` rejects out-of-constraint text (`Violated`). This is the single most
  valuable new test surface — it pins both halves of the codec in one place.
- **Wire-shape assertions:** `Constraint.wire()` produces exactly the captured `extra_body` /
  `response_format` from `VERIFICATION.md` (carried from v2's `test_wire_shapes.py`).
- **`Outcome` algebra:** `then` short-circuits on non-`Ok`; `unwrap` maps each variant to the right
  typed error; `run_batch` surfaces per-item failures as `Failed` with no lost siblings.
- **Authority:** `Allowlist.decide` is total (raising rule → `Decision(False)`); `authorize >> Null`
  performs no effect; `Effect.ok` is false iff the effector raised.
- **Type-level regression:** `assert_type(agent.run(...), Outcome[FileEditPlan])` in a `ty`-checked
  test module, so the static story can't silently erode.
- **`closed`:** loopback rejection, bounded-input rejection, one-request-no-retry, no-retention — all
  runnable without pydantic-ai (Layer 0+1 only).

---

## 18. Open questions

1. **Depth of pydantic-ai coupling.** v3 keeps pydantic-ai as the Layer-2 model loop (retries,
   message handling, `NativeOutput`). Alternative: own the loop on top of `wire/` and drop the
   dependency entirely, making `Constraint.parse` the *only* parser for every mode (maximally
   uniform, but re-implements retry/tool/message machinery). Recommendation: keep pydantic-ai for now;
   the layer boundary makes swapping it later cheap.
2. **`Outcome` sum vs. exceptions (ergonomics).** The full four-variant spine is the least
   idiomatic-Python move. Decide between: (a) commit fully (this document); (b) `Ok`/`Failed` for
   `run`, richer union only for the executed pipeline. Prototype both against real call sites.
3. **Heterogeneous-fleet typing.** `Fleet` is `Agent[Any]`-valued by nature. Is a typed-router
   (`Router[Enum]` narrowing to specific specialist types) worth the machinery, or is `fleet[name] ->
   Agent[T]` re-narrowing sufficient? Leaning: the latter.
4. **Streaming.** v2 is non-streaming. Does `Outcome[T]` gain a streaming sibling
   (`AsyncIterator[Delta] → Outcome[T]`), or is constrained decoding inherently batch-shaped for the
   use case? Deferred.
5. **Tool/function-calling agents.** v2/v3 use `NativeOutput` (response_format), deliberately *not*
   the function-calling tool path. If tool-using agents are ever in scope, does a `Constraint` express
   a tool schema, or is that a different abstraction? Likely different; keep out of scope.
6. **`Choice` and variadic generics.** `Choice(*opts) -> Constraint[Literal[*opts]]` leans on
   `TypeVarTuple`/`Unpack` ergonomics; confirm `ty` handles the literal synthesis, else fall back to a
   `Choice[L]` explicit-parameter form.
7. **Multi-turn sessions & the context axis (§22.8).** A `Session`/`Conversation` that reuses the KV
   of a *growing* history is where PIC pays off most, but it is a bigger abstraction than the
   single-shot `Agent.run`. Decide whether `Context` grows a session sibling or a `Session` wraps a
   sequence of `Agent[T]` runs threading history as `Reuse.PREFIX` segments.
8. **Context model default (settled): neutral, per-segment.** Resolved per design review — the
   `Context` model is *neither* linear-prefix nor chunk-first at the top level; reuse is a per-segment
   policy (§22.1) so both are assembly strategies over one model. Recorded here as a closed question
   for provenance.

---

## 19. Migration map (v2 → v3)

| v2 | v3 | Note |
|---|---|---|
| `ConstrainedOutput` (subclass + dunders) | `Schema(Model)` / `Regex` / `Choice` / `Grammar` | model classes stay plain `BaseModel`; no subclassing |
| `DecoderSpec` + `DecoderApplication` | `Constraint[T]` + `WireSpec` | outbound half of the codec |
| `StructuredAgent._guard` | `Constraint.parse` | inbound half; now co-located, honest, generic |
| `AgentProfile` (string `output_type_ref`) | `AgentSpec[T]` (carries `Constraint[T]`) + `config.spec_from_config` | typed code path primary; strings at the edge |
| `AgentProfile.instructions` (bare `str`) | `AgentSpec.context: Context` (§22) | input becomes a cache-cooperative axis; a lone instruction string is a one-segment `PREFIX` context |
| `AgentProfile.adapter` (bare `str`) | `AgentSpec.adapter: Adapter \| str` (§21) | adapter gains `source`/`base_model` for provisioning + cache-namespacing |
| `BackendCaps{xgrammar, lora}` | `+ chunk_cache` (§22) | new capability for position-independent KV reuse |
| `StructuredAgent` / `AgentResult` | `Agent[T]` / `Outcome[T]` | generic end to end |
| `Backend` (+ per-agent client) | `Backend` (+ shared client) + `wire/` | client lifecycle already fixed in v0.2.0; formalized |
| `Executor`/`DryRun`/`Allowlist`/`Policy` | `Authorizer × Effector` + `Allowlist` + effectors | decomposed; DryRun/Fornix are compositions |
| `AgentSet`/`RoutingTable`/`RoutedResult`/`RoutedExecution`/`BatchResult` | `Fleet`/`Router` + `Outcome[T]` | four result types → one spine |
| `closed.py` (bespoke) | `closed_backend` preset over `wire/`+`constraint` | same guarantees, no duplicated validation |
| `dual_path/` | `observe/` (pipeline observers) | reuses the spine; `[observe]` extra |

Greenfield, but salvageable verbatim: the wire-shape table, `VERIFICATION.md`, the executor
fail-closed tests, the httpx-capture technique, the in-process ASGI mock, the deploy/vLLM verify
scripts.

---

## 20. The three axes: constraint · adapter · context

The library governs exactly three orthogonal things about a request, and keeping them
*separate values* is what keeps each one clean (conflating any two — e.g. making a LoRA a
`Constraint`, or cache policy a mode of the prompt string — is the category error v3 exists to
avoid). Each axis (a) is a first-class value on `AgentSpec`, (b) *cooperates with* a server-side
capability the library never reimplements, and (c) has its own plugin seam:

| Axis | Governs | Value on `AgentSpec` | Server capability (below the library) | Plugin seam |
|---|---|---|---|---|
| **Constraint** (§4) | the **output** — shape/syntax/validity | `constraint: Constraint[T]` | XGrammar (constrained decoding) | any `Constraint[T]` (§23) |
| **Adapter** (§21) | the **weights** that generate | `adapter: Adapter \| str \| None` | LoRA (`caps.lora`) | `AdapterProvider` / capability factory |
| **Context** (§22) | the **input** — what's in the prompt & its KV reuse | `context: Context` | KV-cache reuse / PIC (`caps.chunk_cache`) | `ContextProvider` |

The three compose freely and independently on the wire: the constraint rides
`response_format`/`extra_body` (logits-level), the adapter rides the `model` field, and the
context is the messages — so **switching any one is a full cache hit for the others**. The library's
contribution on every axis is the same shape: *express typed intent + declare the capability + own
the correctness-critical bookkeeping*, and let the backend do the mechanism.

---

## 21. The adapter axis & capability plugins

### 21.1 Adapter is first-class, not a plugin

A per-agent LoRA is `AgentSpec.adapter`. A bare string is the simple case (it becomes the wire
`model` field); an `Adapter` value carries more for providers that resolve/provision:

```python
@dataclass(frozen=True)
class Adapter:
    name: str                       # the wire `model` field value
    source: str | None = None       # HF repo / path, for providers that auto-load
    base_model: str | None = None   # what it was tuned on (for cache-namespacing, §22)
```

### 21.2 The valuable plugin shape: a *capability* = an `AgentSpec[T]` factory

A fine-tuned LoRA is almost always trained to emit *one specific schema*, so the natural unit is a
**specialist shipped whole** — its adapter, the constraint it was trained for, and its instructions,
packaged together. Because `AgentSpec[T]` is just a typed value, a plugin exports a function that
returns one:

```python
def file_edit_specialist(*, adapter: str = "myorg/file-edit@v3") -> AgentSpec[FileEditPlan]:
    return AgentSpec(name="file_edit", constraint=Schema(FileEditPlan),
                     adapter=Adapter(adapter, base_model="qwen3-4b"),
                     instructions="Produce file-edit plans only.")
```

Zero new mechanism: the two axes co-package exactly where reality couples them (the adapter *learned*
that schema). This is the first thing to point plugin authors at.

### 21.3 The operational seam: `AdapterProvider` on the `Backend`

Resolving a logical name → served name/version and provisioning it (e.g. vLLM
`/v1/load_lora_adapter`) is a **wire/backend** concern, deliberately *not* on the `Constraint`:

```python
class AdapterProvider(Protocol):
    def resolve(self, adapter: Adapter | str) -> str: ...   # -> served wire name; ensure loaded
```

### 21.4 Two checks this axis-split unlocks (correctness, not just perf)

- **Adapter↔constraint compatibility at build:** a capability factory can assert "this adapter
  expects `Schema(FileEditPlan)`" and fail loudly if rewired to a `Regex`.
- **Runtime 404 → build error:** an `AdapterProvider.resolve` that verifies the adapter is actually
  served turns "vLLM: unknown model" at request time into a `BackendCapabilityError` at `build` —
  the same fail-fast discipline as XGrammar/LoRA capability gating (§9).

---

## 22. The context axis: cache-cooperative input

`Context` is the input-side mirror of `Constraint[T]`: `Constraint` governs the *output*, `Context`
governs the *input*. The server does the caching (LMCache and the position-independent-caching
family — §22.4); the library **shapes cache-friendly requests, declares the capability, and owns the
correctness-critical identity/namespacing** — it never reimplements KV caching.

### 22.1 The neutral model — reuse policy is *per-segment*, never a global mode

The trap to avoid is baking "linear-prefix *vs* chunk-first" in as a top-level mode of the context;
then one of the two is bolted-on forever. v3 refuses to make it a mode at all: cache-reuse is a
**per-segment policy** over one inert data model, and "linear-prefix" and "chunk-first" are two
*assembly strategies* that read it. A single `Context` can be all-prefix, all-chunk, or **mixed**.

```python
class Reuse(Enum):
    PREFIX = "prefix"   # reusable only as part of the contiguous prefix — classic, EXACT, universal
    CHUNK  = "chunk"    # reusable as a position-independent chunk (PIC) — needs identity, opt-in
    NONE   = "none"     # never cached (the volatile query, secrets)

@dataclass(frozen=True)
class Segment:
    content: str
    role: Role = "user"
    reuse: Reuse = Reuse.PREFIX    # conservative default: exact fidelity, works on every backend
    id: str | None = None          # stable identity; derived from a content hash when reuse == CHUNK

@dataclass(frozen=True)
class Context:
    segments: tuple[Segment, ...]  # order matters for PREFIX; CHUNK is position-independent
    query: str | None = None       # sugar for the trailing NONE segment (the ephemeral suffix)
```

For *this* library the fit is exact: the system/persona block and the **schema-describing few-shot**
(stable, reused across every query for that agent) are textbook reusable units; the user query is the
one non-cached piece.

### 22.2 Why this doesn't trap you — into either strategy, or into a backend

1. **Per-segment, not global.** You annotate pieces; you never declare a `Context` "linear" or
   "chunked." Mixing is the default, not a special case.
2. **`PREFIX` is the safe default; `CHUNK` is opt-in.** Prefix caching is exact-fidelity and
   universal; the approximate/position-independent path is never invoked unless asked for *and*
   `caps.chunk_cache` is present. You can't accidentally trade correctness for a cache hit.
3. **Graceful capability-degradation — no backend lock-in.** The *same* `Context` runs on a plain
   vLLM and a PIC-enabled vLLM. The assembler consults `caps.chunk_cache`: where chunk reuse exists
   it emits boundaries + ids; where it doesn't, `CHUNK` degrades to `PREFIX` (if prefix-positioned)
   or `NONE` (recomputed). **Correctness is invariant; only speed varies.**
4. **`PREFIX → CHUNK` is a one-field upgrade.** Identity is already a *slot* on every segment (unused
   for `PREFIX`, lazily derived for `CHUNK`). Promoting a segment to position-independent reuse later
   changes one field; the model never changes shape. The chunk-first machinery is *latent from day
   one*, dormant until used — satisfying "expandable into chunk-first" structurally.

### 22.3 Two orthogonal knobs, so neither traps the other

- **Reuse policy** (per-segment): *what is cacheable and how.*
- **Fidelity posture** (per-context/agent): `EXACT | BLENDED` — *is approximate reuse permitted at
  all.* `EXACT` forbids blend regardless of annotations (prefix-or-recompute); `BLENDED` unlocks true
  position-independence. `closed` mode pins `EXACT`. Fidelity is a **coarse posture, never a numeric
  recompute ratio** — because the ratio isn't universal (KV Packet, §22.4, has *no* recompute).

### 22.4 The mechanism-neutral contract, grounded in the PIC literature

The load-bearing contract the library must expose is exactly three things — **boundaries + stable
identity + context-dependence** — and *nothing about the algorithm*. This is validated by the
position-independent-caching (PIC) family; the library models the **capability**, not any one system,
so it stays stable as the field evolves:

- **CacheBlend** (LMCache; RAG KV fusion) — precompute each chunk's KV independently, then
  *selectively recompute* the high-attention-deviation tokens (~15%) to blend them. Establishes:
  arbitrary chunks reusable in any order, at a small recompute cost.
- **EPIC** (position-independent caching) — recompute only the *chunk-boundary* tokens statically
  (attention-sink insight), cheaper/more predictable than dynamic deviation.
- **MEPIC** — *Memory Efficient Position Independent Caching for LLM Serving* (arXiv:2512.16822):
  page-aligned KV storage + **block-level** recompute (only the first block is request-specific) +
  RoPE fusion in the attention kernel; ~2× HBM cut, up to 5× on long context, **no model changes and
  no per-request knobs — fully system-managed.** → confirms recompute is *transparent* to the client;
  boundaries + identity are the whole contract.
- **MiniPIC** — *Flexible Position-Independent Caching in <100 LOC* (arXiv:2606.13126, IBM): stores
  unrotated K and applies RoPE per-request logical positions, **realizing Block-Attention, EPIC, and
  Prompt Cache in one vLLM** via three token-level primitives — **Span Separator (SSep)** = chunk
  boundary, **Prompt Depend (PDep)** = per-span context-dependence, block-aligned padding. → the
  "don't commit to one mechanism" thesis *at the serving layer*, and the exact primitives our
  `Segment` maps onto (`Segment` boundary → SSep; `Reuse`/dependence → PDep).
- **KV Packet** — *Recomputation-Free Context-Independent KV Caching* (arXiv:2604.13226): cached docs
  as immutable "packets" wrapped in light **trained soft-token adapters** (self-supervised
  distillation), **zero recompute**, F1 ≈ full recomputation. → proves fidelity is *not* a recompute
  ratio; keep it a coarse posture (§22.3).

**The deeper axis is context-dependence** (MiniPIC's PDep): the real per-segment property is "does
this span's KV depend on the spans before it?" `PREFIX` = fully dependent (reusable only in exact
prior context); `CHUNK` = independent (reusable anywhere). `Reuse` is that dependence, in the two
practically-served forms — extend the enum if a backend exposes a middle ground.

### 22.5 The correctness-critical bookkeeping the library owns

- **Adapter-aware cache namespace.** A chunk's KV is computed under specific weights, so a per-agent
  **LoRA changes the KV** — sharing a chunk's cached KV across two agents with different adapters is
  *wrong* unless the engine explicitly separates base KV from LoRA deltas. The library therefore
  derives chunk identity / `cache_salt` from **`hash(content) + base_model + adapter`**, never
  content alone. This is the strongest reason the library (not the caller) owns identity.
- **Block-alignment guideline.** PIC engines cache at page/block granularity (MEPIC, MiniPIC), so the
  assembler avoids sub-block chunks (they waste a block and can't be shared) — a chunking heuristic,
  not a hard API.
- **Advisory hints, opaque passthrough.** Any fine, mechanism-specific knob (a recompute budget,
  MiniPIC's PDep flag) rides an opaque `cache_options` passthrough (like `extra_body`) — *never*
  hoisted into the core API, because it differs per mechanism and most are system-managed.

### 22.6 The assembler is a pluggable strategy over the neutral model

`Context` is inert data. A `ContextProvider`/assembler reads it + `caps` + fidelity +
`(base_model, adapter)` and produces the wire messages and cache hints (this is where namespacing and
degradation happen):

```python
class ContextProvider(Protocol):
    def assemble(self, ctx: Context, *, caps: BackendCaps, adapter: Adapter | None) -> WireMessages: ...
```

"Linear-prefix" and "chunk-first" are two assemblers over the *identical* model; a future mechanism is
a **third assembler + one `Reuse` variant — no change to `Context`.** That is the anti-trap guarantee
extended to *evolution*: MEPIC/MiniPIC/KV-Packet/whatever-comes-next attaches as an assembler, not a
redesign.

### 22.7 Two properties worth stating explicitly

- **Constraint × cache is safe.** Constrained decoding masks logits (XGrammar) at every step, and
  that masking holds *regardless of KV quality* — so approximate/blended caching **cannot break the
  syntactic/structural guarantee**; only *semantic* choice-quality can degrade. `BLENDED` is
  therefore safe for the constraint contract in a way it isn't for free-form chat — the speed/quality
  trade is *more* acceptable for constrained agents.
- **Cache sharing is a privacy surface.** Shared KV is a cross-request side channel, so cache policy
  joins the `Retention` knob family: **`closed` mode = `Retention.NONE` + `EXACT` + a unique
  `cache_salt`** (no cross-request reuse). One coherent "shares nothing" posture, decided in one place.

### 22.8 Deferred (recorded)

**Multi-turn sessions** — reusing the KV of a *growing* conversation history — is where PIC pays off
most, but it implies a `Session`/`Conversation` abstraction v3 doesn't yet have. A real axis, a bigger
feature; flagged, not built (open question §18).

---

## 23. Extensibility & plugin seams

The library is **open to extension by addition** on every axis, because each seam is a `Protocol`, and
closed to modification (no central enum or `if mode ==` ladder in the core — the defect v2's
`DecodeMode = Literal[...]` union created). Summary of the seams:

| Seam | Layer | In-code registration | Config registration |
|---|---|---|---|
| `Constraint[T]` (new output shape) | 1 | none — implement `wire`/`parse`/`check`, use anywhere | `register_constraint(kind, from_config)` / entry point |
| Capability factory (`AgentSpec[T]`) | 2 | none — export a function returning a spec | it's just a spec; serialize via §11 |
| `AdapterProvider` | 0/2 | pass to `Backend(adapters=…)` | provider names resolved at the config edge |
| `Effector` (new side effect, e.g. fornix) | 3 | compose `authorize(a) >> Effector()` | policy→effector map at the config edge |
| `ContextProvider` (assembly/cache strategy) | 2 | pass to `Backend(context=…)` | strategy named at the config edge |

**The one place extension isn't free is config/serialization.** In-code plugins are registration-free
(structural typing); but to load a *plugin* constraint/effector/provider from YAML/JSON, the config
edge (§11) needs to resolve its `kind` string → constructor. That warrants one small registry per
seam — an explicit `register_*` call or a Python entry point
(`[project.entry-points."structured_agents.constraints"]`) — deliberately localized to the config
layer alongside the import-allowlist, never in the hot path.

**Server-enforced vs client-validated plugins (for `Constraint` authors).** A plugin constraint whose
`wire()` emits a key the backend enforces (regex/grammar/choice/json_schema) is true *constrained
decoding*; one whose `wire()` returns `output_type=str, extra_body={}` and does all its work in
`parse()` is *client-side validation* — still useful (`SemVer`, `IsoDate`, `Json[T]`), needs no server
support, but the model can waste tokens on invalid output that `parse` then rejects. The design makes
which tier you're building **visible** in what `wire()` returns — a distinction v2 couldn't express.

---

## 24. Naming glossary

| Term | Meaning |
|---|---|
| `Constraint[T]` | the codec: `wire()` (outbound) + `parse()` (inbound) + `check()`; carries the output type `T` |
| `WireSpec` | frozen `(output_type, extra_body)` — how a constraint shapes one request |
| `Schema/Regex/Choice/Grammar` | the four constraint constructors, each generic over its true `T` |
| `AgentSpec[T]` | typed, declarative agent definition (constraint + instructions + adapter + authority + settings) |
| `Backend` | server + caps + shared transport; `build(spec) -> Agent[T]`; sole pydantic-ai importer |
| `Agent[T]` | the runnable; `run(prompt) -> Outcome[T]` |
| `Outcome[T]` | `Ok[T] \| Denied \| Violated \| Failed` — the one result spine; `then`/`unwrap` |
| `Authorizer[C]` / `Effector[C]` | decide (pure, fail-closed) / do (the effect); compose to an executor |
| `Decision` / `Effect` | the authorizer's verdict / the effector's result |
| `Fleet` / `Router` | heterogeneous agent collection / serializable validated routing |
| `Observer` | Layer-4 pipeline tap (dual-path, evals) behind the `[observe]` extra |
| `Retention` | wire-layer knob (`FULL/USAGE_ONLY/NONE`) — `closed` mode uses `NONE` |
| **three axes** | constraint (output, §4) · adapter (weights, §21) · context (input, §22) — orthogonal, each cooperating with a server capability |
| `Adapter` | the per-agent LoRA value (`name` + optional `source`/`base_model`); rides the wire `model` field |
| `AdapterProvider` | seam that resolves/provisions a LoRA (e.g. vLLM `load_lora_adapter`); turns runtime 404s into build errors |
| capability factory | a plugin that returns a bundled `AgentSpec[T]` (adapter + its constraint + instructions) — a fine-tuned specialist shipped whole |
| `Context` | the input axis: an ordered tuple of `Segment`s + an ephemeral `query`; mirror of `Constraint` |
| `Segment` | one input piece: `content` + `role` + `Reuse` policy + optional stable `id` |
| `Reuse` | per-segment cache policy `PREFIX / CHUNK / NONE` — encodes context-dependence (MiniPIC PDep); `PREFIX` is the exact/universal default |
| fidelity | per-context posture `EXACT / BLENDED` — whether approximate (position-independent) reuse is permitted; a coarse posture, never a recompute ratio |
| `caps.chunk_cache` | backend capability: position-independent chunk reuse (CacheBlend/EPIC/MEPIC/MiniPIC) is available |
| `cache_salt` | chunk cache namespace = `hash(content) + base_model + adapter` — prevents wrong KV sharing across adapters |
| `ContextProvider` | pluggable assembler: neutral `Context` + `caps` + fidelity + adapter → wire messages + cache hints (linear-prefix / chunk-first are two of these) |

---

*This document is the design target. The recommended build order, by value-per-disruption, is:
(1) `Constraint[T]` codec + `wire/` primitives; (2) `Authorizer × Effector`; (3) config/code split;
(4) `closed` preset over shared wire; (5) the `Outcome` spine (most invasive — last). Item (1) alone
delivers most of the elegance, because the constraint-as-codec is the concept the whole library is
secretly organized around.*
