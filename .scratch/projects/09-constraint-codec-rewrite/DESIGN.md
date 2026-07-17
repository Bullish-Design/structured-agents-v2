# DESIGN — module-by-module specification for constric (structured-agents v3)

This is the implementable heart of the plan. For each module: **responsibility · public surface
(real signatures) · dependencies (must obey one-way layering) · invariants.** Signatures are
written to typecheck under `ty` 0.0.46; every type-level claim traces to `SPIKES.md`. Wire bodies
trace to `../02-library-wrapper/VERIFICATION.md` and v2's `decoder.py` (SALVAGE.md).

Package: `structured_agents/` (import name unchanged even if the distribution is renamed — DECISIONS
H/J). PEP 695 generics throughout (`class Foo[T]`, `def f[T]`), `from __future__ import annotations`
in every module.

```
Layer 4  observe/         pipeline Observers (dual-path, evals)      [observe] extra
Layer 3  fleet.py         Fleet, Router
         authority.py     Authorizer × Effector, Allowlist, effectors, Decision, Effect
Layer 2  agent.py         AgentSpec[T], Backend, Agent[T]   ── SOLE pydantic_ai importer
         context.py       Context, Segment, Reuse, ContextProvider, fidelity
Layer 1  constraint.py    Constraint[T], Schema/Regex/Choice/Grammar, WireSpec, serde
         outcome.py       Outcome[T] base class + Ok/Failed (+ Denied on execute) + combinators
Layer 0  wire/            transport, request, client, retention, errors   (httpx+pydantic only)
         closed.py        preset over wire+constraint (Layer 0+1) — imports NO pydantic_ai (import isolation)
         errors.py        programmer/config error hierarchy (imported everywhere; depends on nothing)
         config.py        serialization edge (depends up to constraint; import-allowlist)
```

Dependency rule: a module may import only from **lower or equal** layers (and `errors.py`, which is
layer-less). `ty`-checkable and enforced by a test (TESTS.md → import-layering test).

---

## `errors.py` — programmer/config error hierarchy (layer-less)

**Responsibility.** The exception tree for *bugs and misconfiguration only*. Runtime domain outcomes
are **never** exceptions here — they are `Outcome` variants (DECISION O). Carries v2's hierarchy
forward, renamed to drop redundancy.

```python
class ConstricError(Exception): ...                     # base (v2 StructuredAgentsError)
class ConfigError(ConstricError): ...                   # invalid spec/config
class ConstraintConfigError(ConfigError): ...           # a Constraint built inconsistently
class ConstraintCompileError(ConstricError): ...        # check() failed (grammar-check extra)
class BackendCapabilityError(ConstricError): ...        # agent needs a cap the backend lacks
class FleetError(ConstricError): ...
class RoutingError(FleetError): ...
class AuthorityError(ConstricError): ...                 # v2 PolicyError; misconfigured authority
class ConstraintViolation(ConstricError): ...           # parse() rejected raw output; carried in Failed.error
```

**Note — `ConstraintViolation` (DECISION B, revised).** v2's `ConstraintViolationError` (bare-string
guard failed = backend didn't enforce) becomes **`ConstraintViolation`, carried inside the `Failed`
outcome's `.error`** (not a separate `Violated` top-level variant — the user chose the lighter
`Ok`/`Failed` spine). `Constraint.parse()` raises it; `Agent.run` wraps it as `Failed(v)`. It stays a
*diagnosable* exception type (callers can `match Failed(error=ConstraintViolation() as v)`), so
"backend not enforcing" is distinguishable from a model/transport error without a fourth variant.

**Invariants.** (i) Nothing in `errors.py` imports any other constric module. (ii) Every message
names the offending agent/spec and states the remedy (v2 strength, kept). (iii) `closed.py` does
**not** use this tree — it raises its own detail-free `ClosedBackendError` (below), deliberately
outside the hierarchy so no structured detail leaks.

---

## Layer 0 — `wire/` (httpx + pydantic only; NO pydantic_ai)

The pydantic-ai-free transport shared by the rich path and `closed`. Extracts v2's duplicated
loopback/bounded-input/`response_format` logic into value objects.

### `wire/transport.py`

**Responsibility.** The single async POST seam, with the three transports as *policy over one
Protocol* — network, loopback-guarded, in-process ASGI (test mock).

```python
@dataclass(frozen=True)
class Response:
    status: int
    json_body: dict[str, Any]

@runtime_checkable
class Transport(Protocol):
    async def post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> Response: ...
    async def aclose(self) -> None: ...

class NetworkTransport:                # wraps one httpx.AsyncClient(follow_redirects=False)
    def __init__(self, *, timeout: float, trust_env: bool = False) -> None: ...
class LoopbackTransport:               # NetworkTransport + validated-loopback URL guard (below)
    def __init__(self, base_url: str, *, timeout: float) -> None: ...
class ASGITransport:                   # httpx.ASGITransport over an in-process app — the test seam
    def __init__(self, app: Any) -> None: ...
```

**Loopback guard (carried VERBATIM from v2 `closed._validated_loopback_url`, SALVAGE.md):**
accepts only `scheme == "http"` and `hostname in {"127.0.0.1", "::1"}`; rejects `localhost`
deliberately (DNS-rebinding) — **keep the "do not add localhost" comment**; rejects any
username/password/query/fragment. Lives here now, used by both `closed` and any loopback rich
backend.

### `wire/request.py`

**Responsibility.** Build request bodies from typed value objects; house the two verified wire
shapes. **Bounded-input value objects replace ad-hoc if-guards** (v2 finding: closed had inline
guards).

```python
class BoundedStr:                      # value object: validate-on-construct, raise ValueError
    def __init__(self, value: str, *, max_bytes: int, pattern: re.Pattern[str] | None = None) -> None: ...
    def __str__(self) -> str: ...

# Concrete instances (limits carried verbatim from v2 closed.py):
def model_name(v: str) -> str:  ...    # re.fullmatch(r"[A-Za-z0-9._:-]{1,128}")
def instructions(v: str) -> str: ...   # nonempty, <= 4096 bytes
def prompt(v: str) -> str: ...         # str, <= 16_384 bytes

def response_format(model: type[BaseModel], *, name: str = "output", strict: bool = True) -> dict[str, Any]:
    """The verified json_schema body (VERIFICATION.md; closed.py):
       {"type":"json_schema","json_schema":{"name":name,"strict":strict,"schema":model.model_json_schema()}}"""

def chat_body(*, model: str, system: str, user: str, response_format: dict | None = None,
              extra_body: dict[str, Any] | None = None, stream: bool = False) -> dict[str, Any]:
    """Assemble the /chat/completions body. extra_body keys land verbatim at top level."""
```

### `wire/client.py`

**Responsibility.** One-shot call + the **retention policy** (what of the response is kept).

```python
class Retention(Enum):
    FULL = "full"              # keep raw + usage
    USAGE_ONLY = "usage_only"  # keep usage, drop raw body
    NONE = "none"              # keep neither (the closed guarantee)

@dataclass(frozen=True)
class WireResult:
    content: str                       # the message content string
    usage: dict[str, Any] | None       # None under Retention.NONE
    raw: dict[str, Any] | None         # None unless Retention.FULL

async def call(transport: Transport, url: str, body: dict, headers: dict, *,
               retention: Retention) -> WireResult: ...
```

**Invariants.** (i) `call` performs **exactly one** POST — no retry, no fallback (the closed
guarantee, now shared). (ii) Under `Retention.NONE`, `WireResult.usage` and `.raw` are `None` —
enforced here, not by the caller. (iii) `wire/` imports **no** pydantic-ai and **no** higher layer.

### `wire/errors.py`

`class WireError(Exception)` — detail-free by default (`str(e)` reveals nothing about the response).
`closed.py` re-exports its own `ClosedBackendError(Exception)` alias so its public error name is
stable and *outside* `ConstricError` (SALVAGE.md; the v2 test asserts detail-freeness).

---

## Layer 1 — `constraint.py` (the linchpin; pure, no pydantic-ai)

**Responsibility.** The bidirectional codec: one value that shapes the wire out (`wire()`) and turns
raw output into typed `T` (`parse()`). Single source of truth; replaces v2's four-way smear
(`ConstrainedOutput` dunders + `DecoderSpec` + `apply()` + `_guard`).

```python
@dataclass(frozen=True)
class WireSpec:
    output_type: Any                                  # what pydantic_ai Agent(output_type=) receives
    extra_body: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Constraint[T](Protocol):
    def wire(self) -> WireSpec: ...
    def parse(self, raw: Any) -> T: ...               # guard/coerce → T; raise _ParseRejected on violation
    def check(self) -> None: ...                      # optional compile-check; default no-op
    def to_config(self) -> dict[str, Any]: ...        # canonical serde form {"kind": ..., ...}
```

### The four constructors (each generic over its **true** output type)

```python
def Schema[M: BaseModel](model: type[M], *, strict: bool = True) -> Constraint[M]: ...
def Regex(pattern: str) -> Constraint[str]: ...
def Choice[S: str](*options: S) -> Constraint[S]: ...        # ← S1: ty infers Constraint[Literal[...]]
def Grammar(ebnf: str) -> Constraint[str]: ...
```

**The wire/parse table (the crown jewel — bodies VERBATIM from v2 `decoder.py`, SALVAGE.md):**

| constructor | `wire().output_type` | `wire().extra_body` | `parse(raw)` | tier (DECISION M) |
|---|---|---|---|---|
| `Schema(M)` | `NativeOutput(M, strict=strict)` | `{}` | identity (pydantic-ai already returned a validated `M`) | server-enforced |
| `Regex(p)` | `str` | `{"structured_outputs": {"regex": p}}` | `re.fullmatch(p, raw)` or raise `ConstraintViolation` → `str` | server-enforced + client-checked |
| `Choice(*o)` | `str` | `{"structured_outputs": {"choice": [*o]}}` | membership or raise `ConstraintViolation` → the `Literal` | server-enforced + client-checked |
| `Grammar(e)` | `str` | `{"structured_outputs": {"grammar": e}}` | passthrough (server-trusted) → `str` | server-enforced (no client check possible) |

**Why `parse` is uniform *and* honest (concept §4.2, confirmed S3).** In `Schema` mode pydantic-ai's
`NativeOutput` enforces `response_format` and validates+retries; `parse` receives an already-valid
`M` and returns it (identity). Its "backend didn't enforce" failure is pydantic-ai's own validation
error → `Failed`. In string modes `output_type=str` returns raw text; `parse`
performs the `fullmatch`/membership guard — v2's missing `_guard` (B4), now co-located with the
constraint and *always run* (DECISION M). `Choice.parse` returns the literal (S1) — v2's deferred
coercion, closed.

### `check()` (DECISION P — A2 structurally gone)

`Schema.check` = `xgr.Grammar.from_json_schema(json.dumps(model.model_json_schema()))`;
`Regex.check` = `xgr.Grammar.from_regex(pattern)`; `Grammar.check` = `xgr.Grammar.from_ebnf(ebnf)`;
`Choice.check` = no-op. All guarded by a try-import of `xgrammar`; absent → no-op. **`Schema` holds a
fully-formed model class**, so `model.model_json_schema()` returns the *real* schema — v2's A2 (parent
empty-schema) cannot recur (no lazy subclass).

### Composability (concept §4.4)

```python
def Nullable[T](c: Constraint[T]) -> Constraint[T | None]: ...
def OneOf[A, B](a: Constraint[A], b: Constraint[B]) -> Constraint[A | B]: ...   # tagged union at wire
```
Ordinary functions returning `Constraint[…]`. `Nullable(Schema(M)).parse(None) is None`;
`OneOf` dispatches on the wire tag. (v3.1 — flagged, not required for the spine.)

### Serde (`to_config` / `constraint_from_config` in `config.py`, DECISION K)

Canonical discriminated form: `{"kind":"schema","ref":"pkg:M"}`, `{"kind":"regex","pattern":"…"}`,
`{"kind":"choice","options":[…]}`, `{"kind":"grammar","ebnf":"…"}`. Round-trips.

**Client-side-only constraints (DECISION M, concept §23).** A plugin constraint whose `wire()`
returns `output_type=str, extra_body={}` and validates in `parse()` (`SemVer`, `IsoDate`, `Json[T]`)
is legal and useful — the tier is *visible* in what `wire()` emits.

**Dependencies:** `errors.py` only (and `pydantic`, `pydantic_ai.NativeOutput` — see note). **NO
Outcome, NO agent.**
> **Layering note on `NativeOutput`.** `Schema.wire()` must produce a `NativeOutput(M)` object, which
> is a pydantic-ai type — a *potential* layer violation (constraint is Layer 1; pydantic-ai is Layer
> 2's dependency). Resolution: `WireSpec.output_type` is typed `Any`, and `constraint.py` imports
> **only** `pydantic_ai.output.NativeOutput` (a pure output-marker, not the model loop, not
> `models.openai`). The **single-importer invariant is specifically about `pydantic_ai.models.openai`**
> (the client), which stays exclusively in `agent.py`. `NativeOutput` is a declarative marker with no
> transport. *Alternative considered and rejected:* have `wire()` return a sentinel
> `SchemaMarker(model)` and let `agent.py` translate it to `NativeOutput` — purer layering, but adds a
> translation table for zero real benefit since `NativeOutput` is inert. Recorded in RISKS.md R2 as
> the one deliberate layering concession, with the mitigation (marker-only import) that keeps it
> honest.

---

## Layer 1 — `outcome.py` (the result spine; pure)

**Responsibility.** The result spine. **USER DECISION B (2026-07-17): the lighter `Ok`/`Failed`
variant** — `Agent.run` is binary; `Denied` appears **only** on the executed pipeline; `Violated`
folds into `Failed` as a distinct error type. Still a **generic base class + method combinators**
(DECISION B / S2 — the encoding finding is independent of variant count; a bare union alias breaks
typed consumption under ty regardless).

```python
class Outcome[T]:
    """Base. Combinators are methods so T flows forward from the class param (S2)."""
    def then[U](self, f: Callable[[T], Outcome[U]]) -> Outcome[U]: ...   # bind; non-Ok short-circuits
    def map[U](self, f: Callable[[T], U]) -> Outcome[U]: ...             # fmap; wraps in Ok
    def unwrap(self) -> T: ...                                           # T or raise the variant's error
    def value_or[D](self, default: D) -> T | D: ...
    def is_ok(self) -> bool: ...

@dataclass(frozen=True)
class Ok[T](Outcome[T]):
    value: T
    usage: dict[str, Any] | None = None
    wire: RequestRecord | None = None          # present iff capture requested (DECISION N)

@dataclass(frozen=True)
class Failed(Outcome[Any]):                    # the run did not produce a valid typed T
    error: Exception                           # model/transport error, OR a ConstraintViolation (see below)

# --- generation spine: Agent.run / run_batch return  Ok[T] | Failed  ---

class ConstraintViolation(Exception):
    """parse() rejected the raw output — the backend did not enforce the constraint.
    Carried inside Failed.error so 'backend not enforcing' stays diagnosable without a top-level
    variant. (was v2 ConstraintViolationError / the concept's Violated)"""
    raw: str

# --- executed pipeline ONLY (fleet.execute): adds authority denial as data ---
@dataclass(frozen=True)
class Denied(Outcome[Any]):                    # authority declined — NOT a failure; the boundary working
    reason: str
    command: Any
# fleet.execute(...) -> Ok[Effect] | Denied | Failed
```

**Why this partition (DECISION B).** A *generation* run has exactly one way to not-succeed —
`Failed` (a model/transport error, or a `ConstraintViolation` when the server didn't enforce). An
*executed* run adds one orthogonal axis — authority — whose one honest representation is `Denied`
(a policy denial is not an error). No domain event is declined two different ways (the v2 wart);
each event has one canonical shape.

**Combinator semantics.**
- `then`: `f(self.value)` if `isinstance(self, Ok)` else `self` (`Failed`/`Denied` pass through,
  re-typed to `Outcome[U]` since they carry no `T`). Verified typed via S2 (method form).
- `map`: `Ok(f(self.value), usage=…, wire=…)` if Ok else self.
- `unwrap`: `Ok`→`value`; `Failed`→`raise self.error` (which may be a `ConstraintViolation`);
  `Denied`→`raise AuthorityError(reason)`. (Escape hatch for imperative callers.)
- `value_or`: `Ok`→`value` else `default`. Returns `T | D` (S2 ✓).

**Ergonomics.** Two supported styles:
```python
# 1. Typed method chain (the ty-verified path):
plan = (await agent.run(msg)).map(lambda p: normalize(p)).value_or(FALLBACK)
# 2. Runtime match (human-readable; runtime-correct, static narrowing pends ty — RISKS.md R1):
match await agent.run(msg):                       # generation: Ok | Failed
    case Ok(value=plan): apply(plan)
    case Failed(error=ConstraintViolation() as v): alert("backend not enforcing: %s", v)
    case Failed(error=e): raise e
match await fleet.execute(msg, executor):         # executed: Ok | Denied | Failed
    case Ok(value=eff): promote(eff)
    case Denied(reason=r): log.info("refused: %s", r)
    case Failed(error=e): raise e
```

**Invariants.** (i) `run`/`run_batch` return `Outcome[T]`/`list[Outcome[T]]` = `Ok|Failed` — failure
is *always* data (DECISION R). (ii) `fleet.execute` returns `Ok|Denied|Failed` — denial is data, not
an exception. (iii) No stage raises for a domain outcome; exceptions only for bugs (DECISION O).
(iv) `Ok.wire` is the *only* capture delivery channel (DECISION N).

**Dependencies:** `errors.py`, and `RequestRecord` from `wire/` (a Layer-0 dataclass — allowed;
outcome is Layer 1 ≥ 0). No pydantic-ai.

---

## Layer 2 — `context.py` (the input axis; pure model + pluggable assembler)

**Responsibility.** The input-side mirror of `Constraint`: inert per-segment data + a pluggable
assembler that reads `caps`/fidelity/adapter to emit wire messages and cache hints. The library
shapes cache-friendly requests and owns the correctness-critical namespace; it never caches.

```python
type Role = Literal["system", "user", "assistant"]

class Reuse(Enum):
    PREFIX = "prefix"   # dependent; reusable only as contiguous prefix — EXACT, universal default
    CHUNK  = "chunk"    # independent; position-independent (PIC) — needs identity, opt-in
    NONE   = "none"     # never cached (volatile query, secrets)

class Fidelity(Enum):
    EXACT   = "exact"   # forbid approximate reuse (prefix-or-recompute); closed pins this
    BLENDED = "blended" # permit position-independent reuse

@dataclass(frozen=True)
class Segment:
    content: str
    role: Role = "user"
    reuse: Reuse = Reuse.PREFIX
    id: str | None = None            # stable identity; derived from content hash when reuse == CHUNK

@dataclass(frozen=True)
class Context:
    segments: tuple[Segment, ...] = ()
    query: str | None = None         # sugar for the trailing NONE segment (ephemeral suffix)
    fidelity: Fidelity = Fidelity.EXACT

    @classmethod
    def of(cls, instructions: str) -> "Context":     # a lone instruction string = one PREFIX segment
        return cls(segments=(Segment(instructions, role="system", reuse=Reuse.PREFIX),))

@dataclass(frozen=True)
class WireMessages:
    messages: list[dict[str, str]]
    cache_options: dict[str, Any] = field(default_factory=dict)   # opaque passthrough (advisory hints)

@runtime_checkable
class ContextProvider(Protocol):
    def assemble(self, ctx: Context, *, caps: "BackendCaps", adapter: "Adapter | None") -> WireMessages: ...
```

**The default assembler `LinearPrefixProvider`** emits `[{role,content}…]` in segment order, drops
`NONE`/appends `query` as the trailing user message, ignores `CHUNK`/degrades it to `PREFIX`. A
`ChunkProvider` (v3.1) consults `caps.chunk_cache`, emits boundaries + ids for `CHUNK` segments, and
degrades to `PREFIX`/`NONE` where the cap is absent — **correctness invariant, only speed varies**
(concept §22.2).

**Cache namespace (correctness-critical, invariant #10, concept §22.5).** Chunk identity /
`cache_salt` = `hash(content) + base_model + adapter` — **never content alone**, so a LoRA's KV is
never wrongly shared across adapters. Owned here, not by the caller. `closed` pins `EXACT` + a unique
`cache_salt` (no cross-request reuse — cache sharing is a privacy surface, concept §22.7).

**Constraint × cache safety (concept §22.7).** XGrammar masks logits at every step regardless of KV
quality, so `BLENDED` cannot break the syntactic/structural guarantee — only semantic choice-quality
degrades. Safe for constrained agents in a way it isn't for free-form chat. Stated as a docstring
invariant.

**Scope for v3.0 (DECISION G).** Ship `Context`/`Segment`/`Reuse`/`Fidelity` + `LinearPrefixProvider`.
`CHUNK` machinery is *latent* (the `id` slot exists, `Reuse.CHUNK` exists, `ChunkProvider` is a v3.1
assembler). No `Session`.

**Dependencies:** `errors.py`; type-only refs to `BackendCaps`/`Adapter` (`TYPE_CHECKING`). No
pydantic-ai (assembly is pure).

---

## Layer 2 — `agent.py` (AgentSpec[T], Backend, Agent[T]) — SOLE pydantic_ai.models.openai importer

**Responsibility.** Bind a typed spec to a pydantic-ai `Agent`, run it, and meet `parse` with
`Outcome`. The only module that imports the pydantic-ai *client*.

```python
@dataclass(frozen=True)
class Settings:                    # validated sampling params (replaces raw model_settings dict)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # ... the _KNOWN_KEYS set (capture.py) becomes this typed struct; unknown keys are impossible,
    #     so v2's "typo passes silently" (finding) is structurally gone — no TypedDict to mis-spell.

@dataclass(frozen=True)
class Adapter:
    name: str                     # the wire `model` field value
    source: str | None = None     # HF repo/path, for providers that auto-load
    base_model: str | None = None # what it was tuned on (cache-namespacing)

@dataclass(frozen=True)
class AgentSpec[T]:
    name: str
    constraint: Constraint[T]     # carries T — the whole type flow originates here
    context: Context              # input axis (a lone str via Context.of)
    adapter: Adapter | str | None = None
    authority: str | None = None  # policy name resolved by an Authorizer at execute time
    settings: Settings = Settings()

class BackendCaps(BaseModel):
    xgrammar: bool = True
    lora: bool = True
    chunk_cache: bool = False     # NEW axis capability (concept §22); default off

@runtime_checkable
class AdapterProvider(Protocol):
    def resolve(self, adapter: Adapter | str) -> str: ...   # logical → served wire name; ensure loaded

class Backend:
    """Server + caps + ONE shared httpx client. Sole importer of pydantic_ai.models.openai."""
    def __init__(self, *, base_url: str, api_key: str = "sk-none", default_model: str,
                 caps: BackendCaps = BackendCaps(), capture: bool = False,
                 adapters: AdapterProvider | None = None,
                 context_provider: ContextProvider | None = None,     # default LinearPrefixProvider
                 transport: Transport | None = None) -> None: ...     # ASGITransport = test seam
    def build[T](self, spec: AgentSpec[T]) -> "Agent[T]": ...
    async def aclose(self) -> None: ...                               # closes the ONE shared client

class Agent[T]:
    def __init__(self, spec: AgentSpec[T], pa: "PydanticAgent", *, capture: bool) -> None: ...
    async def run(self, prompt: str) -> Outcome[T]: ...
    @property
    def raw(self) -> "PydanticAgent": ...                             # escape hatch (v2 `.agent`)
```

**`Backend.build` (the wiring):**
```python
def build[T](self, spec: AgentSpec[T]) -> Agent[T]:
    self._gate(spec)                                  # capability precondition → BackendCapabilityError
    spec.constraint.check()                           # DECISION P: compile-check if grammar-check present
    w = spec.constraint.wire()                        # WireSpec: output_type + extra_body
    model_name = self._adapters.resolve(spec.adapter) if spec.adapter else self.default_model
    settings = self._merge(spec.settings, w.extra_body)  # extra_body MERGE, decoder keys win (A4 fix)
    msgs = self._context.assemble(spec.context, caps=self.caps, adapter=self._as_adapter(spec.adapter))
    pa = PydanticAgent(self._model(model_name), output_type=w.output_type,
                       model_settings=settings, instructions=_system_of(msgs))
    return Agent(spec, pa, capture=self.capture)      # Agent[T] — T from spec.constraint
```

**`Agent.run` (parse meets Outcome — `Ok|Failed`, DECISION B; capture in the return, DECISION N):**
```python
async def run(self, prompt: str) -> Outcome[T]:      # Ok[T] | Failed
    try:
        raw = await self._pa.run(prompt)              # pydantic-ai: model loop + retries
    except PydanticModelError as e:                   # UnexpectedModelBehavior / validation
        return Failed(e)
    try:
        value: T = self._spec.constraint.parse(raw.output)
    except ConstraintViolation as v:                  # server didn't enforce → Failed (diagnosable subtype)
        return Failed(v)
    return Ok(value, usage=raw.usage, wire=self._captured_record())   # raw.usage is a PROPERTY (S3)
```
(`Constraint.parse` raises `ConstraintViolation(raw=…)` on a bare-string guard miss; `Agent.run`
wraps it as `Failed`. `fleet.execute` adds the `Denied` branch — see `authority.py`/`fleet.py`.)

**Gating (DECISION O; carried from v2 `_check_caps`).** `_gate` raises `BackendCapabilityError` when
a `Regex/Choice/Grammar` constraint hits `caps.xgrammar == False`, an adapter hits
`caps.lora == False`, a `CHUNK` context hits `caps.chunk_cache == False` (or the provider degrades —
concept §22.2), or `AdapterProvider.resolve` 404s (concept §21.4). All *config/programmer* errors,
not `Outcome` variants.

**Type flow (no cast anywhere — the whole point):**
`AgentSpec[FileEditPlan]` → `build` → `Agent[FileEditPlan]` → `run` → `Outcome[FileEditPlan]` →
`Ok.value : FileEditPlan`. `ty` sees `FileEditPlan` end to end because `Constraint[T]` carried it
(S2 confirms the method-combinator consumption is typed).

**Invariants.** (i) Only this module imports `pydantic_ai.models.openai` (test-enforced). (ii) `raw.usage`
accessed as a **property**, never called (S3 — A1 impossible). (iii) `extra_body` is **merged**, decoder
keys winning (A4). (iv) `Settings` is a typed struct, so setting typos are a type error, not silent
(v2 finding). (v) One shared client per `Backend`, closed by `aclose` (R).

**Dependencies:** `constraint`, `outcome`, `context`, `wire/`, `errors`, `pydantic_ai`
(+`.models.openai`, +`.output.NativeOutput`). Imports **nothing** from `fleet`/`authority`/`observe`.

---

## Layer 3 — `authority.py` (Authorizer × Effector — decision × effect)

**Responsibility.** Split v2's fused `Executor` into a *decision* (fail-closed by construction) and an
*effect* (the only thing that can fail). An executor is a **composition**, not a subclass hierarchy.
Kills B1/B2/B5 and "where does fornix go" structurally.

```python
@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    def as_outcome[C](self, command: C) -> Outcome[C]:
        return Ok(command) if self.allowed else Denied(self.reason, command)

@dataclass(frozen=True)
class Effect:
    ok: bool
    output: Any = None
    detail: str = ""

@runtime_checkable
class Authorizer[C](Protocol):
    def decide(self, command: C) -> Decision: ...     # TOTAL; fail-closed by construction

@runtime_checkable
class Effector[C](Protocol):
    async def run(self, command: C) -> Effect: ...    # the side effect (async — off-loop, R/B5)

class Allowlist[C](Authorizer[C]):
    """Default-deny. Wraps each rule in try/except → Decision(False) ONCE, at the boundary,
    so 'fail closed' (B2) is structural — no rule must remember it."""
    def __init__(self, rules: dict[str, Callable[[C], bool]]) -> None: ...
    def decide(self, command: C) -> Decision: ...     # unknown/unmatched/raising rule → Decision(False)

def all_of[C](*a: Authorizer[C]) -> Authorizer[C]: ...
def any_of[C](*a: Authorizer[C]) -> Authorizer[C]: ...

# Effectors (the doing side):
class Null[C](Effector[C]):        # dry-run: authorizes, NO side effect, records intent
    async def run(self, command: C) -> Effect: ...    # Effect(ok=True, detail="dry-run: would …")
class Subprocess(Effector[BaseModel]):                # runs validated argv OFF the event loop (asyncio.to_thread)
    async def run(self, command: BaseModel) -> Effect: ...

@dataclass(frozen=True)
class _Executor[C]:                # the composition; not user-facing as a class
    authorizer: Authorizer[C]
    effector: Effector[C]
    async def __call__(self, command: C) -> Outcome[Effect]:
        d = self.authorizer.decide(command)           # decide BEFORE run — B1 impossible
        if not d.allowed:
            return Denied(d.reason, command)
        eff = await self.effector.run(command)
        return Ok(eff) if eff.ok else Failed(RuntimeError(eff.detail))

def authorize[C](a: Authorizer[C]) -> "_Partial[C]": ...   # sugar: authorize(a) >> effector
# authorize(Allowlist(...)) >> Subprocess()   →  _Executor
# DryRun            = authorize(a) >> Null()                 (no special class)
# Fornix            = authorize(a) >> FornixEffector()       (one Effector, one composition)
```

**Why the decomposition dissolves the v2 findings (all *structural*, DECISION-level):**
- **B1** (execute skipped authorize): the `_Executor` *binds* `decide` before `run` — there is no
  `execute` entrypoint that can skip it. Impossible by construction.
- **B2** (raising rule crashed): `Allowlist.decide` wraps each rule once → `Decision(False)`. Fail-
  closed is the type's job, not each rule's.
- **B5** (`Effect.ok` never false): `Effect.ok` is set by the `Effector` (the only fallible party);
  a raising `Subprocess` → `Effect(ok=False)` → `Failed`. The signal is meaningful.
- **"where does fornix go"**: `Fornix = authorize(a) >> FornixEffector()` — a one-line composition +
  one new `Effector` (in `integrations/fornix.py`), not a new executor subclass. Containment (effect)
  and authorization (decision) compose; neither replaces the other (review §G) — literally the type.

**Invariants.** (i) `Authorizer.decide` is total (never raises for a domain reason). (ii) Effects are
async and run off the event loop (B5/R). (iii) Denials/failures are `Outcome` data (DECISION B/O).
(iv) Default-deny (`Allowlist`).

**Dependencies:** `outcome`, `errors`. No pydantic-ai, no agent (authority is command-shape-agnostic —
`C` is any validated command type).

### `integrations/fornix.py` — `FornixEffector` (Layer 3 plugin; stdlib subprocess only)

`class FornixEffector(Effector[BaseModel])`: serializes a validated command model to **argv** (never
a shell string — fornix's `Item.cmd` is validated argv), shells `fornix box --check … -- <argv>`,
parses the one-line JSON `Result` into `Effect`. **Zero new dependency** (subprocess boundary owned
by the app; review §G). Guarded import/availability check → `BackendCapabilityError` if fornix absent.

---

## Layer 3 — `fleet.py` (Fleet, Router)

**Responsibility.** A heterogeneous set of `Agent[Any]` from one `Backend`, plus optional validated
routing. Every entrypoint composes through `Outcome.then` — no bespoke result types (v2 had four).

```python
class Router(BaseModel):
    router: str                      # the router agent name
    routes: dict[str, str]           # route value → specialist name
    default: str | None = None

class Fleet:
    def __init__(self, backend: Backend) -> None: ...
    def build(self, specs: list[AgentSpec[Any]], *, router: Router | None = None) -> None: ...
    def __getitem__(self, name: str) -> Agent[Any]: ...
    def typed[T](self, name: str, constraint_type: type[T]) -> Agent[T]: ...   # DECISION C re-narrow
    def set_routing(self, router: Router) -> None: ...                          # validates (v2 fix)

    async def run_batch(self, calls: list[tuple[str, str]]) -> list[Outcome[Any]]: ...   # DECISION R
    async def route(self, msg: str) -> str: ...                                 # -> specialist name
    async def run(self, msg: str) -> Outcome[Any]: ...                          # route ∘ generate
    async def execute(self, msg: str, executor: "_Executor[Any]") -> Outcome[Effect]: ...
                                     # route ∘ generate ∘ authorize ∘ effect — one Outcome.then chain
    async def aclose(self) -> None: ...                                         # backend.aclose()
```

**`execute` is literally a bind chain** (concept §10) — no `RoutedExecution`/`RoutedResult`/
`BatchResult` types:
```python
async def execute(self, msg, executor):
    name = await self.route(msg)                       # RoutingError on unroutable (config, raises)
    oc = await self[name].run(msg)                     # Outcome[Command]
    return await _bind_async(oc, executor)             # Ok→authorize→effect; non-Ok passes through
```

**`Router` validation (carried from v2 `_check_route_coverage`).** At `build`/`set_routing`: if the
router's constraint is `Choice(...)` (an introspectable `Literal`), **every** route value must have a
`routes` entry or a `default`, else `RoutingError`. Routing stays *data*; no hidden loop (concept §10).

**Heterogeneous typing (DECISION C).** `__getitem__ -> Agent[Any]` is honest (no single fleet `T`);
`fleet.typed("git_ops", GitCommand) -> Agent[GitCommand]` re-narrows for typed call sites without a
`cast` — `typed` verifies at runtime that the built spec's constraint yields that type and returns the
same object typed.

**Invariants.** (i) `run_batch -> list[Outcome[T]]`, per-item failure is `Failed` (R; no lost
siblings). (ii) One shared client via the `Backend` (R). (iii) Routing validated at build/set, never
mutated raw (v2 `set_routing` fix). (iv) Nothing executes implicitly — effects only inside `execute`
(invariant #5).

**Dependencies:** `agent`, `authority`, `outcome`, `errors`. No pydantic-ai directly (via `agent`).

---

## Layer 0/1 — `closed.py` (preset over wire+constraint; NO pydantic-ai)

**Responsibility.** The Lodestar path: loopback-only, json-schema-only, one request, no capture, no
retention, detail-free — **preserved byte-for-byte** (invariant #8) but now a *thin preset* over the
shared `wire/`+`constraint` primitives instead of a bespoke re-implementation.

```python
def closed_backend(*, base_url: str, api_key: str, model: str, timeout: float,
                   output_type: type[BaseModel], instructions: str) -> "ClosedBackend": ...

class ClosedBackend:                         # NO agent/run_sync/build/attach_transport/capture/raw
    async def run(self, prompt: str) -> BaseModel: ...     # validated model from ONE call, or ClosedBackendError
    async def aclose(self) -> None: ...

class ClosedBackendError(Exception): ...     # deliberately detail-free; OUTSIDE ConstricError
```

**Composition (concept §8):**
```python
def closed_backend(...):
    return ClosedBackend(
        transport = LoopbackTransport(base_url, timeout=timeout),  # shared loopback guard (wire/)
        retention = Retention.NONE,                                # drop raw+usage — the privacy guarantee
        constraint = Schema(output_type, strict=True),             # shared Layer-1 codec
        model = model_name(model), instructions = instructions_bounded(instructions), timeout = timeout,
    )
```

**Guarantees, mapped to their now-shared enforcers (from the closed ground-truth report):**
- loopback-only, `localhost` rejected → `wire/transport.LoopbackTransport` (verbatim guard + comment).
- bounded inputs (`model` 1–128 charset; instructions ≤4096B; prompt ≤16_384B; `0 < timeout ≤600`) →
  `wire/request` `BoundedStr` value objects (verbatim limits).
- strict `response_format` json_schema, name `"closed_output"`, `strict:True`, `stream:False`, and
  **nothing else** (no tools/tool_choice/store/user/logprobs/temperature/extra_body) →
  `wire/request.response_format` + a fixed `chat_body`; the closed preset passes no extra_body.
- one request, no retry → `wire/client.call` (exactly one POST, invariant).
- no capture / no retention → `Retention.NONE` (usage/raw forced `None`); no capture wired.
- detail-free error → `ClosedBackendError()` from `(IndexError|KeyError|TypeError|ValueError|WireError)`.
- client owns+closes itself, `follow_redirects=False`, `trust_env=False` → `LoopbackTransport`.

**Compatibility shim (DECISION H).** Also export a `ClosedBackend`-shaped class whose `__init__`
matches v2's keyword signature exactly, so Lodestar migrates with a near-zero diff.

**Invariants.** (i) Imports only `wire/` + `constraint` + `pydantic` + `httpx` — **never pydantic-ai**
(test-enforced, T8; this *import isolation* is the load-bearing guarantee. Per DECISION I, pydantic-ai
is a core dependency, so it is installed-but-unused for a closed-only consumer — the guarantee is that
none of it is on the closed code path, not that it is absent from disk). (ii) Surface has no escape
hatches (test asserts absence of `agent`/`run_sync`/`build`/`attach_transport`). (iii) Errors never
leak detail.

---

## Layer 2-edge — `config.py` (the serialization boundary; DECISION K)

**Responsibility.** The *one* place strings become code. Everything else is typed. Localizes the
import-execution vector behind an explicit allowlist.

```python
def constraint_from_config(d: dict[str, Any], *, allow_modules: frozenset[str]) -> Constraint[Any]: ...
def spec_from_config(d: dict[str, Any], *, allow_modules: frozenset[str]) -> AgentSpec[Any]: ...

# per-seam registries (open-to-extension, DECISION K):
def register_constraint(kind: str, from_config: Callable[[dict], Constraint[Any]]) -> None: ...
# + entry-point group "constric.constraints" discovered lazily.
```

**Invariants.** (i) `constraint_from_config` refuses any `ref` whose module prefix ∉ `allow_modules`
→ `ConfigError` (v2's latent `importlib`-on-data vector, now gated & localized). (ii) No other module
does `importlib` on config data. (iii) In-code plugins need **no** registration (structural typing);
only YAML/JSON-loaded plugins touch the registry.

**Dependencies:** `constraint`, `agent` (for `AgentSpec`), `context`, `errors`. This is the only
Layer-2 module that resolves strings; `agent.py` itself never imports strings.

---

## Layer 4 — `observe/` (pipeline observers; [observe] extra)

**Responsibility.** v2's `dual_path/` reborn as **observers over the pipeline** rather than a parallel
runner — reusing the `Agent[T]`/`Outcome[T]` spine instead of re-implementing runs.

```python
@runtime_checkable
class Observer(Protocol):
    async def observe(self, stage: str, outcome: Outcome[Any]) -> None: ...   # may persist durably

class DualPathObserver(Observer):
    """Fans a message to a second (reference) Agent[T]; records the (Ok[T], Ok[T]) pair.
    DBOS step keyed on run_id → idempotent by construction (fixes v2 durability gap)."""
```

**What carries from v2 §C dual-path list (as implementation notes, not re-litigated):** connection
pooling (psycopg_pool, not connect-per-save), `SetWorkflowID` contextvar-vs-threadlocal verification
(RISKS.md R5), idempotent insert keyed by `run_id`, the jsonb store DDL, `ComparisonRecord` shape,
the SFT-export view. `DualPathDecodeMode` narrowing is gone — the observer records whatever `Outcome`
it sees.

**Invariants.** (i) Behind `[observe]` (dbos + psycopg **declared**, D2 fix). (ii) The persisted
artifact is an `Outcome`; persisting it in a DBOS step keyed on `run_id` is idempotent (fixes v2's
"record itself not durable" — §C). (iii) Observers are pure taps; they never alter the pipeline result.

**Dependencies:** `outcome`, `agent`, `errors`, `dbos`, `psycopg`. No layer below imports `observe`.

---

## Cross-cutting: how every v2 review finding becomes structurally impossible

(Full table in the executive summary; the *mechanism* per finding lives in the module above that
kills it — A1→`agent.py` property access (S3); A2→`constraint.Schema.check` on a formed class;
A4→`Backend._merge`; A5→`Ok.wire` delivery; B1/B2/B5→`authority` decomposition; B4→`Constraint.parse`
always-run; C-typing→`Constraint[T]` end-to-end (S2); C-client→one shared client; C-run_batch→`Outcome`
list; C-settings-typo→typed `Settings`; C-import-vector→`config.allow_modules`; D-packaging→extras
layout (DECISION I); F/G-grail/fornix→gone / `FornixEffector`.)
