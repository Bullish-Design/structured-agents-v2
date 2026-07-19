# v3 — the pared essence (post-simplification)

Supersedes the relevant parts of `00-PLAN.md` / `DECISIONS.md` / `DESIGN.md` after the 2026-07-17
simplification pass. Three user decisions drove it:

1. **Drop `closed`.** Not this library's job; if Lodestar needs it, it lives downstream.
2. **`run -> T`, raise on failure.** No `Outcome` sum-type spine, no combinators.
3. **`instructions: str` for v3.0.** Defer the `Context`/`Reuse` cache axis until a consumer exists.

Scope stays **full** (constraint · agent · authority · fleet · observe · config) — we pared the
*internal complexity*, not the feature set.

---

## The essence, in one sentence

> **Produce a typed, validated Python value from a constrained local-model call — cooperating with
> vLLM/XGrammar (+ per-agent LoRA) — with no cast.**

The type flows mechanically from one value, the `Constraint[T]` codec:

```
Constraint[T] → AgentSpec[T] → Agent[T].run(prompt) -> T          # raises on failure
```

No wrapper at the end. `T` *is* the return type; `ty` types it end-to-end with zero casts because the
constraint carried it the whole way.

---

## What we removed, and why it's safe

| Removed | Was there for | Why it goes |
|---|---|---|
| **`closed.py`** | Lodestar's loopback/privacy path | Not this library's concern (user). Downstream if ever needed. Dissolves REVIEW BLOCKER 1 + 2. |
| **`wire/` (whole layer)** | shared pydantic-ai-free primitives for `closed` + rich | With no `closed`, the rich `Backend` uses pydantic-ai + httpx directly. Only the httpx capture *hook* survives → folded into `agent.py`. |
| **`outcome.py` (whole module)** | `Ok/Failed/Denied/Violated` + `.then/.map/...` | `run -> T` + exceptions is simpler and fully typed. Kills the S2 union-narrowing problem, the `then`-cast, the untyped `match`. Dissolves REVIEW MAJOR 4/4b/5/7, MINOR 12, RISK R1. |
| **`context.py` (whole module)** | `Context/Segment/Reuse/Fidelity` cache axis | Only `PREFIX` (= an instructions string) ever ran. Ship `instructions: str`; add the axis with its `ChunkProvider` consumer later. Dissolves REVIEW MAJOR 8/9. |
| **`Effect`, `_Executor`, `>>` sugar, `Retention`** | authority wrapper + closed privacy knob | Subsumed by plain returns; see authority below. |
| **Import-isolation invariant, R2 `NativeOutput` concession, most of T8** | proving `closed` is pydantic-ai-free | No `closed` ⇒ nothing to isolate. pydantic-ai is just a normal core dep. |

**One review finding survives and still gates Phase 1:** **BLOCKER 3** — the wire bodies were captured
on pydantic-ai **1.87**; v3 pins **2.11**. For `Schema` mode pydantic-ai now *owns* the
`response_format` body entirely (we don't build it ourselves anymore), so re-capturing all four
`.wire()`-driven bodies against 2.11 and pinning them with a test is *more* important, not less.

---

## The pared module stack (5 layers → 3)

```
Layer 2  observe/     pipeline observers (dual-path, evals)          [observe] extra
Layer 1  authority.py Decision · Authorizer · Effector · Allowlist · execute()
         fleet.py     Fleet · Router
         config.py    spec_from_config + allow_modules (serialization edge)
Layer 0  agent.py     AgentSpec[T] · Backend · Agent[T]   ── sole pydantic-ai importer; capture hook inline
         constraint.py Constraint[T] · Schema/Regex/Choice/Grammar · WireSpec   (pure, the linchpin)
         errors.py    exception tree (layer-less)
integrations/fornix.py  FornixEffector (optional)
```

Dependencies flow down. `constraint.py` and `errors.py` are pure. `agent.py` is the only pydantic-ai
importer (kept as a nicety for swappability — no longer a load-bearing guarantee since `closed` is
gone).

---

## Key signatures

### `constraint.py` — unchanged linchpin (this was always the best part)

```python
@dataclass(frozen=True)
class WireSpec:
    output_type: Any                      # what pydantic-ai Agent(output_type=) receives
    extra_body: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Constraint[T](Protocol):
    def wire(self) -> WireSpec: ...
    def parse(self, raw: Any) -> T: ...   # guard/coerce → T; raise ConstraintViolation on a guard miss
    def check(self) -> None: ...          # optional xgrammar compile-check; default no-op

def Schema[M: BaseModel](model: type[M], *, strict: bool = True) -> Constraint[M]: ...
def Regex(pattern: str) -> Constraint[str]: ...
def Choice[S: str](*options: S) -> Constraint[S]: ...    # ty infers Constraint[Literal[...]] (S1 ✓)
def Grammar(ebnf: str) -> Constraint[str]: ...
```

Wire/parse table unchanged (Schema→`NativeOutput(M)`/identity; string modes→`extra_body
structured_outputs`/`fullmatch`|membership|passthrough). **`Schema.parse` = identity is now honest** —
the only path is the rich path where pydantic-ai already returned a validated `M`.

### `agent.py` — `run -> T`, raise on failure

```python
@dataclass(frozen=True)
class Settings:                           # typed sampling params (typo = ty error)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentSpec[T]:
    name: str
    constraint: Constraint[T]             # carries T
    instructions: str                     # ← plain string (context axis deferred)
    adapter: str | None = None            # served LoRA/model name on the wire `model` field
    authority: str | None = None          # policy name resolved by an Authorizer at execute time
    settings: Settings = Settings()

class Backend:                            # sole pydantic-ai importer; one shared httpx client
    def __init__(self, *, base_url: str, default_model: str, caps: BackendCaps = ...,
                 capture: bool = False, adapters: AdapterProvider | None = None,
                 transport: Transport | None = None) -> None: ...
    def build[T](self, spec: AgentSpec[T]) -> Agent[T]: ...
    async def aclose(self) -> None: ...

class Agent[T]:
    async def run(self, prompt: str) -> T: ...
        # raises ConstraintViolation (backend didn't enforce) or a model/transport error
    async def run_batch(self, prompts: list[str]) -> list[T | Exception]: ...   # siblings never lost
    async def run_with_record(self, prompt: str) -> tuple[T, RequestRecord]: ...  # capture on demand
    @property
    def raw(self) -> PydanticAgent: ...   # escape hatch
```

`Agent.run` body:
```python
async def run(self, prompt: str) -> T:
    raw = await self._pa.run(prompt)              # pydantic-ai loop; may raise UnexpectedModelBehavior
    return self._spec.constraint.parse(raw.output)  # identity (schema) or guard (string modes); may raise ConstraintViolation
```

**Capture without the A5 race, without a result wrapper:** per-run attribution is structural because
the record is *returned by the same call* (`run_with_record`). No `ContextVar`, no shared "last" sink.
(This is cleaner than the plan's `Ok.wire`, and needs no `Outcome`.)

### `authority.py` — decision is data; failure is an exception

```python
@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""

@runtime_checkable
class Authorizer[C](Protocol):
    def decide(self, command: C) -> Decision: ...    # total; never raises for a domain reason

@runtime_checkable
class Effector[C](Protocol):
    async def run(self, command: C) -> Any: ...      # the side effect; raises on failure

class Allowlist[C](Authorizer[C]):                   # default-deny; wraps each rule → Decision(False) once (B2)
    def __init__(self, rules: dict[str, Callable[[C], bool]]) -> None: ...

@dataclass(frozen=True)
class Denied:                                        # the one genuine decision-as-data
    reason: str
    command: Any

async def execute[C](authorizer: Authorizer[C], effector: Effector[C], command: C) -> Denied | Any:
    d = authorizer.decide(command)                   # decide BEFORE run — B1
    if not d.allowed:
        return Denied(d.reason, command)             # denial is data, not an exception
    return await effector.run(command)               # allowed → do it; effect failure raises
```

The whole authority model, minus the plan's `Effect` wrapper / `_Executor` class / `>>` operator /
`Outcome` threading. `Denied` is a single small data type at one call site — not a pervasive spine.
Fornix stays: `FornixEffector(Effector)` in `integrations/fornix.py`; use it via `execute(auth, fornix,
cmd)`. (Residual, minor: `Effector.run` is still directly callable without a `Decision` — document it
as the hazard; the blessed path is `execute`.)

### `fleet.py` — heterogeneous, thin

```python
class Router(BaseModel):
    router: str; routes: dict[str, str]; default: str | None = None

class Fleet:
    def build(self, specs: list[AgentSpec[Any]], *, router: Router | None = None) -> None: ...
    def __getitem__(self, name: str) -> Agent[Any]: ...
    def typed[T](self, name: str, t: type[T]) -> Agent[T]: ...    # ONE localized, runtime-checked cast (say so)
    async def route(self, msg: str) -> str: ...
    async def run_batch(self, calls: list[tuple[str, str]]) -> list[Any]: ...   # per-item T-or-Exception
    async def execute(self, msg: str, authorizer, effector) -> Denied | Any: ...
    async def aclose(self) -> None: ...
```

`fleet.execute` = `route → run (may raise) → execute(auth, eff, command)`. No bespoke result types.

---

## What this dissolves from the review

| Review finding | Status now |
|---|---|
| BLOCKER 1 (closed import isolation) | **gone** — no `closed` |
| BLOCKER 2 (`Schema.parse` asymmetry) | **gone** — rich path only; identity is honest |
| BLOCKER 3 (wire capture on 1.87 vs 2.11 pin) | **survives** — must re-capture on 2.11 (now the only Phase-1 gate) |
| MAJOR 4 / 4b / 5 / 7, MINOR 12, RISK R1 (Outcome spine) | **gone** — `run -> T` + exceptions |
| MAJOR 8 / 9 (context axis leak + YAGNI) | **gone** — `instructions: str` |
| MAJOR 6 (Effector.run sans Decision) | **reduced to a documented hazard** |
| MINOR 10 (`fleet.typed` cast) | remains — state it as "one localized cast," don't claim "no cast" |
| A2 / B2 structural, A4 merge, C-client/batch/settings/import, F/G | **kept** — the genuinely good fixes |

---

## Remaining questions (smaller, for the next pass)

1. **Capture in core, or observe-only?** `run_with_record` is clean, but capture mostly serves
   dual-path/evals (`[observe]`). Could live entirely behind `[observe]` and keep `agent.py` to just
   `run`. Leaning: keep a minimal `run_with_record` in core (it's the wire-grounding technique), but
   open.
2. **Is `config.py`/YAML-loading essence, or a later add?** Code-first (`Schema(FileEditPlan)` in
   Python) needs no serde. If every consumer is code-first for v3.0, `config.py` + `allow_modules`
   could be deferred like the context axis. (You kept "full stack," so it stays for now — flag if
   you'd rather defer it.)
3. **`run_batch` on `Agent` (typed `list[T | Exception]`) vs only on `Fleet` (`list[Any]`)** — I'd put
   the typed one on `Agent` and the heterogeneous one on `Fleet`.
4. **Does `check()` (xgrammar compile-check) earn its place** given it only runs under the dev-only
   `grammar-check` extra and the server enforces anyway? Minor; could drop to shrink the surface.
