# structured-agents-v2 — Library Wrapper Concept (rev 2)

## Document status

- **Status:** Implementable concept for the core library wrapper
- **Builds on:** `01-xgrammar-concept/STRUCTURED_AGENT_CONCEPT.md` (thesis),
  `01-xgrammar-concept/spike/FINDINGS.md` (request path), and
  `02-library-wrapper/VERIFICATION.md` (output typing + client-side compile)
- **Package:** `src/structured_agents_v2/`
- **Python:** 3.13 · **PydanticAI:** 1.87.0 · run via `devenv shell -- ...`
- **Core thesis:** A *thin, declarative binding* over PydanticAI where the **output type
  carries its own decoding constraint**. Each small agent ties a constrained output
  model to a model/LoRA adapter and an authority boundary, so specialized agents emit
  validated command objects and run batched against one OpenAI-compatible (vLLM) server.

> **rev 2 change:** the constraint is now a *built-in capability of the response object*
> (`ConstrainedOutput` base class), not a separate `DecoderSpec` wired into the profile.
> This folds open question #1 into the type system and is grounded in VERIFICATION.md.

---

## 1. What we are building — and the facts it rests on

PydanticAI owns the runtime, model client, and validation. This library is the
*binding + composition layer* plus the *authority boundary* PydanticAI leaves to apps.

Verified facts that shape every decision below:

| Fact (verified) | Consequence for the library |
|---|---|
| Model output defaults to the **function-calling tool**, not `response_format` | the library must apply `NativeOutput` itself for json_schema mode |
| `NativeOutput(Model)` → clean `response_format: json_schema` | this is the XGrammar json_schema path |
| `output_type=str` → **text mode** (no rf/tools) | substrate for bare-string grammar/regex/choice via `extra_body` |
| `Literal[...]` → tool/JSON, *not* a bare string | typed choice = JSON; bare-token choice = `str`+`extra_body` |
| `extra_body` keys land verbatim top-level | the grammar/regex/choice hook |
| OpenAI `model` field selects the adapter | per-agent LoRA |
| `asyncio.gather` over `run()` batches (2.4×) | fleet-level concurrency |
| `xgrammar.Grammar.from_json_schema/from_regex` compile client-side (but pull torch+CUDA) | optional **dev-only** compatibility check |

### Non-goals (explicit)

- No custom agent loop, model client, or output parser.
- No Grail/Monty toolset plane (archived 01/02 direction).
- No training/fine-tuning (adapters are produced out-of-band; we only *select* them).
- Not a multi-agent orchestration engine — routing is *data + sugar*, not a workflow runtime.
- The executor is a boundary/protocol, not a full sandbox, in the MVP.

---

## 2. Design tenets

1. **The constraint lives on the output type.** You subclass `ConstrainedOutput`; the
   agent picks up the decode contract automatically. No separate spec to keep in sync.
2. **PydanticAI-native, NativeOutput by default.** json_schema models are auto-wrapped in
   `NativeOutput` so they ride `response_format`, not the default tool path.
3. **Backend-aware.** A `Backend` knows its capabilities (XGrammar? LoRA?) and fails at
   build time if an agent asks for something it can't do.
4. **Authority is separate from generation.** XGrammar guarantees *syntax*; the executor
   guarantees *authority*. The library never executes a side effect implicitly.
5. **Batched by construction.** Agents sharing a backend run concurrently in one call.
6. **Escape hatches always.** Every wrapper exposes the underlying `pydantic_ai.Agent`.

---

## 3. Architecture

```text
   Application: defines ConstrainedOutput command models + executor policies + a RoutingTable
        │
        ▼
   structured_agents_v2
     ConstrainedOutput (base)  ──carries──▶ decode mode + params (+ optional client compile-check)
        │ (used as output_type)
        ▼
     AgentProfile ──build──▶ StructuredAgent ──holds──▶ pydantic_ai.Agent
        │ name / adapter / instructions / output_type / policy
        ▼
     Backend (caps-checked)  ──builds──▶ OpenAIChatModel + Provider
     AgentSet (shares Backend) ──▶ run_batch() ; RoutingTable ──▶ route()/route_and_run()
     Executor (Protocol) ──▶ authorize() / execute()   (never called implicitly)
        │
        ▼  OpenAI /v1  (response_format | extra_body, model=adapter)
   vLLM container (deploy/vllm) — base model + XGrammar + LoRA   (today: llama.cpp; same contract)
```

---

## 4. Core abstractions

### 4.1 `ConstrainedOutput` — the constraint as a built-in capability (primary surface)

A `BaseModel` subclass that carries its own decode contract as class metadata. Subclass
it and the constraint travels with the type wherever it's used as `output_type`.

```python
from typing import Any, ClassVar, Literal
from pydantic import BaseModel

DecodeMode = Literal["json_schema", "grammar", "regex", "choice"]

class ConstrainedOutput(BaseModel):
    # --- decode contract (override per subclass) ---
    __decode_mode__: ClassVar[DecodeMode] = "json_schema"
    __grammar__: ClassVar[str | None] = None      # EBNF/GBNF for mode="grammar"
    __regex__: ClassVar[str | None] = None        # for mode="regex"
    __choices__: ClassVar[list[str] | None] = None  # for mode="choice"
    __strict__: ClassVar[bool] = True             # closed objects + required fields

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        _validate_mode_fields(cls)        # e.g. regex mode requires __regex__
        _maybe_compile_check(cls)         # optional dev-only xgrammar compile (gated import)

    @classmethod
    def decoder_spec(cls) -> "DecoderSpec":
        return DecoderSpec(mode=cls.__decode_mode__, grammar=cls.__grammar__,
                           regex=cls.__regex__, choices=cls.__choices__, strict=cls.__strict__)
```

Examples — the constraint is now a property of the data definition:

```python
class FilePatch(ConstrainedOutput):
    op: Literal["replace", "insert_after", "delete"]
    path: str
    content: str | None = None

class FileEditPlan(ConstrainedOutput):            # json_schema (default)
    action: Literal["edit_file", "refuse"]
    patches: list[FilePatch] = []
    reason: str

class Route(ConstrainedOutput):                   # typed choice via json_schema enum
    route: Literal["file_edit", "git_ops", "answer", "refuse"]

class GitCommandLine(ConstrainedOutput):          # bare-string, regex-constrained
    __decode_mode__ = "regex"
    __regex__ = r"git (status|diff|add|commit|checkout -b) [\w./\- ]*"
    value: str
```

**How #1 is resolved:** mode lives on the model. `json_schema` (incl. `Literal`-enum
routers) is the typed default; `grammar`/`regex`/`choice` are opt-in for bare-string
outputs. The library translates each to the right PydanticAI surface (next section).

### 4.2 `DecoderSpec` + application (derived, mostly internal)

`DecoderSpec` is what `ConstrainedOutput.decoder_spec()` produces; it knows how to apply
itself to an agent. Grounded in the verified wire shapes:

| `mode` | output wrapping | model_settings.extra_body |
|---|---|---|
| `json_schema` | `NativeOutput(model)` → `response_format` | none (XGrammar is server-default) |
| `grammar` | `output_type=str` (text mode) | `{"structured_outputs": {"grammar": ...}}` |
| `regex` | `output_type=str` | `{"structured_outputs": {"regex": ...}}` |
| `choice` | `output_type=str` | `{"structured_outputs": {"choice": [...]}}` |

For the bare-string modes the library validates the returned string against
`regex`/`choices` as a guard, then (for `choice`) coerces to the declared `Literal` so
the caller still gets a typed value. `DecoderSpec` stays public for the rare case of
applying a constraint to an output type you don't own (can't subclass).

### 4.3 `Backend` — server + capabilities (only file importing `pydantic_ai.models.openai`)

```python
class BackendCaps(BaseModel):
    xgrammar: bool = True
    lora: bool = True
    server_default_backend: bool = True   # XGrammar set via server flag, not per-request

class Backend(BaseModel):
    base_url: str
    api_key: str = "sk-none"
    default_model: str
    caps: BackendCaps = BackendCaps()
    # runtime: .model_for(adapter) -> OpenAIChatModel ; .capture() -> http_client hook
```

`build()` raises `BackendCapabilityError` if an agent's decoder/adapter needs a capability
the backend lacks (e.g. `mode!="json_schema"` but `not caps.xgrammar`). Fail at construction.

### 4.4 `AgentProfile` — the serializable binding

```python
class AgentProfile(BaseModel):
    name: str
    adapter: str | None = None            # LoRA name; None → backend.default_model
    instructions: str
    output_type_ref: str | None = None    # dotted path to a ConstrainedOutput (or plain Model)
    decoder: DecoderSpec | None = None     # only to override a non-ConstrainedOutput type
    policy: str | None = None
    model_settings: dict[str, Any] = {}    # temperature, max_tokens, seed, ...
```

When `output_type_ref` resolves to a `ConstrainedOutput`, its `decoder_spec()` is used and
`decoder` may stay `None`. `decoder` is the escape hatch for constraining a model you
can't subclass.

### 4.5 `StructuredAgent` — the wrapper

```python
class StructuredAgent:
    profile: AgentProfile
    @property
    def agent(self) -> pydantic_ai.Agent: ...      # escape hatch
    async def run(self, prompt: str, **kw) -> "AgentResult[OutputT]": ...
    def run_sync(self, prompt: str, **kw) -> "AgentResult[OutputT]": ...
```

Build = `Backend.model_for(adapter)` + decoder application (`NativeOutput` or `str`+extra_body)
+ merged `model_settings` + `instructions`. `AgentResult` is minimal: `output`, `usage`,
`request_body` (when capture is on), `raw` (the PydanticAI result).

### 4.6 `AgentSet` + `RoutingTable` — fleet, batching, routing-as-data (#4 → Option C)

```python
class RoutingTable(BaseModel):
    router: str                         # agent name whose output is the route value
    routes: dict[str, str]              # route value -> specialist agent name
    # validated at build: every value in routes maps to a real agent;
    # every Route literal the router can emit has an entry (or an explicit default).

class AgentSet:
    backend: Backend
    agents: dict[str, StructuredAgent]
    routing: RoutingTable | None = None
    def build(self, profiles: list[AgentProfile], routing: RoutingTable | None = None) -> None: ...
    async def run_batch(self, calls: list[tuple[str, str]]) -> list[AgentResult]: ...   # gather
    async def route(self, msg: str) -> str: ...               # run router, return specialist name
    async def route_and_run(self, msg: str) -> "RoutedResult": ...  # explicit two-step sugar
```

**#4 resolved (Option C):** routing is *data* (validated, serializable, inspectable) but
dispatch is *explicit sugar* — `route()` + `run_batch()` stay visible; `route_and_run()`
is thin convenience returning `RoutedResult(route, output)`. No hidden control flow, no
loops/branches owned by the library.

### 4.7 `Executor` — the authority boundary (Protocol)

```python
class Executor(Protocol):
    def authorize(self, policy: str, command: BaseModel) -> "Decision": ...
    def execute(self, policy: str, command: BaseModel) -> "ExecResult": ...
```

The library guarantees the command is well-formed (decoder + Pydantic); the executor
decides if it's allowed and performs it (allowlists, approval gates, audit). MVP ships a
`DryRunExecutor` + `PolicyError`; real executors are app-provided. **No StructuredAgent
ever calls an executor implicitly.**

---

## 5. Constraint compatibility checking (optional, dev-only)

`ConstrainedOutput.__init_subclass__` *optionally* compiles the model's
`model_json_schema()` (or `__regex__`/`__grammar__`) with `xgrammar` to fail fast on an
un-compilable schema — verified working client-side, no server. Because `xgrammar` drags
in torch + CUDA (~2 GB), it is an **optional extra** (`[grammar-check]`) behind a gated
import: if absent, the check is skipped (vLLM enforces server-side regardless). Enabled in
CI / dev to catch unsupported schemas at definition time.

---

## 6. End-to-end usage sketch

```python
from structured_agents_v2 import Backend, AgentProfile, AgentSet, RoutingTable
from myapp.schemas import Route, FileEditPlan, GitCommand  # ConstrainedOutput subclasses

backend = Backend(base_url="http://remora-server:8000/v1", api_key="...", default_model="base")

profiles = [
    AgentProfile(name="router", adapter="router",
                 instructions="Route to exactly one specialist.",
                 output_type_ref="myapp.schemas:Route"),
    AgentProfile(name="file_edit", adapter="file-edit",
                 instructions="Produce file-edit plans only.",
                 output_type_ref="myapp.schemas:FileEditPlan", policy="repo_file_edit_v1"),
    AgentProfile(name="git_ops", adapter="git-ops",
                 instructions="Translate to a single safe git command.",
                 output_type_ref="myapp.schemas:GitCommand", policy="git_safe_v1"),
]
routing = RoutingTable(router="router",
                       routes={"file_edit": "file_edit", "git_ops": "git_ops"})

fleet = AgentSet(backend=backend)
fleet.build(profiles, routing=routing)

routed = await fleet.route_and_run(user_msg)     # RoutedResult(route="file_edit", output=FileEditPlan(...))
decision = executor.authorize("repo_file_edit_v1", routed.output)
if decision.allowed:
    executor.execute("repo_file_edit_v1", routed.output)

# escape hatch stays open:
raw_agent = fleet.agents["file_edit"].agent       # plain pydantic_ai.Agent
```

---

## 7. Proposed package layout

```text
src/structured_agents_v2/
├── __init__.py          # public exports
├── constrained.py       # ConstrainedOutput, mode validation, optional xgrammar compile-check
├── decoder.py           # DecoderSpec + application (mode → NativeOutput / str+extra_body)
├── backend.py           # Backend, BackendCaps, BackendCapabilityError (sole pydantic_ai.models.openai importer)
├── profile.py           # AgentProfile + output_type_ref resolution (importlib)
├── agent.py             # StructuredAgent, AgentResult
├── fleet.py             # AgentSet, RoutingTable, RoutedResult, run_batch
├── executor.py          # Executor protocol, Decision, ExecResult, DryRunExecutor, PolicyError
├── capture.py           # http_client event-hook recorder (from the spike)
└── errors.py            # error hierarchy
```

`tests/conftest.py` exposes an in-process ASGI mock OpenAI server (`mock_backend`) so the
suite runs without a GPU, plus a `live` marker for runs against `$LLM_BASE_URL` and a
`grammar_check` marker for the optional xgrammar extra.

---

## 8. Build phases

1. **Constraint core (no network):** `ConstrainedOutput` + mode validation, `DecoderSpec`
   + application, `capture.py`. Unit tests assert the wire shape each mode produces
   (NativeOutput→response_format; str+extra_body) against the in-process mock.
2. **Backend + agent:** `Backend`/caps, `AgentProfile` + ref resolution, `StructuredAgent.run`.
   Caps-gate tests; a `live` test reproduces the json_schema round-trip.
3. **Fleet + routing — ✅ BUILT (2026-06-09).** `fleet.py`: `AgentSet.build/run_batch`
   (order-preserving top-level `asyncio.gather`), `RoutingTable` (router/routes/default/route_field,
   validated at build incl. Literal-coverage), `route`/`route_and_run` (explicit dispatch),
   `RoutedResult`; `FleetError`/`RoutingError`. `tests/test_fleet.py` (17 tests, incl. a tracking
   ASGI app that asserts >1 request in flight — concurrency proven on the mock; the `live` 2.4×
   reproduction stays for the vLLM cutover). fleet.py 93% cov, ty+ruff clean.
4. **Executor boundary:** `Executor`, `DryRunExecutor`, router→specialist→executor example.
5. **vLLM cutover + polish:** point `Backend` at `deploy/vllm`; verify grammar/regex/choice
   + LoRA live; enable `[grammar-check]` in CI; YAML/JSON profile loading; examples; mypy+ruff.

---

## 9. MVP acceptance criteria

1. A `ConstrainedOutput` (json_schema) used as `output_type` produces a validated command
   object from a live OpenAI-compatible server, via `response_format` (not the tool path).
2. `grammar`/`regex`/`choice` models emit `output_type=str` + correct `extra_body` (wire-asserted).
3. A different `adapter` changes the wire `model` field.
4. `run_batch` dispatches concurrently (measurably faster than sequential).
5. A backend lacking a capability raises `BackendCapabilityError` at build time.
6. `RoutingTable` validates route→agent coverage; `route_and_run` returns a typed `RoutedResult`.
7. Validated commands flow to an `Executor` the library never calls implicitly.
8. With `[grammar-check]` installed, an un-compilable schema fails at class definition.
9. Every wrapper exposes the underlying `pydantic_ai.Agent`.

---

## 10. Resolved decisions & remaining open questions

**Resolved**
- **#1 (output typing):** folded into `ConstrainedOutput`. Mode is declared on the model;
  json_schema (incl. `Literal`-enum routers) is the typed default; grammar/regex/choice
  are opt-in bare-string modes (`output_type=str` + `extra_body`, library-guarded).
- **#3 (capture):** per-`Backend` opt-in (`Backend(capture=True)`); the last request body is
  attached to `AgentResult.request_body`. **(Implemented in Phase 2.)**
- **#4 (routing):** Option C — routing as validated data + explicit dispatch sugar.
- **Compile check:** feasible client-side, shipped as optional dev extra `[grammar-check]`.

**Phase 2 outcomes (Backend + AgentProfile + StructuredAgent — on `main`)**
- `Backend.build(profile)` is the agent factory; cap-gating raises `BackendCapabilityError`
  at build time (grammar/regex/choice need `caps.xgrammar`; an adapter needs `caps.lora`;
  json_schema is never gated). `backend.py` is the sole importer of `pydantic_ai.models.openai`.
- `AgentProfile.resolve()` resolves `output_type_ref` (`module:Name`) and the `DecoderSpec`:
  a `ConstrainedOutput` supplies its own; a plain Model defaults to json_schema; `decoder`
  overrides / drives a bare-string mode with no output type.
- `AgentResult` is `output` / `usage` (PydanticAI `result.usage()` as-is) / `request_body`
  (capture-gated) / `raw`; `.agent` exposes the underlying `pydantic_ai.Agent`. Tests are
  GPU-free via the ASGI mock, with a `live` marker (`SAV_LIVE=1`) reproducing the round-trip.

**Still open**
1. **Strict-mode schema rewriting.** `__strict__=True` should set `additionalProperties:false`
   + all-required and reshape `Optional` fields. Do we rewrite the schema we send, or rely on
   PydanticAI's `NativeOutput(strict=...)`? (Spike showed NativeOutput sends `strict:false`.)
   *Phase 2 stance:* pass `strict=spec.strict` to `NativeOutput` (in `DecoderSpec.apply`);
   the full schema rewrite is still deferred.
2. **`choice` typing return.** Coerce the guarded string back to the declared `Literal`
   automatically, or return `str` and let the caller cast? (Lean: coerce.)
3. **Sync surface.** Provide `run_batch_sync`/`route_and_run_sync`, or rely on `asyncio.run`?
4. **Backend cutover detail:** confirm vLLM's structured-outputs flag name + whether
   json_schema needs any per-request `guided_decoding_backend` on the pinned tag.
