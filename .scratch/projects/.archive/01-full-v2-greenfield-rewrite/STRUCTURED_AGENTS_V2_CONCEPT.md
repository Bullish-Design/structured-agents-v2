# Structured Agents v2 Concept

## Document status

- **Status:** Final greenfield concept
- **Target repo:** `structured-agents-v2`
- **Compatibility posture:** No legacy compatibility bridge
- **Primary runtime dependency:** PydanticAI
- **Primary constrained execution dependency:** Grail, powered by Monty
- **Core thesis:** `structured-agents-v2` should be a small, opinionated PydanticAI extension library focused on Grail toolsets, policy-controlled tool exposure, stable result/event envelopes, and developer-friendly Pydantic wrappers.

---

## 1. Executive summary

`structured-agents-v2` should be a full greenfield rewrite. It should not migrate or preserve the old `AgentKernel`, `ModelAdapter`, custom response parser, OpenAI-compatible client factory, or custom agent loop.

The new library should instead treat **PydanticAI as the agent runtime** and provide a focused layer around it:

1. **PydanticAI bundling shims** for developer convenience.
2. **First-class Grail `.pym` toolsets** for constrained scripted tools.
3. **Policy wrappers** for tool exposure, approval, deferral, namespacing, limits, and metadata filtering.
4. **Stable run and event envelopes** for apps that need a durable integration contract.
5. **Capability bundles** that make it easy to compose PydanticAI agents without repeatedly wiring low-level details.

The guiding rule is:

> Wrap PydanticAI where it improves ergonomics or stability; never duplicate its runtime responsibilities.

This makes v2 smaller, more maintainable, and more strategically differentiated than the current repo.

---

## 2. Product statement

`structured-agents-v2` is:

> **A PydanticAI extension library for building typed, policy-aware agents with first-class Grail-backed constrained tool execution and stable application-facing run/event envelopes.**

The library should feel like a high-leverage composition layer, not a framework that competes with PydanticAI.

---

## 3. What v2 is

- A greenfield repo.
- A typed Pydantic model layer around common agent configuration.
- A Grail toolset implementation for PydanticAI.
- A policy and capability layer for safe tool composition.
- A stable envelope layer for downstream applications.
- A small set of ergonomic shim objects that bundle PydanticAI functionality.
- A set of examples and tests showing how to build custom agents on top of PydanticAI using these primitives.

---

## 4. What v2 is not

- Not a legacy-compatible rewrite.
- Not a custom agent runtime.
- Not a replacement for `pydantic_ai.Agent`.
- Not a custom model-provider abstraction.
- Not a custom OpenAI-compatible client layer.
- Not a custom tool-call loop.
- Not a response parser framework.
- Not a default grammar-constrained decoding framework.
- Not a general-purpose sandbox independent of Grail/Monty.
- Not a multi-agent orchestration platform by default.

---

## 5. Source findings and design implications

### 5.1 Current `structured-agents`

The current repo centers on:

- `AgentKernel`
- OpenAI-compatible model clients
- response parsers
- `Tool`, `ToolSchema`, `ToolCall`, and `ToolResult`
- optional grammar/structured-output constraints
- observer events
- demos and tests around the above

This architecture is coherent for a small custom agent loop, but in v2 it is the wrong abstraction boundary. PydanticAI already owns the generic runtime loop, model handling, tools, toolsets, structured output, capabilities, and run APIs.

The current repo should be treated as a **source of product lessons**, not code to port.

### 5.2 Grail

Grail provides a Python-facing layer over Monty:

- `.pym` files
- `Input()` declarations
- `@external` declarations
- `GrailScript`
- `load()`, `run()`, and `run_sync()`
- resource `Limits`
- validation, generated artifacts, and structured error mapping

This is highly aligned with `structured-agents-v2`: Grail should become the constrained scripted tool plane.

### 5.3 Monty

Monty is the lower-level secure interpreter. It is intentionally limited and designed for agent-written or agent-used Python-like code. It blocks direct host filesystem, environment, and network access unless explicitly mediated through controlled host bindings. It supports resource limits, type checking, snapshots, async/sync host calls, and a limited standard library.

v2 should not integrate Monty directly except through Grail unless there is a narrow advanced feature that Grail does not expose.

### 5.4 PydanticAI

PydanticAI already provides the agent substrate v2 needs:

- `Agent`
- typed dependencies and outputs
- toolsets
- capabilities
- declarative agent specs
- structured outputs
- deferred tools and approvals
- streaming/iteration APIs
- model-provider abstraction
- local/OpenAI-compatible model integration paths

v2 should integrate with these as native extension points.

---

## 6. Core architecture

```text
Application code
   │
   ▼
structured_agents_v2
   ├── Pydantic wrapper/shim objects
   ├── Agent bundles and profiles
   ├── Capability bundles
   ├── Policy toolset wrappers
   ├── Grail toolsets
   ├── Stable run envelopes
   └── Stable event envelopes
   │
   ├──────────────► PydanticAI
   │                 ├── Agent
   │                 ├── AgentSpec
   │                 ├── Capabilities
   │                 ├── Toolsets
   │                 ├── Deferred tools / approvals
   │                 ├── Structured outputs
   │                 └── Run / stream / iter APIs
   │
   └──────────────► Grail
                     ├── .pym loading
                     ├── Input declarations
                     ├── @external declarations
                     ├── Limits
                     └── Monty-backed constrained execution
```

### Architectural split

| Layer | Owns |
|---|---|
| PydanticAI | Agent runtime, model interaction, tool execution lifecycle, output typing, core streaming APIs |
| Grail | `.pym` parsing/loading, script validation, constrained execution, input/external declarations, Monty limits |
| `structured-agents-v2` | Developer ergonomics, Grail toolset integration, policy composition, stable envelopes, capability bundles |

---

## 7. Main design principles

### 7.1 Be PydanticAI-native

Use PydanticAI concepts directly: `Agent`, `AgentSpec`, `AbstractToolset`, capabilities, deferred tools, structured outputs, and model settings.

### 7.2 Wrap, do not replace

A shim object may hold or construct a PydanticAI agent. It should not implement an independent loop.

### 7.3 Make Grail a first-class tool plane

Grail should not be hidden behind a generic adapter. It should have explicit public types, docs, examples, tests, and policy integration.

### 7.4 Prefer Pydantic models at boundaries

Configuration, policy, manifests, envelopes, tool traces, events, and error surfaces should be Pydantic models.

### 7.5 Keep raw PydanticAI escape hatches

Every wrapper should expose a way to access the underlying PydanticAI object or pass through advanced options.

### 7.6 Keep the core small

If a feature is already cleanly handled by PydanticAI, do not wrap it unless the wrapper materially improves usability, stability, or policy enforcement.

---

## 8. Primary public abstractions

## 8.1 `StructuredAgentProfile`

A serializable profile describing how to assemble a PydanticAI agent with v2 extras.

```python
from typing import Any
from pydantic import BaseModel, Field

class StructuredAgentProfile(BaseModel):
    name: str
    description: str | None = None

    # Prefer PydanticAI-compatible spec fields instead of inventing a parallel spec.
    model: str
    instructions: str | list[str] | None = None
    model_settings: dict[str, Any] | None = None

    # v2 composition layer.
    capability_names: list[str] = Field(default_factory=list)
    grail_toolset_names: list[str] = Field(default_factory=list)
    policy_names: list[str] = Field(default_factory=list)

    # Optional PydanticAI-compatible structured output config.
    output_schema: dict[str, Any] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
```

This should not duplicate the entire PydanticAI `AgentSpec` model. It should be a convenient project-level profile that can compile into a PydanticAI agent.

## 8.2 `StructuredAgent`

A convenience shim around `pydantic_ai.Agent`.

It should provide:

- `.agent` or `.to_pydantic_agent()` to access the underlying PydanticAI agent.
- `.run_raw(...)` to delegate directly to PydanticAI.
- `.run_enveloped(...)` to return a `RunEnvelope`.
- `.iter_event_envelopes(...)` to expose normalized events.
- `.profile` for serializable configuration metadata.

Implementation note: because `pydantic_ai.Agent` is a runtime object, `StructuredAgent` can be a Pydantic model with a private attribute.

```python
from typing import Any
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic_ai import Agent

class StructuredAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: StructuredAgentProfile
    _agent: Agent[Any, Any] = PrivateAttr()

    def __init__(self, *, profile: StructuredAgentProfile, agent: Agent[Any, Any]):
        super().__init__(profile=profile)
        self._agent = agent

    @property
    def agent(self) -> Agent[Any, Any]:
        return self._agent

    async def run_raw(self, prompt: str, **kwargs: Any) -> Any:
        return await self._agent.run(prompt, **kwargs)

    async def run_enveloped(self, prompt: str, **kwargs: Any) -> "RunEnvelope":
        result = await self._agent.run(prompt, **kwargs)
        return RunEnvelope.from_pydantic_ai_result(self.profile.name, result)
```

This object is a shim, not a runtime.

## 8.3 `StructuredAgentFactory`

A builder that composes:

- PydanticAI model config
- capabilities
- Grail toolsets
- policies
- output config
- app defaults

```python
class StructuredAgentFactory(BaseModel):
    default_model: str | None = None
    default_policy_names: list[str] = Field(default_factory=list)
    registry: "StructuredAgentRegistry"

    def build(self, profile: StructuredAgentProfile) -> StructuredAgent:
        ...
```

The factory should compile profiles into PydanticAI agents. It should not create a separate runtime path.

## 8.4 `CapabilityBundle`

A Pydantic config object that bundles PydanticAI capabilities and v2 metadata.

```python
class CapabilityBundle(BaseModel):
    name: str
    description: str | None = None
    instructions: list[str] = Field(default_factory=list)
    pydantic_capability_refs: list[str] = Field(default_factory=list)
    grail_toolset_names: list[str] = Field(default_factory=list)
    policy_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

At runtime, this compiles into one or more PydanticAI capabilities/toolsets.

## 8.5 `GrailToolset`

The most important v2-specific abstraction.

`GrailToolset` should implement PydanticAI’s toolset interface and expose Grail scripts as model-callable tools.

```python
class GrailToolsetConfig(BaseModel):
    name: str
    paths: list[str]
    namespace: str | None = None
    limits: "GrailLimitsConfig" | None = None
    external_registry_name: str | None = None
    include_return_schema: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Expected behavior:

1. Load `.pym` files with `grail.load()`.
2. Extract script metadata: name, declared inputs, declared externals, docstrings, result shape where possible.
3. Build PydanticAI-compatible tool definitions.
4. Validate inputs before execution.
5. Resolve declared externals from `ExternalBindingRegistry`.
6. Execute via `script.run(...)` with Grail limits.
7. Normalize results and errors into stable trace payloads.
8. Support namespacing to avoid tool-name collisions.

## 8.6 `ExternalBindingRegistry`

A host-side registry for functions that Grail scripts may call.

```python
class ExternalBinding(BaseModel):
    name: str
    description: str | None = None
    callable_ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

The runtime registry will map names to actual callables. The serializable config should reference callables by symbolic name, not serialize them.

```python
class ExternalBindingRegistry:
    def register(self, name: str, fn: object, *, metadata: dict[str, Any] | None = None) -> None:
        ...

    def resolve(self, names: list[str]) -> dict[str, object]:
        ...
```

This matters because Grail’s security model depends on explicit host exposure.

## 8.7 `ToolPolicy`

A policy model for tool visibility, approval, deferral, budgets, and metadata filtering.

```python
from typing import Literal

class ToolPolicy(BaseModel):
    name: str

    allow_python_tools: bool = True
    allow_grail_tools: bool = True

    allow_tool_names: list[str] | None = None
    deny_tool_names: list[str] = Field(default_factory=list)

    require_approval_for: list[str] = Field(default_factory=list)
    defer_execution_for: list[str] = Field(default_factory=list)

    max_tool_calls: int | None = None
    max_tool_duration_ms: int | None = None

    required_metadata: dict[str, str] = Field(default_factory=dict)
    default_execution_mode: Literal["direct", "approval", "deferred"] = "direct"

    metadata: dict[str, Any] = Field(default_factory=dict)
```

`ToolPolicy` should compile into PydanticAI toolset wrappers/capabilities instead of inventing a separate policy runtime.

## 8.8 `PolicyToolset`

A wrapper around any PydanticAI toolset.

Responsibilities:

- filter tools before model exposure
- rename or namespace tools
- require approval for selected tools
- defer selected tools
- enforce call budgets where feasible
- attach normalized tracing metadata

## 8.9 `RunEnvelope`

A stable app-facing result object.

```python
from typing import Literal

class UsageEnvelope(BaseModel):
    requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

class DeferredCallEnvelope(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class ToolTraceEnvelope(BaseModel):
    call_id: str | None = None
    tool_name: str
    execution_plane: Literal["python", "grail", "deferred", "builtin", "unknown"]
    status: Literal["requested", "approved", "denied", "started", "succeeded", "failed", "deferred"]
    duration_ms: int | None = None
    input_preview: dict[str, Any] | None = None
    output_preview: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class RunEnvelope(BaseModel):
    agent_name: str
    status: Literal["succeeded", "failed", "deferred", "cancelled"]

    output_text: str | None = None
    output_data: dict[str, Any] | None = None
    output_type: str | None = None

    deferred_calls: list[DeferredCallEnvelope] = Field(default_factory=list)
    tool_trace: list[ToolTraceEnvelope] = Field(default_factory=list)
    usage: UsageEnvelope | None = None

    raw_result_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

The envelope is for application stability. Advanced users should still be able to access raw PydanticAI results.

## 8.10 `EventEnvelope`

A stable event shape for UIs, CLIs, logs, approvals, and telemetry.

```python
class EventEnvelope(BaseModel):
    run_id: str
    sequence: int
    kind: str
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Recommended event kinds:

- `run.started`
- `run.finished`
- `run.failed`
- `model.requested`
- `model.completed`
- `tool.visible`
- `tool.filtered`
- `tool.requested`
- `tool.approval_required`
- `tool.approved`
- `tool.denied`
- `tool.deferred`
- `tool.started`
- `tool.finished`
- `tool.failed`
- `grail.script.loaded`
- `grail.script.failed_validation`
- `grail.script.started`
- `grail.script.finished`
- `output.validated`

---

## 9. Grail integration design

## 9.1 Grail is local-first

The v2 integration should assume local library usage:

```python
script = grail.load("analysis.pym")
result = await script.run(inputs={...}, externals={...}, limits=...)
```

No HTTP service should be required for the default architecture.

## 9.2 Script styles

Support these three patterns:

### Style A: one `.pym` script = one tool

Best for small reusable operations.

Example:

```text
risk_score.pym -> finance__risk_score
```

### Style B: folder/package = one toolset

Best for domain bundles.

Example:

```text
tools/finance/*.pym -> GrailToolset(name="finance")
```

### Style C: internal Grail helper scripts

Best for constrained transformations, validations, scoring, or deterministic helper work not necessarily exposed directly to the model.

## 9.3 Manifest extraction

Every loaded script should produce a manifest.

```python
class GrailManifest(BaseModel):
    script_name: str
    path: str
    tool_name: str
    description: str | None = None
    declared_inputs: list[str] = Field(default_factory=list)
    declared_externals: list[str] = Field(default_factory=list)
    result_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Uses:

- docs generation
- tool registration
- policy matching
- testing
- runtime diagnostics
- approval UIs

## 9.4 Limits

Expose a v2 Pydantic config that maps to Grail limits.

```python
class GrailLimitsConfig(BaseModel):
    memory_bytes: int | None = None
    duration_ms: int | None = None
    recursion_limit: int | None = None
    allocation_limit: int | None = None
```

Provide presets:

- `strict`
- `default`
- `permissive`
- `testing`

## 9.5 Output validation

Support optional output validation for Grail tools:

```python
class GrailOutputConfig(BaseModel):
    output_model_ref: str | None = None
    require_json_serializable: bool = True
```

When an output model is provided, pass it into Grail execution or validate after execution using Pydantic.

## 9.6 Error handling

Normalize Grail errors into stable envelopes without hiding the original exception type.

```python
class GrailErrorEnvelope(BaseModel):
    script_name: str
    error_type: str
    message: str
    lineno: int | None = None
    source_context: str | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

## 10. Policy model

The policy layer should be one of the main reasons v2 exists.

### 10.1 Policy dimensions

| Dimension | Description |
|---|---|
| Exposure | Which tools are visible to the model |
| Approval | Which calls require human or host approval |
| Deferral | Which calls return deferred requests instead of executing immediately |
| Execution plane | Python, Grail, builtin, external/deferred |
| Metadata | Trust level, domain, environment, capability namespace |
| Budget | Call counts, duration, resource limits |
| Namespacing | Prefixes and collision avoidance |
| Audit | Trace output and event emission |

### 10.2 Policy resolution flow

```text
Registered toolsets
   ▼
Capability bundle composition
   ▼
Grail manifests + Python tool metadata
   ▼
Policy resolution
   ▼
PydanticAI wrapper/prepared toolsets
   ▼
Agent run
   ▼
RunEnvelope + EventEnvelope output
```

### 10.3 Safe defaults

Ship these policy presets:

| Policy | Behavior |
|---|---|
| `safe_default` | Allow read-like tools, require approval for writes, prefer Grail for constrained scripts |
| `grail_only` | Hide native Python tools except explicitly allowed builtins |
| `approval_all` | Expose tools but require approval for every call |
| `readonly` | Deny tools marked as mutating/write/destructive |
| `testing_open` | Permissive for tests and local demos |

---

## 11. PydanticAI shim objects

The user-facing wrappers should make common agent construction easy without obscuring PydanticAI.

## 11.1 Why shims exist

PydanticAI is flexible. v2 should provide opinionated bundles for common structured-agent patterns:

- profile-based construction
- policy-aware toolset composition
- Grail tool loading
- stable envelopes
- event normalization
- default model/settings wiring
- reusable app registries

## 11.2 Recommended shim objects

### `StructuredAgentRegistry`

Central registry for names used in profiles.

```python
class StructuredAgentRegistry(BaseModel):
    capability_bundles: dict[str, CapabilityBundle] = Field(default_factory=dict)
    grail_toolsets: dict[str, GrailToolsetConfig] = Field(default_factory=dict)
    policies: dict[str, ToolPolicy] = Field(default_factory=dict)
    external_registries: dict[str, str] = Field(default_factory=dict)
```

Runtime callables should live in runtime registries/private attrs, not serialized config fields.

### `StructuredAgentFactory`

Compiles a profile into a `StructuredAgent`.

### `StructuredAgent`

Holds a PydanticAI agent and provides envelope methods.

### `AgentBundle`

A higher-level object that packages:

- profile
- built agent
- policy metadata
- loaded Grail manifests
- docs/debug metadata

```python
class AgentBundle(BaseModel):
    profile: StructuredAgentProfile
    manifests: list[GrailManifest] = Field(default_factory=list)
    policy_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _agent: object = PrivateAttr(default=None)
```

`AgentBundle` can be useful in CLIs and app startup logs because it can show exactly what was assembled.

---

## 12. Structured output strategy

Default to PydanticAI structured outputs.

Do not reimplement the current repo’s custom `ConstraintPipeline` as a central abstraction. It can be reconsidered later as a narrow provider-specific extension if PydanticAI cannot express a required grammar-constrained decoding feature.

Recommended posture:

| Need | v2 answer |
|---|---|
| Typed final response | PydanticAI output type |
| JSON schema final response | PydanticAI output schema/spec support |
| Tool return typing | PydanticAI tool return schemas where useful |
| Provider-specific constrained decoding | Optional future extension, not MVP |
| Validation of Grail script result | Pydantic model validation in GrailToolset |

---

## 13. Eventing and observability

The current repo’s event model is worth reimplementing conceptually, but not as a custom kernel observer system.

v2 should:

- normalize meaningful PydanticAI run/tool/model events where available
- add Grail-specific events during script load/check/run
- expose `EventEnvelope` streams
- optionally integrate with Logfire through PydanticAI rather than building custom telemetry
- support event callbacks for CLIs, TUIs, web streams, and approval flows

---

## 14. What to reimplement from the current repo

No compatibility bridge should be built. Still, several current repo concepts are worth reimplementing in v2 form.

## 14.1 Reimplement conceptually

### Event concepts

Current repo has kernel/model/tool/turn events. Reimplement the useful idea as stable `EventEnvelope` events over PydanticAI and Grail activity.

Carry forward:

- model request/response events
- tool request/result events
- run start/end events
- duration metadata
- usage metadata where available

Do not carry forward:

- observer system coupled to `AgentKernel`
- turn-based assumptions that only make sense in the custom loop

### Result concepts

Current repo has `RunResult`, `StepResult`, `ToolResult`, and token usage models. Reimplement as:

- `RunEnvelope`
- `ToolTraceEnvelope`
- `UsageEnvelope`
- `DeferredCallEnvelope`

Do not carry forward the old message-history-centered result as the primary API. Raw PydanticAI history/results can remain accessible through escape hatches.

### Tool trace ergonomics

Current repo’s tool call/result model is useful for debugging. Reimplement it as structured tracing around PydanticAI toolsets, especially `GrailToolset` and `PolicyToolset`.

### Demo quality

Recreate demos from scratch using v2 APIs:

- simple text agent
- typed output agent
- native Python toolset agent
- Grail `.pym` toolset agent
- approval/deferred tool flow
- local OpenAI-compatible model example through PydanticAI
- event stream example

### Tests around behavior

Reimplement tests around observable behavior, not old classes:

- tools are exposed or hidden by policy
- Grail scripts validate inputs and externals
- Grail errors normalize correctly
- approval/deferred policies produce deferred calls
- envelopes serialize cleanly
- run events are ordered and stable
- PydanticAI raw access remains available

### Provider lessons

The current repo has practical knowledge around OpenAI-compatible/vLLM usage and optional grammar constraints. Reimplement only as docs/examples or optional future capability, not as a custom client stack.

## 14.2 Do not reimplement

| Current repo section | v2 decision |
|---|---|
| `AgentKernel` | Do not reimplement; use PydanticAI `Agent` |
| `ModelAdapter` | Do not reimplement; use PydanticAI model/provider layer |
| `QwenResponseParser` / parser framework | Do not reimplement; use PydanticAI tool calling/output handling |
| `build_client` / OpenAI-compatible client factory | Do not reimplement; configure PydanticAI models directly |
| custom tool loop | Do not reimplement; use PydanticAI toolsets |
| old `Tool` protocol as primary API | Replace with PydanticAI toolsets and GrailToolset |
| default `ConstraintPipeline` | Do not make central; optional future provider-specific extension only |
| compatibility imports | Do not build |
| legacy `AgentKernel` facade | Do not build |

---

## 15. Proposed package structure

```text
structured_agents_v2/
├── __init__.py
├── profiles.py              # StructuredAgentProfile and profile loading
├── agent.py                 # StructuredAgent shim
├── factory.py               # StructuredAgentFactory
├── registry.py              # StructuredAgentRegistry
├── envelopes.py             # RunEnvelope, EventEnvelope, trace models
├── normalize.py             # PydanticAI result/event normalization
├── capabilities.py          # CapabilityBundle helpers
├── policy/
│   ├── __init__.py
│   ├── models.py            # ToolPolicy and selectors
│   ├── toolset.py           # PolicyToolset wrappers
│   ├── approval.py          # approval/deferred helpers
│   └── presets.py           # safe_default, grail_only, readonly, etc.
├── grail/
│   ├── __init__.py
│   ├── toolset.py           # GrailToolset
│   ├── capability.py        # GrailCapability
│   ├── config.py            # GrailToolsetConfig, GrailLimitsConfig
│   ├── manifests.py         # GrailManifest extraction
│   ├── externals.py         # ExternalBindingRegistry
│   ├── runtime.py           # execution wrapper helpers
│   └── errors.py            # error normalization
├── examples/
│   ├── simple_text_agent.py
│   ├── typed_output_agent.py
│   ├── grail_toolset_agent.py
│   ├── approval_flow.py
│   └── local_model_agent.py
└── tests/
    ├── test_profiles.py
    ├── test_factory.py
    ├── test_grail_toolset.py
    ├── test_policy_toolset.py
    ├── test_envelopes.py
    └── test_events.py
```

No `compat/` package should exist in v2.

---

## 16. Public API sketch

## 16.1 Basic PydanticAI-native usage

```python
from pydantic_ai import Agent
from structured_agents_v2.grail import GrailToolset
from structured_agents_v2.policy import PolicyToolset, ToolPolicy

finance_tools = GrailToolset.from_paths(
    name="finance",
    paths=["tools/expense_analysis.pym", "tools/risk_score.pym"],
)

safe_tools = PolicyToolset(
    finance_tools,
    policy=ToolPolicy.safe_default(),
)

agent = Agent(
    "openai:gpt-5.2",
    instructions="Use finance tools carefully and explain assumptions.",
    toolsets=[safe_tools],
)

result = await agent.run("Analyze this account for budget risk.")
```

This should always remain possible. v2 should not force users through the shim layer.

## 16.2 StructuredAgent shim usage

```python
from structured_agents_v2 import (
    StructuredAgentFactory,
    StructuredAgentProfile,
    StructuredAgentRegistry,
)

profile = StructuredAgentProfile(
    name="finance_assistant",
    model="openai:gpt-5.2",
    instructions="Use available tools carefully and return concise analysis.",
    grail_toolset_names=["finance"],
    policy_names=["safe_default"],
)

factory = StructuredAgentFactory(registry=registry)
agent = factory.build(profile)

result = await agent.run_enveloped("Find budget risks for user 123.")
print(result.status)
print(result.output_text)
```

## 16.3 Raw access escape hatch

```python
pydantic_agent = agent.agent
raw_result = await agent.run_raw("Use the raw PydanticAI result path.")
```

## 16.4 Event envelope usage

```python
async for event in agent.iter_event_envelopes("Run the analysis."):
    print(event.kind, event.message)
```

---

## 17. Configuration examples

## 17.1 Profile YAML

```yaml
name: finance_assistant
model: openai:gpt-5.2
instructions:
  - Use available tools carefully.
  - Ask for approval before write operations.
grail_toolset_names:
  - finance
policy_names:
  - safe_default
metadata:
  domain: finance
```

## 17.2 Grail toolset YAML

```yaml
name: finance
namespace: finance
paths:
  - tools/finance/expense_analysis.pym
  - tools/finance/risk_score.pym
limits:
  duration_ms: 1000
  memory_bytes: 32000000
metadata:
  trust: sandboxed
  domain: finance
```

## 17.3 Policy YAML

```yaml
name: safe_default
allow_python_tools: true
allow_grail_tools: true
require_approval_for:
  - "*write*"
  - "*delete*"
  - "*send*"
deny_tool_names: []
max_tool_calls: 8
required_metadata:
  trust: sandboxed
```

---

## 18. Build plan

## Phase 1: Foundation

Deliver:

- project skeleton
- Pydantic config models
- `RunEnvelope`
- `EventEnvelope`
- basic normalizers
- registry model
- initial docs and examples

Goal: establish the public surface.

## Phase 2: Grail toolset MVP

Deliver:

- `GrailToolset.from_paths(...)`
- script loading and manifest extraction
- external binding registry
- limits config
- result/error normalization
- unit tests with sample `.pym` tools

Goal: prove the core differentiator.

## Phase 3: Policy wrappers

Deliver:

- `ToolPolicy`
- `PolicyToolset`
- exposure filtering
- namespacing
- approval-required mapping
- deferred execution mapping
- policy presets

Goal: make safe composition easy.

## Phase 4: Shim objects

Deliver:

- `StructuredAgentProfile`
- `StructuredAgentFactory`
- `StructuredAgent`
- profile loading from YAML/JSON
- raw PydanticAI escape hatches
- enveloped run helpers

Goal: improve developer ergonomics without owning runtime behavior.

## Phase 5: Events and app integrations

Deliver:

- event normalization
- Grail load/run events
- tool policy events
- CLI/TUI-friendly event streaming example
- approval UI example

Goal: make v2 useful for real applications.

## Phase 6: Polish and release

Deliver:

- complete examples
- API docs
- migration narrative, not compatibility bridge
- test matrix
- package publishing workflow

Goal: ship a clean v2 release.

---

## 19. MVP acceptance criteria

The MVP is successful when:

1. A developer can use `GrailToolset` directly inside a PydanticAI `Agent`.
2. A developer can build an agent from a `StructuredAgentProfile` using a factory.
3. Tool policies can hide, approve, defer, or allow tools without custom agent-loop code.
4. A Grail `.pym` tool can declare inputs and externals, execute with limits, and return a normalized trace.
5. Missing Grail externals fail clearly before unsafe execution.
6. `run_enveloped()` returns a stable JSON-serializable `RunEnvelope`.
7. Event streams use stable `EventEnvelope` objects.
8. Raw PydanticAI access remains available.
9. No legacy compatibility package exists.
10. No custom LLM client or agent loop exists.

---

## 20. Key risks

### 20.1 Duplicating PydanticAI

The largest risk is slowly rebuilding PydanticAI. Avoid this by making PydanticAI-native usage a first-class documented path.

### 20.2 Grail/Monty maturity

Monty is powerful but intentionally limited and still evolving. Version pinning, clear errors, and focused tests are required.

### 20.3 Confusing Grail with PydanticAI Code Mode

PydanticAI Code Mode lets the model write code that calls tools. Grail `.pym` toolsets are curated developer-authored scripts exposed as tools. These should be documented as different patterns.

### 20.4 Too many wrappers

Only add wrappers that reduce real developer friction. Every wrapper should have an escape hatch to the underlying PydanticAI object.

### 20.5 Policy ambiguity

Tool policy should be deterministic and inspectable. Provide debug methods that show which tools were exposed, hidden, approval-gated, or deferred.

---

## 21. Design decisions

| Decision | Choice |
|---|---|
| Repo strategy | Full new v2 repo |
| Legacy compatibility | None |
| Runtime | PydanticAI |
| Constrained execution | Grail via Monty |
| Model providers | PydanticAI model/provider layer |
| Agent declaration | PydanticAI-compatible profile/spec layer |
| Native tools | PydanticAI toolsets |
| Grail tools | First-class `GrailToolset` |
| Policies | PydanticAI wrapper/prepared toolsets and capabilities |
| Results | Stable `RunEnvelope` plus raw result escape hatch |
| Events | Stable `EventEnvelope` plus PydanticAI/Grail normalization |
| Structured output | PydanticAI first |
| Grammar-constrained decoding | Optional future extension only |

---

## 22. Final recommendation

Create `structured-agents-v2` as a new PydanticAI-native library with no compatibility bridge.

The highest-value first release should focus on:

1. `GrailToolset`
2. `ToolPolicy` and `PolicyToolset`
3. `RunEnvelope` and `EventEnvelope`
4. `StructuredAgentProfile`
5. `StructuredAgentFactory`
6. `StructuredAgent` shim with raw PydanticAI access

This gives the project a clear identity:

> **Structured Agents v2 makes PydanticAI easier to package into application-ready, policy-aware agents, with Grail as the preferred constrained tool execution plane.**

That is meaningfully different from both raw PydanticAI and the old `structured-agents` repo, while staying small enough to maintain well.

---

## 23. Reference inputs

Primary project references:

- `Bullish-Design/structured-agents`
- `Bullish-Design/grail`
- `pydantic/monty`
- `STRUCTURED_AGENTS_REFACTOR_CONCEPT.md`
- `STRUCTURED_AGENTS_CONCEPT_DEEP_RESEARCH_v2_1.md`

Primary external design references:

- PydanticAI Agent docs: https://pydantic.dev/docs/ai/core-concepts/agent/
- PydanticAI Agent Specs: https://pydantic.dev/docs/ai/core-concepts/agent-spec/
- PydanticAI Capabilities: https://pydantic.dev/docs/ai/core-concepts/capabilities
- PydanticAI Toolsets: https://pydantic.dev/docs/ai/api/pydantic-ai/toolsets/
- PydanticAI Deferred Tools: https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools
- PydanticAI Code Mode: https://pydantic.dev/docs/ai/harness/code-mode/
- Grail repo: https://github.com/Bullish-Design/grail
- Monty repo: https://github.com/pydantic/monty
