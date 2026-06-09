# structured-agents-v2: Full Implementation Plan

## Context

Greenfield rewrite of `structured-agents` as a focused PydanticAI extension library.
Core thesis: policy-aware agent composition with Grail-backed constrained tool execution
and stable run/event envelopes — without reimplementing any agent runtime.

No legacy compatibility. No custom agent loop. No custom model clients.
CapabilityBundle deferred to post-v0.1.0.

---

## Pre-work: Repo bootstrap

**File: `pyproject.toml`** — replace existing skeleton with full spec:

```toml
[project]
name = "structured-agents-v2"
version = "0.1.0"
description = "PydanticAI extension library for policy-aware agents with Grail-backed constrained tool execution"
requires-python = ">=3.13"
dependencies = [
    "pydantic>=2.12",
    "pydantic-ai>=0.1",
]

[project.optional-dependencies]
grail = ["grail>=3.0"]
yaml = ["pyyaml>=6.0"]
all = ["grail>=3.0", "pyyaml>=6.0"]
dev = [
    "grail>=3.0",
    "pyyaml>=6.0",
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
]

[tool.uv.sources]
grail = { git = "https://github.com/Bullish-Design/grail" }

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py313"
line-length = 120

[tool.mypy]
python_version = "3.13"
strict = true
```

**Create directory skeleton:**
```
src/structured_agents_v2/
├── __init__.py
├── exceptions.py
├── envelopes.py
├── profiles.py
├── agent.py
├── factory.py
├── registry.py
├── config.py
├── _adapters/
│   ├── __init__.py
│   ├── pydantic_ai.py
│   ├── grail.py
│   └── yaml.py
├── policy/
│   ├── __init__.py
│   ├── models.py
│   ├── toolset.py
│   └── presets.py
└── grail/
    ├── __init__.py
    ├── config.py
    ├── manifest.py
    ├── externals.py
    ├── errors.py
    └── toolset.py

tests/
├── conftest.py
├── test_envelopes.py
├── test_policy/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_toolset.py
│   └── test_presets.py
├── test_grail/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_toolset.py
│   ├── test_manifest.py
│   ├── test_externals.py
│   └── test_errors.py
├── test_agent.py
├── test_profiles.py
├── test_factory.py
├── test_registry.py
└── fixtures/
    └── scripts/
        ├── simple_calc.pym         # Inputs: a (int), b (int). Returns a + b.
        ├── with_externals.pym      # Input: query (str). External: fetch_data. Returns result.
        ├── no_inputs.pym           # No inputs. Returns static dict.
        ├── bad_syntax.pym          # Invalid syntax — should raise ParseError at load().
        ├── slow_script.pym         # Loops — tests duration limits.
        └── over_budget.pym         # Heavy allocations — tests memory limits.

examples/
├── 01_simple_text_agent.py
├── 02_typed_output_agent.py
├── 03_grail_toolset_agent.py
├── 04_policy_filtered_agent.py
├── 05_approval_flow.py
├── 06_event_stream.py
└── 07_local_model_agent.py
```

---

## Phase 1: Foundation

**Goal:** Establish public API surface, adapter boundary pattern, envelope models, policy models.

### 1.1 `exceptions.py`

Three exception classes only:
- `StructuredAgentsError(Exception)` — base
- `ToolCallBudgetExceeded(StructuredAgentsError)` — policy budget
- `ConfigError(StructuredAgentsError)` — config load/validation

### 1.2 `_adapters/grail.py`

Wrap all Grail imports behind an availability guard:
- `_GRAIL_AVAILABLE: bool`
- `require_grail() -> None` — raises ImportError with install hint if unavailable
- `grail_available() -> bool`
- Re-export under stable names (only accessible after `require_grail()` call):
  - `GrailScript` (from `grail.script`)
  - `GrailLimits` (from `grail.limits.Limits`)
  - `grail_load` (from `grail.load`)
  - Error types: `GrailError`, `GrailParseError`, `GrailCheckError`, `GrailInputError`,
    `GrailExternalError`, `GrailExecutionError`, `GrailLimitError`, `GrailOutputError`
  - Spec types: `InputSpec`, `ExternalSpec`, `ParameterSpec`

Key Grail API facts (confirmed from source):
- `grail.load(path, *, limits=None, files=None, environ=None, grail_dir=None, ...)` → `GrailScript`
- `GrailScript.inputs: dict[str, InputSpec]` — `InputSpec.name, .type_annotation, .required, .default`
- `GrailScript.externals: dict[str, ExternalSpec]` — `ExternalSpec.name, .is_async, .parameters, .return_type, .docstring`
- `await script.run(inputs, externals, *, output_model=None, limits=None, ...)` → Any
- `grail.Limits` fields: `max_memory`, `max_duration`, `max_recursion`, `max_allocations`
- Error hierarchy: GrailError → ParseError, CheckError, InputError, ExternalError, ExecutionError, LimitError, OutputError
- `LimitError` is NOT a subclass of `ExecutionError`

### 1.3 `_adapters/pydantic_ai.py`

Import and re-export PydanticAI types under stable names. Exact import paths
to be confirmed against installed version during implementation. Expected exports:
- `PydanticAgent` (Agent)
- `PydanticAbstractToolset` (AbstractToolset)
- `ToolsetTool`, `ToolDefinition`
- `AgentRunResult`, `RunUsage`
- `RunContext`
- `ModelSettings`

### 1.4 `_adapters/yaml.py`

```python
try:
    import yaml as _yaml
    yaml_safe_load = _yaml.safe_load
    yaml_safe_dump = _yaml.safe_dump
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

def require_yaml() -> None:
    if not _YAML_AVAILABLE:
        raise ImportError("pyyaml required. Install with: pip install structured-agents-v2[yaml]")
```

### 1.5 `envelopes.py`

Five Pydantic `BaseModel` (frozen=True) classes:

**`UsageEnvelope`**: `requests`, `input_tokens`, `output_tokens`, `total_tokens` (all int=0)
- `from_pydantic_ai(cls, usage: RunUsage) -> UsageEnvelope`

**`ToolTraceEnvelope`**: `call_id`, `tool_name`, `execution_plane` (Literal), `status` (Literal),
`duration_ms`, `input_summary`, `output_summary`, `error`, `metadata`

**`DeferredCallEnvelope`**: `call_id`, `tool_name`, `arguments`, `reason`, `metadata`

**`RunEnvelope`**: `agent_name`, `run_id`, `status` (Literal), `output_text`, `output_data`,
`output_type`, `deferred_calls`, `tool_trace`, `usage`, `metadata`
- `from_pydantic_ai_result(cls, agent_name, result, tool_trace=None) -> RunEnvelope`

**`EventEnvelope`**: `run_id`, `timestamp` (datetime), `sequence`, `kind`, `message`, `payload`, `metadata`

### 1.6 `policy/models.py`

`ToolPolicy(BaseModel, frozen=True)` with all fields from spec:
- `name`, `allow_python_tools`, `allow_grail_tools`
- `allow_tool_names`, `deny_tool_names`
- `require_approval_for`, `defer_execution_for`
- `max_tool_calls`, `max_single_tool_duration_ms`
- `required_metadata`, `default_execution_mode`, `metadata`

Policy resolution helpers (module-level functions):
- `_matches_glob(pattern: str, name: str) -> bool` — uses `fnmatch`
- `should_expose(policy: ToolPolicy, tool_name: str, plane: str, tool_metadata: dict) -> bool`
- `resolve_execution_mode(policy: ToolPolicy, tool_name: str) -> Literal["direct","approval","deferred"]`

### 1.7 `policy/presets.py`

Five factory functions returning `ToolPolicy`:
- `safe_default()` — require approval for `*write*, *delete*, *send*, *remove*, *create*`
- `grail_only()` — `allow_python_tools=False`
- `approval_all()` — `default_execution_mode="approval"`
- `readonly()` — deny `*write*, *delete*, *create*, *update*, *remove*, *send*`
- `open()` — permissive

### 1.8 `policy/__init__.py`

Export: `ToolPolicy`, `PolicyToolset`, `safe_default`, `grail_only`, `approval_all`, `readonly`, `open`

### 1.9 `__init__.py`

Top-level exports (always available):
```python
from .envelopes import RunEnvelope, EventEnvelope, ToolTraceEnvelope, UsageEnvelope, DeferredCallEnvelope
from .policy import ToolPolicy, PolicyToolset
from .agent import StructuredAgent
from .profiles import StructuredAgentProfile
from .factory import StructuredAgentFactory
from .registry import StructuredAgentRegistry
from .exceptions import StructuredAgentsError, ToolCallBudgetExceeded, ConfigError
```

### Phase 1 tests

- `test_envelopes.py`: serialization roundtrip, field validation, `from_pydantic_ai_result` mapping
- `test_policy/test_models.py`: allowlist, denylist, glob matching, plane switches, metadata filtering
- `test_policy/test_presets.py`: each preset produces expected ToolPolicy config

---

## Phase 2: Grail toolset

**Goal:** Prove the core differentiator — Grail .pym scripts as PydanticAI tools.

Every module in `grail/` starts with:
```python
from structured_agents_v2._adapters.grail import require_grail
require_grail()
```

### 2.1 `grail/config.py`

**`GrailLimitsConfig(BaseModel, frozen=True)`**:
- Fields: `max_memory`, `max_duration`, `max_recursion`, `max_allocations`
- `to_grail_limits() -> GrailLimits` — maps to `grail.Limits`
- Class methods: `strict()`, `default()`, `permissive()`

Note: `grail.Limits` accepts strings for memory ("16mb") and duration ("500ms") natively.

**`GrailToolsetConfig(BaseModel, frozen=True)`**:
- `name`, `description`, `paths`, `namespace`, `limits`, `external_registry_name`,
  `include_return_schema`, `metadata`

### 2.2 `grail/manifest.py`

**`GrailInputManifest(BaseModel, frozen=True)`**: `name`, `type_annotation`, `required`, `default`

**`GrailExternalManifest(BaseModel, frozen=True)`**: `name`, `is_async`, `parameters`, `return_type`, `docstring`

**`GrailManifest(BaseModel, frozen=True)`**: `script_name`, `path`, `tool_name`, `description`,
`inputs`, `externals`, `metadata`
- `from_grail_script(cls, script: GrailScript, tool_name: str, description=None) -> GrailManifest`
  - Reads `script.inputs` (dict[str, InputSpec]) and `script.externals` (dict[str, ExternalSpec])

### 2.3 `grail/externals.py`

**`ExternalBindingEntry(BaseModel, frozen=True)`**: `name`, `description`, `is_async`, `metadata`

**`ExternalBindingRegistry`** (NOT a Pydantic model — holds callables):
- `register(name, fn, *, description=None, metadata=None) -> None`
- `resolve(names: list[str]) -> dict[str, Callable]` — raises `KeyError` with clear message
- `list_entries() -> list[ExternalBindingEntry]`

### 2.4 `grail/errors.py`

**`GrailErrorEnvelope(BaseModel, frozen=True)`**:
- `script_name`, `error_category` (Literal 8 values), `error_type`, `message`,
  `lineno`, `source_context`, `suggestion`, `limit_type`, `metadata`
- `from_grail_error(cls, script_name: str, exc: Exception) -> GrailErrorEnvelope`
  - Maps each Grail error subclass to correct category
  - `ParseError` → "parse" (has lineno, col_offset)
  - `CheckError` → "check"
  - `InputError` → "input"
  - `ExternalError` → "external"
  - `ExecutionError` → "execution" (has lineno, source_context, suggestion)
  - `LimitError` → "limit" (has limit_type)
  - `OutputError` → "output"
  - fallback → "unknown"

### 2.5 `grail/toolset.py`

**`GrailToolset(PydanticAbstractToolset[Any])`**:

Constructor:
- `__init__(config: GrailToolsetConfig, *, external_registry=None, event_callback=None)`
- Calls `_load_scripts()` eagerly at construction time

`from_paths(cls, name, paths, *, namespace=None, limits=None, external_registry=None,
event_callback=None, metadata=None) -> GrailToolset`

`_load_scripts()`:
- For each path: call `grail_load(str(path), limits=grail_limits, grail_dir=None)`
  - Note: pass `grail_dir=None` to suppress artifact generation by default
- Build `GrailManifest` from loaded script
- On error: build `GrailErrorEnvelope`, emit event, re-raise

`_make_tool_name(script_name)`: applies namespace prefix if set

`get_tools(ctx) -> dict[str, ToolsetTool]`:
- Build `ToolDefinition` from manifest inputs (JSON schema from InputSpec)
- Return `ToolsetTool` dict

`call_tool(name, tool_args, ctx, tool) -> Any`:
- Look up script and manifest by name
- Resolve externals from registry if script has declared externals
- `await script.run(inputs=tool_args, externals=externals)`
- Build `ToolTraceEnvelope` on success/failure
- On error: build `GrailErrorEnvelope`, emit event, append trace, re-raise

Grail type → JSON schema type mapping:
```python
_TYPE_MAP = {"int": "integer", "float": "number", "str": "string",
             "bool": "boolean", "list": "array", "dict": "object", "None": "null"}
```

Properties: `id`, `manifests`, `tool_traces`

### 2.6 `.pym` test fixtures

```python
# simple_calc.pym
from grail import Input
a: int = Input("a")
b: int = Input("b")
a + b

# no_inputs.pym
{"status": "ok", "value": 42}

# with_externals.pym
from grail import Input, external
query: str = Input("query")
@external
async def fetch_data(q: str) -> str: ...
result = await fetch_data(q=query)
result

# bad_syntax.pym — intentionally broken
this is not valid python !!@#

# slow_script.pym
from grail import Input
n: int = Input("n", default=1000000)
total = 0
for i in range(n):
    total += i
total
```

### Phase 2 tests

- Load valid .pym → script and manifest extracted correctly
- Load bad_syntax.pym → `ParseError` at load, `GrailErrorEnvelope` has category "parse"
- `get_tools()` → correct tool definitions with JSON schema
- `call_tool()` → executes and returns result
- Missing externals → `ExternalError` before execution
- `GrailErrorEnvelope.from_grail_error()` → all 7 error categories mapped correctly
- Namespace prefixing → tool names prefixed correctly
- `GrailManifest.from_grail_script()` → inputs and externals extracted

---

## Phase 3: PolicyToolset

**Goal:** Policy-controlled tool filtering and execution control.

### `policy/toolset.py`

**`PolicyToolset(PydanticAbstractToolset[AgentDepsT])`**:

Constructor: `__init__(inner, policy, *, namespace=None, event_callback=None)`
- `_inner`: the wrapped toolset
- `_policy`: ToolPolicy
- `_namespace`: optional string prefix
- `_call_count`: int (reset per run via `for_run()`)

`get_tools(ctx)`:
1. Get inner tools
2. For each: apply `should_expose()` helper from `policy/models.py`
3. If exposed: apply namespace prefix, resolve execution mode
4. If filtered: emit `tool.filtered` event
5. For approval/deferred mode: modify `ToolDefinition.kind` (exact mechanism TBD, verify against installed PydanticAI)

`call_tool(name, tool_args, ctx, tool)`:
1. Strip namespace from name
2. Check budget: `_call_count >= policy.max_tool_calls` → raise `ToolCallBudgetExceeded`
3. Increment counter, emit `tool.started`
4. Delegate to `_inner.call_tool(inner_name, ...)`
5. Emit `tool.succeeded` or `tool.failed`

`id`: `f"policy:{self._policy.name}"`
`tool_traces`: collect from inner if it exposes `.tool_traces`

### Phase 3 tests

Using a mock inner toolset (simple dict-based stub):
- Allowlist filtering works
- Denylist filtering works
- Glob matching (*write* etc.)
- Plane switch (allow_python_tools=False)
- Budget enforcement: N+1 call raises ToolCallBudgetExceeded
- required_metadata filtering
- Namespace prefixing: tool names prefixed, inner dispatch strips prefix
- Events emitted correctly

---

## Phase 4: Agent shim + factory

**Goal:** Developer ergonomics without owning runtime behavior.

### 4.1 `profiles.py`

**`StructuredAgentProfile(BaseModel, frozen=True)`**:
- `name`, `description`
- `model` (str), `instructions`, `model_settings`, `output_type_ref`
- `grail_toolset_names`, `policy_names`
- `metadata`

### 4.2 `agent.py`

**`StructuredAgent(BaseModel)`** with `model_config = ConfigDict(arbitrary_types_allowed=True)`:
- `profile: StructuredAgentProfile`
- `_agent: PydanticAgent = PrivateAttr()`
- `_tool_traces: list[ToolTraceEnvelope] = PrivateAttr(default_factory=list)`

Methods:
- `agent` property → raw PydanticAI Agent (escape hatch)
- `run(prompt, **kwargs) -> AgentRunResult` — direct delegation
- `run_sync(prompt, **kwargs) -> AgentRunResult` — direct delegation
- `run_enveloped(prompt, **kwargs) -> RunEnvelope` — wraps run() in try/except
- `iter_events(prompt, **kwargs) -> AsyncIterator[EventEnvelope]` — Phase 5

`_collect_tool_traces()`: walk `_agent.toolsets`, collect `.tool_traces` from GrailToolset and PolicyToolset

### 4.3 `registry.py`

**`StructuredAgentRegistry`** (plain class, NOT a Pydantic model — holds runtime objects):
- `_grail_toolset_configs: dict[str, GrailToolsetConfig]`
- `_policies: dict[str, ToolPolicy]`
- `_external_registries: dict[str, ExternalBindingRegistry]`

Registration: `register_grail_toolset(config)`, `register_policy(policy)`, `register_external_registry(name, registry)`

Lookup: `get_grail_toolset_config(name)`, `get_policy(name)`, `get_external_registry(name|None)`
- All raise `KeyError` with available names listed

`export_config() -> dict` — serializable snapshot (no callables)

### 4.4 `factory.py`

**`StructuredAgentFactory`**:
- `__init__(registry, *, default_model=None, default_policy_names=None)`

`build(profile) -> StructuredAgent`:
1. Build `GrailToolset` instances from `profile.grail_toolset_names`
2. Resolve policies, wrap each toolset in `PolicyToolset`
3. Construct `PydanticAgent(model, instructions=..., model_settings=..., toolsets=...)`
4. Return `StructuredAgent(profile=profile, agent=agent)`

### 4.5 `config.py`

```python
def load_profile(path) -> StructuredAgentProfile
def load_policy(path) -> ToolPolicy
def load_grail_toolset_config(path) -> GrailToolsetConfig
def _load_file(path) -> dict  # dispatches yaml/json by extension
```

### Phase 4 tests

- Factory builds agent from profile — correct toolsets and policies attached
- `run_enveloped()` returns complete `RunEnvelope` with correct status
- `agent.agent` returns PydanticAI Agent instance
- Profile with unknown registry names raises `KeyError`
- `load_profile()` from YAML → valid `StructuredAgentProfile`
- `export_config()` → serializable dict

---

## Phase 5: Events + examples + polish

**Goal:** Complete event streaming, all 7 examples, full test matrix.

### 5.1 `iter_events()` on `StructuredAgent`

```python
async def iter_events(self, prompt: str, **kwargs) -> AsyncIterator[EventEnvelope]:
    async with self._agent.iter(prompt, **kwargs) as agent_run:
        yield EventEnvelope(run_id=agent_run.run_id, ..., kind="run.started")
        seq = 1
        async for node in agent_run:
            yield EventEnvelope(..., kind=_node_to_event_kind(node), payload=_node_to_payload(node))
            seq += 1
        yield EventEnvelope(..., kind="run.finished", payload={"usage": ...})
```

Node-to-event mapping (PydanticAI iter nodes):
- `ModelRequestNode` → `"model.requested"`
- `CallToolsNode` → `"tool.called"`
- `End` → `"run.finished"`

### 5.2 Examples

| File | Demonstrates |
|---|---|
| `01_simple_text_agent.py` | PydanticAI Agent directly, no shim |
| `02_typed_output_agent.py` | PydanticAI structured output type |
| `03_grail_toolset_agent.py` | GrailToolset + PydanticAI Agent, no shim |
| `04_policy_filtered_agent.py` | PolicyToolset wrapping GrailToolset |
| `05_approval_flow.py` | approval/deferred policy mode |
| `06_event_stream.py` | iter_events() with EventEnvelope |
| `07_local_model_agent.py` | PydanticAI local/OpenAI-compatible model |

### 5.3 Polish checklist

- `mypy --strict` passes on all src files
- `ruff check` and `ruff format` clean
- `pytest` full matrix passes (unit + grail integration)
- `__all__` defined on every public module
- README quickstart with 3 usage patterns

---

## Adapter boundary rule (enforced throughout)

`_adapters/` is the ONLY code that imports `pydantic_ai` or `grail` directly.
All other modules import from `structured_agents_v2._adapters.*`.
If an upstream changes, exactly one adapter file changes.

---

## Key implementation notes

1. **Grail `grail_dir=None`** — pass this to suppress `.grail/` artifact generation during tool execution; users may not want filesystem side-effects from tool calls
2. **Grail `Limits` string formats** — `max_memory="16mb"`, `max_duration="2s"` work natively; `GrailLimitsConfig` can pass strings through
3. **`LimitError` vs `ExecutionError`** — these are siblings, not parent/child; handle both in error mapping
4. **PydanticAI approval/deferred** — mechanism for marking `ToolDefinition.kind` needs verification against installed version before Phase 3 implementation; defer that specific detail until Phase 3 starts
5. **`for_run()` on toolsets** — PydanticAI may call `for_run()` to reset per-run state; `PolicyToolset` must reset `_call_count` there
6. **`output_type_ref`** in `StructuredAgentProfile` — dotted string path; factory resolves via `importlib` if set

---

## Build order summary

| Phase | Key deliverable | Test tier |
|---|---|---|
| Pre-work | pyproject.toml, directory skeleton | — |
| 1 | adapters, envelopes, policy models, presets, exceptions | Unit (pydantic only) |
| 2 | GrailToolset, manifest, externals, errors, .pym fixtures | Integration (grail) |
| 3 | PolicyToolset | Integration (mock toolset) |
| 4 | profiles, agent, factory, registry, config loaders | Integration (pydantic-ai TestModel) |
| 5 | iter_events, 7 examples, type check, lint | E2E + polish |
