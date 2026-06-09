# structured-agents-v2: Consolidated Implementation Plan

## Status

- **Date:** 2026-04-27
- **Python:** >= 3.13
- **PydanticAI:** 1.87.0 (installed, import paths verified)
- **Grail:** installed from git (import paths verified)
- **Build system:** hatchling via uv
- **Shell:** all commands run inside `devenv shell`

---

## What this library is

A PydanticAI extension library providing:

1. **GrailToolset** — Grail `.pym` scripts as PydanticAI tools
2. **PolicyToolset** — declarative tool filtering, approval gating, and budget enforcement
3. **Stable envelopes** — `RunEnvelope`, `EventEnvelope`, `ToolTraceEnvelope` for app integration
4. **Agent shim** — `StructuredAgent` with enveloped runs and event streaming
5. **Factory + Registry** — profile-driven agent assembly

What it is NOT: a custom agent loop, model client, response parser, or PydanticAI replacement.

---

## Verified API surfaces

These were confirmed against the installed packages on 2026-04-27. **Use these exact import paths.**

### PydanticAI 1.87.0

```python
from pydantic_ai import Agent                                          # Agent class
from pydantic_ai.agent import AgentRunResult                           # run() return type
from pydantic_ai.tools import (
    ToolDefinition,       # dataclass: name, parameters_json_schema, description,
                          #   outer_typed_dict_key, strict, sequential, kind, metadata,
                          #   timeout, defer_loading, prefer_builtin, return_schema,
                          #   include_return_schema
    RunContext,           # context passed to tool functions
    ToolKind,             # Literal["function", "output", "external", "unapproved"]
)
from pydantic_ai.toolsets import (
    AbstractToolset,          # base class — abstract: id, get_tools, call_tool
    WrapperToolset,           # base for toolsets that wrap another toolset
    ToolsetTool,              # dataclass: toolset, tool_def, max_retries, args_validator, args_validator_func
    FilteredToolset,          # .filtered() — filter tools by predicate
    PrefixedToolset,          # .prefixed() — add prefix to tool names
    ApprovalRequiredToolset,  # .approval_required() — mark tools as needing approval
    CombinedToolset,          # combine multiple toolsets
)
from pydantic_ai.usage import Usage, RunUsage           # token usage (dataclasses)
from pydantic_ai.settings import ModelSettings          # model settings type
from pydantic_ai.capabilities import AbstractCapability # capability base class
```

Key facts:
- `AbstractToolset` abstract methods: `id` (property), `get_tools()`, `call_tool()`
- `for_run(ctx)` is a concrete method on `AbstractToolset` — returns self by default, override to reset per-run state
- `for_run_step(ctx)` also exists — resets per-step state
- Built-in combinators: `.filtered()`, `.prefixed()`, `.approval_required()`, `.renamed()`, `.with_metadata()`
- `ToolKind "unapproved"` — marks a tool as requiring approval in PydanticAI's tool execution flow
- `AgentRunResult` attrs: `.output`, `.run_id`, `.usage()`, `.all_messages()`, `.new_messages()`, `.timestamp`, `.metadata`
- `RunUsage` fields: `input_tokens`, `output_tokens`, `cache_write_tokens`, `cache_read_tokens`, `requests`, `tool_calls`, `details`
- `Usage` (per-request) fields: same as RunUsage plus `request_tokens`, `response_tokens`, `total_tokens`

### Grail (from git)

```python
from grail import (
    load,                # load(path, *, limits=None, files=None, environ=None, grail_dir=None, dataclass_registry=None) -> GrailScript
    GrailScript,         # attrs: name, path, inputs, externals, limits, source_lines, source_map, monty_code, files, environ, grail_dir, stubs, dataclass_registry
    Limits,              # Pydantic BaseModel: max_memory, max_duration, max_recursion, max_allocations, gc_interval
    InputSpec,           # dataclass: name, type_annotation, default, required, lineno, col_offset, input_name
    ExternalSpec,        # dataclass: name, is_async, parameters, return_type, docstring, lineno, col_offset
    ParameterSpec,       # dataclass: name, type_annotation, default, has_default, kind
    Input, external,     # script declarations
    DEFAULT, STRICT, PERMISSIVE,  # limit preset dicts
)
from grail.errors import (
    GrailError,          # base
    ParseError,          # __init__(message, lineno, col_offset)
    CheckError,          # __init__(message, ...)
    InputError,          # __init__(message, ...)
    ExternalError,       # __init__(message, ...)
    ExecutionError,      # __init__(message, lineno, col_offset, source_context, suggestion)
    LimitError,          # __init__(message, limit_type) — NOT a subclass of ExecutionError
    OutputError,         # __init__(message, ...)
)
```

Key facts:
- `GrailScript.inputs: dict[str, InputSpec]`
- `GrailScript.externals: dict[str, ExternalSpec]`
- `await script.run(inputs=..., externals=..., *, output_model=None, limits=None)` → Any
- `script.run_sync(inputs=..., externals=...)` → Any
- `Limits` is a Pydantic BaseModel. Presets are plain dicts: `DEFAULT`, `STRICT`, `PERMISSIVE`
- `LimitError` and `ExecutionError` are siblings (both extend `GrailError` directly)
- `grail_dir=None` suppresses `.grail/` artifact generation

---

## Package structure

```
src/structured_agents_v2/
├── __init__.py                     # Public API exports
├── py.typed                        # PEP 561 marker
├── exceptions.py                   # StructuredAgentsError, ToolCallBudgetExceeded, ConfigError
├── _adapters/
│   ├── __init__.py                 # Empty
│   ├── pydantic_ai.py             # Re-exports from pydantic_ai
│   ├── grail.py                   # Re-exports from grail (with availability guard)
│   └── yaml.py                    # Re-exports from pyyaml (with availability guard)
├── envelopes.py                    # UsageEnvelope, ToolTraceEnvelope, DeferredCallEnvelope,
│                                   # RunEnvelope, EventEnvelope
├── policy/
│   ├── __init__.py                 # Exports: ToolPolicy, PolicyToolset, presets
│   ├── models.py                  # ToolPolicy model + resolution helpers
│   ├── toolset.py                 # PolicyToolset (extends WrapperToolset)
│   └── presets.py                 # safe_default, grail_only, approval_all, readonly, open
├── grail/
│   ├── __init__.py                 # Exports: GrailToolset, config, manifest, externals, errors
│   ├── config.py                  # GrailToolsetConfig, GrailLimitsConfig
│   ├── toolset.py                 # GrailToolset (extends AbstractToolset)
│   ├── manifest.py                # GrailManifest, GrailInputManifest, GrailExternalManifest
│   ├── externals.py               # ExternalBindingRegistry, ExternalBindingEntry
│   └── errors.py                  # GrailErrorEnvelope
├── agent.py                        # StructuredAgent shim
├── profiles.py                     # StructuredAgentProfile
├── factory.py                      # StructuredAgentFactory
├── registry.py                     # StructuredAgentRegistry
└── config.py                       # YAML/JSON config loaders

tests/
├── conftest.py                     # Shared fixtures, pytest markers
├── test_exceptions.py
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
├── test_config.py
└── fixtures/
    ├── scripts/
    │   ├── simple_calc.pym
    │   ├── with_externals.pym
    │   ├── no_inputs.pym
    │   ├── bad_syntax.pym
    │   ├── slow_script.pym
    │   └── over_budget.pym
    └── configs/
        ├── sample_profile.yaml
        ├── sample_policy.yaml
        └── sample_grail_toolset.yaml

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

## Architectural rules

1. **Adapter boundary:** Only `_adapters/` imports from `pydantic_ai` or `grail` directly. All other modules import from `structured_agents_v2._adapters.*`.
2. **No custom agent loop:** PydanticAI owns runtime. We never implement tool-call lifecycles, model interaction, or output parsing.
3. **Leverage PydanticAI combinators:** Use `.filtered()`, `.prefixed()`, `.approval_required()` rather than reimplementing them. `PolicyToolset` orchestrates these, it doesn't rewrite them.
4. **Grail errors → error strings:** `GrailToolset.call_tool()` catches Grail exceptions, builds `GrailErrorEnvelope` for tracing, then returns an error string to PydanticAI (not re-raise). PydanticAI feeds the error string back to the model so it can adapt. Only `_load_scripts()` re-raises (construction-time failures are not recoverable).
5. **Pydantic models at boundaries:** Config, policy, manifests, envelopes, traces, errors — all `BaseModel` subclasses. Frozen where immutable semantics make sense.
6. **Escape hatches everywhere:** `StructuredAgent.agent` gives raw PydanticAI access. `GrailToolset` exposes scripts and manifests.

---

## Phase 0: Repository bootstrap

### 0.1 Update `pyproject.toml`

Replace the existing content with:

```toml
[project]
name = "structured-agents-v2"
version = "0.1.0"
description = "PydanticAI extension library for policy-aware agents with Grail-backed constrained tool execution"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }
authors = [
    { name = "Bullish Design", email = "BullishDesignEngineering@gmail.com" },
]

dependencies = [
    "pydantic>=2.12",
    "pydantic-ai>=1.87.0",
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
    "pytest-cov>=4.1",
    "ruff>=0.8",
    "mypy>=1.13",
]

[build-system]
requires = ["hatchling>=1.18"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/structured_agents_v2"]

[tool.hatch.build.targets.sdist]
include = ["src", "tests", "README.md", "pyproject.toml"]

[tool.uv.sources]
grail = { git = "https://github.com/Bullish-Design/grail" }

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py313"
line-length = 120
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"

[tool.mypy]
python_version = "3.13"
packages = ["src/structured_agents_v2"]
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
```

### 0.2 Create directory skeleton

Create all directories and empty `__init__.py` files. Create `src/structured_agents_v2/py.typed` (empty file).

### 0.3 Create `conftest.py`

```python
import pytest

try:
    import grail
    HAS_GRAIL = True
except ImportError:
    HAS_GRAIL = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

requires_grail = pytest.mark.skipif(not HAS_GRAIL, reason="grail not installed")
requires_yaml = pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
```

### 0.4 Verification gate

Run:
```bash
devenv shell -- python -c "import structured_agents_v2; print('OK')"
devenv shell -- pytest --co  # should collect 0 tests with no errors
```

---

## Phase 1: Foundation

**Goal:** Adapter layer, exceptions, envelope models, policy models, presets. All pure-Pydantic, no runtime integration yet.

### Step 1.1: `exceptions.py`

```python
class StructuredAgentsError(Exception):
    """Base exception for structured-agents-v2."""

class ToolCallBudgetExceeded(StructuredAgentsError):
    """Policy max_tool_calls budget exceeded."""

class ConfigError(StructuredAgentsError):
    """Config loading or validation failed."""
```

Three classes. No more.

#### Tests: `test_exceptions.py`

```
- StructuredAgentsError is an Exception subclass
- ToolCallBudgetExceeded is a StructuredAgentsError subclass
- ConfigError is a StructuredAgentsError subclass
- Each can be raised with a message and caught by its parent
```

### Step 1.2: `_adapters/grail.py`

```python
_GRAIL_AVAILABLE = False
try:
    from grail import load as grail_load
    from grail import (
        GrailScript, Limits as GrailLimits,
        InputSpec, ExternalSpec, ParameterSpec,
        DEFAULT as GRAIL_DEFAULT_LIMITS,
        STRICT as GRAIL_STRICT_LIMITS,
        PERMISSIVE as GRAIL_PERMISSIVE_LIMITS,
    )
    from grail.errors import (
        GrailError, ParseError as GrailParseError,
        CheckError as GrailCheckError, InputError as GrailInputError,
        ExternalError as GrailExternalError, ExecutionError as GrailExecutionError,
        LimitError as GrailLimitError, OutputError as GrailOutputError,
    )
    _GRAIL_AVAILABLE = True
except ImportError:
    pass

def require_grail() -> None:
    if not _GRAIL_AVAILABLE:
        raise ImportError(
            "Grail is required for this feature. "
            "Install with: pip install structured-agents-v2[grail]"
        )

def grail_available() -> bool:
    return _GRAIL_AVAILABLE
```

All Grail types get stable names prefixed with `Grail` where needed to avoid collisions. Internal modules import from here, never from `grail` directly.

#### Tests

No dedicated test file — the adapter is tested implicitly by Phase 2 Grail tests. But add a quick smoke test to `test_exceptions.py` or a dedicated `test_adapters.py`:

```
- grail_available() returns True (since grail is installed in dev)
- require_grail() does not raise
- All re-exported names are accessible
```

### Step 1.3: `_adapters/pydantic_ai.py`

```python
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.tools import ToolDefinition, RunContext, ToolKind
from pydantic_ai.toolsets import (
    AbstractToolset as PydanticAbstractToolset,
    WrapperToolset as PydanticWrapperToolset,
    ToolsetTool,
    FilteredToolset,
    PrefixedToolset,
    ApprovalRequiredToolset,
    CombinedToolset,
)
from pydantic_ai.usage import Usage, RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.capabilities import AbstractCapability as PydanticAbstractCapability

__all__ = [
    "PydanticAgent", "AgentRunResult",
    "ToolDefinition", "RunContext", "ToolKind",
    "PydanticAbstractToolset", "PydanticWrapperToolset", "ToolsetTool",
    "FilteredToolset", "PrefixedToolset", "ApprovalRequiredToolset", "CombinedToolset",
    "Usage", "RunUsage", "ModelSettings",
    "PydanticAbstractCapability",
]
```

#### Tests

Smoke test:
```
- All names in __all__ are importable from _adapters.pydantic_ai
```

### Step 1.4: `_adapters/yaml.py`

```python
_YAML_AVAILABLE = False
try:
    import yaml as _yaml
    yaml_safe_load = _yaml.safe_load
    yaml_safe_dump = _yaml.safe_dump
    _YAML_AVAILABLE = True
except ImportError:
    yaml_safe_load = None  # type: ignore[assignment]
    yaml_safe_dump = None  # type: ignore[assignment]

def require_yaml() -> None:
    if not _YAML_AVAILABLE:
        raise ImportError(
            "pyyaml is required for YAML config loading. "
            "Install with: pip install structured-agents-v2[yaml]"
        )

def yaml_available() -> bool:
    return _YAML_AVAILABLE
```

#### Tests

```
- yaml_available() returns True (pyyaml installed in dev)
- require_yaml() does not raise
```

### Step 1.5: `envelopes.py`

Five frozen Pydantic models. This is the stable API boundary for downstream applications.

**`UsageEnvelope`** (`BaseModel, frozen=True`):
- `requests: int = 0`
- `input_tokens: int = 0`
- `output_tokens: int = 0`
- `total_tokens: int = 0`
- `@classmethod from_pydantic_ai(cls, usage: RunUsage) -> UsageEnvelope`
  - Maps `usage.requests`, `usage.input_tokens`, `usage.output_tokens`
  - Computes `total_tokens = input_tokens + output_tokens`

**`ToolTraceEnvelope`** (`BaseModel, frozen=True`):
- `call_id: str | None = None`
- `tool_name: str`
- `execution_plane: Literal["python", "grail", "deferred", "builtin", "unknown"] = "unknown"`
- `status: Literal["requested", "approved", "denied", "started", "succeeded", "failed", "deferred"]`
- `duration_ms: int | None = None`
- `input_summary: dict[str, Any] | None = None`
- `output_summary: str | None = None`
- `error: str | None = None`
- `metadata: dict[str, Any] = Field(default_factory=dict)`

**`DeferredCallEnvelope`** (`BaseModel, frozen=True`):
- `call_id: str`
- `tool_name: str`
- `arguments: dict[str, Any]`
- `reason: str | None = None`
- `metadata: dict[str, Any] = Field(default_factory=dict)`

**`RunEnvelope`** (`BaseModel, frozen=True`):
- `agent_name: str`
- `run_id: str = ""`
- `status: Literal["succeeded", "failed", "deferred", "cancelled"]`
- `output_text: str | None = None`
- `output_data: Any | None = None`
- `output_type: str | None = None`
- `deferred_calls: list[DeferredCallEnvelope] = Field(default_factory=list)`
- `tool_trace: list[ToolTraceEnvelope] = Field(default_factory=list)`
- `usage: UsageEnvelope | None = None`
- `metadata: dict[str, Any] = Field(default_factory=dict)`
- `@classmethod from_pydantic_ai_result(cls, agent_name, result: AgentRunResult, tool_trace=None) -> RunEnvelope`
  - Maps `result.output` → `output_text` (if str) or `output_data` (if dict/model)
  - Maps `result.run_id` → `run_id`
  - Maps `result.usage()` → `UsageEnvelope.from_pydantic_ai()`

**`EventEnvelope`** (`BaseModel, frozen=True`):
- `run_id: str | None = None` — **None for events outside a run context** (e.g., script loading)
- `timestamp: datetime`
- `sequence: int = 0`
- `kind: str` — namespaced event kind (e.g., `"run.started"`, `"tool.filtered"`)
- `message: str | None = None`
- `payload: dict[str, Any] = Field(default_factory=dict)`
- `metadata: dict[str, Any] = Field(default_factory=dict)`

#### Tests: `test_envelopes.py`

```
UsageEnvelope:
- Default construction: all fields are 0
- from_pydantic_ai: maps RunUsage fields correctly, computes total_tokens
- Serialization roundtrip: model_dump() → model_validate() preserves all fields
- Frozen: assignment raises ValidationError

ToolTraceEnvelope:
- Construction with minimal fields (tool_name + status)
- All optional fields default correctly
- Serialization roundtrip

DeferredCallEnvelope:
- Construction with required fields
- Serialization roundtrip

RunEnvelope:
- Default construction: empty lists, None usage
- from_pydantic_ai_result: maps output, run_id, usage correctly
- from_pydantic_ai_result with str output → output_text set, output_data None
- from_pydantic_ai_result with dict output → output_data set
- Serialization roundtrip preserves nested envelopes

EventEnvelope:
- run_id=None is valid (toolset-level events)
- run_id="abc" is valid (run-level events)
- Serialization roundtrip
- Frozen
```

### Step 1.6: `policy/models.py`

**`ToolPolicy`** (`BaseModel, frozen=True`):

```python
class ToolPolicy(BaseModel, frozen=True):
    name: str
    allow_python_tools: bool = True
    allow_grail_tools: bool = True
    allow_tool_names: list[str] | None = None       # None = all allowed
    deny_tool_names: list[str] = Field(default_factory=list)
    require_approval_for: list[str] = Field(default_factory=list)
    defer_execution_for: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    max_single_tool_duration_ms: int | None = None
    required_metadata: dict[str, str] = Field(default_factory=dict)
    default_execution_mode: Literal["direct", "approval", "deferred"] = "direct"
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Convenience class methods for presets
    @classmethod
    def safe_default(cls) -> ToolPolicy: ...
    @classmethod
    def grail_only(cls) -> ToolPolicy: ...
    @classmethod
    def approval_all(cls) -> ToolPolicy: ...
    @classmethod
    def readonly(cls) -> ToolPolicy: ...
    @classmethod
    def open(cls) -> ToolPolicy: ...
```

The class methods delegate to `policy/presets.py` functions.

**Module-level resolution helpers:**

```python
def _matches_glob(pattern: str, name: str) -> bool:
    """fnmatch-style glob matching."""

def should_expose(
    policy: ToolPolicy,
    tool_name: str,
    plane: Literal["python", "grail"],
    tool_metadata: dict[str, Any],
) -> bool:
    """Apply policy filtering rules in order:
    1. Plane switch (allow_python_tools / allow_grail_tools)
    2. deny_tool_names (glob match) → filtered out
    3. allow_tool_names (glob match if set) → must match to pass
    4. required_metadata → tool must have all key-value pairs
    """

def resolve_execution_mode(
    policy: ToolPolicy,
    tool_name: str,
) -> Literal["direct", "approval", "deferred"]:
    """Determine execution mode:
    1. If tool_name matches require_approval_for (glob) → "approval"
    2. If tool_name matches defer_execution_for (glob) → "deferred"
    3. Otherwise → policy.default_execution_mode
    """
```

#### Tests: `test_policy/test_models.py`

```
ToolPolicy construction:
- Default values are correct
- Frozen: assignment raises

should_expose():
- allow_python_tools=False filters Python tools
- allow_grail_tools=False filters Grail tools
- deny_tool_names=["*write*"] filters "db_write", "file_write_csv", does not filter "read_file"
- allow_tool_names=["read_*"] filters "write_data", passes "read_file"
- Both allowlist AND denylist: denylist checked first (deny wins over allow)
- required_metadata={"trust": "sandboxed"}: tool with matching metadata passes, without fails
- Empty policy (defaults) exposes everything

resolve_execution_mode():
- require_approval_for=["*write*"] → "db_write" returns "approval"
- defer_execution_for=["*send*"] → "send_email" returns "deferred"
- No match → returns default_execution_mode
- Both match → require_approval_for takes precedence (checked first)

_matches_glob():
- "*write*" matches "db_write", "write_file", "x_write_y"
- "*write*" does not match "read_file"
- "send_*" matches "send_email", does not match "resend"
- Exact match: "my_tool" matches "my_tool"
```

### Step 1.7: `policy/presets.py`

Five factory functions:

```python
def safe_default() -> ToolPolicy:
    """Allow all tools, require approval for write/delete/send/remove/create."""
    return ToolPolicy(
        name="safe_default",
        require_approval_for=["*write*", "*delete*", "*send*", "*remove*", "*create*"],
    )

def grail_only() -> ToolPolicy:
    """Only Grail tools. Hide all Python tools."""
    return ToolPolicy(name="grail_only", allow_python_tools=False)

def approval_all() -> ToolPolicy:
    """Expose all tools but require approval for every call."""
    return ToolPolicy(name="approval_all", default_execution_mode="approval")

def readonly() -> ToolPolicy:
    """Deny tools matching write/delete/create/update/remove/send."""
    return ToolPolicy(
        name="readonly",
        deny_tool_names=["*write*", "*delete*", "*create*", "*update*", "*remove*", "*send*"],
    )

def open() -> ToolPolicy:
    """Permissive — no restrictions."""
    return ToolPolicy(name="open")
```

#### Tests: `test_policy/test_presets.py`

```
- safe_default(): name is "safe_default", require_approval_for has 5 patterns, allow_python_tools=True
- grail_only(): allow_python_tools=False, allow_grail_tools=True
- approval_all(): default_execution_mode="approval"
- readonly(): deny_tool_names has 6 patterns, all tools matching are filtered
- open(): no restrictions, all defaults
- ToolPolicy.safe_default() classmethod returns same as presets.safe_default()
```

### Step 1.8: `policy/__init__.py`

```python
from .models import ToolPolicy, should_expose, resolve_execution_mode
from .presets import safe_default, grail_only, approval_all, readonly, open

__all__ = [
    "ToolPolicy", "should_expose", "resolve_execution_mode",
    "safe_default", "grail_only", "approval_all", "readonly", "open",
]
```

Note: `PolicyToolset` is added to exports in Phase 3.

### Step 1.9: `__init__.py`

```python
from .envelopes import RunEnvelope, EventEnvelope, ToolTraceEnvelope, UsageEnvelope, DeferredCallEnvelope
from .policy import ToolPolicy
from .exceptions import StructuredAgentsError, ToolCallBudgetExceeded, ConfigError

__all__ = [
    "RunEnvelope", "EventEnvelope", "ToolTraceEnvelope", "UsageEnvelope", "DeferredCallEnvelope",
    "ToolPolicy",
    "StructuredAgentsError", "ToolCallBudgetExceeded", "ConfigError",
]
```

Remaining exports (`StructuredAgent`, `PolicyToolset`, etc.) are added as they're implemented.

### Phase 1 verification gate

```bash
devenv shell -- pytest tests/test_exceptions.py tests/test_envelopes.py tests/test_policy/ -v
devenv shell -- mypy src/structured_agents_v2/exceptions.py src/structured_agents_v2/envelopes.py src/structured_agents_v2/policy/
devenv shell -- ruff check src/structured_agents_v2/
devenv shell -- python -c "from structured_agents_v2 import RunEnvelope, ToolPolicy; print('Phase 1 OK')"
```

All must pass before proceeding to Phase 2.

---

## Phase 2: Grail toolset

**Goal:** Load `.pym` scripts and expose them as PydanticAI tools. This is the core differentiator.

Every `grail/` module starts with:
```python
from structured_agents_v2._adapters.grail import require_grail
require_grail()
```

### Step 2.1: `.pym` test fixtures

Create `tests/fixtures/scripts/`:

**`simple_calc.pym`:**
```python
from grail import Input
a: int = Input("a")
b: int = Input("b")
a + b
```

**`no_inputs.pym`:**
```python
{"status": "ok", "value": 42}
```

**`with_externals.pym`:**
```python
from grail import Input, external
query: str = Input("query")

@external
async def fetch_data(q: str) -> str: ...

result = await fetch_data(q=query)
result
```

**`bad_syntax.pym`:**
```
this is not valid python !!@#
```

**`slow_script.pym`:**
```python
from grail import Input
n: int = Input("n", default=1000000)
total = 0
for i in range(n):
    total += i
total
```

**`over_budget.pym`:**
```python
data = []
for i in range(10000000):
    data.append(list(range(100)))
len(data)
```

### Step 2.2: `grail/config.py`

**`GrailLimitsConfig`** (`BaseModel, frozen=True`):

```python
class GrailLimitsConfig(BaseModel, frozen=True):
    max_memory: int | str | None = None       # bytes or "16mb"
    max_duration: float | str | None = None   # seconds or "500ms"
    max_recursion: int | None = None
    max_allocations: int | None = None

    def to_grail_limits(self) -> GrailLimits:
        from structured_agents_v2._adapters.grail import GrailLimits
        return GrailLimits(
            max_memory=self.max_memory,
            max_duration=self.max_duration,
            max_recursion=self.max_recursion,
            max_allocations=self.max_allocations,
        )

    @classmethod
    def strict(cls) -> GrailLimitsConfig:
        return cls(max_memory="8mb", max_duration="500ms", max_recursion=120)

    @classmethod
    def default(cls) -> GrailLimitsConfig:
        return cls(max_memory="16mb", max_duration="2s", max_recursion=200)

    @classmethod
    def permissive(cls) -> GrailLimitsConfig:
        return cls(max_memory="64mb", max_duration="5s", max_recursion=400)
```

**`GrailToolsetConfig`** (`BaseModel, frozen=True`):

```python
class GrailToolsetConfig(BaseModel, frozen=True):
    name: str
    description: str | None = None
    paths: list[str]
    namespace: str | None = None
    limits: GrailLimitsConfig | None = None
    external_registry_name: str | None = None
    include_return_schema: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
```

#### Tests: `test_grail/test_config.py`

```
GrailLimitsConfig:
- strict() preset values match expectations
- default() preset values match expectations
- permissive() preset values match expectations
- to_grail_limits() returns a Grail Limits object
- to_grail_limits() passes string values through (Grail handles parsing)
- None fields are omitted in Limits construction

GrailToolsetConfig:
- Construction with required fields (name, paths)
- Frozen: assignment raises
- Serialization roundtrip
```

### Step 2.3: `grail/manifest.py`

```python
class GrailInputManifest(BaseModel, frozen=True):
    name: str
    type_annotation: str
    required: bool
    default: Any | None = None

class GrailExternalManifest(BaseModel, frozen=True):
    name: str
    is_async: bool
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    return_type: str = "Any"
    docstring: str | None = None

class GrailManifest(BaseModel, frozen=True):
    script_name: str
    path: str
    tool_name: str
    description: str | None = None
    inputs: list[GrailInputManifest] = Field(default_factory=list)
    externals: list[GrailExternalManifest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_grail_script(
        cls,
        script: GrailScript,
        tool_name: str,
        description: str | None = None,
    ) -> GrailManifest:
        """Extract manifest from a loaded GrailScript.

        If no description is provided, uses the first source line if
        it's a comment, otherwise falls back to a generic description.
        """
        inputs = [
            GrailInputManifest(
                name=spec.name,
                type_annotation=spec.type_annotation,
                required=spec.required,
                default=spec.default,
            )
            for spec in script.inputs.values()
        ]
        externals = [
            GrailExternalManifest(
                name=spec.name,
                is_async=spec.is_async,
                parameters=[
                    {"name": p.name, "type": p.type_annotation, "has_default": p.has_default}
                    for p in spec.parameters
                ],
                return_type=spec.return_type,
                docstring=spec.docstring,
            )
            for spec in script.externals.values()
        ]

        # Try to extract description from first source comment
        desc = description
        if desc is None and script.source_lines:
            first_line = script.source_lines[0].strip()
            if first_line.startswith("#"):
                desc = first_line.lstrip("# ").strip()
        if desc is None:
            desc = f"Grail script: {script.name}"

        return cls(
            script_name=script.name,
            path=str(script.path) if script.path else "",
            tool_name=tool_name,
            description=desc,
            inputs=inputs,
            externals=externals,
        )
```

#### Tests: `test_grail/test_manifest.py`

```
GrailManifest.from_grail_script():
- simple_calc.pym → 2 inputs (a: int required, b: int required), 0 externals
- with_externals.pym → 1 input (query: str), 1 external (fetch_data, is_async=True)
- no_inputs.pym → 0 inputs, 0 externals
- Description extraction: script with "# Calculate sum" as first line → description="Calculate sum"
- Description fallback: script without comment → description="Grail script: <name>"
- Explicit description overrides extraction
- Serialization roundtrip
```

### Step 2.4: `grail/externals.py`

```python
class ExternalBindingEntry(BaseModel, frozen=True):
    name: str
    description: str | None = None
    is_async: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

class ExternalBindingRegistry:
    """Runtime registry for host functions callable by Grail scripts."""

    def __init__(self) -> None:
        self._bindings: dict[str, tuple[Callable[..., Any], ExternalBindingEntry]] = {}

    def register(self, name: str, fn: Callable[..., Any], *, description: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        import asyncio
        is_async = asyncio.iscoroutinefunction(fn)
        entry = ExternalBindingEntry(name=name, description=description, is_async=is_async, metadata=metadata or {})
        self._bindings[name] = (fn, entry)

    def resolve(self, names: list[str]) -> dict[str, Callable[..., Any]]:
        result: dict[str, Callable[..., Any]] = {}
        missing: list[str] = []
        for name in names:
            if name in self._bindings:
                result[name] = self._bindings[name][0]
            else:
                missing.append(name)
        if missing:
            available = sorted(self._bindings.keys())
            raise KeyError(f"Missing external bindings: {missing}. Available: {available}")
        return result

    def list_entries(self) -> list[ExternalBindingEntry]:
        return [entry for _, entry in self._bindings.values()]
```

#### Tests: `test_grail/test_externals.py`

```
ExternalBindingRegistry:
- register() + resolve() → returns callable
- resolve() with missing name → KeyError with available names listed
- register() detects async functions correctly (is_async=True)
- register() detects sync functions correctly (is_async=False)
- list_entries() returns all registered entries
- Empty registry → resolve([]) returns {}
- resolve(["missing"]) on empty registry → KeyError listing empty available
```

### Step 2.5: `grail/errors.py`

```python
class GrailErrorEnvelope(BaseModel, frozen=True):
    script_name: str
    error_category: Literal["parse", "check", "input", "external", "execution", "limit", "output", "unknown"]
    error_type: str
    message: str
    lineno: int | None = None
    source_context: str | None = None
    suggestion: str | None = None
    limit_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_grail_error(cls, script_name: str, exc: Exception) -> GrailErrorEnvelope:
        from structured_agents_v2._adapters.grail import (
            GrailParseError, GrailCheckError, GrailInputError,
            GrailExternalError, GrailExecutionError, GrailLimitError,
            GrailOutputError,
        )
        # Order matters: LimitError before ExecutionError (they're siblings, not parent-child)
        if isinstance(exc, GrailParseError):
            return cls(script_name=script_name, error_category="parse", error_type=type(exc).__name__,
                       message=str(exc), lineno=getattr(exc, "lineno", None))
        elif isinstance(exc, GrailCheckError):
            return cls(script_name=script_name, error_category="check", error_type=type(exc).__name__,
                       message=str(exc))
        elif isinstance(exc, GrailInputError):
            return cls(script_name=script_name, error_category="input", error_type=type(exc).__name__,
                       message=str(exc))
        elif isinstance(exc, GrailExternalError):
            return cls(script_name=script_name, error_category="external", error_type=type(exc).__name__,
                       message=str(exc))
        elif isinstance(exc, GrailLimitError):
            return cls(script_name=script_name, error_category="limit", error_type=type(exc).__name__,
                       message=str(exc), limit_type=getattr(exc, "limit_type", None))
        elif isinstance(exc, GrailExecutionError):
            return cls(script_name=script_name, error_category="execution", error_type=type(exc).__name__,
                       message=str(exc), lineno=getattr(exc, "lineno", None),
                       source_context=getattr(exc, "source_context", None),
                       suggestion=getattr(exc, "suggestion", None))
        elif isinstance(exc, GrailOutputError):
            return cls(script_name=script_name, error_category="output", error_type=type(exc).__name__,
                       message=str(exc))
        else:
            return cls(script_name=script_name, error_category="unknown", error_type=type(exc).__name__,
                       message=str(exc))
```

#### Tests: `test_grail/test_errors.py`

```
GrailErrorEnvelope.from_grail_error():
- ParseError → category="parse", lineno populated
- CheckError → category="check"
- InputError → category="input"
- ExternalError → category="external"
- ExecutionError → category="execution", lineno/source_context/suggestion populated
- LimitError → category="limit", limit_type populated
- OutputError → category="output"
- Generic Exception → category="unknown"
- error_type is the class name string
- Serialization roundtrip
```

### Step 2.6: `grail/toolset.py`

This is the core differentiator. `GrailToolset` implements `AbstractToolset`.

```python
class GrailToolset(PydanticAbstractToolset[Any]):

    def __init__(self, config: GrailToolsetConfig, *, external_registry=None, event_callback=None):
        require_grail()
        self._config = config
        self._external_registry = external_registry
        self._event_callback = event_callback
        self._scripts: dict[str, GrailScript] = {}
        self._manifests: dict[str, GrailManifest] = {}
        self._tool_traces: list[ToolTraceEnvelope] = []
        self._load_scripts()

    @classmethod
    def from_paths(cls, name, paths, *, namespace=None, limits=None, external_registry=None,
                   event_callback=None, metadata=None) -> GrailToolset:
        config = GrailToolsetConfig(name=name, paths=[str(p) for p in paths],
                                    namespace=namespace, limits=limits, metadata=metadata or {})
        return cls(config, external_registry=external_registry, event_callback=event_callback)

    # --- AbstractToolset interface ---

    @property
    def id(self) -> str | None:
        return f"grail:{self._config.name}"

    async def get_tools(self, ctx) -> dict[str, ToolsetTool]:
        # Build ToolDefinition from each manifest's inputs
        # JSON schema from InputSpec type annotations via _TYPE_MAP
        ...

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        # 1. Look up script and manifest
        # 2. Resolve externals from registry
        # 3. Execute: await script.run(inputs=tool_args, externals=externals)
        # 4. On success: build ToolTraceEnvelope, return result
        # 5. On error: build GrailErrorEnvelope + ToolTraceEnvelope,
        #    return error string (DO NOT re-raise)
        ...

    def for_run(self, ctx) -> GrailToolset:
        # Return copy with fresh tool_traces
        ...

    # --- Inspection ---

    @property
    def manifests(self) -> dict[str, GrailManifest]: ...
    @property
    def tool_traces(self) -> list[ToolTraceEnvelope]: ...
```

**Critical design decision:** `call_tool()` returns an error string on failure, it does NOT re-raise. This lets PydanticAI feed the error back to the model so it can adapt. Only `_load_scripts()` re-raises because construction-time failures are not recoverable tool results.

**Type mapping:**
```python
_TYPE_MAP = {
    "int": "integer", "float": "number", "str": "string",
    "bool": "boolean", "list": "array", "dict": "object", "None": "null",
}
```

**Path glob expansion:** Support `*` and `?` in paths for folder-based toolsets:
```python
def _resolve_paths(self) -> list[Path]:
    resolved = []
    for path_str in self._config.paths:
        if "*" in path_str or "?" in path_str:
            resolved.extend(sorted(Path(".").glob(path_str)))
        else:
            resolved.append(Path(path_str))
    return resolved
```

#### Tests: `test_grail/test_toolset.py`

```
GrailToolset construction:
- from_paths() with simple_calc.pym → loads successfully
- from_paths() with bad_syntax.pym → raises at construction (ParseError)
- from_paths() with no_inputs.pym → loads, 0 tool inputs
- id property returns "grail:<name>"

get_tools():
- simple_calc → one tool with parameters_json_schema containing a/b as integer
- Tool description extracted from manifest

call_tool():
- simple_calc with {"a": 2, "b": 3} → returns 5
- no_inputs → returns {"status": "ok", "value": 42}
- with_externals with registered external → resolves and executes
- with_externals without registered external → returns error string (not exception)
- Tool trace recorded on success (status="succeeded", execution_plane="grail")
- Tool trace recorded on failure (status="failed")

for_run():
- Returns new instance with empty tool_traces
- Original instance traces preserved

Namespace:
- namespace="math" → tool name is "math__simple_calc"

Path globbing:
- paths=["tests/fixtures/scripts/*.pym"] resolves to all .pym files
  (NOTE: this will fail for bad_syntax.pym — test with a subdirectory of only valid scripts)

Event callback:
- Event emitted on script load ("grail.script.loaded")
- Event emitted on call success ("grail.script.succeeded")
- Event emitted on call failure ("grail.script.failed")
```

### Step 2.7: `grail/__init__.py`

```python
from structured_agents_v2._adapters.grail import require_grail
require_grail()

from .toolset import GrailToolset
from .config import GrailToolsetConfig, GrailLimitsConfig
from .manifest import GrailManifest, GrailInputManifest, GrailExternalManifest
from .externals import ExternalBindingRegistry, ExternalBindingEntry
from .errors import GrailErrorEnvelope

__all__ = [
    "GrailToolset", "GrailToolsetConfig", "GrailLimitsConfig",
    "GrailManifest", "GrailInputManifest", "GrailExternalManifest",
    "ExternalBindingRegistry", "ExternalBindingEntry",
    "GrailErrorEnvelope",
]
```

### Phase 2 verification gate

```bash
devenv shell -- pytest tests/test_grail/ -v
devenv shell -- mypy src/structured_agents_v2/grail/
devenv shell -- ruff check src/structured_agents_v2/grail/
devenv shell -- python -c "from structured_agents_v2.grail import GrailToolset; print('Phase 2 OK')"
```

---

## Phase 3: PolicyToolset

**Goal:** Declarative tool filtering, approval gating, and budget enforcement.

### Key insight: Leverage PydanticAI combinators

PydanticAI 1.87.0 already provides `.filtered()`, `.prefixed()`, `.approval_required()`. PolicyToolset should **orchestrate** these, not reimplement them. This keeps us aligned with "wrap, don't replace."

### Step 3.1: `policy/toolset.py`

```python
class PolicyToolset(PydanticWrapperToolset[Any]):
    """Wraps an inner toolset and applies ToolPolicy filtering and execution control.

    Uses PydanticAI's native .filtered() and .approval_required() combinators
    internally rather than reimplementing filtering logic.
    """

    def __init__(
        self,
        inner: PydanticAbstractToolset[Any],
        policy: ToolPolicy,
        *,
        event_callback: Callable[[EventEnvelope], None] | None = None,
    ) -> None:
        # Build the combinator chain
        wrapped = inner

        # Apply policy filtering using PydanticAI's FilteredToolset
        wrapped = wrapped.filtered(self._make_filter_func(policy))

        # Apply namespace if the inner toolset doesn't already have one
        # (namespacing is the inner toolset's concern, not policy's;
        #  only apply if policy needs to add identification)

        # Apply approval gating for tools matching require_approval_for
        if policy.require_approval_for or policy.default_execution_mode == "approval":
            wrapped = wrapped.approval_required(self._make_approval_func(policy))

        super().__init__(wrapped)
        self._policy = policy
        self._event_callback = event_callback
        self._call_count = 0

    @property
    def id(self) -> str | None:
        return f"policy:{self._policy.name}"

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        # Budget enforcement
        if self._policy.max_tool_calls is not None and self._call_count >= self._policy.max_tool_calls:
            raise ToolCallBudgetExceeded(
                f"Policy '{self._policy.name}' budget of {self._policy.max_tool_calls} calls exceeded"
            )
        self._call_count += 1
        self._emit("tool.started", tool_name=name)
        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
            self._emit("tool.succeeded", tool_name=name)
            return result
        except Exception as exc:
            self._emit("tool.failed", tool_name=name, error=str(exc))
            raise

    def for_run(self, ctx) -> PolicyToolset:
        """Reset per-run state (call counter)."""
        new = PolicyToolset.__new__(PolicyToolset)
        new._wrapped = self._wrapped.for_run(ctx)  # WrapperToolset stores inner as _wrapped
        new._policy = self._policy
        new._event_callback = self._event_callback
        new._call_count = 0
        return new

    @staticmethod
    def _make_filter_func(policy: ToolPolicy):
        """Build a filter function for PydanticAI's FilteredToolset."""
        def filter_fn(ctx: RunContext, tool_def: ToolDefinition) -> bool:
            # Determine plane from toolset id or metadata
            plane = "grail" if (tool_def.metadata or {}).get("plane") == "grail" else "python"
            tool_meta = dict(tool_def.metadata) if tool_def.metadata else {}
            return should_expose(policy, tool_def.name, plane, tool_meta)
        return filter_fn

    @staticmethod
    def _make_approval_func(policy: ToolPolicy):
        """Build an approval-required function for PydanticAI's ApprovalRequiredToolset."""
        def approval_fn(ctx: RunContext, tool_def: ToolDefinition, tool_args: dict) -> bool:
            mode = resolve_execution_mode(policy, tool_def.name)
            return mode == "approval"
        return approval_fn

    def _emit(self, kind: str, **payload: Any) -> None:
        if self._event_callback:
            from datetime import datetime, timezone
            self._event_callback(EventEnvelope(
                timestamp=datetime.now(timezone.utc), kind=kind, payload=payload,
            ))

    @property
    def tool_traces(self) -> list[ToolTraceEnvelope]:
        inner = self._wrapped
        if hasattr(inner, "tool_traces"):
            return inner.tool_traces
        return []
```

#### Tests: `test_policy/test_toolset.py`

Use a simple mock toolset for isolation:

```python
class MockToolset(AbstractToolset[Any]):
    """Fake toolset exposing named tools for testing."""
    def __init__(self, tools: dict[str, Any], *, plane: str = "python"):
        self._tools = tools
        self._plane = plane

    @property
    def id(self) -> str | None:
        return "mock"

    async def get_tools(self, ctx):
        result = {}
        for name, return_val in self._tools.items():
            tool_def = ToolDefinition(
                name=name, description=f"Mock: {name}",
                parameters_json_schema={"type": "object", "properties": {}},
                metadata={"plane": self._plane},
            )
            result[name] = ToolsetTool(toolset=self, tool_def=tool_def, max_retries=0)
            return result

    async def call_tool(self, name, tool_args, ctx, tool):
        return self._tools[name]
```

```
Filtering:
- deny_tool_names=["*write*"]: "db_write" filtered, "read_file" exposed
- allow_tool_names=["read_*"]: "read_file" exposed, "write_data" filtered
- allow_python_tools=False with plane="python": all tools filtered
- allow_grail_tools=True with plane="grail": tools exposed
- required_metadata={"trust":"sandboxed"}: tool without it filtered

Approval:
- require_approval_for=["*send*"]: "send_email" tool gets approval gating
- default_execution_mode="approval": all tools require approval

Budget:
- max_tool_calls=3: calls 1-3 succeed, call 4 raises ToolCallBudgetExceeded
- max_tool_calls=0: first call raises immediately
- max_tool_calls=None: unlimited calls work

for_run():
- Resets call_count to 0
- Preserves policy

Events:
- tool.started emitted on each call
- tool.succeeded emitted on success
- tool.failed emitted on error
```

### Step 3.2: Update `policy/__init__.py`

Add `PolicyToolset` to exports:
```python
from .toolset import PolicyToolset
# Add to __all__
```

### Step 3.3: Update `__init__.py`

Add `PolicyToolset` to top-level exports.

### Phase 3 verification gate

```bash
devenv shell -- pytest tests/test_policy/ -v
devenv shell -- mypy src/structured_agents_v2/policy/
devenv shell -- ruff check src/structured_agents_v2/
```

---

## Phase 4: Agent shim, factory, registry, config

**Goal:** Developer ergonomics — profile-driven agent assembly with enveloped results.

### Step 4.1: `profiles.py`

```python
class StructuredAgentProfile(BaseModel, frozen=True):
    name: str
    description: str | None = None
    model: str
    instructions: str | list[str] | None = None
    model_settings: dict[str, Any] | None = None
    output_type_ref: str | None = None       # dotted path for importlib resolution
    grail_toolset_names: list[str] = Field(default_factory=list)
    policy_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

#### Tests: `test_profiles.py`

```
- Construction with required fields (name, model)
- Frozen
- Serialization roundtrip
- instructions as str or list[str] both work
```

### Step 4.2: `registry.py`

```python
class StructuredAgentRegistry:
    """Central registry for named configs used in agent assembly."""

    def __init__(self) -> None:
        self._grail_toolset_configs: dict[str, GrailToolsetConfig] = {}
        self._policies: dict[str, ToolPolicy] = {}
        self._external_registries: dict[str, ExternalBindingRegistry] = {}

    def register_grail_toolset(self, config: GrailToolsetConfig) -> None:
        self._grail_toolset_configs[config.name] = config

    def register_policy(self, policy: ToolPolicy) -> None:
        self._policies[policy.name] = policy

    def register_external_registry(self, name: str, registry: ExternalBindingRegistry) -> None:
        self._external_registries[name] = registry

    def get_grail_toolset_config(self, name: str) -> GrailToolsetConfig:
        if name not in self._grail_toolset_configs:
            raise KeyError(f"Unknown Grail toolset: '{name}'. Registered: {sorted(self._grail_toolset_configs)}")
        return self._grail_toolset_configs[name]

    def get_policy(self, name: str) -> ToolPolicy:
        if name not in self._policies:
            raise KeyError(f"Unknown policy: '{name}'. Registered: {sorted(self._policies)}")
        return self._policies[name]

    def get_external_registry(self, name: str | None) -> ExternalBindingRegistry | None:
        if name is None:
            return None
        if name not in self._external_registries:
            raise KeyError(f"Unknown external registry: '{name}'. Registered: {sorted(self._external_registries)}")
        return self._external_registries[name]

    def export_config(self) -> dict[str, Any]:
        return {
            "grail_toolsets": {k: v.model_dump() for k, v in self._grail_toolset_configs.items()},
            "policies": {k: v.model_dump() for k, v in self._policies.items()},
        }
```

#### Tests: `test_registry.py`

```
- register + get roundtrip for each type
- get with unknown name → KeyError with available names listed
- get_external_registry(None) → returns None
- export_config() → serializable dict with correct structure
- Register multiple, list all in export
```

### Step 4.3: `factory.py`

```python
class StructuredAgentFactory:
    def __init__(self, registry: StructuredAgentRegistry, *, default_model: str | None = None,
                 default_policy_names: list[str] | None = None) -> None:
        self._registry = registry
        self._default_model = default_model
        self._default_policy_names = default_policy_names or []

    def build(self, profile: StructuredAgentProfile) -> StructuredAgent:
        from structured_agents_v2._adapters.pydantic_ai import PydanticAgent

        # 1. Build Grail toolsets
        toolsets: list[PydanticAbstractToolset] = []
        for name in profile.grail_toolset_names:
            grail_config = self._registry.get_grail_toolset_config(name)
            ext_reg = self._registry.get_external_registry(grail_config.external_registry_name)
            toolset = GrailToolset(grail_config, external_registry=ext_reg)
            toolsets.append(toolset)

        # 2. Apply policies (each policy wraps ALL toolsets)
        policy_names = profile.policy_names or self._default_policy_names
        if policy_names:
            policies = [self._registry.get_policy(name) for name in policy_names]
            for policy in policies:
                toolsets = [PolicyToolset(ts, policy) for ts in toolsets]

        # 3. Resolve output type if specified
        output_type = None
        if profile.output_type_ref:
            output_type = self._resolve_type(profile.output_type_ref)

        # 4. Build PydanticAI Agent
        model = profile.model or self._default_model
        if model is None:
            raise ConfigError("No model specified in profile or factory default_model")

        agent = PydanticAgent(
            model,
            instructions=profile.instructions,
            model_settings=profile.model_settings,
            output_type=output_type,
            toolsets=toolsets or None,
        )

        return StructuredAgent(profile=profile, agent=agent)

    @staticmethod
    def _resolve_type(dotted_path: str) -> type:
        import importlib
        module_path, _, attr_name = dotted_path.rpartition(".")
        if not module_path:
            raise ConfigError(f"output_type_ref must be a dotted path: {dotted_path}")
        module = importlib.import_module(module_path)
        try:
            return getattr(module, attr_name)
        except AttributeError:
            raise ConfigError(f"Cannot find '{attr_name}' in module '{module_path}'")
```

#### Tests: `test_factory.py`

```
- build() with minimal profile (no toolsets, no policies) → returns StructuredAgent
- build() with grail_toolset_names → GrailToolset attached
- build() with policy_names → PolicyToolset wrapping inner toolsets
- build() with unknown grail toolset name → KeyError
- build() with unknown policy name → KeyError
- build() with no model and no default_model → ConfigError
- default_model used when profile.model is empty
- default_policy_names used when profile.policy_names is empty
- agent.agent returns a PydanticAI Agent instance
- agent.profile matches the input profile
```

### Step 4.4: `agent.py`

```python
class StructuredAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: StructuredAgentProfile
    _agent: PydanticAgent = PrivateAttr()

    def __init__(self, *, profile: StructuredAgentProfile, agent: PydanticAgent) -> None:
        super().__init__(profile=profile)
        self._agent = agent

    @property
    def agent(self) -> PydanticAgent:
        """Raw PydanticAI Agent escape hatch."""
        return self._agent

    async def run(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        return await self._agent.run(prompt, **kwargs)

    def run_sync(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        return self._agent.run_sync(prompt, **kwargs)

    async def run_enveloped(self, prompt: str, **kwargs: Any) -> RunEnvelope:
        try:
            result = await self._agent.run(prompt, **kwargs)
            traces = self._collect_tool_traces()
            return RunEnvelope.from_pydantic_ai_result(
                agent_name=self.profile.name, result=result, tool_trace=traces,
            )
        except Exception as exc:
            return RunEnvelope(
                agent_name=self.profile.name, run_id="",
                status="failed",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )

    def _collect_tool_traces(self) -> list[ToolTraceEnvelope]:
        """Walk agent toolsets and collect traces from GrailToolset/PolicyToolset."""
        traces: list[ToolTraceEnvelope] = []
        def visitor(ts: PydanticAbstractToolset) -> None:
            if hasattr(ts, "tool_traces"):
                traces.extend(ts.tool_traces)
        self._agent.toolsets.apply(visitor)  # PydanticAI's apply() walks the tree
        return traces
```

Note: `iter_events()` is implemented in Phase 5.

#### Tests: `test_agent.py`

```
- agent property returns PydanticAI Agent
- profile property returns StructuredAgentProfile
- run_enveloped() with failing agent → RunEnvelope with status="failed" and error in metadata
- StructuredAgent is not fully serializable (model_dump only has profile)
```

(Full integration tests with real model calls are in Phase 5 examples.)

### Step 4.5: `config.py`

```python
def load_profile(path: str | Path) -> StructuredAgentProfile:
    data = _load_file(path)
    return StructuredAgentProfile.model_validate(data)

def load_policy(path: str | Path) -> ToolPolicy:
    data = _load_file(path)
    return ToolPolicy.model_validate(data)

def load_grail_toolset_config(path: str | Path) -> GrailToolsetConfig:
    data = _load_file(path)
    return GrailToolsetConfig.model_validate(data)

def _load_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    content = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        from structured_agents_v2._adapters.yaml import require_yaml, yaml_safe_load
        require_yaml()
        return yaml_safe_load(content)
    else:
        import json
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
```

Create YAML test fixtures:

**`tests/fixtures/configs/sample_profile.yaml`:**
```yaml
name: test_agent
model: test
instructions: Be helpful.
grail_toolset_names:
  - finance
policy_names:
  - safe_default
```

**`tests/fixtures/configs/sample_policy.yaml`:**
```yaml
name: test_policy
deny_tool_names:
  - "*delete*"
max_tool_calls: 5
```

#### Tests: `test_config.py`

```
- load_profile() from YAML → valid StructuredAgentProfile
- load_policy() from YAML → valid ToolPolicy
- load_grail_toolset_config() from JSON → valid GrailToolsetConfig
- _load_file() with nonexistent path → ConfigError
- _load_file() with malformed JSON → ConfigError
- _load_file() dispatches by extension (.yaml vs .json)
```

### Step 4.6: Update `__init__.py`

Add all remaining exports:

```python
from .agent import StructuredAgent
from .profiles import StructuredAgentProfile
from .factory import StructuredAgentFactory
from .registry import StructuredAgentRegistry
```

### Phase 4 verification gate

```bash
devenv shell -- pytest tests/ -v
devenv shell -- mypy src/structured_agents_v2/
devenv shell -- ruff check src/structured_agents_v2/
devenv shell -- python -c "
from structured_agents_v2 import (
    StructuredAgent, StructuredAgentProfile, StructuredAgentFactory,
    StructuredAgentRegistry, RunEnvelope, ToolPolicy, PolicyToolset,
)
print('Phase 4 OK — all public exports work')
"
```

---

## Phase 5: Events, examples, polish

**Goal:** Event streaming, 7 runnable examples, full test matrix, type-check clean.

### Step 5.1: `iter_events()` on `StructuredAgent`

```python
async def iter_events(self, prompt: str, **kwargs: Any) -> AsyncIterator[EventEnvelope]:
    from datetime import datetime, timezone

    async with self._agent.iter(prompt, **kwargs) as agent_run:
        yield EventEnvelope(
            run_id=agent_run.run_id,
            timestamp=datetime.now(timezone.utc),
            sequence=0,
            kind="run.started",
            message=f"Agent '{self.profile.name}' started",
        )
        seq = 1
        async for node in agent_run:
            yield EventEnvelope(
                run_id=agent_run.run_id,
                timestamp=datetime.now(timezone.utc),
                sequence=seq,
                kind=_node_to_event_kind(node),
                payload=_node_to_payload(node),
            )
            seq += 1

        yield EventEnvelope(
            run_id=agent_run.run_id,
            timestamp=datetime.now(timezone.utc),
            sequence=seq,
            kind="run.finished",
        )
```

Node-to-event mapping helpers (adjust to actual PydanticAI node types during implementation):
```python
def _node_to_event_kind(node: Any) -> str:
    type_name = type(node).__name__
    mapping = {
        "ModelRequestNode": "model.requested",
        "CallToolsNode": "tool.called",
        "End": "run.finishing",
    }
    return mapping.get(type_name, f"node.{type_name.lower()}")
```

### Step 5.2: Examples

Each example should be a standalone runnable script with clear docstring.

| File | What it demonstrates | Dependencies |
|---|---|---|
| `01_simple_text_agent.py` | Plain PydanticAI Agent, no v2 shim | pydantic-ai |
| `02_typed_output_agent.py` | PydanticAI structured output type | pydantic-ai |
| `03_grail_toolset_agent.py` | GrailToolset + PydanticAI Agent, no shim | pydantic-ai, grail |
| `04_policy_filtered_agent.py` | PolicyToolset wrapping GrailToolset | pydantic-ai, grail |
| `05_approval_flow.py` | Approval/deferred policy mode | pydantic-ai, grail |
| `06_event_stream.py` | iter_events() with EventEnvelope | pydantic-ai, grail |
| `07_local_model_agent.py` | PydanticAI with local/OpenAI-compatible model | pydantic-ai |

### Step 5.3: Polish checklist

```
- [ ] `__all__` defined on every public module
- [ ] `mypy --strict` passes on all src/ files
- [ ] `ruff check` clean
- [ ] `ruff format --check` clean
- [ ] `pytest` full matrix passes
- [ ] All public classes and functions have docstrings
- [ ] README.md updated with:
      - One-line description
      - Install instructions (pip install, extras)
      - Three usage patterns (PydanticAI-native, shim, config-driven)
      - Link to examples/
```

### Phase 5 verification gate (final)

```bash
devenv shell -- pytest tests/ -v --cov=structured_agents_v2 --cov-report=term-missing
devenv shell -- mypy src/structured_agents_v2/ --strict
devenv shell -- ruff check src/structured_agents_v2/
devenv shell -- ruff format --check src/structured_agents_v2/
```

---

## Summary: what the intern should know

1. **Always run commands inside `devenv shell`** — this project uses devenv for environment management.
2. **Follow the phases in order.** Each phase has a verification gate. Do not proceed until it passes.
3. **Import discipline is the most important rule.** If you're writing code outside `_adapters/` and you type `from pydantic_ai` or `from grail`, stop. Import from `_adapters` instead.
4. **Grail errors in `call_tool()` → return error string.** Do NOT re-raise. PydanticAI handles error results gracefully.
5. **Leverage PydanticAI combinators.** Don't reimplement `.filtered()`, `.prefixed()`, or `.approval_required()`. PolicyToolset orchestrates them.
6. **Test each step before moving on.** The test descriptions in this plan are specifications — implement them first (TDD) or immediately after the source code.
7. **Check `mypy` and `ruff` at each phase gate.** Fix issues immediately, don't accumulate tech debt.
8. **When in doubt, read the installed PydanticAI source.** The verified API surface at the top of this document is your source of truth for import paths and type signatures.
