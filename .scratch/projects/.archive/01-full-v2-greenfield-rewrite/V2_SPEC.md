# Structured Agents v2 — Implementation Specification

## Document status

- **Status:** Implementation spec (derived from STRUCTURED_AGENTS_V2_CONCEPT.md)
- **Target repo:** `structured-agents-v2` (new greenfield repo)
- **Python:** 3.12+
- **Runtime dependency:** PydanticAI (agent loop, model providers, tool execution)
- **Optional dependency:** Grail 3.x (constrained `.pym` tool execution via Monty)
- **Core thesis:** A small, opinionated PydanticAI extension library providing Grail-backed constrained toolsets, policy-controlled tool exposure, stable result/event envelopes, and ergonomic agent composition — with hard API boundaries isolating every upstream dependency.

---

## 1. Design tenets

1. **PydanticAI is the runtime.** We never implement an agent loop, model client, tool-call lifecycle, or output parser. PydanticAI owns all of that.
2. **Grail is the default constrained tool plane, but optional.** The core package works without Grail installed. Grail support ships as an extras group (`pip install structured-agents-v2[grail]`).
3. **Hard API boundaries everywhere.** Every upstream integration (PydanticAI toolset interface, PydanticAI capability interface, Grail script API, Grail error types) is wrapped behind a thin adapter. If an upstream changes, exactly one adapter file changes; no public API breaks.
4. **Wrap where it adds value; pass through where it doesn't.** Configuration, policy, envelopes, and Grail integration merit wrapping. Model selection, system prompts, and output types are PydanticAI's job — expose them, don't re-abstract them.
5. **Escape hatches always available.** Every wrapper exposes the underlying PydanticAI or Grail object for advanced users.
6. **Pydantic models at every boundary.** Config, policy, manifests, envelopes, traces, errors — all Pydantic `BaseModel` subclasses. Serializable, validatable, versionable.

---

## 2. Dependency and import boundary strategy

### 2.1 Dependency tiers

| Tier | Package | Install | Import boundary |
|---|---|---|---|
| Required | `pydantic >= 2.12` | always | Direct import everywhere |
| Required | `pydantic-ai >= 0.1` | always | Imported only through `_adapters/pydantic_ai.py` |
| Optional | `grail >= 3.0` | `pip install .[grail]` | Imported only through `_adapters/grail.py` |
| Optional | `pyyaml >= 6.0` | `pip install .[yaml]` | Imported only in `config.py` loader functions |

### 2.2 Adapter layer contract

Every upstream runtime dependency has exactly one adapter file in `src/structured_agents_v2/_adapters/`. Each adapter:

1. Imports the upstream library.
2. Re-exports only the types and functions we use.
3. Provides thin wrapper types where needed to stabilize the interface.
4. Raises `ImportError` with a clear install hint if the optional dependency is missing.

**The rest of the codebase imports from `_adapters`, never from `pydantic_ai` or `grail` directly.** This is the single most important architectural rule. It means:

- If PydanticAI renames `AbstractToolset` to `BaseToolset`, we update one file.
- If Grail changes `GrailScript.run()` signature, we update one file.
- Our public API and all internal modules remain stable.

### 2.3 Adapter file inventory

```
_adapters/
├── __init__.py
├── pydantic_ai.py      # Agent, AbstractToolset, AbstractCapability, RunContext,
│                        # ToolsetTool, ToolDefinition, AgentRunResult, RunUsage,
│                        # ModelRequestNode, CallToolsNode, End,
│                        # DeferredToolRequests, HandleDeferredToolCalls,
│                        # ApprovalRequired, CallDeferred, ModelSettings
├── grail.py             # grail.load, GrailScript, Limits, InputSpec, ExternalSpec,
│                        # ParameterSpec, GrailError, ParseError, CheckError,
│                        # InputError, ExternalError, ExecutionError, LimitError,
│                        # OutputError
└── yaml.py              # yaml.safe_load, yaml.safe_dump (for config loading)
```

Each adapter exports under our own names where stabilization is needed:

```python
# _adapters/pydantic_ai.py
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.toolsets import AbstractToolset as PydanticAbstractToolset
from pydantic_ai.toolsets import ToolsetTool, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability as PydanticAbstractCapability
from pydantic_ai.run import AgentRunResult, RunUsage
from pydantic_ai.tools import RunContext
from pydantic_ai.settings import ModelSettings
# ... etc

# Re-export for internal use
__all__ = [
    "PydanticAgent",
    "PydanticAbstractToolset",
    "PydanticAbstractCapability",
    "ToolsetTool",
    "ToolDefinition",
    "AgentRunResult",
    "RunUsage",
    "RunContext",
    "ModelSettings",
    # ...
]
```

```python
# _adapters/grail.py
_GRAIL_AVAILABLE = False
try:
    import grail as _grail
    from grail import (
        GrailScript as _GrailScript,
        Limits as _GrailLimits,
    )
    from grail.errors import (
        GrailError as _GrailError,
        ParseError as _ParseError,
        CheckError as _CheckError,
        InputError as _InputError,
        ExternalError as _ExternalError,
        ExecutionError as _ExecutionError,
        LimitError as _LimitError,
        OutputError as _OutputError,
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

# Re-export everything under stable names, only when available
# Internal code calls require_grail() before using any of these
```

---

## 3. Package structure

```
structured-agents-v2/
├── pyproject.toml
├── README.md
├── src/
│   └── structured_agents_v2/
│       ├── __init__.py                  # Public API re-exports
│       ├── _adapters/                   # Upstream dependency isolation
│       │   ├── __init__.py
│       │   ├── pydantic_ai.py           # PydanticAI adapter
│       │   ├── grail.py                 # Grail adapter (optional)
│       │   └── yaml.py                  # YAML adapter (optional)
│       ├── envelopes.py                 # RunEnvelope, EventEnvelope, trace models
│       ├── policy/
│       │   ├── __init__.py
│       │   ├── models.py               # ToolPolicy config model
│       │   ├── toolset.py              # PolicyToolset (AbstractToolset impl)
│       │   └── presets.py              # Built-in policy presets
│       ├── grail/
│       │   ├── __init__.py
│       │   ├── config.py               # GrailToolsetConfig, GrailLimitsConfig
│       │   ├── toolset.py              # GrailToolset (AbstractToolset impl)
│       │   ├── manifest.py             # GrailManifest extraction
│       │   ├── externals.py            # ExternalBindingRegistry
│       │   └── errors.py               # GrailErrorEnvelope normalization
│       ├── agent.py                     # StructuredAgent shim
│       ├── profiles.py                  # StructuredAgentProfile config model
│       ├── factory.py                   # StructuredAgentFactory builder
│       ├── registry.py                  # StructuredAgentRegistry
│       └── config.py                    # YAML/JSON config loading helpers
├── tests/
│   ├── conftest.py
│   ├── test_envelopes.py
│   ├── test_policy/
│   │   ├── test_models.py
│   │   ├── test_toolset.py
│   │   └── test_presets.py
│   ├── test_grail/
│   │   ├── test_config.py
│   │   ├── test_toolset.py
│   │   ├── test_manifest.py
│   │   ├── test_externals.py
│   │   └── test_errors.py
│   ├── test_agent.py
│   ├── test_profiles.py
│   ├── test_factory.py
│   ├── test_registry.py
│   └── fixtures/
│       └── scripts/                     # Sample .pym files for testing
│           ├── simple_calc.pym
│           ├── with_externals.pym
│           └── bad_syntax.pym
└── examples/
    ├── 01_simple_text_agent.py
    ├── 02_typed_output_agent.py
    ├── 03_grail_toolset_agent.py
    ├── 04_policy_filtered_agent.py
    ├── 05_approval_flow.py
    ├── 06_event_stream.py
    └── 07_local_model_agent.py
```

---

## 4. Public API surface

### 4.1 Top-level exports (`__init__.py`)

```python
# Core — always available
from structured_agents_v2.envelopes import (
    RunEnvelope,
    EventEnvelope,
    ToolTraceEnvelope,
    UsageEnvelope,
    DeferredCallEnvelope,
)
from structured_agents_v2.policy import ToolPolicy, PolicyToolset
from structured_agents_v2.agent import StructuredAgent
from structured_agents_v2.profiles import StructuredAgentProfile
from structured_agents_v2.factory import StructuredAgentFactory
from structured_agents_v2.registry import StructuredAgentRegistry

# Grail — available only when grail extra is installed
# Users import explicitly: from structured_agents_v2.grail import GrailToolset
```

### 4.2 What users should never import

- Anything from `_adapters/` — internal only
- Private helpers prefixed with `_`

---

## 5. Envelopes (`envelopes.py`)

Stable, serializable result and event contracts. These are the primary API boundary between this library and downstream applications. They must never expose PydanticAI or Grail types directly.

### 5.1 `UsageEnvelope`

```python
class UsageEnvelope(BaseModel, frozen=True):
    """Token usage summary for a run."""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_pydantic_ai(cls, usage: "RunUsage") -> "UsageEnvelope":
        """Adapter: convert PydanticAI RunUsage to stable envelope."""
        ...
```

### 5.2 `ToolTraceEnvelope`

```python
from typing import Literal, Any

class ToolTraceEnvelope(BaseModel, frozen=True):
    """Trace record for a single tool invocation."""
    call_id: str | None = None
    tool_name: str
    execution_plane: Literal["python", "grail", "deferred", "builtin", "unknown"] = "unknown"
    status: Literal[
        "requested", "approved", "denied",
        "started", "succeeded", "failed", "deferred"
    ]
    duration_ms: int | None = None
    input_summary: dict[str, Any] | None = None
    output_summary: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 5.3 `DeferredCallEnvelope`

```python
class DeferredCallEnvelope(BaseModel, frozen=True):
    """Record of a tool call that was deferred for external resolution."""
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 5.4 `RunEnvelope`

```python
class RunEnvelope(BaseModel, frozen=True):
    """Stable application-facing result of a structured agent run."""
    agent_name: str
    run_id: str
    status: Literal["succeeded", "failed", "deferred", "cancelled"]

    # Output
    output_text: str | None = None
    output_data: Any | None = None
    output_type: str | None = None

    # Traces
    deferred_calls: list[DeferredCallEnvelope] = Field(default_factory=list)
    tool_trace: list[ToolTraceEnvelope] = Field(default_factory=list)
    usage: UsageEnvelope | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_pydantic_ai_result(
        cls,
        agent_name: str,
        result: "AgentRunResult",
        tool_trace: list[ToolTraceEnvelope] | None = None,
    ) -> "RunEnvelope":
        """Adapter: convert PydanticAI AgentRunResult to stable envelope."""
        ...
```

### 5.5 `EventEnvelope`

```python
from datetime import datetime

class EventEnvelope(BaseModel, frozen=True):
    """Stable event shape for UIs, CLIs, logs, and telemetry."""
    run_id: str
    timestamp: datetime
    sequence: int
    kind: str       # Namespaced event kind, e.g. "run.started", "tool.finished"
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 5.6 Event kind taxonomy

Events are namespaced strings. The set is extensible; these are the initial kinds:

| Namespace | Kind | Emitted when |
|---|---|---|
| `run` | `run.started` | Agent run begins |
| `run` | `run.finished` | Agent run completes successfully |
| `run` | `run.failed` | Agent run errors |
| `tool` | `tool.called` | Tool invocation requested by model |
| `tool` | `tool.started` | Tool execution begins |
| `tool` | `tool.succeeded` | Tool execution succeeds |
| `tool` | `tool.failed` | Tool execution errors |
| `tool` | `tool.filtered` | Tool hidden by policy before model sees it |
| `tool` | `tool.approval_required` | Tool requires human approval |
| `tool` | `tool.approved` | Tool approved |
| `tool` | `tool.denied` | Tool denied |
| `tool` | `tool.deferred` | Tool deferred to external system |
| `grail` | `grail.script.loaded` | Grail script loaded and validated |
| `grail` | `grail.script.load_failed` | Grail script failed to load |
| `grail` | `grail.script.started` | Grail script execution begins |
| `grail` | `grail.script.succeeded` | Grail script execution succeeds |
| `grail` | `grail.script.failed` | Grail script execution errors |
| `grail` | `grail.limit_exceeded` | Grail script hit a resource limit |

---

## 6. Policy layer (`policy/`)

### 6.1 `ToolPolicy` (`policy/models.py`)

A declarative, serializable policy for controlling tool visibility, approval, deferral, and budgets.

```python
from typing import Literal, Any

class ToolPolicy(BaseModel, frozen=True):
    """Declarative tool exposure and execution policy."""
    name: str

    # Plane-level switches
    allow_python_tools: bool = True
    allow_grail_tools: bool = True

    # Name-level filtering (applied after plane switches)
    # None = allow all; explicit list = allowlist (only these pass)
    allow_tool_names: list[str] | None = None
    deny_tool_names: list[str] = Field(default_factory=list)

    # Execution mode overrides per tool name (supports glob patterns)
    require_approval_for: list[str] = Field(default_factory=list)
    defer_execution_for: list[str] = Field(default_factory=list)

    # Budget constraints
    max_tool_calls: int | None = None
    max_single_tool_duration_ms: int | None = None

    # Metadata-based filtering
    # Tool must have all of these metadata key-value pairs to pass
    required_metadata: dict[str, str] = Field(default_factory=dict)

    # Default execution mode for tools not matched by specific overrides
    default_execution_mode: Literal["direct", "approval", "deferred"] = "direct"

    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Policy resolution rules (applied in order):**

1. Check plane switch (`allow_python_tools` / `allow_grail_tools`). If the tool's plane is disabled, filter it out.
2. Check `deny_tool_names` (glob match). If matched, filter out.
3. Check `allow_tool_names`. If set and tool not in list (glob match), filter out.
4. Check `required_metadata`. If tool lacks any required key-value pair, filter out.
5. Determine execution mode:
   a. If tool name matches `require_approval_for` (glob) → `approval`
   b. If tool name matches `defer_execution_for` (glob) → `deferred`
   c. Otherwise → `default_execution_mode`

Glob matching uses `fnmatch` semantics: `*write*` matches `db_write`, `file_write_csv`, etc.

### 6.2 `PolicyToolset` (`policy/toolset.py`)

An `AbstractToolset` wrapper that wraps any inner toolset and applies a `ToolPolicy`.

```python
class PolicyToolset(PydanticAbstractToolset[AgentDepsT]):
    """Wraps an inner toolset and applies policy-based filtering and execution control."""

    def __init__(
        self,
        inner: PydanticAbstractToolset[AgentDepsT],
        policy: ToolPolicy,
        *,
        namespace: str | None = None,
        event_callback: Callable[[EventEnvelope], None] | None = None,
    ) -> None:
        self._inner = inner
        self._policy = policy
        self._namespace = namespace
        self._event_callback = event_callback
        self._call_count = 0

    @property
    def id(self) -> str | None:
        return f"policy:{self._policy.name}"

    async def get_tools(self, ctx) -> dict[str, ToolsetTool]:
        """Get tools from inner toolset, apply policy filtering."""
        inner_tools = await self._inner.get_tools(ctx)
        filtered = {}
        for name, tool in inner_tools.items():
            display_name = f"{self._namespace}__{name}" if self._namespace else name
            if self._should_expose(name, tool):
                # Modify tool definition based on execution mode
                mode = self._resolve_execution_mode(name)
                if mode == "approval":
                    # Mark as requiring approval via ToolDefinition.kind
                    ...
                filtered[display_name] = tool
            else:
                self._emit_event("tool.filtered", tool_name=name)
        return filtered

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        """Execute tool call with policy enforcement."""
        # Strip namespace prefix for inner dispatch
        inner_name = self._strip_namespace(name)

        # Budget check
        if self._policy.max_tool_calls and self._call_count >= self._policy.max_tool_calls:
            raise ToolCallBudgetExceeded(...)

        self._call_count += 1
        self._emit_event("tool.started", tool_name=name)

        try:
            result = await self._inner.call_tool(inner_name, tool_args, ctx, tool)
            self._emit_event("tool.succeeded", tool_name=name)
            return result
        except Exception as exc:
            self._emit_event("tool.failed", tool_name=name, error=str(exc))
            raise
```

**Key design decisions:**

- `PolicyToolset` is a standard `AbstractToolset` wrapper. It delegates all execution to the inner toolset.
- Namespacing is applied at the `get_tools` level by prefixing tool names with `{namespace}__`.
- Approval and deferral are implemented by modifying the `ToolDefinition.kind` field, which tells PydanticAI to treat the tool as requiring approval or being deferred. We do NOT implement our own approval loop.
- Budget enforcement is per-run (the `_call_count` resets via `for_run()`).

### 6.3 Policy presets (`policy/presets.py`)

```python
def safe_default() -> ToolPolicy:
    """Allow all tools, require approval for write/delete/send operations."""
    return ToolPolicy(
        name="safe_default",
        require_approval_for=["*write*", "*delete*", "*send*", "*remove*", "*create*"],
    )

def grail_only() -> ToolPolicy:
    """Only allow Grail tools. Hide all Python tools."""
    return ToolPolicy(
        name="grail_only",
        allow_python_tools=False,
        allow_grail_tools=True,
    )

def approval_all() -> ToolPolicy:
    """Expose all tools but require approval for every call."""
    return ToolPolicy(
        name="approval_all",
        default_execution_mode="approval",
    )

def readonly() -> ToolPolicy:
    """Deny tools that match write/delete/mutate patterns."""
    return ToolPolicy(
        name="readonly",
        deny_tool_names=["*write*", "*delete*", "*create*", "*update*", "*remove*", "*send*"],
    )

def open() -> ToolPolicy:
    """Permissive policy for testing and local development."""
    return ToolPolicy(name="open")
```

---

## 7. Grail integration (`grail/`)

This entire subpackage is guarded by the `grail` extras group. Every module begins with:

```python
from structured_agents_v2._adapters.grail import require_grail
require_grail()
```

### 7.1 `GrailLimitsConfig` (`grail/config.py`)

A Pydantic config model that maps to Grail's `Limits`.

```python
class GrailLimitsConfig(BaseModel, frozen=True):
    """Resource limits for Grail script execution."""
    max_memory: int | str | None = None       # bytes or "16mb"
    max_duration: float | str | None = None   # seconds or "500ms"
    max_recursion: int | None = None
    max_allocations: int | None = None

    def to_grail_limits(self) -> "_GrailLimits":
        """Convert to Grail Limits object."""
        from structured_agents_v2._adapters.grail import GrailLimits
        return GrailLimits(
            max_memory=self.max_memory,
            max_duration=self.max_duration,
            max_recursion=self.max_recursion,
            max_allocations=self.max_allocations,
        )

    @classmethod
    def strict(cls) -> "GrailLimitsConfig":
        return cls(max_memory="8mb", max_duration="500ms", max_recursion=120)

    @classmethod
    def default(cls) -> "GrailLimitsConfig":
        return cls(max_memory="16mb", max_duration="2s", max_recursion=200)

    @classmethod
    def permissive(cls) -> "GrailLimitsConfig":
        return cls(max_memory="64mb", max_duration="5s", max_recursion=400)
```

### 7.2 `GrailToolsetConfig` (`grail/config.py`)

```python
class GrailToolsetConfig(BaseModel, frozen=True):
    """Configuration for a collection of Grail scripts exposed as tools."""
    name: str
    description: str | None = None
    paths: list[str]
    namespace: str | None = None
    limits: GrailLimitsConfig | None = None
    external_registry_name: str | None = None
    include_return_schema: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 7.3 `GrailManifest` (`grail/manifest.py`)

Extracted metadata from a loaded Grail script, usable for documentation, registration, policy matching, and debugging.

```python
class GrailInputManifest(BaseModel, frozen=True):
    """Manifest entry for a single declared Input()."""
    name: str
    type_annotation: str
    required: bool
    default: Any | None = None

class GrailExternalManifest(BaseModel, frozen=True):
    """Manifest entry for a single declared @external."""
    name: str
    is_async: bool
    parameters: list[dict[str, Any]]    # [{name, type, default?, required}]
    return_type: str
    docstring: str | None = None

class GrailManifest(BaseModel, frozen=True):
    """Complete manifest for a loaded Grail script."""
    script_name: str
    path: str
    tool_name: str                       # Resolved tool name (with namespace if set)
    description: str | None = None
    inputs: list[GrailInputManifest] = Field(default_factory=list)
    externals: list[GrailExternalManifest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_grail_script(
        cls,
        script: "_GrailScript",
        tool_name: str,
        description: str | None = None,
    ) -> "GrailManifest":
        """Extract manifest from a loaded GrailScript."""
        ...
```

### 7.4 `GrailErrorEnvelope` (`grail/errors.py`)

Normalized error surface for Grail execution failures.

```python
class GrailErrorEnvelope(BaseModel, frozen=True):
    """Normalized error envelope for Grail script failures."""
    script_name: str
    error_category: Literal[
        "parse", "check", "input", "external",
        "execution", "limit", "output", "unknown"
    ]
    error_type: str                      # Original exception class name
    message: str
    lineno: int | None = None
    source_context: str | None = None
    suggestion: str | None = None
    limit_type: str | None = None        # For limit errors: "memory", "duration", etc.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_grail_error(cls, script_name: str, exc: Exception) -> "GrailErrorEnvelope":
        """Map any Grail exception to a normalized envelope.

        This is the single point where Grail error types are mapped.
        If Grail changes its error hierarchy, only this method changes.
        """
        from structured_agents_v2._adapters.grail import (
            GrailParseError, GrailCheckError, GrailInputError,
            GrailExternalError, GrailExecutionError, GrailLimitError,
            GrailOutputError,
        )

        if isinstance(exc, GrailParseError):
            return cls(
                script_name=script_name,
                error_category="parse",
                error_type=type(exc).__name__,
                message=str(exc),
                lineno=getattr(exc, "lineno", None),
            )
        elif isinstance(exc, GrailLimitError):
            return cls(
                script_name=script_name,
                error_category="limit",
                error_type=type(exc).__name__,
                message=str(exc),
                limit_type=getattr(exc, "limit_type", None),
            )
        elif isinstance(exc, GrailExecutionError):
            return cls(
                script_name=script_name,
                error_category="execution",
                error_type=type(exc).__name__,
                message=str(exc),
                lineno=getattr(exc, "lineno", None),
                source_context=getattr(exc, "source_context", None),
                suggestion=getattr(exc, "suggestion", None),
            )
        # ... remaining mappings for CheckError, InputError, ExternalError, OutputError
        else:
            return cls(
                script_name=script_name,
                error_category="unknown",
                error_type=type(exc).__name__,
                message=str(exc),
            )
```

### 7.5 `ExternalBindingRegistry` (`grail/externals.py`)

Host-side registry for functions that Grail scripts can call via `@external` declarations.

```python
class ExternalBindingEntry(BaseModel, frozen=True):
    """Metadata for a registered external binding."""
    name: str
    description: str | None = None
    is_async: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalBindingRegistry:
    """Runtime registry mapping external function names to host callables.

    This is intentionally NOT a Pydantic model — callables are not serializable.
    The registry is a runtime object, not a config object.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, tuple[Callable, ExternalBindingEntry]] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a host function as an available external binding."""
        is_async = asyncio.iscoroutinefunction(fn)
        entry = ExternalBindingEntry(
            name=name,
            description=description,
            is_async=is_async,
            metadata=metadata or {},
        )
        self._bindings[name] = (fn, entry)

    def resolve(self, names: list[str]) -> dict[str, Callable]:
        """Resolve a list of external names to callables.

        Raises KeyError with a clear message if any name is unregistered.
        """
        result = {}
        missing = []
        for name in names:
            if name in self._bindings:
                result[name] = self._bindings[name][0]
            else:
                missing.append(name)
        if missing:
            available = list(self._bindings.keys())
            raise KeyError(
                f"Missing external bindings: {missing}. "
                f"Available: {available}"
            )
        return result

    def list_entries(self) -> list[ExternalBindingEntry]:
        """Return metadata for all registered bindings."""
        return [entry for _, entry in self._bindings.values()]
```

### 7.6 `GrailToolset` (`grail/toolset.py`)

The core differentiator. Implements PydanticAI's `AbstractToolset` interface, exposing Grail `.pym` scripts as model-callable tools.

```python
from __future__ import annotations

from typing import Any, Sequence
from pathlib import Path

from structured_agents_v2._adapters.pydantic_ai import (
    PydanticAbstractToolset,
    ToolsetTool,
    ToolDefinition,
    RunContext,
)
from structured_agents_v2._adapters.grail import require_grail
from structured_agents_v2.grail.config import GrailToolsetConfig, GrailLimitsConfig
from structured_agents_v2.grail.manifest import GrailManifest
from structured_agents_v2.grail.externals import ExternalBindingRegistry
from structured_agents_v2.grail.errors import GrailErrorEnvelope
from structured_agents_v2.envelopes import ToolTraceEnvelope, EventEnvelope


class GrailToolset(PydanticAbstractToolset[Any]):
    """PydanticAI toolset backed by Grail .pym scripts.

    Each .pym script becomes one tool. The script's Input() declarations
    become the tool's parameters. The script's @external declarations
    are resolved from an ExternalBindingRegistry at construction time.
    """

    def __init__(
        self,
        config: GrailToolsetConfig,
        *,
        external_registry: ExternalBindingRegistry | None = None,
        event_callback: Callable[[EventEnvelope], None] | None = None,
    ) -> None:
        require_grail()
        self._config = config
        self._external_registry = external_registry
        self._event_callback = event_callback
        self._scripts: dict[str, "_GrailScript"] = {}
        self._manifests: dict[str, GrailManifest] = {}
        self._tool_traces: list[ToolTraceEnvelope] = []
        self._load_scripts()

    @classmethod
    def from_paths(
        cls,
        name: str,
        paths: Sequence[str | Path],
        *,
        namespace: str | None = None,
        limits: GrailLimitsConfig | None = None,
        external_registry: ExternalBindingRegistry | None = None,
        event_callback: Callable[[EventEnvelope], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GrailToolset":
        """Convenience constructor from a list of .pym file paths."""
        config = GrailToolsetConfig(
            name=name,
            paths=[str(p) for p in paths],
            namespace=namespace,
            limits=limits,
            metadata=metadata or {},
        )
        return cls(config, external_registry=external_registry, event_callback=event_callback)

    def _load_scripts(self) -> None:
        """Load all .pym scripts at construction time.

        Eagerly loads and validates so errors surface early, not at first tool call.
        """
        from structured_agents_v2._adapters.grail import grail_load, GrailLimits

        grail_limits = self._config.limits.to_grail_limits() if self._config.limits else None

        for path_str in self._config.paths:
            path = Path(path_str)
            try:
                script = grail_load(str(path), limits=grail_limits)
                tool_name = self._make_tool_name(script.name)
                self._scripts[tool_name] = script
                self._manifests[tool_name] = GrailManifest.from_grail_script(
                    script, tool_name=tool_name,
                )
                self._emit("grail.script.loaded", tool_name=tool_name, path=str(path))
            except Exception as exc:
                envelope = GrailErrorEnvelope.from_grail_error(path.stem, exc)
                self._emit("grail.script.load_failed", tool_name=path.stem, error=envelope.model_dump())
                raise

    def _make_tool_name(self, script_name: str) -> str:
        if self._config.namespace:
            return f"{self._config.namespace}__{script_name}"
        return script_name

    # --- AbstractToolset interface ---

    @property
    def id(self) -> str | None:
        return f"grail:{self._config.name}"

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        """Build PydanticAI tool definitions from loaded Grail scripts."""
        tools = {}
        for tool_name, manifest in self._manifests.items():
            # Build JSON schema from Input() declarations
            parameters_schema = self._build_parameters_schema(manifest)
            tool_def = ToolDefinition(
                name=tool_name,
                description=manifest.description or f"Grail script: {manifest.script_name}",
                parameters_json_schema=parameters_schema,
                kind="function",
                strict=False,
                defer_loading=False,
                return_schema=None,
            )
            tools[tool_name] = ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=0,
                args_validator=self._build_input_validator(manifest),
            )
        return tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        """Execute a Grail script with the model-provided arguments."""
        import time

        script = self._scripts[name]
        manifest = self._manifests[name]

        # Resolve externals
        externals = None
        if script.externals and self._external_registry:
            external_names = list(script.externals.keys())
            externals = self._external_registry.resolve(external_names)

        self._emit("grail.script.started", tool_name=name)
        start = time.monotonic()

        try:
            result = await script.run(
                inputs=tool_args,
                externals=externals,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            self._emit("grail.script.succeeded", tool_name=name, duration_ms=duration_ms)

            self._tool_traces.append(ToolTraceEnvelope(
                tool_name=name,
                execution_plane="grail",
                status="succeeded",
                duration_ms=duration_ms,
                input_summary=_summarize_args(tool_args),
                output_summary=_summarize_output(result),
            ))
            return result

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            error_envelope = GrailErrorEnvelope.from_grail_error(manifest.script_name, exc)
            self._emit("grail.script.failed", tool_name=name, error=error_envelope.model_dump())

            self._tool_traces.append(ToolTraceEnvelope(
                tool_name=name,
                execution_plane="grail",
                status="failed",
                duration_ms=duration_ms,
                error=error_envelope.message,
            ))
            raise

    # --- Inspection ---

    @property
    def manifests(self) -> dict[str, GrailManifest]:
        """All loaded script manifests, keyed by tool name."""
        return dict(self._manifests)

    @property
    def tool_traces(self) -> list[ToolTraceEnvelope]:
        """Accumulated tool traces for this toolset instance."""
        return list(self._tool_traces)

    # --- Internal helpers ---

    def _build_parameters_schema(self, manifest: GrailManifest) -> dict[str, Any]:
        """Build a JSON Schema object from Input() declarations."""
        properties = {}
        required = []
        for inp in manifest.inputs:
            properties[inp.name] = {"type": _grail_type_to_json_type(inp.type_annotation)}
            if inp.required:
                required.append(inp.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_input_validator(self, manifest: GrailManifest):
        """Build a validator for tool arguments based on Input() specs.

        Returns a validator compatible with ToolsetTool.args_validator.
        """
        ...

    def _emit(self, kind: str, **payload: Any) -> None:
        if self._event_callback:
            self._event_callback(EventEnvelope(
                run_id="",  # Filled by caller or left empty for toolset-level events
                timestamp=datetime.now(),
                sequence=0,
                kind=kind,
                payload=payload,
            ))
```

### 7.7 Grail type mapping

```python
_TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "None": "null",
}

def _grail_type_to_json_type(annotation: str) -> str:
    """Map Grail type annotation strings to JSON Schema types."""
    return _TYPE_MAP.get(annotation, "string")
```

---

## 8. Agent shim (`agent.py`)

A thin convenience wrapper around a PydanticAI `Agent`. It does NOT own any runtime behavior.

```python
from typing import Any, AsyncIterator
from pydantic import BaseModel, ConfigDict, PrivateAttr

from structured_agents_v2._adapters.pydantic_ai import PydanticAgent, AgentRunResult
from structured_agents_v2.envelopes import RunEnvelope, EventEnvelope, ToolTraceEnvelope
from structured_agents_v2.profiles import StructuredAgentProfile


class StructuredAgent(BaseModel):
    """Convenience wrapper around a PydanticAI Agent.

    Provides enveloped runs, event iteration, and profile access
    while keeping the underlying PydanticAI agent fully accessible.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: StructuredAgentProfile
    _agent: PydanticAgent = PrivateAttr()
    _tool_traces: list[ToolTraceEnvelope] = PrivateAttr(default_factory=list)

    def __init__(self, *, profile: StructuredAgentProfile, agent: PydanticAgent) -> None:
        super().__init__(profile=profile)
        self._agent = agent
        self._tool_traces = []

    # --- Escape hatch ---

    @property
    def agent(self) -> PydanticAgent:
        """Access the underlying PydanticAI Agent directly."""
        return self._agent

    # --- Run methods ---

    async def run(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        """Delegate directly to PydanticAI. Returns raw PydanticAI result."""
        return await self._agent.run(prompt, **kwargs)

    def run_sync(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        """Synchronous run. Delegates to PydanticAI."""
        return self._agent.run_sync(prompt, **kwargs)

    async def run_enveloped(self, prompt: str, **kwargs: Any) -> RunEnvelope:
        """Run and return a stable RunEnvelope."""
        try:
            result = await self._agent.run(prompt, **kwargs)
            return RunEnvelope.from_pydantic_ai_result(
                agent_name=self.profile.name,
                result=result,
                tool_trace=self._collect_tool_traces(),
            )
        except Exception as exc:
            return RunEnvelope(
                agent_name=self.profile.name,
                run_id="",
                status="failed",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )

    async def iter_events(self, prompt: str, **kwargs: Any) -> AsyncIterator[EventEnvelope]:
        """Run with node-by-node iteration, yielding stable EventEnvelopes.

        Uses PydanticAI's iter() API internally.
        """
        async with self._agent.iter(prompt, **kwargs) as agent_run:
            yield EventEnvelope(
                run_id=agent_run.run_id,
                timestamp=datetime.now(),
                sequence=0,
                kind="run.started",
                message=f"Agent '{self.profile.name}' started",
            )
            seq = 1
            async for node in agent_run:
                yield EventEnvelope(
                    run_id=agent_run.run_id,
                    timestamp=datetime.now(),
                    sequence=seq,
                    kind=_node_to_event_kind(node),
                    payload=_node_to_payload(node),
                )
                seq += 1

            yield EventEnvelope(
                run_id=agent_run.run_id,
                timestamp=datetime.now(),
                sequence=seq,
                kind="run.finished",
                payload={"usage": agent_run.usage()},
            )

    # --- Internal ---

    def _collect_tool_traces(self) -> list[ToolTraceEnvelope]:
        """Collect tool traces from all toolsets that support it."""
        traces = []
        # GrailToolset and PolicyToolset expose .tool_traces
        # Walk the agent's toolsets and collect
        ...
        return traces
```

---

## 9. Profile and factory (`profiles.py`, `factory.py`)

### 9.1 `StructuredAgentProfile`

```python
class StructuredAgentProfile(BaseModel, frozen=True):
    """Serializable profile describing how to assemble a StructuredAgent."""
    name: str
    description: str | None = None

    # PydanticAI agent config (passed through, not re-abstracted)
    model: str
    instructions: str | list[str] | None = None
    model_settings: dict[str, Any] | None = None
    output_type_ref: str | None = None    # Optional dotted path to output type

    # v2 composition references (resolved by factory via registry)
    grail_toolset_names: list[str] = Field(default_factory=list)
    policy_names: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 9.2 `StructuredAgentFactory`

```python
class StructuredAgentFactory:
    """Builds StructuredAgent instances from profiles using a registry."""

    def __init__(
        self,
        registry: "StructuredAgentRegistry",
        *,
        default_model: str | None = None,
        default_policy_names: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._default_model = default_model
        self._default_policy_names = default_policy_names or []

    def build(self, profile: StructuredAgentProfile) -> StructuredAgent:
        """Compile a profile into a StructuredAgent.

        Resolution order:
        1. Resolve Grail toolset configs from registry, build GrailToolsets.
        2. Resolve policy configs from registry.
        3. Wrap toolsets in PolicyToolsets.
        4. Construct PydanticAI Agent with toolsets, model, instructions.
        5. Wrap in StructuredAgent.
        """
        from structured_agents_v2._adapters.pydantic_ai import PydanticAgent

        # 1. Build toolsets
        toolsets = []
        for name in profile.grail_toolset_names:
            grail_config = self._registry.get_grail_toolset_config(name)
            external_reg = self._registry.get_external_registry(
                grail_config.external_registry_name
            )
            toolset = GrailToolset(grail_config, external_registry=external_reg)
            toolsets.append(toolset)

        # 2. Apply policies
        policy_names = profile.policy_names or self._default_policy_names
        for policy_name in policy_names:
            policy = self._registry.get_policy(policy_name)
            toolsets = [PolicyToolset(ts, policy) for ts in toolsets]

        # 3. Build PydanticAI agent
        model = profile.model or self._default_model
        agent = PydanticAgent(
            model,
            instructions=profile.instructions,
            model_settings=profile.model_settings,
            toolsets=toolsets,
        )

        # 4. Wrap
        return StructuredAgent(profile=profile, agent=agent)
```

---

## 10. Registry (`registry.py`)

```python
class StructuredAgentRegistry:
    """Central registry for named configs used in profile-based agent assembly.

    Serializable configs are stored as Pydantic models.
    Runtime objects (external registries with callables) are stored separately.
    """

    def __init__(self) -> None:
        self._grail_toolset_configs: dict[str, GrailToolsetConfig] = {}
        self._policies: dict[str, ToolPolicy] = {}
        self._external_registries: dict[str, ExternalBindingRegistry] = {}

    # --- Registration ---

    def register_grail_toolset(self, config: GrailToolsetConfig) -> None:
        self._grail_toolset_configs[config.name] = config

    def register_policy(self, policy: ToolPolicy) -> None:
        self._policies[policy.name] = policy

    def register_external_registry(self, name: str, registry: ExternalBindingRegistry) -> None:
        self._external_registries[name] = registry

    # --- Lookup ---

    def get_grail_toolset_config(self, name: str) -> GrailToolsetConfig:
        if name not in self._grail_toolset_configs:
            raise KeyError(f"Unknown Grail toolset: '{name}'. Registered: {list(self._grail_toolset_configs)}")
        return self._grail_toolset_configs[name]

    def get_policy(self, name: str) -> ToolPolicy:
        if name not in self._policies:
            raise KeyError(f"Unknown policy: '{name}'. Registered: {list(self._policies)}")
        return self._policies[name]

    def get_external_registry(self, name: str | None) -> ExternalBindingRegistry | None:
        if name is None:
            return None
        if name not in self._external_registries:
            raise KeyError(f"Unknown external registry: '{name}'. Registered: {list(self._external_registries)}")
        return self._external_registries[name]

    # --- Serializable snapshot ---

    def export_config(self) -> dict[str, Any]:
        """Export serializable config (policies and toolset configs, not callables)."""
        return {
            "grail_toolsets": {k: v.model_dump() for k, v in self._grail_toolset_configs.items()},
            "policies": {k: v.model_dump() for k, v in self._policies.items()},
        }
```

---

## 11. Config loading (`config.py`)

Optional YAML/JSON loading for profiles, policies, and toolset configs.

```python
def load_profile(path: str | Path) -> StructuredAgentProfile:
    """Load a StructuredAgentProfile from a YAML or JSON file."""
    data = _load_file(path)
    return StructuredAgentProfile.model_validate(data)

def load_policy(path: str | Path) -> ToolPolicy:
    """Load a ToolPolicy from a YAML or JSON file."""
    data = _load_file(path)
    return ToolPolicy.model_validate(data)

def load_grail_toolset_config(path: str | Path) -> GrailToolsetConfig:
    """Load a GrailToolsetConfig from a YAML or JSON file."""
    data = _load_file(path)
    return GrailToolsetConfig.model_validate(data)

def _load_file(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON based on file extension."""
    path = Path(path)
    content = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        from structured_agents_v2._adapters.yaml import yaml_safe_load
        return yaml_safe_load(content)
    else:
        import json
        return json.loads(content)
```

---

## 12. Usage patterns

### 12.1 PydanticAI-native (no shim layer)

This path must always work. It's the escape hatch and the simplest integration.

```python
from pydantic_ai import Agent
from structured_agents_v2.grail import GrailToolset
from structured_agents_v2.policy import PolicyToolset, ToolPolicy
from structured_agents_v2.policy.presets import safe_default

# Load Grail tools directly
finance_tools = GrailToolset.from_paths(
    name="finance",
    paths=["tools/expense_analysis.pym", "tools/risk_score.pym"],
)

# Wrap with policy
safe_tools = PolicyToolset(finance_tools, policy=safe_default())

# Use with a standard PydanticAI Agent
agent = Agent(
    "anthropic:claude-sonnet-4-6",
    instructions="Use finance tools carefully.",
    toolsets=[safe_tools],
)

result = await agent.run("Analyze this account for budget risk.")
print(result.output)
```

### 12.2 StructuredAgent shim (enveloped results)

```python
from structured_agents_v2 import (
    StructuredAgentFactory,
    StructuredAgentProfile,
    StructuredAgentRegistry,
)
from structured_agents_v2.grail import GrailToolsetConfig, GrailLimitsConfig
from structured_agents_v2.policy.presets import safe_default

# Set up registry
registry = StructuredAgentRegistry()
registry.register_grail_toolset(GrailToolsetConfig(
    name="finance",
    paths=["tools/expense_analysis.pym", "tools/risk_score.pym"],
    namespace="finance",
    limits=GrailLimitsConfig.default(),
))
registry.register_policy(safe_default())

# Define profile
profile = StructuredAgentProfile(
    name="finance_assistant",
    model="anthropic:claude-sonnet-4-6",
    instructions="Use available tools carefully. Return concise analysis.",
    grail_toolset_names=["finance"],
    policy_names=["safe_default"],
)

# Build and run
factory = StructuredAgentFactory(registry=registry)
agent = factory.build(profile)

envelope = await agent.run_enveloped("Find budget risks for user 123.")
print(envelope.status)        # "succeeded"
print(envelope.output_text)   # The agent's response
print(envelope.tool_trace)    # List of ToolTraceEnvelope
print(envelope.usage)         # UsageEnvelope
```

### 12.3 Raw PydanticAI escape hatch

```python
# From a StructuredAgent, access the underlying PydanticAI Agent
pydantic_agent = agent.agent
raw_result = await pydantic_agent.run("Use any PydanticAI feature directly.")
raw_result.all_messages()     # Full message history
raw_result.usage()            # PydanticAI RunUsage object
```

### 12.4 Event streaming

```python
async for event in agent.iter_events("Run the full analysis."):
    if event.kind == "tool.started":
        print(f"  Tool: {event.payload['tool_name']}...")
    elif event.kind == "grail.script.succeeded":
        print(f"  Script done in {event.payload['duration_ms']}ms")
    elif event.kind == "run.finished":
        print(f"Run complete. Tokens: {event.payload.get('usage', {})}")
```

### 12.5 Config-driven (YAML profiles)

```yaml
# agents/finance.yaml
name: finance_assistant
model: anthropic:claude-sonnet-4-6
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

```python
from structured_agents_v2.config import load_profile

profile = load_profile("agents/finance.yaml")
agent = factory.build(profile)
```

---

## 13. Testing strategy

### 13.1 Test tiers

| Tier | What | Mocking | Requires |
|---|---|---|---|
| Unit | Envelopes, policies, manifests, config loading | No external deps | `pydantic` only |
| Integration — Grail | GrailToolset loading, execution, error mapping | No LLM | `grail` extra |
| Integration — PydanticAI | PolicyToolset, StructuredAgent with mock model | `TestModel` from pydantic-ai | `pydantic-ai` |
| E2E | Full agent run with Grail tools and real/mock model | Optional | All deps |

### 13.2 Key test cases

**Envelopes:**
- `RunEnvelope` serializes to/from JSON cleanly
- `EventEnvelope` kinds match taxonomy
- `from_pydantic_ai_result` maps all fields correctly

**Policy:**
- `allow_tool_names` allowlist filters correctly
- `deny_tool_names` denylist filters correctly
- Glob patterns match expected names
- `require_approval_for` marks tools correctly
- `max_tool_calls` budget triggers after N calls
- Plane switches (`allow_python_tools=False`) filter correctly
- Policy presets produce expected configurations

**Grail toolset:**
- `.from_paths()` loads valid `.pym` files
- `.from_paths()` raises on invalid `.pym` (parse error)
- `get_tools()` returns correct tool definitions
- `call_tool()` executes script and returns result
- Missing externals fail before execution with clear error
- `GrailErrorEnvelope.from_grail_error()` maps all error categories
- `GrailManifest.from_grail_script()` extracts inputs and externals
- Limits are applied (duration, memory)
- Namespace prefixing works

**Agent / Factory:**
- Factory builds agent with Grail toolsets and policies
- `run_enveloped()` returns complete `RunEnvelope`
- `agent.agent` returns the underlying PydanticAI Agent
- Profile with unknown registry names raises `KeyError`

**Adapter boundary:**
- All imports from `_adapters/` work when deps are installed
- Grail adapter raises clear `ImportError` when grail is not installed
- YAML adapter raises clear `ImportError` when pyyaml is not installed

### 13.3 Test fixtures

```
tests/fixtures/scripts/
├── simple_calc.pym          # Inputs: a (int), b (int). Returns a + b.
├── with_externals.pym       # Inputs: query (str). External: fetch_data. Returns result.
├── no_inputs.pym            # No inputs. Returns static dict.
├── bad_syntax.pym           # Invalid Python syntax. Should fail at load().
├── slow_script.pym          # Loops to test duration limits.
└── over_budget.pym          # Allocates heavily to test memory limits.
```

---

## 14. Error handling strategy

### 14.1 Error hierarchy

We do NOT create a deep custom exception hierarchy. We use standard Python exceptions plus one thin layer:

```python
# exceptions.py

class StructuredAgentsError(Exception):
    """Base exception for structured-agents-v2."""
    pass

class ToolCallBudgetExceeded(StructuredAgentsError):
    """Raised when a policy's max_tool_calls budget is exceeded."""
    pass

class ConfigError(StructuredAgentsError):
    """Raised when profile/policy/config validation fails."""
    pass
```

Grail errors are caught, wrapped in `GrailErrorEnvelope` for tracing, and then either:
- Re-raised as-is (when using `GrailToolset` directly — the caller handles them)
- Returned as a tool error result (when inside PydanticAI's tool call lifecycle — PydanticAI handles tool errors)

### 14.2 Error boundary rules

| Layer | Catches | Produces |
|---|---|---|
| `GrailToolset.call_tool()` | Grail exceptions | `GrailErrorEnvelope` for tracing, then returns error string to PydanticAI |
| `PolicyToolset.call_tool()` | Budget violations | `ToolCallBudgetExceeded` |
| `StructuredAgent.run_enveloped()` | Any exception from run | `RunEnvelope` with `status="failed"` |
| `StructuredAgentFactory.build()` | Missing registry keys | `KeyError` with clear message |
| Config loaders | File/parse errors | `ConfigError` |
| Adapters | Missing imports | `ImportError` with install hint |

---

## 15. pyproject.toml

```toml
[project]
name = "structured-agents-v2"
version = "0.1.0"
description = "PydanticAI extension library for policy-aware agents with Grail-backed constrained tool execution"
requires-python = ">=3.12"
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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/structured_agents_v2"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.mypy]
python_version = "3.12"
strict = true
```

---

## 16. Build phases

### Phase 1: Foundation (envelopes + adapters + policy models)

**Deliver:**
- Repository skeleton with `pyproject.toml`, src layout, test config
- `_adapters/pydantic_ai.py` — PydanticAI adapter
- `_adapters/grail.py` — Grail adapter with availability guard
- `envelopes.py` — All envelope models with `from_pydantic_ai_result()`
- `policy/models.py` — `ToolPolicy` model
- `policy/presets.py` — Built-in presets
- `exceptions.py` — Error base classes
- Unit tests for envelopes, policy models, presets

**Goal:** Establish the public API surface and prove the adapter boundary pattern.

### Phase 2: Grail toolset

**Deliver:**
- `grail/config.py` — `GrailToolsetConfig`, `GrailLimitsConfig`
- `grail/manifest.py` — `GrailManifest` extraction
- `grail/externals.py` — `ExternalBindingRegistry`
- `grail/errors.py` — `GrailErrorEnvelope` normalization
- `grail/toolset.py` — `GrailToolset` implementing `AbstractToolset`
- Test fixtures (`.pym` scripts)
- Integration tests: load, execute, error mapping, limits, externals

**Goal:** Prove the core differentiator works end-to-end.

### Phase 3: PolicyToolset

**Deliver:**
- `policy/toolset.py` — `PolicyToolset` implementing `AbstractToolset`
- Policy filtering (plane, name, metadata)
- Namespace prefixing
- Approval/deferral marking
- Budget enforcement
- Integration tests with mock toolsets

**Goal:** Policy-controlled tool composition works correctly.

### Phase 4: Agent shim + factory

**Deliver:**
- `profiles.py` — `StructuredAgentProfile`
- `agent.py` — `StructuredAgent`
- `factory.py` — `StructuredAgentFactory`
- `registry.py` — `StructuredAgentRegistry`
- `config.py` — YAML/JSON loaders
- Integration tests: build from profile, run_enveloped, escape hatch

**Goal:** Developer ergonomics layer complete.

### Phase 5: Events + examples + polish

**Deliver:**
- `iter_events()` implementation on `StructuredAgent`
- Event normalization from PydanticAI nodes
- Complete examples (all 7 listed in package structure)
- Full test matrix passing
- Type checking clean
- README with quickstart

**Goal:** Ship-ready v0.1.0.

---

## 17. API boundary change scenarios

These scenarios validate that the adapter boundary pattern works.

### Scenario: PydanticAI renames `AbstractToolset`

**Impact:** `_adapters/pydantic_ai.py` import line changes.
**Files changed:** 1 (`_adapters/pydantic_ai.py`)
**Public API change:** None.

### Scenario: PydanticAI changes `ToolDefinition` fields

**Impact:** `_adapters/pydantic_ai.py` re-export may need a thin wrapper. `GrailToolset.get_tools()` and `PolicyToolset.get_tools()` update their construction calls.
**Files changed:** 1-3 (`_adapters/pydantic_ai.py`, `grail/toolset.py`, `policy/toolset.py`)
**Public API change:** None.

### Scenario: Grail changes `GrailScript.run()` signature

**Impact:** `_adapters/grail.py` adapter wraps the new signature. `GrailToolset.call_tool()` may update its call.
**Files changed:** 1-2 (`_adapters/grail.py`, `grail/toolset.py`)
**Public API change:** None.

### Scenario: Grail adds a new error type

**Impact:** `_adapters/grail.py` adds the new import. `grail/errors.py` adds a new branch in `from_grail_error()`.
**Files changed:** 2 (`_adapters/grail.py`, `grail/errors.py`)
**Public API change:** New `error_category` literal value in `GrailErrorEnvelope` (additive, non-breaking).

### Scenario: Grail changes `Limits` field names

**Impact:** `GrailLimitsConfig.to_grail_limits()` updates its field mapping.
**Files changed:** 1 (`grail/config.py`)
**Public API change:** None. `GrailLimitsConfig` uses our own field names.

### Scenario: PydanticAI changes streaming/iteration API

**Impact:** `StructuredAgent.iter_events()` updates its iteration logic. `_adapters/pydantic_ai.py` updates imports.
**Files changed:** 2 (`_adapters/pydantic_ai.py`, `agent.py`)
**Public API change:** None. `EventEnvelope` shape is stable.

---

## 18. Design decisions summary

| Decision | Choice | Rationale |
|---|---|---|
| Repo strategy | New greenfield repo | No legacy baggage |
| Runtime | PydanticAI | Avoids reimplementing agent loop |
| Grail dependency | Optional extra | Core works without it; Grail users get first-class support |
| API boundary strategy | `_adapters/` layer | Isolates all upstream changes to one file per dependency |
| Agent wrapper | Thin shim, not runtime | `StructuredAgent` adds envelopes, not behavior |
| Policy implementation | `AbstractToolset` wrapper | Composes naturally with any inner toolset |
| Error strategy | Normalize into envelopes, re-raise originals | Tracing gets stable data; callers get real exceptions |
| Config format | Pydantic models + optional YAML | Programmatic-first, config-file-friendly |
| Event model | Stable `EventEnvelope` over PydanticAI iteration | Downstream consumers never depend on PydanticAI internals |
| Structured output | PydanticAI `output_type` (pass-through) | Not our problem to solve |
| Model providers | PydanticAI model strings (pass-through) | Not our problem to solve |
| Testing | Tiered: unit / integration-grail / integration-pai / e2e | Each tier runs independently |

---

## 19. Acceptance criteria for v0.1.0

1. `GrailToolset.from_paths()` loads `.pym` scripts and exposes them as PydanticAI tools.
2. `GrailToolset` can be used directly in a PydanticAI `Agent` with no shim layer.
3. `PolicyToolset` wraps any toolset and filters/gates tools based on `ToolPolicy`.
4. `StructuredAgentFactory.build()` assembles agents from `StructuredAgentProfile` and registry.
5. `StructuredAgent.run_enveloped()` returns a stable, JSON-serializable `RunEnvelope`.
6. `StructuredAgent.iter_events()` yields stable `EventEnvelope` objects.
7. `StructuredAgent.agent` provides direct PydanticAI access.
8. Missing Grail dependency raises `ImportError` with install instructions (not a crash).
9. All Grail error types map to `GrailErrorEnvelope` with correct categories.
10. Missing externals fail before script execution with a clear message.
11. `_adapters/` layer is the only code that imports `pydantic_ai` or `grail`.
12. No custom agent loop, model client, or response parser exists anywhere.
13. All tests pass. Type checking passes. Linting passes.
