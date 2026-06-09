# Implementation Review: structured-agents-v2

## Document status

- **Reviewer:** Claude (Opus 4.6)
- **Date:** 2026-04-27
- **Documents reviewed:**
  - `STRUCTURED_AGENTS_V2_CONCEPT.md` (concept, 1191 lines)
  - `V2_SPEC.md` (implementation specification, 1704 lines)
  - `IMPLEMENTATION_PLAN.md` (build plan, 587 lines)

---

## 1. Overall assessment

The three documents form a coherent progression from concept to spec to plan. The core thesis is sound: build a thin PydanticAI extension library focused on Grail toolsets, policy-controlled tool exposure, and stable envelopes — without reimplementing any agent runtime. The adapter boundary pattern is well-reasoned and the phased build order is logical.

**Strengths:**
- Clear architectural boundaries — PydanticAI owns runtime, Grail owns constrained execution, v2 owns composition and policy
- Adapter layer is the single best design decision; it isolates all upstream coupling to one file per dependency
- Phased delivery is ordered correctly: foundation → differentiator (Grail) → composition (policy) → ergonomics (shim) → polish
- Concept-to-spec evolution is clean: `AgentBundle`, `GrailCapability`, `normalize.py`, `runtime.py`, and `approval.py` were correctly pruned
- CapabilityBundle deferral to post-v0.1.0 is the right call — it's the least differentiated abstraction

**Weaknesses:**
- Several unresolved technical details that could block implementation
- Some inconsistencies between documents that need reconciliation
- Error propagation strategy has conflicting specifications
- Event callback model has an async/sync tension
- Namespace stacking behavior is unspecified

---

## 2. Cross-document alignment

### 2.1 Intentional divergences (good)

These are cases where later documents correctly refined earlier ones:

| Topic | Concept | Spec/Plan | Assessment |
|---|---|---|---|
| `CapabilityBundle` | Included (§8.4) | Deferred to post-v0.1.0 | Correct — not differentiated enough for MVP |
| `AgentBundle` | Included (§11.2) | Dropped | Correct — premature convenience object |
| `GrailCapability` | In package structure (§15) | Dropped | Correct — capabilities can compose toolsets directly |
| `normalize.py` | In package structure (§15) | Dropped | Correct — normalization lives in envelopes and adapters |
| `approval.py` | In package structure (§15) | Dropped | Correct — approval via `ToolDefinition.kind`, not custom module |
| `runtime.py` | In package structure (§15) | Dropped | Correct — no custom runtime needed |
| Build phases | 6 phases (§18) | 5 phases | Correct — phases 5+6 merged into a single polish phase |
| Profile `output_schema` | `dict[str, Any]` | `output_type_ref: str` (dotted path) | Better — dotted path is more precise and doesn't duplicate PydanticAI |

### 2.2 Unintentional inconsistencies (need resolution)

| Topic | Document A | Document B | Recommendation |
|---|---|---|---|
| Python version | Spec: `>=3.12` (§1, §15) | Plan: `>=3.13` | **Use 3.13+** — confirmed in clarification session |
| Ruff line-length | Spec: `100` (§15) | Plan: `120`, existing pyproject: `120` | **Use 120** — matches existing config |
| Ruff target-version | Spec: `py312` (§15) | Plan: `py313` | **Use py313** — matches Python version decision |
| `ToolTraceEnvelope.input_preview` | Concept (§8.9) | Spec/Plan: `input_summary` | **Use `input_summary`** — spec is authoritative |
| `ToolTraceEnvelope.output_preview` | Concept (§8.9) | Spec/Plan: `output_summary` | **Use `output_summary`** — spec is authoritative |
| `ToolPolicy.max_tool_duration_ms` | Concept (§8.7) | Plan: `max_single_tool_duration_ms` | **Use `max_single_tool_duration_ms`** — plan is more precise |
| `pyproject.toml` build-system | Spec includes it (§15) | Plan omits it | **Include it** — required for packaging |
| `pytest-cov` | Existing pyproject includes it | Spec/Plan omit it | **Include it** — useful for coverage reporting |

### 2.3 Missing from plan (present in concept/spec)

| Item | Where specified | Impact |
|---|---|---|
| `_build_input_validator()` implementation | Spec §7.6 (mentioned, body `...`) | Medium — needed for `ToolsetTool.args_validator` |
| `_summarize_args()` and `_summarize_output()` helpers | Spec §7.6 (referenced) | Low — simple utility functions |
| `_node_to_event_kind()` and `_node_to_payload()` helpers | Spec §8 (referenced) | Medium — needed for Phase 5 `iter_events()` |
| `GrailOutputConfig` | Concept §9.5 | Low — optional output validation, can be post-v0.1.0 |
| `testing` preset for `GrailLimitsConfig` | Concept §9.4 mentions 4 presets | Low — only 3 presets in plan (strict/default/permissive) |

---

## 3. Technical analysis

### 3.1 Error propagation conflict

**The problem:** The spec and plan give conflicting guidance on how `GrailToolset.call_tool()` handles errors.

- Spec §14.2 (error boundary table): says `GrailToolset.call_tool()` should "return error string to PydanticAI"
- Spec §7.6 (code): `call_tool()` catches exception, builds `GrailErrorEnvelope`, then `raise`
- Plan §2.5: "On error: build `GrailErrorEnvelope`, emit event, append trace, re-raise"

These are fundamentally different behaviors:
- **Return error string**: PydanticAI sends the error text back to the model as a tool result. The model can try a different approach. No exception propagates.
- **Re-raise**: PydanticAI's tool execution handler catches it and may retry or abort the run.

**Recommendation:** Return error string to PydanticAI, NOT re-raise. This is the more PydanticAI-native pattern. PydanticAI's tool call lifecycle is designed to handle tool error results gracefully — the model sees the error and can adjust. Re-raising breaks this loop. The implementation should:

```python
async def call_tool(self, name, tool_args, ctx, tool) -> Any:
    try:
        result = await script.run(...)
        # ... trace success ...
        return result
    except Exception as exc:
        envelope = GrailErrorEnvelope.from_grail_error(...)
        # ... trace failure, emit event ...
        return f"Tool error ({envelope.error_category}): {envelope.message}"
```

The exception should only re-raise for truly unrecoverable situations (e.g., `require_grail()` failure, which wouldn't happen inside `call_tool`).

**Exception:** `_load_scripts()` at construction time SHOULD re-raise — load failures are not recoverable tool results. This part is correct in both spec and plan.

### 3.2 Namespace stacking

**The problem:** Both `GrailToolset` and `PolicyToolset` support namespace prefixing. If a `PolicyToolset(namespace="safe")` wraps a `GrailToolset(namespace="finance")`, the resulting tool name would be `safe__finance__risk_score`. Neither spec nor plan addresses this interaction.

**Recommendation:** Document the expected behavior and add a test case. Options:
1. **Allow stacking** — `safe__finance__risk_score`. Simple, predictable, but verbose.
2. **PolicyToolset skips namespace if inner toolset already namespaced** — requires introspection.
3. **Remove namespace from PolicyToolset entirely** — namespacing is a GrailToolset concern, policy is about filtering.

I recommend **option 3**: remove `namespace` from `PolicyToolset`. Namespacing is fundamentally about tool identity (a Grail concern), not about policy enforcement. PolicyToolset should filter and gate, not rename. If a user needs renaming, that's a separate concern.

If namespace must stay on PolicyToolset for other reasons, go with option 1 and document the stacking behavior explicitly.

### 3.3 `for_run()` lifecycle method

**The problem:** PydanticAI's `AbstractToolset` likely has a `for_run()` method that creates a per-run copy of the toolset with fresh state. Both `PolicyToolset` (which tracks `_call_count`) and `GrailToolset` (which accumulates `_tool_traces`) need per-run state reset. The plan mentions this in implementation notes (#5) but doesn't specify it in the class definitions.

**Recommendation:** Explicitly specify `for_run()` on both:

```python
# PolicyToolset
def for_run(self, ctx) -> PolicyToolset:
    """Create a per-run copy with reset call counter."""
    copy = PolicyToolset(self._inner.for_run(ctx), self._policy, ...)
    copy._call_count = 0
    return copy

# GrailToolset
def for_run(self, ctx) -> GrailToolset:
    """Create a per-run copy with fresh tool traces."""
    copy = ... # shallow copy, reset _tool_traces
    return copy
```

This should be promoted from an implementation note to a first-class design element, since incorrect `for_run()` behavior will cause budget counts to persist across runs and tool traces to accumulate indefinitely.

### 3.4 Async event callbacks

**The problem:** Event callbacks are typed as `Callable[[EventEnvelope], None]` (synchronous). But tool execution happens in async contexts. If a callback does any I/O (logging to a file, sending to a websocket, posting to a queue), it blocks the event loop.

**Recommendation:** Support both sync and async callbacks:

```python
EventCallback = Callable[[EventEnvelope], None] | Callable[[EventEnvelope], Awaitable[None]]
```

The `_emit()` helper can detect and handle both:

```python
async def _emit(self, kind: str, **payload: Any) -> None:
    if self._event_callback:
        event = EventEnvelope(...)
        result = self._event_callback(event)
        if asyncio.iscoroutine(result):
            await result
```

Alternatively, keep callbacks synchronous and document that they must not block. This is simpler and matches how most logging/telemetry callbacks work. I lean toward **keeping it simple and synchronous** for v0.1.0, with a note that async callback support can be added later.

### 3.5 `EventEnvelope.run_id` for toolset-level events

**The problem:** `EventEnvelope` requires a `run_id`, but toolset-level events (emitted during `_load_scripts()`, `get_tools()`) don't have a run context and therefore no run_id. The spec handles this with `run_id=""` in `GrailToolset._emit()`, which is a code smell.

**Recommendation:** Make `run_id` optional:

```python
class EventEnvelope(BaseModel, frozen=True):
    run_id: str | None = None  # None for toolset-level events outside a run
    ...
```

This is cleaner than empty string and lets consumers distinguish between "no run context" and "run context present." Toolset-level events like `grail.script.loaded` genuinely don't belong to a run — they happen at construction time.

### 3.6 `GrailToolsetConfig.paths` lacks glob support

**The problem:** The concept describes "Style B: folder/package = one toolset" (§9.2) where `tools/finance/*.pym` maps to a toolset. But `GrailToolsetConfig.paths` is `list[str]` with no glob expansion.

**Recommendation:** Add glob expansion in `_load_scripts()`:

```python
from pathlib import Path

def _resolve_paths(self) -> list[Path]:
    resolved = []
    for path_str in self._config.paths:
        path = Path(path_str)
        if '*' in path_str or '?' in path_str:
            resolved.extend(sorted(Path('.').glob(path_str)))
        else:
            resolved.append(path)
    return resolved
```

This is a small addition with high ergonomic value. Without it, users managing toolsets with many scripts have to enumerate every file.

### 3.7 Adapter boundary purity for envelopes

**The problem:** `envelopes.py` has `from_pydantic_ai()` and `from_pydantic_ai_result()` class methods that accept PydanticAI types (`RunUsage`, `AgentRunResult`). These are imported from `_adapters.pydantic_ai`, so the adapter rule is technically followed. But it means `envelopes.py` has a runtime dependency on pydantic-ai types, not just pydantic.

**Assessment:** This is acceptable. The alternative — putting these factory methods in a separate adapter or in `agent.py` — would split the envelope logic across files for minimal benefit. The current design is pragmatic: envelopes import from `_adapters`, never from `pydantic_ai` directly. The adapter boundary is respected.

No change recommended.

### 3.8 `StructuredAgent` as `BaseModel`

**The problem:** `StructuredAgent` extends `BaseModel` with `arbitrary_types_allowed=True` and stores the PydanticAI agent in a `PrivateAttr`. This means:
- `StructuredAgent.model_dump()` only serializes `profile`, not the agent
- `StructuredAgent.model_validate()` can't reconstruct the agent
- The agent isn't truly a "model" in the Pydantic sense — it's a runtime wrapper

**Assessment:** This is a valid design choice. The spec acknowledges it explicitly. The profile IS serializable and represents the declarative configuration. The agent is the runtime instantiation. Using BaseModel gives free validation on the profile and consistent model semantics for the serializable part.

**Minor recommendation:** Add a clear note in the class docstring that this model is not fully round-trippable via `model_dump()`/`model_validate()`. Users should use the factory to reconstruct agents from profiles.

### 3.9 Eager script loading failure mode

**The problem:** `GrailToolset._load_scripts()` loads ALL scripts eagerly at construction time and raises on the first failure. If a toolset has 10 scripts and script #3 has a syntax error, scripts #4-10 are never loaded. The user gets one error at a time.

**Recommendation:** Consider a "collect all errors" mode:

```python
def _load_scripts(self) -> None:
    errors: list[GrailErrorEnvelope] = []
    for path_str in self._config.paths:
        try:
            # ... load script ...
        except Exception as exc:
            errors.append(GrailErrorEnvelope.from_grail_error(path.stem, exc))
    if errors:
        # Raise with all errors collected
        raise ConfigError(f"Failed to load {len(errors)} script(s): ...")
```

This is especially useful during development when multiple scripts may have issues. The current fail-fast behavior is fine for production but frustrating during development.

This could be a constructor flag: `fail_fast: bool = True`. Default behavior stays fail-fast; developers can set `fail_fast=False` to collect all errors.

### 3.10 Missing `__init__.py` exports for `grail/`

**The problem:** The plan specifies `__init__.py` for `policy/` (§1.8) with explicit exports, but doesn't specify what `grail/__init__.py` exports.

**Recommendation:** Add to the plan:

```python
# grail/__init__.py
from structured_agents_v2._adapters.grail import require_grail
require_grail()

from .toolset import GrailToolset
from .config import GrailToolsetConfig, GrailLimitsConfig
from .manifest import GrailManifest, GrailInputManifest, GrailExternalManifest
from .externals import ExternalBindingRegistry, ExternalBindingEntry
from .errors import GrailErrorEnvelope

__all__ = [
    "GrailToolset",
    "GrailToolsetConfig",
    "GrailLimitsConfig",
    "GrailManifest",
    "GrailInputManifest",
    "GrailExternalManifest",
    "ExternalBindingRegistry",
    "ExternalBindingEntry",
    "GrailErrorEnvelope",
]
```

Note the `require_grail()` call at package level — importing `from structured_agents_v2.grail import GrailToolset` will fail early with a clear message if grail isn't installed.

---

## 4. Design recommendations

### 4.1 Add `ToolPolicy` class methods for presets

The plan has presets as module-level functions in `policy/presets.py`. Consider also adding them as class methods on `ToolPolicy`:

```python
class ToolPolicy(BaseModel, frozen=True):
    ...

    @classmethod
    def safe_default(cls) -> "ToolPolicy":
        from .presets import safe_default
        return safe_default()
```

This enables `ToolPolicy.safe_default()` in addition to `from structured_agents_v2.policy.presets import safe_default`. Both the concept (§16.1) and spec (§12.1) show `ToolPolicy.safe_default()` in usage examples, but the plan doesn't implement it this way.

### 4.2 Add pytest markers for optional dependency tests

```python
# conftest.py
import pytest

try:
    import grail
    HAS_GRAIL = True
except ImportError:
    HAS_GRAIL = False

requires_grail = pytest.mark.skipif(not HAS_GRAIL, reason="grail not installed")
```

This lets the test suite run partially even without grail installed, which matters for CI environments where grail might not be available.

### 4.3 Consider `GrailToolset` description extraction

The plan uses `manifest.description or f"Grail script: {manifest.script_name}"` as the tool description. Grail `.pym` scripts likely support docstrings or comments at the top of the file. The manifest extraction should attempt to pull a description from:
1. Script-level docstring (if Grail exposes it)
2. First comment line
3. Fallback to `f"Grail script: {manifest.script_name}"`

Tool descriptions are what the LLM sees when deciding which tool to call. Generic descriptions like "Grail script: simple_calc" give the model almost no information. This deserves explicit attention in the plan.

### 4.4 Type-check the adapter layer early

The adapter layer (`_adapters/pydantic_ai.py`) should be implemented and type-checked first, before any dependent code. The plan puts it in Phase 1, which is correct, but should emphasize that the **exact PydanticAI import paths must be confirmed** during this step. The plan acknowledges this ("to be confirmed against installed version") but should make it a blocking gate before proceeding to Phase 2.

If any adapter import paths are wrong, everything downstream is built on a faulty assumption.

### 4.5 Add `GrailToolset.reload()` method

For development workflows, being able to reload scripts without recreating the toolset would be valuable:

```python
def reload(self) -> None:
    """Reload all scripts from disk. Useful during development."""
    self._scripts.clear()
    self._manifests.clear()
    self._load_scripts()
```

This is a small addition with high ergonomic value for iterative `.pym` development. Not critical for v0.1.0 but worth considering.

### 4.6 Specify `ToolDefinition` field mapping explicitly

The spec shows `ToolDefinition` construction with fields like `kind`, `strict`, `defer_loading`, `return_schema`. These fields may not exist or may have different names in the installed PydanticAI version. The plan should call out that this is a high-risk area requiring verification against the actual PydanticAI API, and list the specific fields that need confirmation:

- `ToolDefinition.kind` — critical for approval/deferral mechanism
- `ToolDefinition.strict` — may not exist
- `ToolDefinition.defer_loading` — may not exist
- `ToolDefinition.return_schema` — may not exist
- `ToolsetTool.args_validator` — may have different signature

### 4.7 Consider `PolicyToolset` as a decorator pattern

The current design wraps toolsets: `PolicyToolset(inner_toolset, policy)`. But the factory applies policies by wrapping ALL toolsets in a loop:

```python
for policy_name in policy_names:
    policy = self._registry.get_policy(policy_name)
    toolsets = [PolicyToolset(ts, policy) for ts in toolsets]
```

If there are 2 policies and 3 toolsets, this creates 3 PolicyToolsets for the first policy, then wraps those 3 in 3 more for the second policy — 6 wrappers total, 2 layers deep. Each `get_tools()` call traverses 2 layers of filtering.

**Recommendation:** Consider merging multiple policies into a single `PolicyToolset`:

```python
# factory.py
merged_policy = merge_policies(policies)  # Combine into one
toolsets = [PolicyToolset(ts, merged_policy) for ts in toolsets]
```

Or apply multiple policies in a single wrapper that holds a list. This avoids the nesting and is more efficient. However, it requires defining policy merge semantics (most restrictive wins? first match wins?). If merge semantics are complex, the nesting approach is simpler and correct — just document the performance implication.

---

## 5. Risk assessment

### 5.1 High risk: PydanticAI API stability

The entire library depends on PydanticAI's `AbstractToolset` interface, `ToolDefinition` fields, `Agent.iter()` API, and `RunUsage`/`AgentRunResult` types. PydanticAI is version `>=0.1`, which signals pre-1.0 instability. The adapter layer mitigates this well, but the rate of breaking changes could still create maintenance burden.

**Mitigation:** Pin to specific PydanticAI version ranges in CI. Run adapter compatibility tests on PydanticAI upgrades before bumping.

### 5.2 High risk: `ToolDefinition.kind` for approval/deferral

This is the most significant unresolved design question. The plan says "exact mechanism TBD, verify against installed PydanticAI." If PydanticAI doesn't support marking tools as requiring approval via `ToolDefinition.kind`, the entire approval and deferral flow needs a fundamentally different approach.

**Mitigation:** Verify this BEFORE starting Phase 3. If the mechanism doesn't exist, consider:
- Using PydanticAI's deferred tools API directly (if it provides one)
- Implementing approval as a separate concern outside the toolset (e.g., in `StructuredAgent.run_enveloped()`)
- Deferring approval/deferral to post-v0.1.0

### 5.3 Medium risk: Grail API assumptions

The plan documents specific Grail API details (e.g., `GrailScript.inputs`, `ExternalSpec.is_async`, `LimitError` not being a subclass of `ExecutionError`). These were confirmed from Grail source in the previous session, but Grail is also pre-stable. Changes to Grail's `InputSpec`, `ExternalSpec`, or error hierarchy would affect the adapter and manifest layers.

**Mitigation:** The adapter layer handles this. Confirm API details against installed Grail version at the start of Phase 2.

### 5.4 Low risk: Over-abstraction creep

The concept document shows signs of wanting more abstraction than needed (`AgentBundle`, `GrailCapability`, `CapabilityBundle`). The spec and plan correctly pruned these. But during implementation, there's a temptation to re-introduce convenience wrappers. Stay disciplined about the "keep the core small" principle.

**Mitigation:** Every new class should pass the test: "Does this materially improve usability, stability, or policy enforcement compared to using PydanticAI directly?" If no, don't add it.

---

## 6. Testing gaps

### 6.1 Missing test scenarios

The plan lists good test cases but misses several important scenarios:

| Missing test | Why it matters |
|---|---|
| Multiple policies applied to one toolset (nesting) | Verifies correct filter composition |
| Policy with both allowlist AND denylist | Verifies precedence rules |
| GrailToolset with mixed valid/invalid scripts | Verifies error handling granularity |
| `run_enveloped()` when tool errors are returned (not raised) | Verifies RunEnvelope captures tool error traces |
| `EventEnvelope` ordering guarantees | Verifies sequence numbers are monotonic |
| Config loading with malformed YAML | Verifies `ConfigError` is raised |
| `ExternalBindingRegistry` with async/sync function mismatch | Verifies async external called from sync context |
| Empty toolset (no scripts) | Edge case: `GrailToolset` with empty `paths` |
| `StructuredAgentProfile` with empty `grail_toolset_names` and empty `policy_names` | Verifies factory handles vanilla agent |
| `ToolPolicy` with `max_tool_calls=0` | Edge case: should block all calls immediately |

### 6.2 Integration test infrastructure

The plan should specify how to run Grail-dependent tests in CI. Options:
- Install grail from git in CI (matches `uv` source config)
- Use pytest markers to skip grail tests when not installed
- Have separate CI jobs for "core only" and "full" test suites

---

## 7. Suggestions for plan improvements

### 7.1 Pre-work additions

The plan's pre-work section should also include:
1. `CLAUDE.md` with project conventions (adapter boundary rule, import discipline, test tier expectations)
2. `py.typed` marker file in `src/structured_agents_v2/` for PEP 561 compliance
3. `.github/workflows/` or equivalent CI config skeleton (even if empty initially)
4. Verify installed PydanticAI and Grail versions, document actual import paths

### 7.2 Phase 1 additions

Add to Phase 1:
- `_adapters/__init__.py` — should it be empty or export convenience functions?
- Verify all PydanticAI adapter imports against installed version (blocking gate)
- Define `__all__` for every module (plan mentions this only in Phase 5 polish)

### 7.3 Phase 2 additions

Add to Phase 2:
- `grail/__init__.py` exports (currently unspecified)
- Description extraction strategy for tool definitions
- Glob path expansion in `_resolve_paths()`
- `_build_input_validator()` implementation detail

### 7.4 Phase 3 additions

Add to Phase 3:
- `for_run()` implementation on `PolicyToolset` (promote from implementation note to spec)
- Namespace stacking behavior documentation
- Verification of `ToolDefinition.kind` mechanism (blocking gate — if this doesn't work, the phase needs redesign)

### 7.5 Phase 4 additions

Add to Phase 4:
- `output_type_ref` resolution implementation (importlib-based, with error handling)
- Factory behavior when `profile.model` is `None` and no `default_model` is set
- Registry `register_preset_policies()` convenience method to register all presets at once

---

## 8. Architectural questions for the implementer

These are questions that should be answered before or during implementation:

1. **Should `GrailToolset.call_tool()` return error strings or re-raise?** The spec is internally inconsistent. (Recommendation: return error strings — see §3.1)

2. **What happens when `PolicyToolset` wraps a namespaced `GrailToolset`?** Double prefix, single prefix, or error? (Recommendation: remove namespace from PolicyToolset — see §3.2)

3. **Should `EventEnvelope.run_id` be optional?** Toolset-level events don't have run context. (Recommendation: yes, make it `str | None = None` — see §3.5)

4. **Does PydanticAI's `ToolDefinition` actually have a `kind` field?** This is the mechanism for approval/deferral. If not, the entire Phase 3 approach needs rethinking. (Must be verified before Phase 3 starts.)

5. **Should `GrailToolset` support partial loading?** Fail-fast vs collect-all-errors. (Recommendation: add optional `fail_fast=True` parameter — see §3.9)

6. **How are multiple policies composed?** Nesting (current plan) or merging? (Current nesting approach is correct but document the performance implication — see §4.7)

7. **Should presets also be exposed as `ToolPolicy` class methods?** Concept and spec usage examples show `ToolPolicy.safe_default()`. (Recommendation: yes — see §4.1)

8. **What PydanticAI version are we actually targeting?** The plan says `>=0.1` but PydanticAI is currently at a much higher version. The adapter imports need to be verified against the real installed version.

---

## 9. Implementation priority adjustments

The current phase order is correct. Within phases, I'd suggest these priority adjustments:

### Phase 1: Adapter verification first

Before writing any envelope or policy code, verify every import in `_adapters/pydantic_ai.py` against the installed PydanticAI version. If imports are wrong, nothing else works. Spend the first hour of Phase 1 getting the adapter layer to import cleanly with all type annotations passing.

### Phase 2: Description extraction before toolset

Implement `GrailManifest.from_grail_script()` thoroughly — especially description extraction — before building the full `GrailToolset`. Good tool descriptions are critical for LLM tool selection. A tool with a bad description is a tool the model won't use correctly.

### Phase 3: `ToolDefinition.kind` verification first

Before writing any PolicyToolset code, verify the approval/deferral mechanism. If `ToolDefinition.kind` doesn't exist or works differently, redesign before implementing.

### Phase 5: Examples before polish

Write examples before doing lint/type-check polish. Examples are the best test of API ergonomics. If the examples feel awkward, the API needs adjustment — and it's better to discover that before spending time on polish.

---

## 10. Summary of recommendations

### Must-fix before implementation

1. Resolve error propagation conflict in `GrailToolset.call_tool()` (§3.1)
2. Add `for_run()` to PolicyToolset and GrailToolset specifications (§3.3)
3. Add `pyproject.toml` build-system section to plan (§2.2)
4. Make `EventEnvelope.run_id` optional (§3.5)
5. Verify `ToolDefinition.kind` mechanism before Phase 3 (§5.2)

### Should-fix during implementation

6. Add glob path expansion to `GrailToolsetConfig.paths` (§3.6)
7. Add `grail/__init__.py` exports to plan (§3.10)
8. Add pytest markers for grail-dependent tests (§4.2)
9. Add `ToolPolicy` class methods for presets (§4.1)
10. Address namespace stacking behavior (§3.2)

### Nice-to-have for v0.1.0

11. `GrailToolset.reload()` for development workflows (§4.5)
12. Partial loading mode (`fail_fast` parameter) (§3.9)
13. Async event callback support (§3.4)
14. Policy merge semantics for multi-policy composition (§4.7)
15. `GrailLimitsConfig.testing()` preset (§2.3)
