# REFACTORING_GUIDE.md — Add SGLang & llama.cpp via selectable per-engine plugins

**Project:** `11-sglang-provider-abstraction`
**Companion:** `SGLANG_ANALYSIS.md` (why) · `ARCHITECTURE_REVIEW.md` (the design decision this guide implements)
**Baseline:** `structured_agents` **v0.3.0**, working tree tip `ca89e3f` (package `src/structured_agents/`,
NOT the superseded `structured_agents_v2` of `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md`).

**Approach (owner-confirmed):** support all three engines as **selectable in-tree plugins** — one engine
per `Backend`, chosen by name. **Not** out-of-tree third-party discovery; **not** three engines live at
once. Each engine is a self-contained module that renders every constraint kind and declares its own
capabilities. vLLM stays the default and its wire bytes are preserved **exactly**; SGLang and llama.cpp
are added but their wire shapes are **doc-derived and unverified against any server in this repo** — they
ship labeled as such and the SGLang live path stays gated behind the GGUF-load blocker (SGLANG_ANALYSIS
§7 / R1).

This deliberately drops three pieces of the earlier proposal that the requirement does not need:
a neutral `ConstraintSpec` IR (redundant with the codecs), entry-point discovery, and a public
`Provider`/`Capabilities`/`load_provider`/`register_provider` surface. See `ARCHITECTURE_REVIEW.md`
§Q1/§Q2/§Q5 for the reasoning.

Conventions (`pyproject.toml`): line length 120, double quotes, `from __future__ import annotations`,
Python 3.13, ruff `E,F,I,UP,B`, `ty` for typing. Run everything through devenv:

```bash
devenv shell -- ruff check src tests
devenv shell -- ty check src
devenv shell -- pytest
```

---

## Target end-state (file map)

```
src/structured_agents/
  constraint.py       # MODIFIED: codecs drop wire(); gain a `kind` tag. WireSpec stays here.
  agent.py            # MODIFIED: Backend takes engine=; generic capability gate. BackendCaps removed.
  __init__.py         # MODIFIED: drop BackendCaps export. (WireSpec stays exported.)
  engine/             # NEW package — three plugins + an internal selector
    __init__.py       # NEW: internal _BUILTINS dict + select(name). NO entry points, NOT public.
    base.py           # NEW: Engine Protocol (internal).
    vllm.py           # NEW: reproduces today's exact bytes.
    sglang.py         # NEW: regex/ebnf fields; choice lowered to regex. Unverified.
    llama_cpp.py      # NEW: json_schema + GBNF only; regex/lora off. Unverified.
tests/
  test_constraint.py  # MODIFIED: replace the wire() golden with a `kind`/to_config neutrality test.
  test_engine.py      # NEW: vLLM golden bytes + engine dialects + caps gating + unknown-engine.
  test_live.py        # MODIFIED: select engine from env; neutral docstring.
pyproject.toml        # MODIFIED: move the ty unresolved-import override to engine/; neutral marker text.
README.md             # MODIFIED: wording.
```

No `providers/` package, no `ConstraintSpec`, no `[project.entry-points]` table.

---

## Phase 0 — Baseline & branch

```bash
git switch -c 11-sglang-provider-abstraction
devenv shell -- pytest            # capture a green baseline
```

The wire bytes you must preserve exactly (today's behavior, pinned by `tests/test_constraint.py:40-52`
and `.scratch/projects/09-constraint-codec-rewrite/REVIEW_SPIKES/bodies.json`):

| Constraint | `WireSpec` today |
|---|---|
| `Schema(M)` | `output_type=NativeOutput(M, strict=True)`, `extra_body={}` |
| `Regex(P)` | `output_type=str`, `extra_body={"structured_outputs": {"regex": P}}` |
| `Choice(a,b)` | `output_type=str`, `extra_body={"structured_outputs": {"choice": [a, b]}}` |
| `Grammar(G)` | `output_type=str`, `extra_body={"structured_outputs": {"grammar": G}}` |

The vLLM engine (Phase 2.2) must reproduce this table byte-for-byte; the golden test (Phase 5.2) proves it.

---

## Phase 1 — Constraint codecs stop knowing about vLLM (`constraint.py`)

Today each codec bakes the vLLM wire shape in `wire()` (`constraint.py:40-43,72-73,95-96,118-119`). Remove
that. Keep the concrete frozen dataclasses (`_Schema`/`_Regex`/`_Choice`/`_Grammar`) and their public
fields — the engines read those fields directly. Add a small `kind` tag so `Backend` can gate without
knowing wire shapes. **`WireSpec` stays in `constraint.py`** (engines produce it now).

### 1.1 Update the `Constraint` protocol

Drop `wire()`, add `kind`:

```python
from typing import Any, ClassVar, Protocol, cast, runtime_checkable  # add ClassVar

@runtime_checkable
class Constraint[T](Protocol):
    """A constrained-output codec. Describes *what* is constrained, not how it goes on the wire."""

    kind: str

    def parse(self, raw: Any) -> T: ...

    def check(self) -> None: ...

    def to_config(self) -> dict[str, Any]: ...
```

### 1.2 Delete each codec's `wire()`; add a `kind` ClassVar

For each concrete codec, add a `kind` class variable and remove the `wire()` method (and, in `_Schema`,
the `from pydantic_ai.output import NativeOutput` import that lived inside `wire()` — it moves to the
engine modules). `parse()`, `check()`, `to_config()` are untouched.

```python
@dataclass(frozen=True)
class _Schema[M: BaseModel]:
    kind: ClassVar[str] = "schema"
    model: type[M]
    strict: bool
    # remove wire(); keep parse/check/to_config

@dataclass(frozen=True)
class _Regex:
    kind: ClassVar[str] = "regex"
    pattern: str
    # remove wire(); keep __post_init__/parse/check/to_config

@dataclass(frozen=True)
class _Choice[T: str]:
    kind: ClassVar[str] = "choice"
    options: tuple[T, ...]
    # remove wire(); keep __post_init__/parse/check/to_config

@dataclass(frozen=True)
class _Grammar:
    kind: ClassVar[str] = "grammar"
    ebnf: str
    # remove wire(); keep __post_init__/parse/check/to_config
```

`ClassVar` is not a dataclass field, so it does not affect the frozen constructors. Optionally collapse
the hardcoded `"kind": ...` strings in each `to_config()` to `self.kind` — nice, not required.

> The concrete codecs stay underscore-private; the engine modules (same package) import them to `match`.
> This is a deliberate within-package coupling (`ARCHITECTURE_REVIEW.md` §5): it keeps full typing and
> avoids a redundant `ConstraintSpec`. If you later dislike importing `_`-names, add typed getters on
> each codec — but do **not** reintroduce an all-optional tagged-union IR.

**Verify (constraint.py compiles/lints in isolation; `agent.py` stays red until Phase 3):**

```bash
devenv shell -- ruff check src/structured_agents/constraint.py
```

---

## Phase 2 — The `engine/` package

Each engine is one small stateless module: a `name`, a `supports` set (constraint kinds it can render,
plus `"lora"` if it supports adapter selection), and a `render(constraint) -> WireSpec`. Selection is a
plain internal dict — no entry points, nothing public.

### 2.1 `src/structured_agents/engine/base.py`

```python
"""Engine plugins translate a neutral Constraint onto one inference engine's wire shape."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..constraint import Constraint, WireSpec


@runtime_checkable
class Engine(Protocol):
    """One inference engine's constrained-decoding dialect. Internal; not part of the public API."""

    name: str
    supports: frozenset[str]  # constraint kinds + "lora"

    def render(self, constraint: Constraint) -> WireSpec: ...
```

### 2.2 `src/structured_agents/engine/vllm.py` — reproduce today's bytes exactly

```python
"""vLLM engine: the reference dialect, byte-for-byte identical to the pre-refactor wire."""

from __future__ import annotations

from ..constraint import Constraint, WireSpec, _Choice, _Grammar, _Regex, _Schema
from ..errors import BackendCapabilityError


class VLLMEngine:
    name = "vllm"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})

    def render(self, constraint: Constraint) -> WireSpec:
        match constraint:
            case _Schema():
                from pydantic_ai.output import NativeOutput

                return WireSpec(output_type=NativeOutput(constraint.model, strict=constraint.strict))
            case _Regex():
                return WireSpec(output_type=str, extra_body={"structured_outputs": {"regex": constraint.pattern}})
            case _Choice():
                return WireSpec(output_type=str, extra_body={"structured_outputs": {"choice": list(constraint.options)}})
            case _Grammar():
                return WireSpec(output_type=str, extra_body={"structured_outputs": {"grammar": constraint.ebnf}})
        raise BackendCapabilityError(f"vllm engine cannot render constraint {type(constraint).__name__}.")
```

This is the only engine whose output is contractually pinned (the golden test). Do not touch its shapes.

### 2.3 `src/structured_agents/engine/sglang.py`

```python
"""SGLang engine: response_format json_schema; extra_body regex/ebnf; choice lowered to a regex.

UNVERIFIED: no constrained request is exercised against SGLang anywhere in this repo (deploy/sglang/
native/verify.sh only checks health/models/one chat; its README disclaims wire-shape compatibility).
Field names and choice lowering follow SGLang's published API (SGLANG_ANALYSIS §3, §12), not a live run.
"""

from __future__ import annotations

import re

from ..constraint import Constraint, WireSpec, _Choice, _Grammar, _Regex, _Schema
from ..errors import BackendCapabilityError


def _regex_alternation(options: tuple[str, ...]) -> str:
    # The alternation *semantic* (which options, in order) comes from the constraint; only the target
    # syntax (regex) is the engine's concern. Options are regex-escaped so literals stay literal.
    return "(" + "|".join(re.escape(option) for option in options) + ")"


class SGLangEngine:
    name = "sglang"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})

    def render(self, constraint: Constraint) -> WireSpec:
        match constraint:
            case _Schema():
                from pydantic_ai.output import NativeOutput

                return WireSpec(output_type=NativeOutput(constraint.model, strict=constraint.strict))
            case _Regex():
                return WireSpec(output_type=str, extra_body={"regex": constraint.pattern})
            case _Grammar():
                return WireSpec(output_type=str, extra_body={"ebnf": constraint.ebnf})
            case _Choice():
                return WireSpec(output_type=str, extra_body={"regex": _regex_alternation(constraint.options)})
        raise BackendCapabilityError(f"sglang engine cannot render constraint {type(constraint).__name__}.")
```

> LoRA: SGLang also accepts the `base:adapter` model-name syntax, which matches the library's current
> `spec.adapter or default_model` behavior (`agent.py:62`) — so no special adapter handling is needed.
> If a deployment ever needs the request-body form, inject `{"lora_path": adapter}` here for the sglang
> path only; do **not** add a public adapter protocol (`Settings.extra_body` already exists as a seam).

### 2.4 `src/structured_agents/engine/llama_cpp.py`

```python
"""llama.cpp engine: json_schema + GBNF grammar only. No regex, no LoRA over the OpenAI API.

UNVERIFIED: llama.cpp does not implement vLLM's XGrammar extension (deploy/llama-cpp/native/verify.sh:3),
and its verify.sh exercises no grammar surface at all. GBNF is close to but not the same dialect as the
EBNF that XGrammar accepts, so the Grammar constraint's EBNF is passed through WITHOUT a parity claim.
"""

from __future__ import annotations

from ..constraint import Constraint, WireSpec, _Choice, _Grammar, _Schema
from ..errors import BackendCapabilityError


def _gbnf_alternation(options: tuple[str, ...]) -> str:
    # Escape backslashes first, then quotes, so GBNF string literals stay well-formed.
    quoted = " | ".join('"' + option.replace("\\", "\\\\").replace('"', '\\"') + '"' for option in options)
    return f"root ::= {quoted}"


class LlamaCppEngine:
    name = "llama_cpp"
    supports = frozenset({"schema", "choice", "grammar"})  # no regex, no lora

    def render(self, constraint: Constraint) -> WireSpec:
        match constraint:
            case _Schema():
                from pydantic_ai.output import NativeOutput

                return WireSpec(output_type=NativeOutput(constraint.model, strict=constraint.strict))
            case _Grammar():
                return WireSpec(output_type=str, extra_body={"grammar": constraint.ebnf})  # GBNF, not EBNF-parity
            case _Choice():
                return WireSpec(output_type=str, extra_body={"grammar": _gbnf_alternation(constraint.options)})
        raise BackendCapabilityError("llama_cpp engine does not support regex constraints.")
```

Regex is unreachable here because `caps` gate it out at `Backend.build` (Phase 3), but the final `raise`
keeps `render` honest if called directly. (This fixes the earlier draft's GBNF escaping, which missed
backslashes.)

### 2.5 `src/structured_agents/engine/__init__.py` — internal selector (no entry points)

```python
"""Selectable inference-engine plugins. Built-ins only; no out-of-tree discovery."""

from __future__ import annotations

from ..errors import ConfigError
from .base import Engine
from .llama_cpp import LlamaCppEngine
from .sglang import SGLangEngine
from .vllm import VLLMEngine

_BUILTINS: dict[str, Engine] = {
    "vllm": VLLMEngine(),
    "sglang": SGLangEngine(),
    "llama_cpp": LlamaCppEngine(),
}


def select(name: str) -> Engine:
    """Resolve a built-in engine by name."""
    try:
        return _BUILTINS[name]
    except KeyError:
        raise ConfigError(f"Unknown engine {name!r}.") from None


__all__ = ["Engine", "select"]
```

Engines are stateless, so module-level singletons are fine. If you ever need out-of-tree engines, add an
entry-point fallback inside `select` behind the same signature — a backward-compatible follow-up, not
part of this change.

**Verify:**

```bash
devenv shell -- ruff check src/structured_agents/engine
```

---

## Phase 3 — Rewire `Backend` (`agent.py`)

Replace the vLLM-shaped gate (`agent.py:53-58`) with a generic, engine-driven one. Behavior is identical
when `engine="vllm"` (the default). Remove `BackendCaps` entirely — nothing in the repo uses it
(`grep` confirms it appears only in `agent.py` and `__init__.py`), and it is not carried forward under a
misleading alias.

### 3.1 Imports; delete `BackendCaps`

- Add `from .engine import Engine, select`.
- Delete `class BackendCaps(BaseModel)` (`agent.py:38-40`).
- Remove `from pydantic import BaseModel` (agent.py used it *only* for `BackendCaps`; confirm no other
  use before deleting the import).

### 3.2 New `Backend.__init__`

```python
class Backend:
    """The sole importer of pydantic_ai.models.openai. Builds durable agents against a selected engine."""

    def __init__(self, *, engine: str | Engine = "vllm", base_url: str = "http://localhost:8000/v1",
                 api_key: str = "sk-none", default_model: str = "test",
                 http_client: httpx.AsyncClient | None = None, model: Any | None = None) -> None:
        self.engine = engine if not isinstance(engine, str) else select(engine)
        self.base_url, self.api_key, self.default_model = base_url, api_key, default_model
        self.client, self.model = http_client, model
```

(`caps` is gone; capabilities now come from the selected engine.)

### 3.3 New `Backend.build`

```python
    def build[T](self, spec: AgentSpec[T]) -> Agent[T]:
        constraint = spec.constraint
        if constraint.kind not in self.engine.supports:
            raise BackendCapabilityError(
                f"Agent {spec.name!r} requires {constraint.kind} constraints, "
                f"unsupported by engine {self.engine.name!r}."
            )
        if spec.adapter and "lora" not in self.engine.supports:
            raise BackendCapabilityError(
                f"Agent {spec.name!r} requires LoRA, unsupported by engine {self.engine.name!r}."
            )
        constraint.check()
        wire = self.engine.render(constraint)
        settings = {k: v for k, v in spec.settings.__dict__.items() if v is not None and k != "extra_body"}
        settings["extra_body"] = {**spec.settings.extra_body, **wire.extra_body}
        model = cast(Model[Any], self.model) if self.model is not None else OpenAIChatModel(
            spec.adapter or self.default_model,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key, http_client=self.client),
        )
        return Agent(spec, DBOSAgent(PydanticAgent(  # type: ignore[no-matching-overload]
            model, output_type=wire.output_type, model_settings=settings,
            instructions=spec.instructions, name=spec.name), name=spec.name))
```

Changes vs. the original `build` (`agent.py:53-68`):
- `spec.constraint.wire()` → `self.engine.render(constraint)`.
- `structured_outputs`-keyed gate → `constraint.kind not in self.engine.supports`. Note this now also
  "gates" `schema` (every engine lists it in `supports`), which is correct and harmless — today schema
  is ungated because its `extra_body` is empty.
- The adapter→model-name mapping (`spec.adapter or self.default_model`) is **unchanged** — vLLM and
  SGLang both accept an adapter name in the model field, so no `adapter_wire` seam is needed.
- DBOS wrapping, settings filtering, and the `self.model` test-injection path are unchanged.

**Verify:**

```bash
devenv shell -- ruff check src && devenv shell -- ty check src
```

---

## Phase 4 — Exports (`__init__.py`)

- Line 3: drop `BackendCaps` from the `from .agent import ...` line.
- `__all__`: remove `"BackendCaps"`.
- **Keep** `WireSpec` exported (still a real public type: the engine render result).
- Do **not** add `Engine`, `select`, `Provider`, `Capabilities`, `ConstraintSpec`, `load_provider`, or
  `register_provider`. The only new public capability is the `engine=` string argument to `Backend`.
  (Optional: export `Engine` purely as a typing aid if callers want to annotate a custom engine object
  passed to `Backend(engine=...)`. Skip unless asked — keep the surface minimal.)

**Verify:**

```bash
devenv shell -- python -c "import structured_agents as s; b = s.Backend(engine='sglang'); print(b.engine.name)"
devenv shell -- python -c "import structured_agents as s; s.Backend(engine='nope')" 2>&1 | grep -q "Unknown engine" && echo OK
```

---

## Phase 5 — Tests

### 5.1 Update `tests/test_constraint.py`

`wire()` is gone, so replace `test_wire_shapes_match_the_verified_table` (and the `_wire` helper plus its
`NativeOutput`/`WireSpec` imports at `tests/test_constraint.py:9,56`) with a provider-neutral check of the
codec identity. The wire-byte assertions move to `test_engine.py` (5.2).

```python
from structured_agents import Choice, Grammar, Regex, Schema


def test_constraints_carry_a_neutral_kind_and_config() -> None:
    assert Schema(Person).kind == "schema"
    assert Regex(r"\d{4}").kind == "regex"
    assert Choice("keep", "skip").kind == "choice"
    assert Grammar('root ::= "a"').kind == "grammar"
    assert Choice("keep", "skip").to_config() == {"kind": "choice", "options": ["keep", "skip"]}
```

Keep the existing `parse`/round-trip/`__post_init__` tests unchanged (they never touched `wire()`).

### 5.2 New `tests/test_engine.py` — golden bytes, dialects, caps gating, selection

```python
from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai.output import NativeOutput

from structured_agents import AgentSpec, Backend, Choice, Grammar, Regex, Schema
from structured_agents.engine import select
from structured_agents.errors import BackendCapabilityError, ConfigError


class Person(BaseModel):
    name: str


def test_vllm_bytes_are_unchanged() -> None:
    """Regression guard: the vLLM engine must reproduce the pre-refactor wire exactly."""
    vllm = select("vllm")
    assert vllm.render(Regex(r"\d{4}")).extra_body == {"structured_outputs": {"regex": r"\d{4}"}}
    assert vllm.render(Choice("keep", "skip")).extra_body == {"structured_outputs": {"choice": ["keep", "skip"]}}
    assert vllm.render(Grammar('root ::= "a" | "b"')).extra_body == {
        "structured_outputs": {"grammar": 'root ::= "a" | "b"'}
    }
    schema_wire = vllm.render(Schema(Person))
    assert isinstance(schema_wire.output_type, NativeOutput)
    assert schema_wire.extra_body == {}


def test_sglang_dialect() -> None:
    sglang = select("sglang")
    assert sglang.render(Regex(r"\d{4}")).extra_body == {"regex": r"\d{4}"}
    assert sglang.render(Grammar('root ::= "a"')).extra_body == {"ebnf": 'root ::= "a"'}
    assert sglang.render(Choice("a.b", "c")).extra_body == {"regex": r"(a\.b|c)"}  # escaped alternation
    assert isinstance(sglang.render(Schema(Person)).output_type, NativeOutput)


def test_llama_cpp_narrow_caps() -> None:
    llama = select("llama_cpp")
    assert "regex" not in llama.supports and "lora" not in llama.supports
    assert llama.render(Grammar('root ::= "a"')).extra_body == {"grammar": 'root ::= "a"'}
    assert llama.render(Choice("a", "b")).extra_body == {"grammar": 'root ::= "a" | "b"'}
    with pytest.raises(BackendCapabilityError):
        llama.render(Regex(r"\d"))


def test_backend_gate_rejects_unsupported_constraint() -> None:
    backend = Backend(engine="llama_cpp", http_client=httpx.AsyncClient())
    with pytest.raises(BackendCapabilityError, match="regex"):
        backend.build(AgentSpec("r", Regex(r"\d"), "x"))


def test_backend_gate_rejects_lora_when_unsupported() -> None:
    backend = Backend(engine="llama_cpp", http_client=httpx.AsyncClient())
    with pytest.raises(BackendCapabilityError, match="LoRA"):
        backend.build(AgentSpec("s", Schema(Person), "x", adapter="my-lora"))


def test_unknown_engine_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown engine"):
        select("does-not-exist")
```

> `test_vllm_bytes_are_unchanged` is the regression guard for the whole refactor. If it passes, existing
> vLLM behavior is preserved.

### 5.3 `tests/test_agent.py` — no change

`tests/test_agent.py:25` builds `Backend(http_client=...)` with no `engine=` argument, so it defaults to
`"vllm"` and the `Schema` render path is byte-identical. It should pass **unchanged**. (Confirm; do not
edit.)

**Verify (full suite; live tests skip without `SAV_LIVE=1`):**

```bash
devenv shell -- pytest
```

---

## Phase 6 — Live conformance (opt-in)

Generalize `tests/test_live.py` to select an engine from the environment, turning the vLLM-only cutover
suite into a cross-engine conformance suite. Minimal change:

- Add near the other env reads (`tests/test_live.py:38-41`):
  ```python
  ENGINE = os.environ.get("LLM_ENGINE", "vllm")
  ```
- Change the backend construction (`tests/test_live.py:49`) to:
  ```python
  backend = Backend(engine=ENGINE, base_url=BASE_URL, api_key=API_KEY, default_model=MODEL)
  ```
- Update the module docstring (`:1`) and skip message (`:35`) to say "the configured inference endpoint"
  instead of "vLLM".

Run against each server independently:

```bash
# vLLM (proven)
SAV_LIVE=1 LLM_ENGINE=vllm   LLM_BASE_URL=http://127.0.0.1:8000/v1 devenv shell -- pytest tests/test_live.py
# SGLang — ONLY once the server actually loads the model (SGLANG_ANALYSIS §7 / R1)
SAV_LIVE=1 LLM_ENGINE=sglang LLM_BASE_URL=http://127.0.0.1:8002/v1 devenv shell -- pytest tests/test_live.py
```

> **Gate (R1):** do not expect the SGLang live run to pass until the deploy layer can serve the target
> model (the prior spike shows SGLang fails to load the Gemma-4 GGUF *before inference*). Until then,
> SGLang and llama.cpp are verified only by the unit/golden tests in Phase 5.

---

## Phase 7 — Docs, marker & ty override cleanup

1. `README.md:4` — change "on DBOS and vLLM/XGrammar." to
   "on DBOS and OpenAI-compatible engines (vLLM, SGLang, llama.cpp) with XGrammar-family constrained
   decoding."
2. `pyproject.toml:46` — marker text:
   `"live: hits the configured inference server (skipped unless SAV_LIVE=1)."`
3. **ty override — move, don't delete.** `pyproject.toml:51-55` ignores `unresolved-import` for
   `constraint.py` because of its in-method `from pydantic_ai.output import NativeOutput`. That import
   now lives in `engine/vllm.py`, `engine/sglang.py`, `engine/llama_cpp.py`. Repoint the override:
   ```toml
   [[tool.ty.overrides]]
   include = ["src/structured_agents/engine/**"]

   [tool.ty.overrides.rules]
   unresolved-import = "ignore"
   ```
   Leave the separate `agent.py` `no-matching-overload` override as-is. Then confirm `ty check src` is
   green; if ty resolves `pydantic_ai.output` fine without the override, drop it entirely.
4. (Optional) `tests/live_crash_worker.py` artifact names `raw-vllm-*` → `raw-llm-*` for neutrality.

**Final verify:**

```bash
devenv shell -- ruff check src tests
devenv shell -- ty check src
devenv shell -- pytest
```

---

## Commit sequencing (suggested)

| Commit | Contents | Green? |
|---|---|---|
| 1 | Phase 1 (codecs drop `wire()`, gain `kind`) + Phase 2 (engine pkg) | engines unit-testable; `agent.py` still red |
| 2 | Phase 3 (`Backend` rewire) + Phase 4 (exports) | full suite green |
| 3 | Phase 5 (tests, incl. golden guard) | green + regression guard |
| 4 | Phase 6 (live parametrize) + Phase 7 (docs/marker/ty) | green |

Keep commits 1–2 together if you prefer a single always-green step; the golden test in commit 3 is what
proves non-regression.

---

## Rollback / safety notes

- **Behavior-preserving for vLLM by construction:** `VLLMEngine.render` emits the exact bytes asserted by
  `test_vllm_bytes_are_unchanged`. If that test fails, stop — the dialect drifted.
- No change touches `plane.py`, `authority.py`, `approval.py`, `errors.py`, or `integrations/fornix.py`
  (verified LLM-neutral).
- Default `engine="vllm"` means every existing caller and serialized `AgentSpec` behaves identically with
  no code change (`AgentSpec` carries no engine identity; engine is a `Backend` construction choice).
- SGLang/llama.cpp engines ship "dark": selectable but never the default, so merging them cannot regress
  the proven path.

## What this refactor deliberately does NOT do

- It does **not** add `ConstraintSpec`, entry-point discovery, a public engine registry, or a
  `Provider`/`Capabilities` public API — the requirement is *select one of three built-ins*, not
  out-of-tree extensibility (`ARCHITECTURE_REVIEW.md` §Q1/§Q2/§Q5).
- It does **not** try to compose with `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md` — that guide targets the
  superseded `structured_agents_v2` @ v0.2.0 (`ClosedBackend`/`provider_extra`/`AgentProfile` do not
  exist in this v0.3.0 tree). The engine seam lands directly on the current `Backend`.
- It does **not** make SGLang the default, nor claim SGLang/llama.cpp wire shapes work — they are
  doc-derived and unverified in this repo, and the SGLang live path stays gated on R1.
- It does **not** fix the SGLang GGUF serving blocker (upstream Transformers; tracked in
  `08-unsloth-gemma4-gguf-compatibility/`), nor add streaming, tool-calling, or non-OpenAI transports.
