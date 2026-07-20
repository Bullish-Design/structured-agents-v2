# SGLANG_ANALYSIS.md

**Project:** `11-sglang-provider-abstraction`
**Date:** 2026-07-19
**Author:** investigation pass (read-only; no code changed)
**Question:** Fully replace vLLM with SGLang in the `structured-agents` library — *or* abstract the
LLM provider into a plugin system that supports vLLM, SGLang, and llama.cpp simultaneously.

---

## 0. TL;DR

1. **The library is barely coupled to vLLM.** The only vLLM-specific code in `src/structured_agents/`
   is the structured-output *wire shape* emitted by four constraint codecs in `constraint.py`
   (`extra_body={"structured_outputs": {...}}`) plus two capability flags in `agent.py`. Everything
   else (transport, durability, authority, approval, plane) is engine-neutral OpenAI-over-HTTP + DBOS.
   A **direct vLLM→SGLang swap in the library is ~1 file of real change** (`constraint.py`), plus caps
   and docs.

2. **The hard part is not the library — it is the server.** Per the prior spike
   (`08-unsloth-gemma4-gguf-compatibility`, `artifacts/sglang-gemma4-spike/…`), **SGLang cannot yet
   load the production GGUF** (Unsloth Gemma 4 12B QAT `UD-Q4_K_XL`). It fails *before weight load* on
   an upstream Transformers GGUF-config converter bug. So a *full operational cutover* replacing vLLM is
   **currently blocked at the serving layer**, independent of any library change. SGLang's
   structured-output surface has **never been runtime-verified** against this model.

3. **Recommendation: build the multi-engine abstraction (Option B), not the throwaway swap (Option A) —
   but in the *narrowed* form settled in `ARCHITECTURE_REVIEW.md`.** Keep the good idea (each engine is a
   self-contained plugin module; the constraint describes *what*, the engine decides *how*), but drop the
   machinery the requirement does not need: **no** neutral `ConstraintSpec` IR (redundant with the
   codecs), **no** entry-point discovery, **no** public `Provider`/`Capabilities`/`load_provider`/
   `register_provider` surface. The requirement is *select one of three built-in engines per `Backend`*,
   not out-of-tree extensibility and not three engines at once. vLLM stays the proven default; SGLang and
   llama.cpp are added but ship **labeled unverified** (see §3.1, §9).

> **Correction (verified against the tree):** an earlier draft of this section claimed the design should
> "compose with the closed-backend refactor" and ride its `provider_extra` seam. That is wrong.
> `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md` targets the **superseded** `structured_agents_v2` @ v0.2.0
> (`ClosedBackend`, `provider_extra`, `AgentProfile`, `StrictConfig` — none exist in the current v0.3.0
> `structured_agents` tree; `grep` confirms zero hits). The engine seam lands **directly on the current
> `Backend`**; there is no `provider_extra` to fold into.

The rest of this document is the evidence and the concrete design.

---

## 1. What "vLLM integration" actually means in this repo

There are **three distinct layers**, and only the first is what a library refactor touches:

| Layer | Location | vLLM coupling | In scope for a library refactor? |
|---|---|---|---|
| **Library** (the shipped package) | `src/structured_agents/` | Thin — one wire shape + two caps | **Yes** — this is the refactor surface |
| **Deployment** (how a server is launched) | `deploy/vllm/`, `deploy/sglang/`, `deploy/llama-cpp/` | Heavy, but per-engine and already separated | No (already engine-specific dirs) |
| **Artifacts / notes** (spikes, benchmarks) | `artifacts/`, `.scratch/` | Historical | No |

The library **does not import vllm**. It talks to any OpenAI-compatible `/v1` endpoint over `httpx`
via `pydantic-ai`'s `OpenAIChatModel`. "vLLM" enters the library only as (a) a specific
`extra_body` JSON shape for regex/choice/grammar constraints, and (b) two capability booleans that
assume vLLM semantics. That is the entire coupling.

---

## 2. Exhaustive inventory of vLLM coupling in the library

Verified by reading every file under `src/structured_agents/` and grepping the tree.

### 2.1 Real (behavioral) coupling — must change

| # | File:line | What it is | vLLM-specific? | Notes |
|---|---|---|---|---|
| C1 | `constraint.py:73` | `Regex.wire()` → `extra_body={"structured_outputs": {"regex": pattern}}` | **Yes** | SGLang wants `{"regex": pattern}` (un-nested) |
| C2 | `constraint.py:96` | `Choice.wire()` → `extra_body={"structured_outputs": {"choice": [...]}}` | **Yes** | SGLang has **no `choice`** param — must emulate |
| C3 | `constraint.py:119` | `Grammar.wire()` → `extra_body={"structured_outputs": {"grammar": ebnf}}` | **Yes** | SGLang wants `{"ebnf": ebnf}` |
| C4 | `agent.py:55-56` | `build()` gate: `if wire.extra_body.get("structured_outputs") and not caps.xgrammar: raise` | **Yes** | Gate keys off the *vLLM* dict shape |
| C5 | `agent.py:38-40` | `BackendCaps(xgrammar=True, lora=True)` defaults | Partly | "xgrammar" is a vLLM-ism; SGLang calls it `grammar-backend`, llama.cpp has neither |

### 2.2 Portable (already engine-neutral) — no change needed

| # | File:line | What it is | Why it is fine |
|---|---|---|---|
| P1 | `constraint.py:40-43` | `Schema.wire()` → `NativeOutput(model, strict)` | Compiles to OpenAI `response_format:{type:"json_schema"}`, which **vLLM, SGLang, and llama.cpp all support**. Verified request body: `.scratch/projects/09-constraint-codec-rewrite/REVIEW_SPIKES/bodies.json`. |
| P2 | `constraint.py:48-56, 126-134` | `check()` uses `xgrammar` to *pre-compile* schema/grammar | Compile-time validation only; `xgrammar` is the default grammar backend of **both** vLLM and SGLang, and the import is optional (`except ImportError: return`). Portable. |
| P3 | `agent.py:44-68` | `Backend` builds `OpenAIChatModel` + `OpenAIProvider(base_url=…)` | Pure OpenAI-compatible client. Points at any `/v1`. Engine-neutral. |
| P4 | `agent.py:62-65` | `spec.adapter or default_model` → OpenAI `model` field | LoRA/model selection by name. vLLM and SGLang both accept a model/adapter name here (see §4). |
| P5 | `config.py` (whole) | Config→constraint factories + entry-point discovery | Builds the same constraints; no engine shape. Already has a plugin pattern (`structured_agents.constraints` entry-point group, `config.py:74`). |
| P6 | `plane.py`, `authority.py`, `approval.py`, `errors.py`, `integrations/fornix.py` | Durable plane, authorization, approval, effects | **Zero** LLM coupling (grep confirmed). |

### 2.3 Cosmetic / documentation coupling — trivial

| # | Location | Text |
|---|---|---|
| D1 | `README.md:4` | "…on DBOS and vLLM/XGrammar." |
| D2 | `pyproject.toml:46` | pytest marker: `"live: hits the configured vLLM server…"` |
| D3 | `tests/test_live.py:1,35,137` | Docstrings/skip messages say "vLLM" |
| D4 | `tests/live_crash_worker.py:76-78` | Artifact filenames `raw-vllm-request.json` etc. |

**Conclusion of the inventory:** the entire behavioral swap is **C1–C5**, i.e. one codec file plus a
capability gate. Everything else is either already neutral or cosmetic.

---

## 3. The structured-output wire gap (verified)

This is the crux of any replacement. All three engines are OpenAI-compatible but express *constrained
decoding* differently.

> **Verification status (important):** the **vLLM** column below is the only one exercised end-to-end in
> this repo — `deploy/vllm/verify.sh` sends live json_schema/regex/choice/grammar requests and asserts
> the outputs. The **SGLang** and **llama.cpp** columns are **doc-derived, not repo-verified**: their
> `verify.sh` scripts only check health/models/one plain chat, and `deploy/sglang/native/README.md`
> explicitly states its xgrammar backend "does not establish compatibility with … the library's
> `structured_outputs` wire shape." So the SGLang/llama.cpp field names, choice lowering, and GBNF
> handling below follow published APIs (§12) and must ship **labeled unverified** until a live conformance
> run (Phase 6 of `REFACTORING_GUIDE.md`), which for SGLang is gated on the GGUF-load blocker (§7 / R1).

### 3.1 Per-constraint mapping

| Constraint | vLLM (what the library emits today) | SGLang | llama.cpp server |
|---|---|---|---|
| **Schema / JSON** | `response_format:{type:json_schema, json_schema:{…, strict:true}}` (via `NativeOutput`) | **Same** `response_format` json_schema | `response_format` json_schema **or** `json_schema` field |
| **Regex** | `extra_body={"structured_outputs":{"regex":P}}` | `extra_body={"regex":P}` | **No native regex** (must convert to GBNF) |
| **Choice** | `extra_body={"structured_outputs":{"choice":[…]}}` | **No `choice`** → emulate as `extra_body={"regex": "opt1|opt2|…"}` (regex-escaped, alternated) or an EBNF alternation | **No native choice** → GBNF alternation |
| **Grammar (EBNF)** | `extra_body={"structured_outputs":{"grammar":G}}` | `extra_body={"ebnf":G}` | `grammar=<GBNF>` (GBNF ≈ EBNF but not identical dialect) |

Key facts:
- SGLang: JSON Schema via `response_format`; **regex** and **EBNF** via `extra_body` fields named
  `regex` / `ebnf`; default grammar backend is **XGrammar** (`--grammar-backend xgrammar`, also
  `outlines`/`llguidance`). **Only one** of `json_schema`/`regex`/`ebnf` may be set per request — the
  library already uses exactly one constraint per agent, so this is not a limitation.
- SGLang has **no `choice` parameter** in its OpenAI server. The library's `Choice` constraint must be
  lowered to a regex alternation (`"(a|b|c)"` with each option `re.escape`-d) or an EBNF rule. This is a
  semantic translation, not a passthrough — the most important non-trivial part of a SGLang provider.
- llama.cpp's server implements **GBNF grammars** and JSON-schema, but **not** vLLM's XGrammar
  `structured_outputs` extension and **not** regex. The repo already documents this:
  `deploy/llama-cpp/native/verify.sh:3` — *"vLLM verifier because llama.cpp does not implement vLLM's
  XGrammar extension."* A llama.cpp provider therefore has a **narrower capability set** and would need
  regex/choice→GBNF conversion (or to reject those constraints).

### 3.2 Why this matters for the design

The vLLM wire shape is currently **baked into the constraint codecs** (`constraint.py`). That is the
single design decision that makes the library "a vLLM library." The fix — for either option — is to
stop letting the *constraint* decide the *provider dialect*. The constraint should describe *what* is
constrained; a provider should decide *how* to put it on the wire.

---

## 4. LoRA / model-identity differences

- **vLLM native profile** (`deploy/vllm/native/serve.sh:16-19`): LoRA is **prohibited** (GGUF weights
  don't support LoRA in vLLM). The vLLM *container* profile (`deploy/vllm/entrypoint.sh:54-61`) serves
  each adapter as a distinct OpenAI **model id** (`--lora-modules name=path`), selected via the `model`
  field — which is exactly how the library's `spec.adapter` works (`agent.py:62`).
- **SGLang** selects LoRA either by `model:"base:adapter"` syntax **or** a `lora_path` field in the
  request body (newer releases). The `base:adapter` model-name syntax is drop-in compatible with the
  library's current `adapter`-as-model-name approach; `lora_path` would need a provider that injects it
  into `extra_body`.
- **llama.cpp**: no multi-LoRA-over-OpenAI story here; `caps.lora=False`.

**Implication:** `caps.lora` must become **per-provider**, and a SGLang provider *may* need to translate
`adapter` into a `lora_path` extra_body field rather than a model name. This is a provider concern, not
a constraint concern.

---

## 5. Option A — Direct replacement (vLLM → SGLang)

**Goal:** the library speaks SGLang, vLLM support is dropped.

### 5.1 Concrete changes
1. `constraint.py`
   - `Regex.wire()` → `extra_body={"regex": self.pattern}`
   - `Grammar.wire()` → `extra_body={"ebnf": self.ebnf}`
   - `Choice.wire()` → `extra_body={"regex": "(" + "|".join(re.escape(o) for o in options) + ")"}`
     (or an EBNF alternation). **Parsing is unaffected** — `Choice.parse()` still checks membership.
   - `Schema.wire()` unchanged (`NativeOutput` / `response_format`).
2. `agent.py`
   - `build()` gate (C4): key off a provider-neutral notion of "needs constrained decoding" instead of
     the literal `structured_outputs` dict key (which will no longer exist).
   - `BackendCaps`: rename/re-mean `xgrammar` → e.g. `grammar` (or keep the name, redefine as
     "grammar-backend available").
3. Docs/markers: D1–D4.
4. Deploy default flips to `deploy/sglang/native/` (already exists; already sets
   `--grammar-backend xgrammar`).

### 5.2 Effort / risk
- **Effort:** small — essentially one codec file + caps + docs. A day of coding.
- **Risk — HIGH at the operational layer, LOW at the code layer.** The code change is trivial and
  testable in isolation, but:
  - The prior spike proves **SGLang cannot currently serve the production GGUF** (see §7). Dropping
    vLLM would leave the project with **no working production endpoint** for the target model.
  - Loses the only **runtime-verified** structured-output path (vLLM `verify.sh` exercises json_schema,
    regex, choice, EBNF end-to-end; SGLang's `verify.sh` tests only health/models/one chat).
  - Throwaway: if you later want vLLM or llama.cpp back, you re-do the work.

**Verdict:** Option A is a false economy. It is barely cheaper than Option B and strictly worse.

---

## 6. Option B — Engine plugin abstraction (recommended, narrowed form)

**Goal:** the library is engine-neutral; vLLM, SGLang, and llama.cpp are interchangeable plugin modules
that each (a) render a constraint onto their own wire and (b) declare their own capabilities. One engine
is **selected per-`Backend`** (not run simultaneously). See `ARCHITECTURE_REVIEW.md` for why this
narrowed form (no second IR, no entry points, no public registry) beats the original proposal.

### 6.1 Design principle

> The **constraint** describes *what* is constrained (the existing typed codecs).
> The **engine** decides *how* it goes on the wire and *whether* it can.

This inverts the one coupling that makes the library vLLM-shaped (§3.2), and mirrors the plugin pattern
the repo already uses for constraints (`structured_agents.constraints` entry points, `config.py:74`).

### 6.2 The constraint codecs stay — no second IR

The four codecs (`_Schema`/`_Regex`/`_Choice`/`_Grammar`, `constraint.py:35-137`) are **already** typed,
provider-neutral descriptions. Do **not** introduce a parallel `ConstraintSpec` dataclass — it would be a
weaker-typed tagged union (`pattern: str | None`, correlated with `kind` only by convention) duplicating
what the concrete codecs already guarantee, and a third place the kind-set must track in lockstep
(`ARCHITECTURE_REVIEW.md` §Q2). Instead:

- **Remove** the vLLM wire shape from the codecs (delete each `wire()`).
- **Add** a small `kind` tag (`ClassVar[str]`) to each codec and to the `Constraint` protocol, so
  `Backend` can gate without knowing wire shapes.
- The engine consumes the **concrete constraint directly** and reads its real, non-optional fields.

```python
# constraint.py — codecs stop knowing about vLLM; WireSpec stays here (engines produce it)

@runtime_checkable
class Constraint[T](Protocol):
    kind: str                               # "schema" | "regex" | "choice" | "grammar"
    def parse(self, raw: Any) -> T: ...
    def check(self) -> None: ...
    def to_config(self) -> dict[str, Any]: ...
    # wire() is gone.

@dataclass(frozen=True)
class _Regex:
    kind: ClassVar[str] = "regex"
    pattern: str
    # parse/check/to_config unchanged
```

`parse()`, `check()`, `to_config()` are already provider-neutral and stay as-is.

### 6.3 Engine protocol (internal)

Each engine is a small stateless object with a `name`, a `supports` set (constraint kinds it can render,
plus `"lora"`), and a `render(constraint) -> WireSpec`. The protocol is **internal** — not exported. No
public `Capabilities` model, no `adapter_wire` seam (adapter selection stays `spec.adapter or
default_model`, which vLLM and SGLang both accept as a model name; the `Settings.extra_body` seam already
covers the rare `lora_path` case).

```python
# engine/base.py — internal
@runtime_checkable
class Engine(Protocol):
    name: str
    supports: frozenset[str]                 # constraint kinds + "lora"
    def render(self, constraint: Constraint) -> WireSpec: ...
```

### 6.4 Reference engine implementations (sketch)

Each engine matches on the **concrete constraint** and reads its real fields (no `ConstraintSpec`):

```python
# engine/vllm.py — reproduces today's exact bytes (the golden guard)
class VLLMEngine:
    name = "vllm"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})
    def render(self, c):
        match c:
            case _Schema():  return WireSpec(NativeOutput(c.model, strict=c.strict))
            case _Regex():   return WireSpec(str, {"structured_outputs": {"regex": c.pattern}})
            case _Choice():  return WireSpec(str, {"structured_outputs": {"choice": list(c.options)}})
            case _Grammar(): return WireSpec(str, {"structured_outputs": {"grammar": c.ebnf}})

# engine/sglang.py — regex/ebnf fields; choice lowered to a regex alternation. UNVERIFIED.
class SGLangEngine:
    name = "sglang"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})
    def render(self, c):
        match c:
            case _Schema():  return WireSpec(NativeOutput(c.model, strict=c.strict))
            case _Regex():   return WireSpec(str, {"regex": c.pattern})
            case _Grammar(): return WireSpec(str, {"ebnf": c.ebnf})
            case _Choice():  return WireSpec(str, {"regex": "(" + "|".join(re.escape(o) for o in c.options) + ")"})

# engine/llama_cpp.py — json_schema + GBNF only; regex/lora off. UNVERIFIED.
class LlamaCppEngine:
    name = "llama_cpp"
    supports = frozenset({"schema", "choice", "grammar"})            # no regex, no lora
    def render(self, c):
        match c:
            case _Schema():  return WireSpec(NativeOutput(c.model, strict=c.strict))
            case _Grammar(): return WireSpec(str, {"grammar": c.ebnf})          # GBNF, not EBNF-parity
            case _Choice():  return WireSpec(str, {"grammar": _gbnf_alt(c.options)})
        raise BackendCapabilityError("llama_cpp engine does not support regex constraints.")
```

The choice-lowering *semantic* (which options, in order) comes from the constraint; only the target
syntax differs per engine (regex alternation vs GBNF alternation). Escape correctly per target:
`re.escape` for regex; escape both `\` and `"` for GBNF (`REFACTORING_GUIDE.md` §2.4 fixes the earlier
draft's missing-backslash bug). *(GBNF vs EBNF dialect caveat for llama.cpp grammars — see §9 risk R4.)*

### 6.5 `Backend` becomes engine-parameterized

```python
class Backend:
    def __init__(self, *, engine: str | Engine = "vllm", base_url=..., api_key=..., default_model=..., ...):
        self.engine = engine if not isinstance(engine, str) else select(engine)

    def build(self, spec):
        c = spec.constraint
        if c.kind not in self.engine.supports:
            raise BackendCapabilityError(f"{spec.name!r} needs {c.kind}; {self.engine.name} lacks it.")
        if spec.adapter and "lora" not in self.engine.supports:
            raise BackendCapabilityError(f"{spec.name!r} needs LoRA; {self.engine.name} lacks it.")
        c.check()
        wire = self.engine.render(c)
        # …assemble settings.extra_body = {**spec.settings.extra_body, **wire.extra_body}
        # …build OpenAIChatModel(spec.adapter or self.default_model, provider=OpenAIProvider(base_url=…)) as today
```

The capability gate (C4/C5) becomes **generic**: it checks `kind in engine.supports` instead of the
literal vLLM `structured_outputs` key. This is strictly cleaner than today's code, and it drops the
mutable `BackendCaps` model entirely.

### 6.6 Engine selection — an internal dict, not entry points

The three engines are built-in and closed-set. Resolve them with a plain dict; **do not** add
entry-point discovery or a public registry (that would only serve out-of-tree third parties, which the
requirement excludes — `ARCHITECTURE_REVIEW.md` §Q1). Note the contrast with the *constraint* plugin
mechanism (`config.py:68-83`): constraints are built from untrusted serialized config across a trust
boundary (hence entry points + `register_constraint` + the module allowlist); engine choice is a
deployment constant with none of that, so the parallel does not apply.

```python
# engine/__init__.py — internal
_BUILTINS: dict[str, Engine] = {"vllm": VLLMEngine(), "sglang": SGLangEngine(), "llama_cpp": LlamaCppEngine()}

def select(name: str) -> Engine:
    try: return _BUILTINS[name]
    except KeyError: raise ConfigError(f"Unknown engine {name!r}.") from None
```

If out-of-tree engines ever become a real need, add an entry-point fallback inside `select` behind the
same signature — a backward-compatible follow-up, not part of this change.

### 6.7 Config surface

`spec_from_config` (`config.py:133`) is unaffected — constraints are unchanged from the caller's view.
Engine selection lives on `Backend`, not on `AgentSpec`, so serialized specs stay portable across engines
(the same agent runs on vLLM or SGLang by pointing a different `Backend` at it). Optionally add a
top-level `engine: "sglang"` key to whatever bootstraps a `Backend`.

### 6.8 Capability matrix (what each engine declares in its `supports` set)

| Constraint | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| schema (json) | ✅ passthrough | ✅ passthrough | ✅ passthrough |
| regex | ✅ `structured_outputs.regex` | ✅ `regex` | ❌ (or GBNF-convert) |
| choice | ✅ `structured_outputs.choice` | ➖ lowered to `regex` | ➖ lowered to GBNF |
| grammar (EBNF) | ✅ `structured_outputs.grammar` | ✅ `ebnf` | ➖ GBNF (dialect risk) |
| lora | ✅ (container profile) | ✅ (`base:adapter`/`lora_path`) | ❌ |

✅ native · ➖ emulated/translated · ❌ unsupported (gate raises `BackendCapabilityError`)

### 6.9 Effort / risk
- **Effort:** small–moderate — a new `engine/` package (3 tiny engine modules + a base protocol + an
  internal `select`), drop `wire()`/add `kind` on the codecs, generalize the `build()` gate, remove
  `BackendCaps`, tests. **No** second IR, **no** entry points, **no** public registry to build. ~1–2 days.
- **Risk — LOW in code.** Backwards-compatible: `engine="vllm"` is the default and existing behavior is
  byte-for-byte preserved by `VLLMEngine` (golden test). The one deliberate public break is removing the
  unused `BackendCaps` export — handle it honestly (no field-mismatched alias). SGLang/llama.cpp engines
  merge **without** a live server, because the vLLM path (the only proven one) is untouched; but they are
  **unverified** and must be labeled so (§3.1, §9 R3/R4).

---

## 7. The blocking reality: SGLang can't serve the target model yet

This is the single most important operational finding and it constrains *both* options. From
`08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md`, `MINIMAL_REPRODUCTION.md`, and
`artifacts/sglang-gemma4-spike/…`:

- SGLang 0.5.14 **never reached weight load** for Unsloth Gemma 4 12B QAT `UD-Q4_K_XL` GGUF. It fails in
  a chain of **pre-inference** blockers:
  1. Upstream Transformers GGUF→config converter copies Gemma 4's **per-layer** `head_count_kv` list into
     the **scalar** `num_key_value_heads` (`StrictDataclassFieldValidationError: expected int, got list`).
  2. After a config shim: `Gemma4ForCausalLM does not support … scaled_dot_product_attention`.
  3. After an eager-attention patch: `Gemma4ForCausalLM does not support setting experts implementation.`
- **Native (safetensors) Gemma 4 works** in SGLang (control experiment served health/models/chat 200) —
  so the problem is the **GGUF converter**, not SGLang's CUDA path. Root cause is **upstream
  Transformers**, and the notes recommend an upstream fix over permanent monkeypatches.
- Consequently: **SGLang structured outputs, LoRA, MTP, and performance are all UNVERIFIED** against this
  model. The deploy dir ships a temporary Transformers shim (`deploy/sglang/native/gemma4_gguf_compat.py`,
  `sitecustomize.py`) that is explicitly a stopgap.
- Prior guidance (`07-sglang-gguf-spike/PROMPT.md`): *do not make SGLang the default; do not claim GGUF /
  structured outputs / MTP / perf works until runtime-tested.*

**Therefore:** a *full* replacement of vLLM (Option A) would strand the production model on a
non-loading engine. The provider abstraction (Option B) lets the library be ready for SGLang while the
serving blocker is resolved (upstream Transformers fix, a non-GGUF quant like the working
`compressed-tensors` safetensors QAT checkpoint, or a different model).

---

## 8. Recommendation

**Adopt Option B in its narrowed form (selectable in-tree engine plugins), with vLLM as the
default/reference engine.** See `ARCHITECTURE_REVIEW.md` for the full scorecard and
`REFACTORING_GUIDE.md` for the step-by-step.

Rationale:
1. It is only marginally more work than the throwaway swap, and it is not throwaway.
2. It keeps the **only runtime-proven** path (vLLM) intact and default — respecting the spike's "don't
   make SGLang default until proven" mandate.
3. It makes SGLang a droppable second engine that can be validated on its own timeline once the GGUF
   serving blocker clears (or on the working safetensors QAT checkpoint).
4. It cleanly encodes llama.cpp's **narrower** capabilities (no regex/LoRA, GBNF grammars) instead of
   pretending all engines are identical.
5. It lands **directly on the current `Backend`**. (An earlier draft said to fold this into the
   "closed backend" refactor's `provider_extra` seam — that recommendation is withdrawn:
   `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md` targets the superseded `structured_agents_v2` @ v0.2.0 and its
   `provider_extra`/`ClosedBackend`/`Capabilities` do not exist in this v0.3.0 tree. Nothing to compose
   with.)

If the goal really is *only* SGLang and vLLM will truly never be needed again, Option A is viable — but
do it **as** a single `SGLangEngine` behind the same seam, so the choice is a default flip, not a
rewrite.

---

## 9. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | SGLang can't load the production GGUF (upstream Transformers bug) | **High / blocking** | Track upstream fix; validate against working safetensors QAT checkpoint; keep vLLM default |
| R2 | SGLang has no `choice` param | Medium | Lower `Choice` → regex alternation in `SGLangEngine.render` (options are `re.escape`d); `parse()` still validates membership |
| R3 | SGLang structured-output wire shape never runtime-verified against this model | Medium | Add SGLang cases to an engine-parameterized live suite (§10) before flipping any default |
| R4 | llama.cpp grammar is **GBNF**, not the same EBNF dialect as XGrammar; regex unsupported | Medium | `LlamaCppEngine.supports` omits `regex`; translate `choice`/`grammar` to GBNF or gate off. Don't claim EBNF parity |
| R5 | SGLang LoRA uses `base:adapter` / `lora_path`, not vLLM's model-id-per-adapter | Low | `base:adapter` matches today's `spec.adapter or default_model`; the rare `lora_path` form goes in the sglang render path or `Settings.extra_body` — no public `adapter_wire` seam needed |
| R6 | Silent behavior drift if `VLLMEngine` isn't byte-identical to current wire | Low | Golden-body test: assert `VLLMEngine.render()` reproduces the exact bodies in `REVIEW_SPIKES/bodies.json` / `test_constraint.py:40-52` |
| R7 | ~~Provider abstraction fights the "closed backend" refactor~~ — **not a real risk**: that refactor targets superseded `structured_agents_v2` @ v0.2.0, absent from this tree | Low | Land the engine seam directly on the current `Backend`. Only public break is removing the unused `BackendCaps` export — no misleading alias |
| R8 | SGLang/llama.cpp wire shapes are doc-derived, unverified in-repo (§3.1) | Medium | Ship engines labeled unverified; enable SGLang live conformance only after R1 clears; keep vLLM default |

---

## 10. Test strategy

- **Unit (no server):** golden-body tests per engine — feed each concrete constraint to
  `engine.render()` and assert the exact `extra_body`/`response_format`. Seed vLLM goldens from
  `.scratch/projects/09-constraint-codec-rewrite/REVIEW_SPIKES/bodies.json` so the refactor is provably
  non-regressive. Add SGLang/llama.cpp goldens from their documented shapes (labeled unverified).
- **Capability gating:** assert `build()` raises `BackendCapabilityError` for
  (llama.cpp + regex), (llama.cpp + lora), etc.; `select("nope")` raises `ConfigError`.
- **Live (opt-in):** generalize `tests/test_live.py` to parameterize over
  `LLM_ENGINE ∈ {vllm, sglang, llama_cpp}` (env-selected, `SAV_LIVE=1`). Reuse the existing four
  constraint agents (schema/regex/choice/grammar). This turns the current vLLM-only cutover suite into a
  cross-engine conformance suite. **Do not enable SGLang live tests until the server loads the model
  (R1).**
- **Deploy `verify.sh`:** already the de-facto integration test per engine; keep them as the source of
  truth for what each server actually accepts.

---

## 11. Phased implementation plan (Option B, narrowed)

Full step-by-step with exact code is in `REFACTORING_GUIDE.md`. Summary:

1. **Phase 1 — Codecs drop `wire()`, gain `kind` (no behavior change yet).** Delete each codec's
   `wire()`; add a `kind` ClassVar and a `kind` field to the `Constraint` protocol; keep `WireSpec` in
   `constraint.py`. No `ConstraintSpec`.
2. **Phase 2 — `engine/` package.** Add `base.py` (internal `Engine` protocol), `vllm.py` (byte-identical
   to today), `sglang.py` (regex/ebnf/choice-as-regex), `llama_cpp.py` (GBNF + json schema; regex/lora
   off), and `__init__.py` with an internal `_BUILTINS` dict + `select()`. No entry points.
3. **Phase 3 — Rewire `Backend`.** Add `engine=` (default `"vllm"`); replace the `structured_outputs`
   gate (C4) with `kind in engine.supports`; remove `BackendCaps`. Behavior byte-identical on vLLM.
4. **Phase 4 — Exports + tests.** Drop `BackendCaps` from `__init__`; keep `WireSpec`. Golden test
   (`VLLMEngine.render` == today's bodies) + engine-dialect + caps-gating + unknown-engine tests. **Do
   not** flip any default.
5. **Phase 5 — Live conformance.** Parameterize `test_live.py` by an `LLM_ENGINE` env var. Run vLLM
   (proven). Run SGLang **only when R1 clears**. SGLang/llama.cpp remain unit-verified only until then.
6. **Phase 6 — Docs & cleanup.** Update README/marker/docstrings (D1–D4) to "OpenAI-compatible engines
   (vLLM/SGLang/llama.cpp)"; move the `constraint.py` ty `unresolved-import` override to `engine/**`.

---

## 12. Appendix — verified references

**Library files read:** `src/structured_agents/{__init__,agent,constraint,config,plane,authority,`
`approval,errors,integrations/fornix}.py`; `tests/test_live.py`, `tests/live_crash_worker.py`;
`pyproject.toml`; `README.md`.

**Captured request bodies (ground truth for vLLM wire shape):**
`.scratch/projects/09-constraint-codec-rewrite/REVIEW_SPIKES/bodies.json`,
`.scratch/projects/01-xgrammar-concept/spike/captured_request.json`.

**Deploy configs:** `deploy/vllm/{entrypoint.sh,native/serve.sh,native/verify.sh}`,
`deploy/sglang/native/{serve.sh,run.sh,README.md,gemma4_gguf_compat.py,sitecustomize.py}`,
`deploy/llama-cpp/native/{serve.sh,verify.sh,README.md}`.

**Prior spike evidence (SGLang GGUF blocker):**
`.scratch/projects/07-sglang-gguf-spike/PROMPT.md`,
`.scratch/projects/08-unsloth-gemma4-gguf-compatibility/{ANALYSIS,MINIMAL_REPRODUCTION,METADATA_REPORT}.md`,
`artifacts/sglang-gemma4-spike/20260714T205109Z/AUDIT.md`,
`.scratch/projects/05-small-model-survey/REPORT.md`,
`CODE_REVIEW_FINAL_REFACTOR_GUIDE.md`.

**External (SGLang API shape, verified 2026-07):**
- SGLang Structured Outputs: https://docs.sglang.ai/advanced_features/structured_outputs.html
  (`response_format` json_schema; `extra_body` `regex`/`ebnf`; `--grammar-backend xgrammar|outlines|llguidance`; one constraint per request)
- SGLang LoRA Serving: https://docs.sglang.io/advanced_features/lora.html
  (`base:adapter` model syntax and `lora_path` request field)
- vLLM Structured Outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
