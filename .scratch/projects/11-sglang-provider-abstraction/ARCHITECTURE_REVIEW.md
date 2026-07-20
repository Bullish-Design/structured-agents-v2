# ARCHITECTURE_REVIEW.md — Multi-Backend Design Review

**Project:** `11-sglang-provider-abstraction`
**Date:** 2026-07-19
**Mode:** Read-only, skeptical staff-level design review. No source or planning docs edited.
**Reviewed against tree:** `structured_agents` **v0.3.0** (working dir `main`, tip `ca89e3f`).
**Requirement (owner-confirmed after first draft):** support **all three** engines as plugins and
**choose one per `Backend`** — *selection among the three built-ins*, **not** out-of-tree third-party
discovery, and **not** three engines running simultaneously. §1/§2/§5/§7 below reflect this; the
scorecard and per-question findings (§3–§4) are unchanged and explain *why* this shape wins.

---

## 1. Verdict

**Keep the proposal's structure; cut its speculative machinery.** The proposal correctly identifies the
one real coupling (constraint codecs bake the vLLM `extra_body={"structured_outputs": {...}}` shape) and
correctly wants to move the *how-it-goes-on-the-wire* decision out of the constraint into per-engine
modules. With the requirement now confirmed as **select-one-of-three-built-ins** (not out-of-tree
extensibility, not simultaneity), the recommended design is **Option B with three things removed**:

- **Remove the neutral IR (`ConstraintSpec`).** It duplicates the already-typed codecs with a weaker,
  all-optional tagged-union (Q2). Each engine's `render` consumes the concrete constraint directly.
- **Remove entry-point discovery + the public `load_provider`/`register_provider` registry.** Selecting
  among three in-tree engines needs a plain internal `dict`, not a setuptools-entry-point scan; that
  machinery only earns its keep for out-of-tree third parties, which the owner has ruled out (Q1, Q7).
- **Keep the public surface to one knob:** `Backend(engine="vllm"|"sglang"|"llama_cpp")`, default
  `"vllm"`. Do **not** export `Provider`, `Capabilities`, `ConstraintSpec`, `load_provider`,
  `register_provider` (Q5).

Keep the good part: **each engine is a self-contained module** (`engine/vllm.py`, `engine/sglang.py`,
`engine/llama_cpp.py`) that renders every constraint kind and declares its own capabilities — the
"plugin" the owner wants, minus the discovery framework. **Build all three now.** vLLM is reproduced
byte-for-byte (the golden guard); SGLang and llama.cpp are shipped **labeled unverified** — no
constrained request is exercised against either engine anywhere in this repo (§4.8), and SGLang is
additionally gated behind the live GGUF-load blocker (R1). **Schema needs no per-engine logic at all** —
all three accept `response_format` json_schema (pydantic-ai `NativeOutput`), so only regex/choice/grammar
differ across engines.

Two of the proposal's load-bearing premises do not survive verification and must be corrected before any
implementation (details in §4): (a) the "compose with the closed-backend refactor / build the seam
inside CR-01's `provider_extra`" recommendation is **stale** — `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md`
targets a different, superseded codebase (`structured_agents_v2` @ v0.2.0) whose `ClosedBackend`,
`provider_extra`, `AgentProfile`, and `StrictConfig` **do not exist** in the current v0.3.0 tree; (b) the
SGLang/llama.cpp wire shapes are **doc-derived, not repo-verified** — nothing in this repo exercises a
constrained request against SGLang or llama.cpp.

---

## 2. Requirement clarification — RESOLVED

The design's shape hinged on one question, **now answered by the owner**:

> **"Support all three engines as plugins, and be able to choose which one to run with"** — and,
> clarified: *select among the three built-in engines per `Backend`*; **not** out-of-tree third-party
> registration, and **not** three engines live simultaneously in one deployment.

This is the middle reading, and it is the one the code should serve:

- **All three are in scope** (so llama.cpp is built now, not deferred).
- **Selection, not simultaneity** — one `Backend` = one engine, chosen at construction. Confirmed by
  the fact that a serialized `AgentSpec` carries no engine identity and engine choice is a deployment
  constant (the deploy tree already separates engines into their own dirs).
- **In-tree, not open** — the owner explicitly does *not* need third parties to add an engine without
  editing the library. That single answer is what removes entry-point discovery and the public registry
  from the design (Q1).

The rest of the original evidence still stands and reinforces the "selection" reading (kept for the
record):

- There is **one production model** and the target engine (SGLang) **cannot currently load it**
  (`08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md`; SGLANG_ANALYSIS.md §7). No one is running three
  engines side by side today, and cannot.
- The deploy tree already separates engines into their own dirs; engine choice is a *deployment*
  decision, not a per-request or per-agent one.
- A serialized `AgentSpec` carries no engine identity, and the proposal itself keeps provider selection
  on `Backend`, not `AgentSpec` (SGLANG_ANALYSIS.md §6.7) — i.e. one `Backend` = one engine.

**Because the owner chose "select among the three built-ins" (not out-of-tree discovery), the following
are cut:** entry-point discovery, `load_provider`/`register_provider`, the runtime `_registry`, and the
public `Provider`/`Capabilities` protocol surface. What remains is an *engine selector* on a
single-engine `Backend` plus three self-contained engine modules. If the owner later needs out-of-tree
engines, adding entry-point resolution behind the same `engine=` seam is a small, backward-compatible
follow-up — so nothing is lost by deferring it.

**llama.cpp is in scope and will be built** — but note it ships *unverified*: no production requirement
is recorded, its wire shape is doc-derived (§4.8), and its verify.sh tests no grammar surface. The
engine module must therefore be honest about capabilities (no regex, no LoRA, GBNF≠EBNF) and must not
advertise parity it hasn't demonstrated.

---

## 3. Scorecard

Axes: complexity added · new public surface · testability · how it handles the choice/regex/EBNF gap ·
how it composes with the (stale) closed-backend refactor.

| Option | Complexity added | New public surface | Testability | choice/regex/EBNF gap | Composes w/ closed-backend |
|---|---|---|---|---|---|
| **A. As proposed** (IR + Provider protocol + entry-point registry) | **High** — new `providers/` pkg (base+3), 2nd IR, registry, entry-point scan, caps model | **+5**: `Provider`, `Capabilities`, `ConstraintSpec`, `load_provider`, `register_provider` | Good, but half the tests exercise machinery (registry/entry points) not behavior | In each provider's `render()`; **duplicates** the alternation logic across sglang/llama_cpp | **Moot** — target refactor is stale (§4.6); premise is false |
| **B. Minimal dialect** (`render(constraint,dialect)` + `dialect` Literal on `Backend`) | **Low** — 1 new file (~50 LoC), no registry, no entry points, no 2nd IR | **+1**: `dialect` param (a `Literal`) | **Excellent** — one golden render-table test; vLLM bytes pinned exactly | One central dialect table; alternation helper shared | Cleanly — nothing to fight; rides `Backend` |
| **C. Constraint-owned lowering** (each `Constraint` lowers itself per dialect) | Medium | Small | Good | **Scatters** dialect knowledge onto every constraint; couples constraints to all engines | Neutral |
| **D. Lean on pydantic-ai** | Lowest | 0 | n/a | Covers **schema only**; regex/choice/grammar still unmodeled by pydantic-ai | Neutral |
| **E. Do less now** (SGLang behind same `Backend`, defer llama.cpp) | Lowest | +1 (or 0) | Excellent | vLLM+SGLang only; llama.cpp deferred | Cleanly |

**Winner: B's mechanism inside A's per-engine module layout, informed by D and E.** Keep A's good part —
three self-contained engine modules (the "plugin" the owner wants) — but drive them with B's minimal
mechanism (no second IR, no entry-point registry, one public knob). This delivers the exact same
capability as A-as-proposed (any engine's wire shape, per-`Backend` selection, fail-fast gating,
byte-for-byte vLLM preservation) at a fraction of the public surface. D justifies cutting schema out of
per-engine logic entirely. E supplies the risk posture: all three ship, but SGLang/llama.cpp are labeled
unverified and the SGLang live path stays gated on R1 — honesty, not deferral. C is rejected — it
inverts the scatter problem (couples every constraint to every engine), the one thing the proposal got
right to avoid.

---

## 4. Findings (questions 1–8)

### Q1 — Is the abstraction proportionate? **No. Entry-point discovery + registry is speculative generality.**

There are exactly three known, closed-set, **in-tree** backends. The proposal justifies entry points by
analogy to the *existing* constraint plugin mechanism (`config.py:68-83`, group
`structured_agents.constraints`; SGLANG_ANALYSIS.md §6.1/§6.6). **That analogy is cargo-culted.** The
constraint entry-point mechanism exists for a real reason the providers don't share: constraints are
built from **untrusted serialized config** across a trust boundary, which is exactly why
`constraint_from_config` threads a `frozenset` module allowlist (`config.py:86-109,112-130`) and why
custom constraint *kinds* must be registerable. Providers have **none** of that: there is no
"provider-from-config" trust boundary, engine choice is a deployment constant, and the three engines are
shipped in the library itself. The proposal even resolves built-ins from a hardcoded `_BUILTINS` dict
first (REFACTORING_GUIDE.md:190-234) — so the entry-point scan earns its `importlib.metadata` cost,
global `_registry`, `_entry_points_discovered` flag, and two public functions for a use case
(third-party out-of-tree providers) **that the requirement never asked for**.

A `dict[str, …]` of built-ins, or a `match` on a `dialect: Literal[...]`, is equally capable and
strictly simpler. The line between "extensible" and "over-built" here is precisely: **selection among a
closed in-tree set = fine as a Literal/dict; discovery of unknown out-of-tree implementations = the part
to cut.**

### Q2 — Is `ConstraintSpec` the right seam? **No. It is a redundant, weaker-typed second representation.**

The four codecs (`constraint.py:35-137`) are *already* typed, provider-neutral descriptions: `_Regex`
guarantees `pattern: str`, `_Choice` guarantees `options: tuple[str,...]`, etc. `ConstraintSpec`
(SGLANG_ANALYSIS.md §6.2 / REFACTORING_GUIDE.md:79-88) flattens all of them into **one dataclass with
five all-optional fields keyed by a `kind` Literal** — a hand-rolled tagged union that *loses* the
type guarantees the codecs already provide (`pattern: str | None`, `model: type|None`, correlated only
by convention with `kind`). The provider's `render()` then does `if spec.kind == "regex": … spec.pattern`
(REFACTORING_GUIDE.md:257-268), re-deriving at runtime what the concrete type already knew statically.
It also introduces a **third** place the constraint-kind set must be maintained in lockstep (codecs, the
`ConstraintSpec.kind` Literal, and `Capabilities`' one-bool-per-kind fields).

Cleaner: the dialect consumes the **concrete constraint** directly. A single
`render(constraint, dialect) -> WireSpec` in one `dialect.py` can `match` on the concrete codec types
(which already exist and are already frozen), reading their real, non-optional fields. No second IR, no
lockstep enum. This is also the alternative the kickoff flagged ("giving `wire()`/`render()` a
`provider`/`dialect` argument"), and it wins.

### Q3 — Where should engine-specific *translation* live? **In one dialect table; the alternation semantic is shared, the target syntax is per-dialect.**

The proposal puts all lowering in each provider's `render()`. That scatters and **duplicates** a
constraint *semantic*: `choice` → "an alternation of the escaped literal options" is identical for
SGLang and llama.cpp; only the *target syntax* differs (regex `(a|b)` in `sglang.py:302-305` vs GBNF
`root ::= "a" | "b"` in `llama_cpp.py:327-329,345-346`). Writing the escape/ordering logic twice invites
drift (note the two implementations already escape differently: `re.escape` vs a bare `"` replace that
misses backslashes).

The elegant boundary: the **constraint** owns the fact that a choice is an alternation of specific
escaped literals (a tiny shared helper or a method like `_Choice.alternatives()`), and the **dialect**
owns only *field naming and target syntax* (regex vs GBNF vs `structured_outputs.choice` passthrough).
That is not full "constraint-owns-lowering" (Alternative C, which would couple the constraint to every
engine's grammar dialect — rejected); it is "constraint owns the *semantic*, dialect owns the *syntax*,"
co-located in one greppable `dialect.py`. When you add an engine you read one file, not four.

### Q4 — Static caps vs runtime negotiation. **Static is the right, non-gold-plated answer. Keep fail-fast at build; do NOT add probing.**

Today's gate is a build-time raise (`agent.py:53-58`) — good taste, keep it. The proposal's per-kind
`Capabilities` is *more honest* than today's single `xgrammar` bool (which also misnames a vLLM-ism;
C5). Runtime capability probing against the live server is **gold-plating** for a closed three-set:
`deploy/*/verify.sh` and the opt-in `test_live.py` already are the runtime check, and pydantic-ai/OpenAI
will surface a 4xx if a server rejects a shape. The one legitimate worry the kickoff raises — vLLM's
structured-outputs flag/field name drifting across releases — is a *deployment/serve.sh* concern, not a
library concern (the repo already pins it: `deploy/vllm/native/serve.sh:63`
`--structured-outputs-config.backend xgrammar`). Verdict: a small **internal** per-dialect caps table +
build-time `BackendCapabilityError`. No probing, and `Capabilities` need not be public (§Q5).

### Q5 — Consistency with the codebase's taste. **The proposal over-exposes. Minimal public API = one Literal.**

The library is terse, dataclass/protocol-driven, and narrow. The proposal adds **five** public names
(`Provider`, `Capabilities`, `ConstraintSpec`, `load_provider`, `register_provider`;
REFACTORING_GUIDE.md:362-364, 462-469) for a three-engine need. The minimal public surface that meets
the goal is **one**: `dialect: Literal["vllm","sglang","llama_cpp"] = "vllm"` on `Backend`. `render()`,
the caps table, and the dialect modules stay **internal** (mirroring how the concrete codecs `_Schema`/
`_Regex`/… are private and only the `Schema()`/`Regex()` factories are public). Note the "no escape
hatch" self-image is already softened by `Settings.extra_body` (`agent.py:26,61`) and `Backend(model=…)`
injection (`agent.py:48,62-63`) — so a per-engine `extra_body` need (e.g. SGLang `lora_path`) already
has a home and does **not** justify a public `Provider.adapter_wire` protocol method.

### Q6 — Integration with the closed-backend refactor. **The premise is false: that refactor is stale relative to this tree. Do NOT fold into a non-existent `provider_extra`.**

This is the proposal's biggest unverified claim (SGLANG_ANALYSIS.md §0.3, §8.5, R7;
"build the provider seam *there*, not as a separate mechanism"). Verified against the code:

- `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md` targets **baseline `v0.2.0` (`70d97fa`)** and package
  **`structured_agents_v2`**, with `ClosedBackend`, `AgentProfile`, `AgentSet`, `DualPathRuntime`,
  `StrictConfig`, `SecretStr` api_key, and the `provider_extra` seam (CR-01/CR-02, guide §3.1–3.2, §8).
- **None of those exist in the current tree.** `grep -rn` across `src/` for
  `ClosedBackend|provider_extra|AgentProfile|AgentSet|DualPathRuntime|SecretStr|StrictConfig|
  structured_agents_v2` returns **zero hits**. The package is `structured_agents` (v0.3.0), `Backend` is
  a plain class (not a strict BaseModel) with `api_key: str = "sk-none"` in **plaintext**
  (`agent.py:46,49,64`), and `BackendCaps(BaseModel)` has **no** `extra="forbid"` (`agent.py:38-40`) —
  so CR-01's own acceptance test (`BackendCaps(xgramar=False)` raises) would *fail* today.
- The git history shows why: after `70d97fa` (v0.2.0) came a series of `plan(v3)` commits and a
  from-scratch **"durable agent plane"** rewrite (`425c55a Add pure constraint codec`, `87c00e4 Add
  durable agent primitive`, …). The v3 rewrite **superseded** the v2 code the closed-backend guide was
  written against; it did not carry CR-01/CR-02's mechanisms forward.

**Consequence:** "ride CR-01's `provider_extra`" is not actionable — there is no CR-01 in this tree.
The provider design neither helps nor fights the closed-backend refactor; that refactor has been
overtaken by events. Sequencing recommendation: land the dialect seam **directly on the current
`Backend`**, independently. If the owner still intends to apply parts of the v2 review (secret api_key,
strict config) to v0.3.0, that is a separate workstream; the dialect selector is orthogonal and should
not wait on it. **Delete the "compose with closed-backend / CR-01 `provider_extra`" claims from
SGLANG_ANALYSIS.md** (§0.3, §8.5, R7).

### Q7 — Does it deliver the goal, and which goal? **It delivers "portability," but pays for "simultaneity" the requirement doesn't need.**

Covered in §2. The machinery that only simultaneity would justify (registry, entry points, "plugins")
is dead weight under the actual goal. Under "portability to SGLang," a dialect selector delivers 100% of
the value. If the owner confirms simultaneity is *not* required, the design collapses to Alternative B/E.

### Q8 — Backward-compat & risk. **Byte-for-byte vLLM preservation is achievable; the only real breakage is the `BackendCaps`/`WireSpec` public exports and the `test_constraint` golden — all manageable, none hidden.**

Confirmed against the test contract:

- **Golden bytes are pinned in exactly one place:** `tests/test_constraint.py:40-52` asserts `.wire()`
  shapes (`{"structured_outputs":{"regex":…}}`, `{…"choice":[…]}`, `{…"grammar":…}`, and `NativeOutput`
  for schema). Ground truth matches `09-…/REVIEW_SPIKES/bodies.json` (regex/choice/grammar under
  `structured_outputs`, schema under `response_format`). A `render(constraint,"vllm")` that reproduces
  these exact `WireSpec`s makes the refactor provably non-regressive — same guarantee as the proposal's
  `test_vllm_bytes_are_unchanged`, minus the provider layer. **This test must be rewritten** whichever
  design lands (it calls `.wire()` directly).
- **`test_agent.py:25` and `test_live.py:49` construct `Backend` with no `provider=`/`dialect=`
  argument** and never reference `BackendCaps` — so a default `dialect="vllm"` keeps both green
  untouched. `BackendCaps` and `xgrammar` appear in **zero** tests.
- **Public-surface breakage:** `BackendCaps` and `WireSpec` are exported (`__init__.py:3,19,39,62`).
  Option B keeps `WireSpec` (still the render return type). `BackendCaps` (fields `xgrammar`,`lora`)
  either stays as-is (if caps stay internal, the minimal design) or is replaced — a deliberate v0.3.x
  API change, not hidden, and pre-1.0. Do **not** silently alias `BackendCaps = Capabilities` with
  different fields (the proposal's suggestion, REFACTORING_GUIDE.md:394) — that is a semver lie.
- **ty overrides:** `constraint.py`'s `unresolved-import` override (`pyproject.toml:51-55`) exists for
  the in-method `from pydantic_ai.output import NativeOutput`. Moving `NativeOutput` construction into
  `dialect.py` moves that override, it doesn't remove the need for it. Re-check, don't assume.

**Unverified risk the proposal understates:** the SGLang and llama.cpp wire shapes are **doc-derived,
not repo-verified.** `deploy/vllm/verify.sh` exercises json_schema/regex/choice/grammar end-to-end
(the proven path), but SGLang's and llama.cpp's `verify.sh` do **only** health/models/one plain chat —
no constrained request ever hits them — and `deploy/sglang/native/README.md` explicitly states the
xgrammar backend "does not establish compatibility with … the library's `structured_outputs` wire
shape." So shipping SGLang/llama.cpp dialects "dark" means shipping **guesses** (un-nested `regex`/`ebnf`
field names, choice→regex, GBNF alternation) that no test in this repo can confirm. That is fine for
SGLang *if* labeled unverified and gated behind the R1 server blocker; it is an argument to **defer
llama.cpp** until there is a reason and a way to verify it.

---

## 5. Recommended design (the winning shape)

Concrete enough to rewrite `REFACTORING_GUIDE.md` against; not the guide itself. This is Option B's
per-engine-module structure with the second IR and the discovery machinery removed.

**File map.** Keep the good bones of the proposal (`providers/`, renamed `engine/` to match the owner's
and the deploy tree's vocabulary — cosmetic, pick one):

```
src/structured_agents/
  constraint.py     # codecs UNCHANGED except: drop the vLLM wire knowledge from wire()
  agent.py          # Backend gains engine=…; generic capability gate
  engine/           # NEW package — three self-contained "plugins" + an internal selector
    __init__.py     # internal _BUILTINS dict + select(name) -> Engine ; NO entry-point scan, NOT public
    base.py         # Engine Protocol + a tiny Caps type (both INTERNAL)
    vllm.py         # reproduces today's bytes exactly
    sglang.py       # regex/ebnf field names; choice→regex; labeled unverified
    llama_cpp.py    # json_schema + GBNF only; regex/lora off; labeled unverified
```

**No `ConstraintSpec`.** The four codecs stay concrete frozen dataclasses. Remove only the vLLM wire
shape from them; each engine's `render` matches the concrete constraint and reads its real, non-optional
fields:

```python
# engine/sglang.py  (illustrative; each engine is one small module)
from ..constraint import _Schema, _Regex, _Choice, _Grammar, WireSpec  # concrete codecs, same package

class SGLangEngine:
    name = "sglang"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})

    def render(self, c) -> WireSpec:
        match c:
            case _Schema():  return WireSpec(NativeOutput(c.model, strict=c.strict))  # engine-independent
            case _Regex():   return WireSpec(str, {"regex": c.pattern})
            case _Grammar(): return WireSpec(str, {"ebnf": c.ebnf})
            case _Choice():  return WireSpec(str, {"regex": _alternation_regex(c.options)})
```

- **Schema is engine-independent** — every engine returns `NativeOutput`/`response_format` for it
  (Q2/D). Don't branch schema per engine.
- **Choice lowering** uses one *shared* alternation helper so the escape/ordering logic lives once; the
  engine only picks the target syntax — `structured_outputs.choice` (vLLM) / regex `(a|b)` (SGLang) /
  GBNF `root ::= "a" | "b"` (llama.cpp) (Q3). Escaping must be correct per target: `re.escape` for
  regex; escape both `"` and `\` for GBNF (the proposal's `llama_cpp` helper misses `\`).
- Matching the private `_Regex`/… codecs from a sibling module is a within-package coupling and is
  acceptable given the codecs are stable. If the owner dislikes reaching into underscored classes, add a
  single neutral accessor on the codec (`kind` + typed getters) — but **not** the all-optional
  `ConstraintSpec` union, which is the weak version.

**Engine selection is a plain internal dict — no entry points:**

```python
# engine/__init__.py
_BUILTINS = {"vllm": VLLMEngine, "sglang": SGLangEngine, "llama_cpp": LlamaCppEngine}

def select(name: str):
    try: return _BUILTINS[name]()          # or cache instances; they're stateless
    except KeyError: raise ConfigError(f"Unknown engine {name!r}.")
```

**`Backend`.** One new parameter; generic gate:

```python
class Backend:
    def __init__(self, *, engine: str = "vllm", base_url=..., api_key=..., default_model=..., ...):
        self.engine = select(engine)
        ...
    def build[T](self, spec: AgentSpec[T]) -> Agent[T]:
        kind = spec.constraint.kind                 # cheap "which codec am I"; add if not present
        if kind not in self.engine.supports:
            raise BackendCapabilityError(f"{spec.name!r} needs {kind}; {self.engine.name} lacks it.")
        if spec.adapter and "lora" not in self.engine.supports:
            raise BackendCapabilityError(f"{spec.name!r} needs LoRA; {self.engine.name} lacks it.")
        spec.constraint.check()
        wire = self.engine.render(spec.constraint)
        # …identical settings assembly, OpenAIChatModel(...), DBOSAgent(...) as agent.py:60-68
```

Adapter/LoRA field naming (vLLM model-id-per-adapter vs SGLang `lora_path`) is a deployment concern that
already has a seam in `Settings.extra_body`; keep it inside the `sglang` engine's `render`/`build` path
if ever needed — no public `adapter_wire` protocol.

**Public API surface (the whole delta):** add `engine="…"` to `Backend`. Optionally export an `Engine`
*type* for annotation. **Nothing else** — no `Provider`, `Capabilities`, `ConstraintSpec`,
`load_provider`, `register_provider`, no entry points. Keep `WireSpec` exported. Decide `BackendCaps`
deliberately: either keep it, or remove it with an honest deprecation — **do not** alias
`BackendCaps = Capabilities` with different fields (a semver lie).

**Build all three now, but honestly:** `vllm` reproduces today's bytes byte-for-byte (the golden guard).
`sglang` and `llama_cpp` ship with correct capability declarations but their wire shapes are
**doc-derived and unverified in this repo** (§4.8) — mark them so in docstrings, and do not enable the
SGLang live path until the R1 GGUF-load blocker clears. Correct capabilities are what make an unverified
engine safe: llama.cpp declares `regex`/`lora` off, so a regex agent fails fast rather than mis-rendering.

**Tests:** a golden render-table test (`VLLMEngine().render(c) == <today's WireSpec>` for the four
constraints — the regression guard); per-engine shape tests for `sglang`/`llama_cpp` from their
documented shapes; capability-gating tests (llama_cpp+regex raises, llama_cpp+lora raises,
unknown-engine raises `ConfigError`). Parameterize `test_live.py` by an engine env var for `vllm`
(proven) and `sglang` (when R1 clears) — the proposal's Phase 7 stands. One new test file, no
registry/entry-point matrix.

---

## 6. Deltas to the existing plan

Changes to make if the owner accepts this review:

**SGLANG_ANALYSIS.md**
- §0.3 / §8.5 / R7 (§9): **Remove** the "composes with the closed-backend refactor / build the seam
  inside CR-01's `provider_extra`" recommendation — that refactor targets superseded code
  (`structured_agents_v2` @ v0.2.0) absent from the v0.3.0 tree (Q6). Replace with: "land the dialect
  seam directly on the current `Backend`; the v2 closed-backend guide is stale and orthogonal."
- §6 (Option B) / §6.1–6.8: **Downgrade** from "Provider protocol + registry + entry points + neutral
  IR" to "dialect selector + one `render()` table." Drop `ConstraintSpec` (Q2), drop entry-point
  discovery / `load_provider` / `register_provider` (Q1), drop the public `Provider`/`Capabilities`
  surface (Q5). Keep: the design principle (constraint = *what*, dialect = *how*), the choice→regex
  lowering, the per-engine capability matrix (as an internal table).
- §3.1 / §6.4 / §6.8: **Label SGLang and llama.cpp wire shapes as UNVERIFIED (doc-derived).** No
  constrained request is exercised against either engine in-repo; `deploy/sglang/native/README.md`
  disclaims wire-shape compatibility (Q8). Add this to the risk register as a first-class item, not a
  footnote.
- §6 scope: **Recommend deferring llama.cpp** (no requirement, unverified shape, no grammar surface in
  its verify.sh) rather than building it in Phase 4.

**REFACTORING_GUIDE.md**
- **Keep** the per-engine module structure (`engine/`—or `providers/`—with `vllm.py`/`sglang.py`/
  `llama_cpp.py`); that part matches the owner's "plugins." **Build all three** (do not defer
  llama.cpp), but label SGLang/llama.cpp wire shapes *unverified* in their docstrings.
- Phase 1: **Do not add `ConstraintSpec`.** Keep the concrete codecs; each engine's `render` matches
  them and reads real fields. Move only the vLLM wire shape out of `constraint.py`.
- Phase 2: **Delete** `base.py`'s entry-point scan and `load_provider`/`register_provider`. Replace with
  a plain internal `_BUILTINS` dict + `select(name)`. Keep the `Engine` protocol and `Caps`
  **internal** (not exported).
- Phase 3: `Backend(engine=...)` (not `provider=...`); gate on the engine's `supports` set.
- Phase 4 (`__init__`): add **only** `engine=` support (and optionally an `Engine` type for
  annotation); do **not** export `Provider`/`Capabilities`/`ConstraintSpec`/`load_provider`/
  `register_provider`. Handle `BackendCaps` removal honestly (no field-mismatched alias).
- Phase 5: **Drop** the `[project.entry-points."structured_agents.providers"]` table entirely — there is
  no discovery to declare. (Optional extras for engine client libs can stay.)
- Phase 6: keep the golden render-table + gating tests; drop the registry/entry-point tests.

---

## 7. Open questions / unverifiable items

1. ~~Simultaneity vs portability~~ — **RESOLVED**: select one of three built-ins per `Backend`; no
   simultaneity, no out-of-tree discovery. (§2)
2. ~~Is llama.cpp required?~~ **RESOLVED**: yes, in scope — build all three, but ship SGLang/llama.cpp
   labeled unverified. (§2)
3. **SGLang/llama.cpp wire shapes are unverified in this repo** — un-nested `regex`/`ebnf`, no `choice`,
   GBNF alternation, and SGLang `lora_path` all come from external docs (SGLANG_ANALYSIS.md §12), not a
   running server here. Cannot be confirmed until the R1 GGUF blocker
   (`08-unsloth-gemma4-gguf-compatibility`) clears and a live conformance run happens. (§Q8)
4. **SGLang cannot load the production GGUF today** (upstream Transformers converter bug) — verified
   only via the prior spike's write-up, not re-run here. Any "SGLang works" claim remains gated on it.
5. **Whether the v2 closed-backend review (secret api_key, strict config) will be re-applied to
   v0.3.0** is unknown; it's a separate workstream. The dialect seam does not depend on it either way.
   (§Q6)
6. **`BackendCaps`/`WireSpec` external consumers** — I verified no *in-repo* test depends on
   `BackendCaps`; downstream/out-of-repo importers (if any) are unknown and would feel a `BackendCaps`
   removal. (§Q8)
