# structured-agents v3 (constric) — the plan, in one page

**Deliverable of the planning session.** A complete, buildable architecture + phased plan for the
ground-up rewrite of `structured-agents-v2` around a single central abstraction — the
**`Constraint[T]` codec**. Read order: **this page → DECISIONS → DESIGN → PHASES → TESTS → SALVAGE →
RISKS**, with **SPIKES** as the empirical spine underneath. Success metric (per kickoff): the
*coherence* of the architecture, not the cost to build it.

---

## The shape of v3

One value organizes the whole library: a **`Constraint[T]`** is a bidirectional codec — `wire()`
shapes the request going out, `parse()` turns raw output into a typed `T` coming back. Because the
constraint *carries* `T`, the type flows mechanically and **without a single `cast`**:

```
Constraint[T] → AgentSpec[T] → Agent[T] → Outcome[T] → Ok.value : T
```

Everything else falls out of taking "explicit, typed, composable value" literally on **three
orthogonal axes** — output (`Constraint`), weights (`Adapter`), input (`Context`) — each cooperating
with a server capability (XGrammar / LoRA / KV-reuse) it never reimplements, and a **one-way layer
stack**:

```
Layer 4  observe/     pipeline Observers (dual-path, evals)            [observe]
Layer 3  fleet · authority   Fleet/Router · Authorizer × Effector      [agent]
Layer 2  agent · context     AgentSpec[T]/Backend/Agent[T] (sole pydantic_ai) · Context
Layer 1  constraint · outcome   Constraint[T] codec · Outcome[T] spine
Layer 0  wire/ · closed        pydantic-ai-free transport · the closed preset
```

Three things become one each:
- **The constraint** — v2 smeared it across four places (`ConstrainedOutput` dunders, `DecoderSpec`,
  `apply()`, `_guard`). v3: one `Constraint[T]` value (`wire()` + `parse()`).
- **The result** — v2 had four result types and declined two ways (raise *and* data). v3: one
  `Outcome[T]` spine (`Ok/Denied/Violated/Failed`), decisions-as-data uniformly.
- **Authority** — v2 fused decide+do into one `Executor`. v3: `Authorizer × Effector`, an executor is
  the composition `authorize(a) >> effector`; DryRun and Fornix are compositions, not subclasses.

---

## The decisions taken (full rationale in DECISIONS.md)

| | Decision | Verdict |
|---|---|---|
| A | pydantic-ai coupling | **keep** as the Layer-2 loop, sole importer; swappable by the layer boundary |
| B | `Outcome[T]` spine | **full 4-variant, encoded as a generic base class + method combinators** (not a bare union) |
| C | fleet typing | `Agent[Any]` by nature + `fleet.typed(name, T)` re-narrow; no typed-router machinery |
| D/E | streaming / tools | **out of scope**; seams reserved (`Constraint` ≠ tool schema) |
| F | `Choice` generics | **`Choice[S: str](*o: S) -> Constraint[S]`** — ty infers the literal; concept's `Literal[*Opts]` is broken |
| G | context / session | ship the neutral per-segment `Context`; **defer `Session`** |
| H | repo & name | **new repo**, proposed name **`constric`** (fallback: keep `structured-agents`) — *needs user confirm* |
| I | extras | lean core `pydantic+httpx`; **pydantic-ai behind `[agent]`** so `closed` installs pydantic-ai-free |
| J | naming | lock the vocabulary; `Agent` vs `pydantic_ai.Agent` resolved by import discipline |
| K | config/plugins | per-seam registry + entry points; the import vector localized behind one `allow_modules` allowlist |
| L–R | (surfaced) | Backend/Closed separate but share `wire/`; `parse()` always runs; capture via `Ok.wire`; cap-errors are exceptions; `check()` at build; adapter provisioning on `Backend`; shared client + results-as-data |

**Three decisions were settled *empirically*, not by argument** (SPIKES.md, run against the repo's
real `ty 0.0.46` / `pydantic-ai 2.11.0`):
1. **Choice generics (F):** the concept's own constructor doesn't compile; a bounded `TypeVar S: str`
   is cleaner and infers the literal. Better than the concept.
2. **Outcome encoding (B):** the concept's bare `type Outcome[T] = Ok[T] | …` alias **fails to deliver
   its own flagship promise** — `ty` can't recover `T` from the union via `match`/`isinstance`/
   `TypeGuard`/`fold`. A generic **base class with method combinators** types perfectly. This is the
   single biggest correction to the concept.
3. **pydantic-ai surface (A/I):** `NativeOutput` importable, both `output_type` shapes accepted,
   `usage` is a **property** — so the v2 A1 bug is *structurally impossible* at the `>=2.11` floor.

---

## Where v3 departs from or improves on CONCEPT.md

1. **`Choice` constructor rewritten.** `Choice[*Opts] -> Constraint[Literal[*Opts]]` is a category
   error `ty` rejects (S1). Replaced by `Choice[S: str](*options: S) -> Constraint[S]`. *Improvement:*
   simpler, and it closes v2's deferred choice→Literal coercion for free.
2. **`Outcome[T]` is a class, not a union alias.** The concept's §5 code contradicts itself (a `class
   Outcome` with `.then` *and* a `type Outcome = …` alias). Only the class-with-methods form typechecks
   under ty (S2). *Correction:* the typed, no-cast promise holds **only** via method combinators;
   `match` stays a runtime-correct convenience (RISKS R1).
3. **pydantic-ai moves to an `[agent]` extra (I.2).** The concept keeps it a core dep; making it an
   extra lets the privacy-critical `closed` path install with **no pydantic-ai at all**, sharpening the
   very attack-surface goal the concept prizes (§8). *Improvement:* the layering becomes physical in the
   dependency graph, not just the imports.
4. **`NativeOutput` layering concession made explicit (R2).** The concept doesn't address that
   `Schema.wire()` must return a pydantic-ai type from Layer 1. v3 bounds it (marker-only import,
   test-enforced) rather than leaving it implicit.
5. **Composites (`Nullable`/`OneOf`) deferred to v3.1 (R3).** The concept sketches them as free, but
   their discriminated-union **wire shape was never captured** — shipping them would violate the "never
   guess a request shape" rule. Deferred until captured against tower.
6. **Everything else: adopted and sharpened**, not restated — the three axes, the `closed` preset over
   shared `wire/`, capture via `Ok.wire`, authority decomposition, the neutral per-segment `Context`,
   observers over the spine.

---

## Every v2 review finding is now *structurally impossible* (mechanism, one line each)

| Finding | v2 defect | v3 mechanism that kills it |
|---|---|---|
| **A1** | `raw.usage()` broke on newer pydantic-ai | `agent.py` accesses `raw.usage` as a **property** at the `>=2.11` floor (S3) |
| **A2** | grammar-check saw the parent's empty schema (subclass timing) | `Schema` holds a **fully-formed** model class; there is no lazy subclass (DESIGN §constraint) |
| **A4** | `build` clobbered user `extra_body` | `Backend._merge` merges, decoder keys win |
| **A5** | capture misattributed under concurrency; unbounded | capture delivered in `Ok.wire` — no shared "last" to race; bounded (DECISION N) |
| **B1** | `execute` skipped authorize | the `_Executor` composition **binds `decide` before `run`** — no skippable entrypoint |
| **B2** | a raising allow-rule crashed | `Allowlist.decide` wraps each rule → `Decision(False)` once, structurally |
| **B4** | promised bare-string guard never implemented | `Constraint.parse` **always runs**; is the inbound half of the codec |
| **B5** | `Effect.ok` never false | the `Effector` is the only fallible party; a raise → `Effect(ok=False)` → `Failed` |
| **C: typing** | `run()` returned `AgentResult[Any]` | `Constraint[T]` carries `T` end-to-end; `run -> Outcome[T]`, no cast (S2) |
| **C: client** | N unclosed clients | one shared `httpx.AsyncClient` per `Backend` + `aclose()` |
| **C: run_batch** | `gather` lost siblings | `run_batch -> list[Outcome[T]]`; per-item `Failed` |
| **C: settings typo** | silently dropped by a TypedDict | typed `Settings` dataclass — a typo is a **type error** |
| **C: import vector** | `output_type_ref` `importlib` in the hot path | localized to `config.constraint_from_config` behind `allow_modules` |
| **D** | grail dead dep; psycopg undeclared; pydantic-ai unbounded | grail dropped; psycopg declared in `[observe]`; pin `>=2.11,<3` |
| **F/G** | grail; fornix-as-executor | grail gone; **`FornixEffector`** = one effector + one composition (zero new dep) |

---

## Status of this planning session (definition-of-done)

- **Every decision (A–R) resolved** with recommendation + rationale + rejected alternatives
  (DECISIONS.md). Three settled by spike, not assertion (SPIKES.md).
- **Every module implementable from DESIGN.md** — real signatures, one-way dependencies, invariants.
- **PHASES.md leaves a green, demonstrable state at every step** (P1–P6 each independently green; P4
  and P6 natural release points) and covers the whole concept (three axes, closed, observe).
- **Every v2 finding shown structurally impossible** (table above) with a one-line mechanism.
- **Departures from CONCEPT.md stated explicitly** (six, above), each with why.

**Two things need a human before building:**
1. **DECISION H — the name/repo.** `constric` is a recommendation, not a derivation; new-repo-vs-branch
   is a strategy call. *Do not create the repo (Phase 0) until confirmed.* (RISKS R8.)
2. **The elegance-over-everything mandate** was honored literally in two places a pragmatist might
   veto: committing to the **full four-variant `Outcome`** (B) and moving **pydantic-ai to an extra**
   (I.2). Both are recommended and defended; both are reversible; flagged here for a conscious sign-off.
