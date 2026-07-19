# REVIEW — adversarial architecture review of structured-agents v3 (the Constraint-Codec plan)

**Session:** review kickoff (`REVIEW_KICKOFF.md`). Success metric: *coherence of the resulting
architecture*, not build cost. Everything labelled "settled/invariant/resolved" was put back on the
table. Spikes were independently re-run in devenv (see `REVIEW_SPIKES/` and §Independent spike
results). Ground rules honored: no edits to `src/` or `main`; all work in `.scratch/`.

**One-paragraph verdict.** The central bet — *make the constraint a first-class bidirectional value* —
is genuinely good and worth building; the `wire/` extraction, the authority decomposition, and the
typed `Constraint[T] → AgentSpec[T] → Agent[T]` flow are real improvements over v2 that survive
scrutiny. But three load-bearing claims **do not hold as written** and gate Phase 1: (1) the **closed
path is not pydantic-ai-free** the way the plan states — it transitively imports `pydantic_ai` via
`Schema`/`constraint.py`, and the T8 test as specified would *pass while the guarantee is false*;
(2) **`Schema.parse` as "identity" is a no-op that pretends symmetry** and is *wrong* for the one path
(closed) the plan uses it to unify — closed shares the `wire/` layer, **not** the `Constraint` codec;
(3) the **wire bodies were captured on pydantic-ai 1.87 and are being carried to a 2.11 pin without
re-capture**, repeating the exact "trust an unverified pydantic-ai surface" mistake that produced the
A1 bug. Beyond those, the `Outcome` sum-type spine is the weakest big idea in the plan: the S2 finding
("ty can't narrow the union") is used to *justify* method-combinators, but it more honestly *undercuts
having a sum-type result for `run` at all* — a `run -> T` (raise on failure) + a two-variant
`execute` result is more coherent and needs none of the combinator apparatus. Details, evidence, and
the alternative below.

---

## Severity-ranked findings

Legend: **BLOCKER** (gates Phase 1) · **MAJOR** (fix before the surface it touches ships) · **MINOR**
(tighten; not a gate). Each finding: the claim challenged (`file:section`), why it's wrong/weak, the
evidence, and the fix.

---

### BLOCKER 1 — The `closed` path is **not** pydantic-ai-free; the import-isolation guarantee is hollow as specified

**Claim challenged.** DESIGN §closed inv. (i): *"Imports only `wire/` + `constraint` + `pydantic` +
`httpx` — **never pydantic-ai** … this import isolation is the load-bearing guarantee … none of it is
on the closed code path."* DECISION I/L, T8, PHASES P4, RISKS R2 all repeat it.

**Why it's wrong.** The closed composition (DESIGN §closed, and CONCEPT §8) builds
`constraint = Schema(output_type)`. `Schema` lives in `constraint.py`, which — per R2 / DESIGN
§constraint / T8 — imports `pydantic_ai.output.NativeOutput` **at module level**. Importing
`constraint.py` therefore imports `pydantic_ai`. So `closed.py → constraint.py → import pydantic_ai`.
The guarantee "closed imports no pydantic-ai / none of it is on the closed code path" is **false
transitively**.

**Evidence (re-run, devenv, pydantic-ai 2.11.0):**
```
from pydantic_ai import NativeOutput            # what constraint.py does at module level
→ pydantic_ai.models.openai loaded?  False      # the *client* is NOT pulled  ✓ (narrow claim holds)
→ but `pydantic_ai` (top-level package) IS now in sys.modules, plus httpx/httpx2
```
So the only true, narrow claim is **"closed does not import `pydantic_ai.models.openai`"** (the
client). The plan states the much stronger "no pydantic-ai," which is false. Worse: **T8 is specified
as an AST test on each module's own source** (TESTS.md T8: *"parses each module's imports (AST)"*).
An AST test of `closed.py` sees no literal `import pydantic_ai` and **passes** — certifying a
guarantee the running process violates. A green T8 would give false confidence.

**Why it matters.** For Lodestar the whole point of `closed` is a minimal attack surface / no
pydantic-ai on the privacy path. "Installed-but-unused" (DECISION I) is defensible only if *unused*
means *not imported*; here it **is** imported the moment `closed_backend()` constructs its `Schema`.

**Fix (pick one, state it plainly):**
- **(a) Make the guarantee real.** `constraint.py` imports `NativeOutput` **lazily inside
  `Schema.wire()`**, not at module level. Then importing `constraint.py` / constructing `Schema` /
  calling `Schema.parse` never loads `pydantic_ai`; only `Schema.wire()` (which closed never calls)
  does. Change T8 to a **runtime** assertion: import `structured_agents.closed` in a fresh subprocess
  and assert `"pydantic_ai" not in sys.modules`. This contradicts R2's "module-level marker import"
  framing — so R2 must be rewritten, not just footnoted.
- **(b) Sever closed from `constraint.py` entirely** (this is what v2 already does — `closed.py`
  imports no decoder). Closed builds its body via `wire/request.response_format(model, ...)` and
  validates via `model_validate_json` directly; it never touches `Schema`. Then the honest statement
  is **"closed shares `wire/`, not `constraint`"** (see BLOCKER 2). This is cleaner and makes the
  isolation trivially true.

Recommend **(b)** as primary (it also resolves BLOCKER 2) with **(a)** as the mechanism if you insist
`Schema` be the shared value. Either way: **restate the guarantee as testable-transitive and fix
T8**, or the plan's headline privacy property is unproven.

---

### BLOCKER 2 — `Schema.parse` "identity" is a no-op that pretends symmetry, and is *wrong* for the closed path; the "closed rides the same `Constraint[T]` codec" claim is false

**Claim challenged.** DESIGN §constraint wire/parse table: `Schema(M).parse` = *"identity (pydantic-ai
already returned a validated `M`)."* CONCEPT §8: closed *"rides the **same** `Constraint[T]` codec as
everything else."* DECISION M/§4.2: *"`parse()` **always runs** … the inbound half of every codec …
the safety net when the server claims to enforce but doesn't (B4)."* SALVAGE: closed = *"preset over
shared `wire/`+`constraint`, zero duplicated validation."*

**Why it's wrong — two ways:**

1. **In the rich path, `Schema.parse` does literally nothing.** It receives an already-validated `M`
   from pydantic-ai and returns it. So for the *most important* mode, the "always-runs inbound guard
   that kills B4" is **vacuous** — the actual enforcement is pydantic-ai's `NativeOutput` validation,
   not the codec. The "`parse` always runs as the B4 safety net" story is true only for
   `Regex`/`Choice`/`Grammar`. For `Schema` there is **no client-side safety net in the codec**; the
   plan should say so rather than imply uniform coverage.

2. **In the closed path, "identity" is actively wrong.** Closed does not run pydantic-ai; it gets a
   raw JSON **string** back from `wire/client.call` (`WireResult.content: str`). To produce the `M`
   that `ClosedBackend.run -> BaseModel` promises, *something must `model_validate_json(content)`*. If
   `Schema.parse` is identity, closed returns the raw string, not a model — broken. So either:
   - `Schema.parse` must **branch on its input** (`return raw if isinstance(raw, M) else
     M.model_validate_json(raw)`) — which makes "identity" a lie and is **untested** (T1 only asserts
     `parse(valid_M_instance) is that_instance`, the rich-path identity — never the closed-path
     string→model path); **or**
   - closed does its **own** `model_validate_json` outside the codec — which makes *"closed rides the
     same `Constraint[T]` codec"* **false**. What closed genuinely shares is `wire/` (transport,
     bounded inputs, `response_format` builder, retention) — real and valuable — **not** the codec.

**Why it matters.** The codec's bidirectional symmetry (`wire()` out / `parse()` in) is the plan's
linchpin and its headline unification of rich+closed. That symmetry **breaks precisely at `Schema`** —
the only mode `closed` uses and the dominant rich mode. `wire()` for `Schema` returns a pydantic-ai
type (unusable by closed); `parse()` for `Schema` is a no-op in rich and a different operation in
closed. The abstraction is real for the three string modes and *illusory for `Schema`*.

**Fix.**
- State honestly: **closed shares the `wire/` layer, not the `Constraint` codec.** Update CONCEPT §8,
  DECISION L, SALVAGE, and 00-PLAN accordingly (this is a downgrade of a headline claim — do it
  loudly, not in a footnote).
- Decide what `Schema.parse` *is*. The cleanest is: `Schema.parse(raw)` **accepts either a validated
  `M` or a JSON string** and returns a validated `M` (call `model_validate`/`model_validate_json`).
  Then it is genuinely the inbound half, does real work in *both* paths, and closed can use it — which
  also makes the codec truly shared. But then the rich path's "identity" comment is wrong and T1 must
  test the string→model direction. Pick this and rewrite the table, or admit the asymmetry.

---

### BLOCKER 3 — Wire bodies were captured on pydantic-ai **1.87**; v3 pins **2.11** and carries them "verbatim" without re-capture — the A1 mistake, repeated

**Claim challenged.** SPIKES §"What the spikes did **not** need to test": *"The wire bodies … are
already empirically captured in VERIFICATION.md … they carry over **verbatim** … No re-capture needed
to plan."* SALVAGE marks the wire table and `response_format` body **VERBATIM**. T2 asserts
`Schema.wire().output_type is NativeOutput(M)` — **not** the actual on-wire `response_format` body for
the rich path.

**Why it's weak.** `VERIFICATION.md` is dated 2026-06-09 and its capture table is explicitly *"How
PydanticAI **1.87** emits each output type."* v3 pins `pydantic-ai-slim[openai] >=2.11,<3` (a **major**
bump). The rich `Schema` path's actual `response_format` JSON is produced **inside pydantic-ai's
`NativeOutput` translation**, not by library code — so a change in how 2.11 serializes
`NativeOutput → response_format` (name, `strict` placement, `$defs`/`$ref` handling,
`additionalProperties`) would silently drift the wire shape. The 1.87→2.11 jump **already moved one
surface** (`usage` method→property — the A1 bug). Declining to re-capture the wire bodies on 2.11 is
the same class of error: trusting an unverified pydantic-ai surface across a version boundary.

**Evidence.** S3 (re-run, confirmed) shows `usage` is now a property and `NativeOutput` importable at
2.11 — i.e. the surface *did* move between the captured version and the pinned one. The plan verified
the *type* surface at 2.11 but **not** the *wire* surface at 2.11.

**Fix (gates Phase 1 sign-off).** Re-capture all four `.wire()`-driven `/chat/completions` bodies —
**including the rich `Schema` path's real `response_format`** — against pydantic-ai **2.11** (ASGI mock
is enough; the byte-shape is deterministic), diff against VERIFICATION.md, update VERIFICATION.md with
a 2.11 column, and make **T2 assert the on-wire body for the rich Schema path**, not just that
`output_type is NativeOutput(M)`. Add the R6 canary. Until then, "verbatim" is an assumption, not a
fact — and it is load-bearing for server enforcement.

---

### MAJOR 4 — The `Outcome` sum-type spine is the weakest big idea; the S2 finding argues for *not having a sum-type result for `run` at all*

**Claim challenged.** DECISION B / DESIGN §outcome: model the result as a **generic base class with
method combinators** because *"ty can't narrow the union"* (S2). 00-PLAN calls S2 *"the single biggest
correction to the concept."* The kickoff explicitly asks (item 4): does the finding argue for a
different result shape entirely?

**The finding is solid; the response is a non-sequitur.** S2 (independently reproduced — see
§spikes) confirms: a bare `type Outcome[T] = Ok[T] | Failed` union **cannot** recover `T` via
`match`/`isinstance`/`TypeGuard`/`fold` under `ty` 0.0.46, while a base class with methods types
perfectly. **But the reason the class-with-methods form works is that `T` flows *forward* from the
class parameter — i.e. you only need forward flow.** That is exactly what a plain return type gives
you *for free, with no wrapper at all*:

- **`run` has two outcomes: it produced a `T`, or it didn't.** "It didn't" (model error, transport
  error, a `ConstraintViolation` because the backend didn't enforce) is a **failure**, not a
  *decision* — DECISION O itself says *"exceptions are for … runtime failure of a correctly-built
  pipeline"* is a grey area, but a backend that won't obey `response_format` is far closer to "failure"
  than to "a policy chose no." So `async def run(...) -> T` that **raises** `ConstraintViolation` /
  model errors is honest, fully typed (it's just `T` — `ty` needs no combinators, no union, no
  match-narrowing workaround), and Pythonic. `T | None` + a captured error is the softer variant.
- **`Denied` is the one genuine decision-as-data** (the safety boundary saying "no" is not an error).
  Keep it — but it only occurs on `execute`, whose result is then a **two-variant**
  `Ok[Effect] | Denied` (or `Effect | Denied`), consumed by a single `match`. No four-way spine, no
  base-class combinator library, no `.then/.map/.value_or/.unwrap` apparatus.

**Why the current shape is cosplay in Python.** (a) The typed path is method-combinators, which almost
no Python caller reaches for; the *natural* consumption is `match`, which the plan itself admits is
**not statically narrowed** (R1) — so the concept's flagship `match oc: case Ok(value=cmd)` (§15) is
*untyped*, at the single most important call site. (b) The executed pipeline's 3-way result
(`Ok|Denied|Failed`) is consumed by `match` regardless — the combinators don't help there. (c) So the
entire combinator apparatus exists to work around union-narrowing, a problem you only have *because you
chose a union*. Drop the union for `run` and the problem, the apparatus, and the S2 "correction" all
evaporate.

**Recommendation (a real cleaner-architecture delta — see §Cleaner architecture).** `run -> T`
(raise `ConstraintViolation`/model errors), `run_batch -> list[T | ConstraintViolation]` or
`list[Result]` where results-as-data genuinely matters (batch, to avoid losing siblings — that's a
legitimate reason to keep a tiny wrapper *there*), and `execute -> Ok[Effect] | Denied`. This keeps
"denials are data" (the invariant that actually earns its keep) and drops the sum-type spine that
doesn't. If you keep `Outcome`, at least **down-rank R1 from MEDIUM to HIGH** and stop calling the
`match` ergonomic delivered — it isn't.

---

### MAJOR 4b — The `Outcome.then` combinator itself needs an internal `cast` — falsifying "no cast anywhere" *inside the apparatus S2 was invoked to justify* (re-run evidence)

**Claim challenged.** DESIGN §outcome, combinator semantics: `then` — *"`Failed`/`Denied` pass
through, re-typed to `Outcome[U]` since they carry no `T`. **Verified typed via S2 (method form).**"*
Principle #3 / DESIGN §agent: *"No casts that assert what the checker can't verify … no cast anywhere —
the whole point."*

**Why it's wrong (reproduced).** The pass-through body `def then[U](self, f) -> Outcome[U]: return
self` does **not** typecheck under `ty` 0.0.46:
```
error[invalid-return-type]: Return type does not match returned value
   expected `Outcome[U@then]`, found `Self@then`
```
`Self` (an `Outcome[T]`) is not accepted as `Outcome[U]`. The Failed/Denied short-circuit therefore
requires an **internal `cast`** (or a per-subclass override that duplicates the method). So the
`Outcome` spine — the apparatus the S2 finding was used to justify precisely *because* it avoids casts
— **cannot be implemented without one**. The "verified typed via S2" annotation in DESIGN is false for
`then` specifically (S2b tested `map`/`unwrap`/`value_or`, which *do* type — but **not** `then`'s
pass-through, which is the one the pipeline composition depends on).

**Why it matters.** `then` is the bind that `fleet.execute` is *"literally a bind chain"* built on
(DESIGN §fleet). If the bind needs a cast, the pipeline's "typed, no-cast" composition is a cast behind
a method. Combined with MAJOR 4 (the union shouldn't exist for `run`) and MAJOR 10 (`fleet.typed`
cast), the "no cast anywhere" invariant is contradicted in **three** places, all in the result/typing
spine.

**Fix.** Two honest options: (a) accept **one localized, audited `cast`** in `Outcome.then` (and say so
— it's the standard way to write a monadic bind in a nominal type system; there's no shame in it, only
in claiming otherwise); or (b) drop `then`/the sum type for `run` (MAJOR 4), which removes the need for
the bind entirely. Either way, strike "no cast anywhere" — it is not achievable with this spine.

---

### MAJOR 5 — The "no same-event-two-ways" claim is over-stated: the lighter spine relocates the asymmetry rather than dissolving it

**Claim challenged.** DECISION B: *"There is no same-event-two-ways split — each event has one
canonical shape … `Violated`-as-a-`Failed`-subtype keeps the signal without a fourth variant."*
00-PLAN: *"More idiomatic; no same-event-two-ways split."*

**Why it's weak.** After the user's lighter spine, "the pipeline correctly declined for a non-error
reason" has **two** structural encodings:
- **policy denial** → a top-level **variant** `Denied` (data);
- **backend-didn't-enforce** → an **error type** `ConstraintViolation` **inside** `Failed.error`.

Both are "not a hard model/transport failure; the boundary caught something." A caller who wants to
treat "declined-but-not-crashed" uniformly must check a *variant* in one case and an *error subtype* in
the other — two shapes for one conceptual category. That is a milder cousin of exactly the v2 wart
("two ways to decline") the plan claims to have dissolved. It is **defensible** (the partition
run-is-binary / execute-adds-authority is clean), but the plan **over-claims** it as fully dissolved.
This is also downstream of MAJOR 4: if a `ConstraintViolation` is a *failure* (and it reads as one),
raising it is more honest than boxing it as data-inside-Failed.

**Fix.** Soften the claim to what's true: *run is binary; execute adds authority as data; a
constraint-violation is a diagnosable failure.* Don't assert the wart is gone — it's reduced. And
reconcile with DECISION O: state explicitly why a non-enforcing backend is "failure" (Failed) but a
policy "no" is "decision" (Denied) — the line is real but currently asserted, not argued.

---

### MAJOR 6 — B1's footgun is **relocated**, not eliminated: `Effector.run` is publicly callable with no `Decision`

**Claim challenged.** DESIGN §authority / 00-PLAN: *"B1 (execute skipped authorize) … there is **no**
`execute` entrypoint that can skip [decide] — impossible by construction."*

**Why it's weak.** The `_Executor` composition does bind `decide` before `run` — good. But `Effector`
is a public `Protocol` with `async def run(self, command: C) -> Effect`. **Any caller can call
`some_effector.run(cmd)` directly**, performing the side effect with no authorization — the *exact*
shape of v2's B1 ("a public entrypoint that skips the check"). The composition makes the *blessed* path
safe; it does not make the *unsafe* path unrepresentable. v2's B1 was "`execute()` skipped
`authorize()`"; v3's is "`Effector.run()` skips `Decision`." Renamed, not removed.

**Fix.** If B1 is to be *structurally* impossible (the plan's word), the effect must not be invocable
without a decision as a **type**. Options: make `Effector.run` take a proof-of-authorization token
(`def run(self, authorized: Authorized[C]) -> Effect`, where only the composition can mint
`Authorized`), or keep `Effector.run` but document that it is a **hazard** and route all use through
`authorize(a) >> e`. At minimum, stop claiming "impossible by construction" — it's "safe on the blessed
path." (Same critique applies to `Subprocess`/`FornixEffector` being directly runnable.)

---

### MAJOR 7 — `Effect` is redundant with `Outcome`, and the B5 "fix" reintroduces the very success-bool it faults

**Claim challenged.** DESIGN §authority: `Effect{ok, output, detail}`; B5 fixed by making `Effect.ok`
*"meaningful."* `_Executor` then maps `Effect(ok=False) -> Failed(RuntimeError(detail))`,
`Effect(ok=True) -> Ok(eff)`.

**Why it's weak.** `Effect` is a second result type that exists only to be *immediately re-wrapped*
into `Outcome` one line later. The library already has a canonical "did it work" value — `Outcome`.
Two things:
- The v2 B5 finding was *"`ExecResult.ok` is a dead success-bool."* The v3 fix keeps a success-bool
  (`Effect.ok`) and merely wires it — i.e. it re-adopts the shape whose misuse it's citing, then
  bridges it to `Outcome`. If `Effector.run` returned `Outcome[EffectOutput]` directly (raise→`Failed`
  handled by the composition, success→`Ok`), there would be **no `ok` bool to be dead**, and one fewer
  result type.
- `Effect.output: Any` and `Effect.detail: str` re-encode what `Ok.value` / `Failed.error` already
  carry.

**Fix.** Drop `Effect`; `Effector[C]` returns `Outcome[R]` (or raises, and the composition catches →
`Failed`). One result spine, not "Outcome for generation, Effect for effects." (If you keep the
`Outcome` spine at all — see MAJOR 4 — this is where it should be *reused*, not paralleled.)

---

### MAJOR 8 — Three-axis orthogonality **leaks**: the context axis's cache identity depends on the adapter axis

**Claim challenged.** CONCEPT §20 / principle #9: *"three orthogonal axes … switching any one is a
full cache hit for the others."* Invariant listed as non-negotiable.

**Why it's weak.** §22.5 (correctness bookkeeping the library owns): `cache_salt = hash(content) +
base_model + adapter` — *"never content alone,"* precisely because *"a per-agent LoRA changes the
KV."* So **changing the adapter changes the context's cache namespace by construction.** "Switching the
adapter is a full cache hit for the context" is therefore **false** for chunk-cached context: a new
adapter → new salt → cache miss. The plan asserts orthogonality (§20) and its own violation (§22.5) two
sections apart. The axes are orthogonal *in expression* (three separate values on `AgentSpec`) but
**not** *in cache behavior* (context KV is adapter-scoped). The kickoff's suspicion — "real
orthogonality or a tidy story that will leak" — is confirmed: it leaks exactly here.

**Impact for v3.0:** LOW (CHUNK is deferred; PREFIX cache is naturally adapter-scoped anyway). But the
**invariant as stated is false**, and it's on the non-negotiable list. Fix the wording: axes are
orthogonal in *expression and wire placement*; cache *identity* deliberately couples context to
(base_model, adapter) for correctness. Don't sell independence you don't have.

---

### MAJOR 9 — Shipping the full `Context`/`Reuse`/`Fidelity` axis in v3.0 is speculative generality with no consumer (YAGNI)

**Claim challenged.** DECISION G / DESIGN §context / PHASES P5: ship
`Context/Segment/Reuse/Fidelity/LinearPrefixProvider` + `cache_salt` bookkeeping now; `CHUNK`,
`ChunkProvider`, `Session` deferred.

**Why it's weak.** The *only* behavior exercised in v3.0 is: a lone instruction string → one `PREFIX`
system segment; `LinearPrefixProvider` emits messages in order + `query` as trailing user;
`Reuse.CHUNK` is *degraded to PREFIX*; `Fidelity` has one live value (`EXACT`). That is
**functionally identical to v2's `instructions: str`**. Everything else — the `Reuse` enum's `CHUNK`
arm, `Fidelity.BLENDED`, `Segment.id`, the `cache_salt = hash+base_model+adapter` computation, the
`WireMessages.cache_options` opaque passthrough — is **machinery with no reader**. The plan even says
the salt is *"implemented even though the default provider is prefix-only, so the invariant is enforced
from day one"* — i.e. a correctness invariant enforced on a code path that **cannot be taken** until
v3.1's `ChunkProvider` exists. That's the definition of speculative generality the kickoff warns about.

**Fix.** For v3.0 ship the **seam** without the **machinery**: `AgentSpec.context: Context`, where
`Context` for now is just an ordered list of `(role, content)` + `Context.of(str)` sugar and a
`ContextProvider` Protocol. Defer `Reuse`, `Fidelity`, `Segment.id`, and `cache_salt` to the phase that
actually lands `ChunkProvider` (they arrive *with their consumer*, fully testable against real chunk
behavior, instead of as unexercised invariants). You lose nothing — `Context` is still the value, the
axis still exists — and you stop shipping correctness code you can't run.

---

### MINOR 10 — `fleet.typed(name, T)` "re-narrows without a cast" — the cast is just hidden inside the library

**Challenge.** DESIGN §fleet / DECISION C: `fleet.typed("git_ops", GitCommand) -> Agent[GitCommand]`
*"re-narrows for typed call sites **without a `cast`**."* But `__getitem__` returns `Agent[Any]`;
producing `Agent[GitCommand]` from `Agent[Any]` is exactly a `cast` (or an equivalent unchecked
`assert_type`). The plan's own DESIGN §agent boasts *"no cast anywhere — the whole point."* `fleet.typed`
contradicts it. It's **honest-ish** (localized, runtime-checked against the spec's constraint) but it is
the "typing dodge" the kickoff flags — a cast relocated from caller to library. Say so: "one localized,
runtime-verified narrowing cast," not "without a cast."

---

### MINOR 11 — A1/A5 "structurally impossible" is over-stated ("structural" vs "pinned + spiked-later")

- **A1** is impossible only *at the `>=2.11` floor* and guarded by a canary (R6); a `2.x` bump could
  move `usage` again. That's "fixed by a pin + canary," not "structural." (Contrast A2, which really is
  structural — no subclass, formed schema. That one holds; confirmed.)
- **A5**: delivery via `Ok.wire` removes the *shared* "last" sink, but RISKS R4 admits the **per-run
  correlation mechanism is unspiked** (httpx `Request.extensions` vs a run-scoped sink, Phase-6
  backlog). Until that spike lands, "misattribution is structurally impossible" is *designed-to-be*,
  not *proven*. If the internal correlation is keyed wrong, the A5 race recurs. Down-grade the wording
  to "designed race-free, pending the R4 spike."

**Fix.** Reserve "structurally impossible" for A2/B2 (genuinely structural). For A1/A4/A5/B1/B5 say
"fixed by construction on the blessed path, guarded by test/pin" — accurate and still strong.

---

### MINOR 12 — DECISION B contains a live self-contradiction (internal-consistency defect in the spine doc)

DECISIONS §B has **two** "Rejected" blocks. The second (`DECISIONS.md:81-83`) reads: *"(b) Lighter
Ok/Failed only — resurrects the raise-vs-data split. (c) Exceptions throughout — the v2 status quo…"* —
i.e. it **rejects the user's chosen design**, contradicting the "USER DECISION: the lighter Ok/Failed
variant" at the top of the same section (`:41`) and the first Rejected block (`:77-79`). This is
leftover from the pre-user-decision recommendation. A reader of the spine doc cannot tell which way B
resolves. **Fix:** delete the stale second Rejected block. (Also note: MAJOR 4 argues (c)
"exceptions for `run`" is not obviously wrong — so don't just delete it, *reconsider* it.)

---

### MINOR 13 — Wire-name/shape drift between the rich `Schema` path and `closed`

The rich path's `response_format` is emitted by pydantic-ai's `NativeOutput` (name/schema chosen by
pydantic-ai); `closed`'s is built by `wire/request.response_format(model, name="closed_output",
strict=True)` from `model.model_json_schema()`. SALVAGE lists "the `response_format` json_schema body"
as one **VERBATIM** shared shape, but the two paths generate it via **different code** and at least the
`name` differs (`"output"`/pydantic-ai default vs `"closed_output"`). They are **not** guaranteed
byte-identical, and nothing tests that they agree. This compounds BLOCKER 3. **Fix:** either route both
through `wire/request.response_format` (only possible if closed doesn't use `Schema.wire()` — BLOCKER
1(b)/BLOCKER 2), or explicitly document that the rich and closed json_schema bodies differ and test
each against its own capture.

---

### MINOR 14 — PHASES P1 acceptance "the whole layer imports **no** pydantic-ai" is unsatisfiable as written

PHASES P1 lands `constraint.py` and asserts *"the whole layer imports **no** pydantic-ai (T8
partial)"* and *"Demo … no server, no pydantic-ai."* But `constraint.py` (landed in P1) imports
`pydantic_ai.output.NativeOutput` per R2/DESIGN. So P1's own acceptance criterion is false unless the
lazy-import fix (BLOCKER 1a) is adopted. Reconcile P1 with the R2 concession.

---

### MINOR 15 — Optimistic risk severities

R1 (`match` not narrowed) is rated **MEDIUM impact / retired-by-design**, but it defeats the concept's
flagship ergonomic (§15) and is the natural consumption path for the 3-way `execute` result → **HIGH**
in practice. R2 (`NativeOutput` layering) is rated **LOW impact**, but it is the *root of BLOCKER 1*
(closed isolation) → materially higher. Re-score both; a risk register that under-rates its own
load-bearing items misleads the build.

---

## What survives scrutiny (upheld with fresh evidence)

Not everything breaks; several bets are genuinely sound and should be built as-is:

- **`Constraint[T]` carrying `T` for the *string* modes** (`Regex`/`Choice`/`Grammar`) is a clean, real
  win: `wire()`+`parse()` on one value, `parse` doing the actual client-side guard (kills B4 *for those
  modes*), and `Choice[S: str]` inferring the literal (S1 upheld — see spikes). This is the best part
  of the plan and needs no change.
- **The `wire/` extraction** (transport/loopback-guard/bounded-inputs/retention as shared value
  objects) is the right DRY: it removes v2's duplicated closed-vs-rich validation *without* a subclass
  hierarchy. Genuinely shared, genuinely cleaner. (It is what closed *actually* shares — see BLOCKER 2.)
- **Authority as `Authorizer × Effector`** is the right decomposition for B2 (structural fail-closed via
  `Allowlist` wrapping each rule) and for "where does fornix go" (a composition, not a subclass). B2 is
  genuinely structural. (B1/B5 are weaker — MAJOR 6/7 — but the *split* is right.)
- **A2 is genuinely, structurally dead:** no `ConstrainedOutput` subclass, `Schema` holds a formed
  class, `model_json_schema()` is complete. The best of the "structurally impossible" claims.
- **The config/allowlist localization** (one `constraint_from_config` gated by `allow_modules`) is the
  right shape for the import-execution vector; C-import-vector is real.
- **Results-as-data for `run_batch`** (no lost siblings) and **one shared client** (R) are correct and
  should carry.
- **`Choice[S: str]`** replacing the concept's `Literal[*Opts]` is a real improvement (S1). Upheld.

---

## Every v2 finding's "structurally impossible" claim, checked

00-PLAN asserts every v0.2.0 review finding is now *structurally impossible*. Audited one by one
(**Confirmed** = truly impossible by construction; **Relocated** = footgun moved, not removed;
**Pinned** = fixed but by a version/test guard, not structure; **Partial** = holds for some modes/paths
only):

| Finding | Plan's mechanism | Verdict | Note |
|---|---|---|---|
| **A1** `usage()` call | `raw.usage` property at `>=2.11` | **Pinned** | Real at the floor (S3 ✓), but a pin+canary, not structural (MIN11). |
| **A2** parent empty schema | `Schema` holds a formed class; no subclass | **Confirmed** | Genuinely structural — the strongest claim. |
| **A4** `extra_body` clobber | `Backend._merge`, decoder keys win | **Confirmed** | Straightforward merge; holds. |
| **A5** capture misattribution | `Ok.wire` delivery, no shared sink | **Partial/Pinned** | Removes the shared "last", but per-run correlation is **unspiked** (R4, Phase 6) — designed-race-free, not yet proven (MIN11). |
| **B1** execute skips authorize | `_Executor` binds decide before run | **Relocated** | `Effector.run` is publicly callable with no `Decision` — same footgun, renamed (MAJOR 6). |
| **B2** raising rule crashes | `Allowlist.decide` wraps each rule | **Confirmed** | Structural fail-closed; holds. |
| **B4** bare-string guard missing | `Constraint.parse` always runs | **Partial** | True for Regex/Choice/Grammar; **vacuous for Schema** (parse=identity) — the safety net there is pydantic-ai, not the codec (BLOCKER 2). |
| **B5** `Effect.ok` never false | effector is the only fallible party | **Confirmed-but-redundant** | Works, but re-introduces the success-bool shape it faults; `Effect` should be `Outcome` (MAJOR 7). |
| **C: typing** `Any` at seam | `Constraint[T]` end-to-end | **Partial** | Real for the value; undercut by `then`-cast (4b), `fleet.typed` cast (10), `match` non-narrowing (4). |
| **C: client** N pools | one shared client + `aclose` | **Confirmed** | Promoted v0.2.0 fix; holds. |
| **C: run_batch** lost siblings | `list[Outcome[T]]`, per-item Failed | **Confirmed** | Holds. |
| **C: settings typo** silent drop | typed `Settings` dataclass | **Confirmed** | A typo is a `ty` error (T7 negative test); holds. |
| **C: import vector** | `config.allow_modules` at one fn | **Confirmed** | Localized + gated; holds. |
| **D** packaging | grail dropped, psycopg declared, pin | **Confirmed** | Correct; note pin makes A1 "pinned" not "structural". |
| **F/G** grail/fornix | gone / `FornixEffector` composition | **Confirmed** | Right disposition. |

**Net:** of 15 claims, ~8 are genuinely structural, 3 are pinned/partial (A1, A5, C-typing), 2 are
partial-by-mode (B4, B5), and **1 is relocated not removed (B1)**. The plan's blanket "every finding
structurally impossible" is true for the majority but **over-stated** for A1/A5/B1/B4. Reserve
"structural" for A2/B2/C-client/C-batch/C-settings/C-import.

---

## Cleaner architecture delta

After trying to break the plan, one coherent alternative shape emerges. It is **not** a wholesale
redesign — it keeps the plan's best bets (the `Constraint` codec for string modes, the `wire/`
extraction, `Authorizer × Effector`, config allowlist) and changes **three** things where the plan
over-reaches. Net: **fewer concepts, fewer result types, fewer false claims.**

**Δ1 — Collapse the result spine to what each entrypoint actually needs.**
- `Agent.run(prompt) -> T` — **raises** `ConstraintViolation` / model+transport errors. `T` flows
  natively; `ty` types it perfectly with **zero wrapper, zero combinator, zero cast, zero
  match-narrowing problem** (the S2 problem only exists because the plan chose a union). A
  non-enforcing backend is a *failure*, and Python spells failure with an exception (DECISION O even
  admits exceptions are for runtime failure of a correct pipeline).
- `run_batch -> list[RunResult[T]]` where `RunResult[T] = Ok[T] | Failed` — **here** results-as-data
  genuinely earns its keep (don't lose siblings), and it's a *two-variant* consumed by one `match`.
  This is the one place a small wrapper is justified; keep it *only* here.
- `fleet.execute(...) -> Ok[Effect] | Denied` — `Denied` is the single genuine decision-as-data (the
  safety boundary saying "no" is not an error). Two variants, one `match`. `Effector.run -> R` or
  raises (the composition turns a raise into the failure path).
- **Dropped:** the `Outcome[T]` base class, `.then/.map/.value_or/.unwrap`, `Effect`, `Violated`, and
  the `then`-cast (MAJOR 4b). What's kept — `Denied`-as-data and batch-results-as-data — is exactly
  the subset that pays for itself.
- **Argument vs the plan:** the plan spends a base class + 5 combinators + a documented ty limitation
  (R1) + an internal cast (MAJOR 4b) to deliver a typed `Ok.value`. This delta delivers a typed value
  *as the return type itself*. The monadic spine is elegant in ML; in Python, over `ty`, it is
  net-negative — the combinators are the only typed path, callers reach for the untyped `match`, and
  the bind needs a cast. Less machinery, more honesty.

**Δ2 — Make the codec/wire split honest: `closed` shares `wire/`, and `Schema.parse` does real work.**
- State plainly: **`closed` shares the `wire/` layer, not the `Constraint` codec.** That is true,
  testable, and still eliminates v2's duplication (the duplication was in `wire/`-level validation, not
  in the codec).
- `Schema.parse(raw)` **validates** (`model_validate`/`model_validate_json`) rather than being
  identity — so it does real work in *both* paths and closed can genuinely reuse it if desired. This
  makes the "inbound half always runs" claim *true* instead of vacuous, and closes the closed-path
  test gap.
- `NativeOutput` imported **lazily inside `Schema.wire()`** → the closed isolation guarantee becomes
  literally true and transitively testable. (Δ2 subsumes BLOCKERs 1 & 2.)

**Δ3 — Ship seams, not speculative machinery.**
- `Context` is a value (an ordered `(role, content)` list + `Context.of(str)` sugar) and a
  `ContextProvider` Protocol — the *seam*. Defer `Reuse`/`Fidelity`/`Segment.id`/`cache_salt` to the
  phase that lands `ChunkProvider` (their only consumer). Same for keeping `Nullable`/`OneOf` out until
  their wire shape is captured (the plan already does this — good; apply the same discipline to the
  context axis).

**Where the current plan already *is* the cleanest — don't churn these:**
- The `Constraint` codec for `Regex`/`Choice`/`Grammar` (one value, `wire`+`parse`, literal-typed
  `Choice`). Leave it exactly as designed.
- `wire/` value objects (loopback guard, bounded inputs, retention). Leave as designed.
- `Authorizer × Effector` *split* + `Allowlist` structural fail-closed (B2) + fornix-as-composition.
  Leave the split; only fix the `Effector.run`-sans-decision hazard (MAJOR 6) and drop `Effect` in
  favor of the result type (MAJOR 7).
- Config allowlist localization, shared client, results-as-data batch. Leave as designed.

**Honest counter-argument (why the plan's spine might still win):** a uniform `Outcome` at *every*
stage means one mental model and one `then`-composed pipeline expression — real value if the pipeline
grows more stages (retry, cache-check, post-process) later. If the roadmap genuinely expects a
multi-stage pipeline where uniform bind pays off, the sum-type spine amortizes. But *today* there are
three stages (route/generate/authorize+effect), two of them consumed by `match` regardless, and the
bind needs a cast — so the amortization is speculative and the cost is present. On the kickoff's own
metric (coherence *now*, not future-proofing), Δ1 wins. Present it to the user as a re-open of
DECISION B, with this counter-argument stated.

---

## Verdicts on every "settled" item

*(Independent-evidence one-liners. "Upheld" = survived; "Weakened" = holds but over-claimed; "Overturned"
= should change.)* Spike-conclusion rows finalized in §Independent spike results.

| Item | Verdict | Evidence (one line) |
|---|---|---|
| **Kickoff invariant: one first-class constraint value** | **Upheld (string modes) / Weakened (Schema)** | Real for Regex/Choice/Grammar; for `Schema`, `wire()`→pydantic-ai type & `parse()`→no-op break the codec symmetry (BL2). |
| **Type-honesty / no-cast** | **Weakened** | True for the `Constraint[T]→Ok.value` chain; violated by `fleet.typed` (hidden cast, MIN10) and undercut by `match` non-narrowing (MAJ4). |
| **Three orthogonal axes** | **Weakened** | Orthogonal in expression; cache identity couples context→adapter by design (MAJ8). |
| **Decisions-as-data / explicit-effects** | **Upheld (denials) / Weakened (violations)** | `Denied`-as-data earns its keep; `ConstraintViolation`-in-`Failed` gives two shapes for "declined" (MAJ5). |
| **Authority = decision × effect** | **Upheld (split) / Weakened (B1,B5 claims)** | Split is right; `Effector.run` still callable sans decision (MAJ6); `Effect` redundant (MAJ7). |
| **One-way layering; pydantic-ai confined to agent** | **Overturned as stated** | `constraint.py` imports `pydantic_ai` at module level → `closed` transitively imports it (BL1); only the *client* (`models.openai`) is truly confined. |
| **`closed` guarantees (no pydantic-ai / shares codec)** | **Overturned** | Not pydantic-ai-free transitively (BL1); shares `wire/`, not the codec (BL2). |
| **Wire mode table verbatim** | **Weakened / unproven** | Captured on 1.87; pinned to 2.11 without re-capture; rich `Schema` body is pydantic-ai-produced & untested (BL3, MIN13). |
| **Library-owns-cache-bookkeeping** | **Weakened (YAGNI)** | Correct in principle; shipped ahead of any consumer, enforced on an untakeable path (MAJ9). |
| **User decision B — lighter Ok/Failed spine** | **Weakened; re-open recommended** | Coherent but over-claimed ("no two-ways" — MAJ5); S2 arguably kills the sum type for `run` entirely (MAJ4). Present the `run -> T` alternative for re-confirmation. |
| **User decision I — pydantic-ai in core** | **Upheld** | Correct given import isolation is the real guarantee — *once BL1 makes that guarantee true*. In-core vs `[agent]` doesn't change the closed code path; ergonomics win stands. |
| **Spike S1 (Choice generics)** | **Upheld** | Reproduced: `Choice[S: str]` → `Constraint[Literal["keep","skip"]]`; `Literal[*Opts]` rejected. (§spikes) |
| **Spike S2 (Outcome encoding)** | **Finding upheld / response overturned** | Reproduced: union doesn't narrow, class-with-methods does — but this argues against a sum-type `run` result (MAJ4), not for combinators. (§spikes) |
| **Spike S3 (pydantic-ai surface)** | **Upheld (type surface) / incomplete (wire surface)** | Reproduced: `NativeOutput` importable, `usage` is a property. Did **not** verify the 2.11 *wire* body (BL3). (§spikes) |

---

## Independent spike results

*(From-scratch reproduction in devenv against Python 3.13.13 · ty 0.0.46 · pydantic-ai 2.11.0 ·
pydantic 2.13.3. Raw output in `REVIEW_SPIKES/`. This section is finalized from the independent
re-run — see the appended agent transcript summary.)*

Files: `REVIEW_SPIKES/s1_*.py`, `s2*.py`, `s3_*.py` (all `ruff`-clean; `ty`-clean except the three
where a type error is the point). Every headline spike claim **reproduced exactly**. Two glossed
points and several edge cases were newly surfaced — one of them (the `then` cast) directly weakens the
plan.

**My own probe (run directly):** importing `NativeOutput` as `constraint.py` does
(`from pydantic_ai import NativeOutput`) pulls the top-level `pydantic_ai` package and httpx into
`sys.modules` but **not** `pydantic_ai.models.openai`. `isinstance(AgentRunResult.usage, property)` →
`True`. Grounds BLOCKER 1 (transitive import) and confirms S3.

**S1 — Choice generics — CONFIRMED, with edges the plan missed.**
- `(a) Choice[*Opts] -> Constraint[Literal[*Opts]]`: `ty` emits `error[invalid-type-form]: Type
  arguments for Literal must be … a literal value`; call site reveals `Constraint[Unknown]`,
  `parse()` → `Unknown`. **Rejected, as claimed.**
- `(b) Choice[S: str](*options: S)`: reveals `_C[Literal["keep","skip"]]`; `parse()` →
  `Literal["keep","skip"]`. **Winner, as claimed.**
- **New edges:** single `Choice("only")` → `Literal["only"]`. A `str`-annotated *local initialized
  from a literal* **stays** `Literal` (does *not* widen) — so R7's "runtime `str` variable →
  `Constraint[str]`" is **imprecise**: only a *genuinely opaque* `str` (e.g. a function return) widens
  to `_C[str]`. Non-str args (`Choice(1,2)`) are a **hard `invalid-argument-type` error** at the call
  (bound `S: str` violated), not a silent widen. → tighten R7's wording; the bound is stronger than
  the plan says (good news, but state it right).

**S2 — Outcome encoding — FINDING CONFIRMED, RESPONSE WEAKENED.**
- `(a)` bare union `type Outcome[T] = Ok[T] | Failed`: `match case Ok(value=v)` → `v: @Todo`;
  `isinstance` → `.value: object`; `TypeGuard[Ok[T]]` → `Ok[Unknown]`; free `unwrap` → `Unknown`;
  `fold` → `Unknown`. **T lost everywhere — confirmed.**
- `(b)` base class + methods: `oc.unwrap()` → `Plan`; `oc.map(λ)` → `Outcome[list[str]]`;
  `oc.value_or(None)` → `Plan | None`; inherited `ok.unwrap()` → `Plan`. **Typed — confirmed.**
- **NEW, load-bearing (this weakens the plan):** the plan's own `then` combinator does **not**
  typecheck. `def then[U](self, f) -> Outcome[U]: return self` yields
  `error[invalid-return-type]: expected Outcome[U@then], found Self@then`. The Failed/Denied
  **pass-through requires a `cast`** (or per-subclass overrides). DESIGN §outcome claims `then`'s
  pass-through is *"Verified typed via S2 (method form)"* — it is **not**; the "no-cast" promise is
  violated *inside the combinator apparatus S2 was invoked to justify*. (Call sites still see the right
  `Outcome[str]`; the cast is internal — same "relocated cast" pattern as `fleet.typed`.)
- **NEW:** base-class `match oc: case Ok(value=v)` → `v: @Todo` (same degradation as the alias) —
  confirms R1 and that the concept's flagship `match` is untyped even in encoding (b).
- **NEW:** a single tagged `Result[T]` dataclass (no subclasses) types **just as well** as the
  base-class form (`unwrap → Plan`, `map → Result[list[str]]`, `value_or → Plan | None`). So S2 does
  **not** even mandate the subclass hierarchy — a plain tagged class works. This further supports
  MAJOR 4: the encoding question is wide open, and "sum-type + subclasses + combinators" is one of
  several typed options, not the forced one.

**S3 — pydantic-ai 2.11 surface — CONFIRMED (type surface); wire surface untested.**
- `from pydantic_ai import NativeOutput` → `pydantic_ai.output.NativeOutput`. ✓
- `isinstance(AgentRunResult.usage, property)` → **True**; `usage.fget: (self) -> RunUsage`;
  `callable(usage) == False` — **no callable back-compat** in 2.11 (the `callable()` shim really is
  unneeded at this floor; A1 fixed *at the pin*). ✓
- `Agent(output_type=str)` and `Agent(output_type=NativeOutput(M))` both construct. ✓
- **New:** `AgentRunResult.output` is a **generic dataclass field** (`OutputDataT`), not a property —
  so `raw.output`'s static type is whatever `output_type` implied, which is exactly what the codec
  relies on. Fine — but note this was verified at the *type* level only; the **on-wire
  `response_format` body at 2.11 was not captured** (BLOCKER 3).

---

## Must-fix-before-building (ordered — gates Phase 1)

1. **BLOCKER 1** — Make the closed import-isolation guarantee *true and transitively tested* (lazy
   `NativeOutput` import, or sever closed from `constraint.py`), and rewrite R2/T8 accordingly. Until
   this, the headline privacy property is unproven.
2. **BLOCKER 2** — Resolve `Schema.parse`: either it does real `model_validate` work in both paths
   (and T1 tests the string→model direction), or admit closed shares `wire/`, not the codec. Rewrite
   CONCEPT §8 / DECISION L / SALVAGE / 00-PLAN to match reality. (Resolves with BL1 if you pick
   sever-closed-from-constraint.)
3. **BLOCKER 3** — Re-capture the four `.wire()` bodies (esp. the rich `Schema` `response_format`) on
   pydantic-ai **2.11**, add a 2.11 column to VERIFICATION.md, make T2 assert the on-wire rich body,
   add the R6 canary. Don't carry 1.87 captures as "verbatim."
4. **MAJOR 4 (decision for the user)** — Re-open the `Outcome` spine: present `run -> T` (raise) +
   `execute -> Ok[Effect] | Denied` against the current class-combinator spine, and let the user
   re-confirm. This is the largest coherence lever in the plan.
5. **MAJOR 12/5** — Delete the stale second Rejected block in DECISION B; soften "no same-event-two-ways"
   to what's true.
6. **MAJOR 6/7** — Either make `Effector.run`-without-decision unrepresentable (auth token) or stop
   calling B1 "structural"; drop `Effect` in favor of `Outcome` reuse.
7. **MAJOR 8/9** — Fix the orthogonality wording (cache identity couples context→adapter); ship the
   `Context` *seam* without the unexercised `Reuse`/`Fidelity`/`cache_salt` machinery until
   `ChunkProvider` has a consumer.

Items 1–3 are true gates (a false privacy guarantee, a broken/illusory codec claim, an unverified wire
surface). Item 4 is the biggest *design* call and should be made before Phase 1 commits to the
combinator spine. 5–7 can be folded into their respective phases but should be decided now.
