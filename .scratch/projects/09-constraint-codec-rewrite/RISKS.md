# RISKS & spike list — what to verify empirically before/while committing

Spikes already run are in `SPIKES.md` (and `spikes/*.py`); their results are folded into DECISIONS.md.
This file is the *residual* risk register: what remains uncertain, how likely/severe it is, and the
concrete check that retires it. Ordered by severity.

Legend: **Likelihood × Impact** → **Mitigation / spike.**

---

## R1 — `ty` cannot narrow `T` out of the `Outcome` union via `match`/`isinstance` (HIGH likelihood · MEDIUM impact) — **retired-by-design, monitored**

**Risk.** The concept's flagship ergonomic — `match oc: case Ok(value=cmd): # cmd: GitCommand` —
returns `cmd: @Todo` under ty 0.0.46 (SPIKES S2a). If the design leaned on `match` for the *typed*
path, the "typed outputs, no cast" promise would be **false at the most important call site**.

**Status: retired at design time (DECISION B).** The typed path is the **method-combinator API**
(`.unwrap`/`.map`/`.value_or`/`.then`), which S2b proves types perfectly (`oc.unwrap() → Plan`,
`oc.map(λ) → Outcome[U]`). `match` remains a **runtime-correct** convenience (T3 runtime-tests it),
not the static contract. Residual risk is only *ergonomic* (some users will reach for `match` and see
a weaker type than they'd like).

**Monitor.** T7 carries an `xfail`/`expect-error` on the `match`-narrowing case that **flips green when
ty implements generic match-capture narrowing** — an early signal we can then promote `match` to a
first-class typed path. Re-run `spikes/spike3.py` on each ty bump.

**Would a different checker help?** Pyright/mypy likely narrow this correctly today — but the repo's
quality gate is `ty` (no pyright/mypy in devenv). We design to the gate we're graded on. If the fleet
ever adds pyright, revisit whether `match` can become the documented typed path.

---

## R2 — `constraint.py` importing `pydantic_ai.output.NativeOutput` is a layering concession (MEDIUM likelihood · LOW impact) — **accepted, bounded**

**Risk.** `Schema.wire()` must return a `NativeOutput(M)`, a pydantic-ai type, from Layer 1 —
nominally the layer below where pydantic-ai lives. A careless reading of "single-importer" could call
this a violation, or a future refactor could let *more* of pydantic-ai leak into Layer 1.

**Mitigation (DESIGN §constraint).** The single-importer invariant is specifically about
`pydantic_ai.models.openai` (the **client/transport**), which stays exclusively in `agent.py`.
`NativeOutput` is an inert declarative **marker** (no transport, no loop). `constraint.py` imports
**only** `pydantic_ai.output.NativeOutput` and nothing from `.models`. **T8 asserts exactly this** (an
AST test: `constraint.py` may import `pydantic_ai.output.NativeOutput` and nothing else from
pydantic-ai; `agent.py` is the sole `models.openai` importer). The concession is bounded and
test-enforced.

**Alternative if the concession is judged unacceptable.** `wire()` returns a sentinel
`SchemaMarker(model)`; `agent.py` translates it to `NativeOutput`. Purer layering, one translation
table, zero functional benefit (NativeOutput is inert). Recorded, not recommended.

---

## R3 — `Nullable`/`OneOf` composite constraints & the wire tag (MEDIUM likelihood · MEDIUM impact) — **spike before shipping composites (v3.1, not v3.0)**

**Risk.** `OneOf(Schema(A), Schema(B)) -> Constraint[A | B]` implies a **tagged union on the wire**
(concept §4.4). The exact `response_format`/`extra_body` shape a vLLM+XGrammar backend accepts for a
discriminated union is **not** in VERIFICATION.md — it was never captured. `Nullable` needs the schema
to admit `null`. Guessing the wire shape violates the "never guess a request shape" rule.

**Mitigation.** Composites are **out of the v3.0 spine** (DESIGN marks them v3.1). Before shipping:
capture the real discriminated-union body against `deploy/vllm/verify.sh` (or the ASGI mock with a
known-good XGrammar union schema), and add it to VERIFICATION.md + a T2 wire-shape test. Until
captured, `OneOf`/`Nullable` stay design sketches. **Spike:** `Grammar.from_json_schema` on a
`Union`/`anyOf` schema client-side (xgrammar accepts recursion per VERIFICATION §2 — does it accept
`anyOf`?), then a live round-trip on tower.

---

## R4 — Per-run capture correlation without the ContextVar (MEDIUM likelihood · LOW impact) — **spike the httpx request-tag path**

**Risk.** DECISION N moves capture *delivery* to `Ok.wire`, but the httpx event hook fires on the
*request*, before the run's result exists — so `Agent.run` still needs to correlate "the request my
awaited call produced" with "the record I attach to *this* `Ok`". v2 used a per-run `ContextVar` sink
for this. Removing the ContextVar as the *public* channel is clean; but the *internal* correlation
must still be race-free under `run_batch` (two overlapping runs of one agent).

**Mitigation / spike.** Two candidate internal mechanisms, spike both against the v2 in-flight
concurrency test (ported): (a) tag the outbound request via httpx `Request.extensions` with a per-run
token, and have the hook file the record under that token — no ambient state; (b) keep a per-run sink
but *scoped to and read only by* the single `run` coroutine (never cross-run), delivered into `Ok.wire`
and then discarded. Prefer (a) if httpx surfaces `extensions` in the event hook; else (b). **Acceptance:**
the ported concurrency proof (T3/T6-style) shows two overlapping same-agent runs each carry their own
distinct `Ok.wire` — the exact A5 scenario, now structurally correct.

---

## R5 — DBOS `SetWorkflowID` isolation (LOW likelihood · MEDIUM impact) — **carry v2's open verification into Phase 8**

**Risk.** The dual-path observer runs two legs concurrently on one event loop, each wrapping its
`await` in `with SetWorkflowID(wid)`. If DBOS implements this with **contextvars**, task isolation
makes it safe; if **thread-locals**, the ids can cross legs and mis-attribute workflow ids. v2 flagged
this as unverified (§C); v3 inherits it unchanged (Phase 8, `[observe]`).

**Mitigation.** Before trusting workflow-id attribution in `observe/`, verify against the *pinned* DBOS
version: a focused test that runs two `SetWorkflowID`-wrapped coroutines concurrently and asserts each
leg sees its own id. If thread-local, switch to explicit per-step id passing rather than the context
manager. Purely an `[observe]`-extra concern; does not touch the core.

---

## R6 — `pydantic-ai` output-typing API drift across the pinned range (LOW likelihood · MEDIUM impact) — **the pin + a canary test**

**Risk.** v3 depends on pydantic-ai for (a) `NativeOutput(M)` enforcing `response_format` and
validating+retrying, (b) `output_type=str` returning raw text, (c) `raw.usage` as a property, (d)
`raw.output` shape. S3 confirms all four at 2.11.0. A future 2.x could shift one (as 1.87→2.11 shifted
`usage` method→property, the A1 bug).

**Mitigation.** Pin `>=2.11,<3` (DECISION I). Add a **canary test** that asserts the four surface facts
directly (imports `NativeOutput`; builds `Agent(output_type=str)` and `=NativeOutput(M)`;
`isinstance(AgentRunResult.usage, property)`) — so a range-widening bump fails loudly at *this* seam
instead of deep in a run. The whole coupling is one module (DECISION A), so a break is localized.

---

## R7 — `Choice[S: str]` inference edge cases (LOW likelihood · LOW impact) — **already spiked; note the widening case**

**Risk.** `Choice[S: str](*options: S)` infers the literal *join* (S1). Two edge cases: (a) a
**single** option `Choice("only")` → `Constraint[Literal["only"]]` (fine); (b) a **runtime `str`
variable** `Choice(x)` where `x: str` → `Constraint[str]` (widens — correct and honest, but the caller
loses the literal). Neither is wrong; document (b) so users pass literals when they want the literal
type.

**Mitigation.** A T7 assertion for the single-option and the widening case, plus one docstring line.
No further spike needed (covered by `spike2.py`).

---

## R8 — Repo/name decision is taste, not architecture (certainty · LOW impact) — **open; user is choosing**

**Risk.** The only unresolved decision is the **name** (DECISION H) — the user is picking it
themselves (`constric` was only a suggestion). Proceeding on a guess would create a repo the user has
to rename. *(The two elegance-mandate calls that were also flagged here — the `Outcome` variant count
and pydantic-ai packaging — are now **resolved by the user**: lighter `Ok`/`Failed` spine, pydantic-ai
in core. See DECISIONS B & I.)*

**Mitigation.** **Do not create the repo (Phase 0) until the user settles the name.** Everything else
in the plan is
name-agnostic (import package is `structured_agents` regardless).

---

## Spike backlog (cheap, do-before-or-early)

| # | Spike | Retires | When | Cost |
|---|---|---|---|---|
| done | Choice generics (`S: str`) | R7, DECISION F | done (S1) | — |
| done | Outcome encoding (class+methods) | R1, DECISION B | done (S2) | — |
| done | pydantic-ai 2.11 surface | R6, DECISION A/I | done (S3) | — |
| 1 | httpx `Request.extensions` in event hook for per-run capture tag | R4 | Phase 6 | small |
| 2 | XGrammar discriminated-union / `anyOf` wire body (client + tower) | R3 | before v3.1 composites | medium (needs tower) |
| 3 | DBOS `SetWorkflowID` contextvar-vs-threadlocal on pinned version | R5 | Phase 8 | small |
| 4 | pydantic-ai canary (four surface facts) as a standing test | R6 | Phase 6 | trivial |

**Nothing on this list blocks the core spine (Phases 1–6).** R3 blocks only the v3.1 composites; R5
blocks only the `[observe]` extra; R8 blocks only repo creation and needs a human, not a spike.
