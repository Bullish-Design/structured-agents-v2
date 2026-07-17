# Review Kickoff — interrogate the structured-agents v3 concept **and** its plan

**Use this to start a clean *review* session. The deliverable is a critique + a concrete set of
improvements (and, where warranted, a better architecture), not code.**

---

## Your role & the one metric

You are an adversarial architecture reviewer. Two documents-sets are in front of you: the **v3
concept** (`CONCEPT.md`) and the **plan built from it** (`00-PLAN.md`, `DECISIONS.md`, `DESIGN.md`,
`PHASES.md`, `TESTS.md`, `SALVAGE.md`, `RISKS.md`, `SPIKES.md` + `spikes/*.py`). Your job is to
**try to break both**, and where they survive, say why; where they don't, propose the fix or the
cleaner alternative.

**The only success metric is the quality of the resulting architecture: the cleanest, most elegant,
most internally-coherent design possible.** Implementation difficulty, effort, migration cost, and
timeline are explicitly **not** factors. Do not defend a design because it's already written or
because rewriting the plan is work. When two designs trade "less-rework" against
"more-coherent-to-reason-about," choose coherence every time. Token/agent budget for the *review* is
not a constraint — be exhaustive.

---

## Assume nothing is settled

This is the core instruction. The plan labels many things "settled," "non-negotiable,"
"invariant," "resolved by spike," or "resolved by the user." **Treat none of those labels as
binding.** Every one is a claim to be re-examined on its merits. In particular, put all of the
following explicitly back on the table:

1. **The concept's central abstractions.** Is `Constraint[T]`-as-bidirectional-codec actually the
   right linchpin, or is it one good idea among several the design over-orbits? Is the **three-axis**
   framing (constraint/adapter/context) real orthogonality or a tidy story that will leak? Is the
   `Authorizer × Effector` split genuinely cleaner than one object, or ceremony? Is the `Outcome`
   spine worth its weight *at all*, or would plain returns + targeted exceptions be more honest in
   Python? Challenge the thesis, not just the details.

2. **The planning kickoff's "non-negotiable invariants."** The plan inherited ~10 invariants it was
   forbidden to challenge (one first-class constraint value; type-honesty/no-cast; three orthogonal
   axes; decisions-as-data; explicit-effects; authority = decision × effect; one-way layering with
   pydantic-ai confined to the agent layer; the `closed` guarantees; the wire mode table verbatim;
   library-owns-cache-bookkeeping). **Interrogate each.** Which are truly load-bearing, which are
   taste dressed as law, and which quietly cost more elegance than they buy? If one should be
   relaxed or dropped, make that case with evidence.

3. **The user's two decisions.** The user chose (B) the **lighter `Ok`/`Failed` Outcome spine**
   (Denied only on `execute`, Violated folded into Failed as a `ConstraintViolation` subtype) over
   the session's original full-four-variant, and (I) **pydantic-ai in core** (no `[agent]` extra).
   These are *choices*, not proofs. Re-open both: is the lighter spine coherent, or does
   "Denied-only-on-execute + Violated-inside-Failed" quietly reintroduce the v2 "two ways to decline"
   wart it claims to avoid? Is pydantic-ai-in-core right, or does the closed path's import-isolation
   argument actually favor the extra? **Do not silently revert a user decision** — but if the
   architecture is cleaner the other way, build the strongest case and present it as a decision for
   the user to re-confirm, with the tradeoff stated plainly.

4. **The spike *conclusions* (not just the spike *findings*).** Three spikes (S1 Choice generics, S2
   Outcome encoding, S3 pydantic-ai surface) drove decisions. Separate two things: the **finding**
   (an empirical fact about `ty`/pydantic-ai) and the **response** (the design choice made because of
   it). A finding can be solid while the response is wrong. Re-run every spike *from scratch* (don't
   trust the transcript or the committed outputs), then ask whether the response is the most elegant
   reaction to the finding or merely *a* reaction. E.g. S2 concluded "model `Outcome` as a class with
   method combinators because ty can't narrow the union" — is that the cleanest response, or does it
   argue for *not having a sum-type result at all*, or for a different result shape entirely?

5. **The module boundaries and layer stack.** Are `wire / constraint / outcome / context / agent /
   authority / fleet / config / closed / observe` the right cuts? Is anything mis-layered (e.g. the
   `NativeOutput`-in-Layer-1 concession, R2)? Should two modules merge or one split? Is the
   one-way-dependency rule bought at the price of awkward seams anywhere?

6. **Whether the rewrite's shape is even right.** Should `closed` be *a* path or *the* path? Should
   the library own the model loop rather than wrap pydantic-ai (concept open-question A)? Is the
   `Fleet`/`Router` abstraction earning its place, or is it scope the core doesn't need? Is the whole
   thing better as a smaller, sharper library? Nothing about scope is fixed.

7. **The name/repo (H).** Open; the user is choosing. Not your call to finalize, but if the
   naming/role framing (fleet families) suggests something, note it.

The point is not to churn for its own sake — it's to make sure that *everything* which survives this
review survives because it's genuinely the most coherent option, not because it was written down
first or labeled "settled."

---

## Read first (in this order)

1. **`.scratch/projects/09-constraint-codec-rewrite/00-PLAN.md`** — the one-page synthesis; the map
   of what the plan claims.
2. **`DECISIONS.md`** — decisions A–R. This is the spine; most of your findings will attach to a
   decision, its rationale, or its rejected alternatives (check the alternatives are fairly stated).
3. **`DESIGN.md`** — the module-by-module spec with real signatures. Verify the signatures typecheck
   and the invariants hold; hunt for hand-waving.
4. **`SPIKES.md` + `spikes/*.py`** — the empirical claims. **Re-run these independently** (see
   Verification below). This is where the plan is most falsifiable.
5. **`CONCEPT.md`** — the design target the plan is built from. Where the plan departs from it (six
   departures listed in 00-PLAN), judge whether the departure improved things or lost something; where
   the plan *followed* it, judge whether it should have.
6. **`PHASES.md`, `TESTS.md`, `SALVAGE.md`, `RISKS.md`** — sequencing, test architecture, the
   verbatim/rewrite/drop ledger, and the residual-risk register. Check PHASES leaves a genuinely
   green+demonstrable state each step; check SALVAGE doesn't drop something load-bearing or copy
   something unsafe "verbatim"; check RISKS severities are honest and nothing real is missing.
7. **`../07-library-code-review/CODE_REVIEW.md`** — the v2 review. The plan claims every finding
   (A1–A5, B1–B5, section C/D/F/G) is now *structurally impossible*. **Verify each claim** — is the
   defect truly impossible by construction, or merely relocated/renamed?
8. **`../02-library-wrapper/VERIFICATION.md`** — the captured wire shapes. Any wire-facing claim in
   the plan must match these (or a fresh capture); flag drift.
9. **The shipped v0.2.0 code** in `src/structured_agents_v2/` — ground truth for what's being
   replaced and what "salvage verbatim" actually copies.

---

## Independent verification required (verify, don't trust)

The plan's credibility rests on three spikes and a set of wire/type claims. Reproduce them yourself;
do not accept the committed outputs.

- **Re-run S1 (Choice generics)** in devenv: does `def Choice[S: str](*options: S) -> Constraint[S]`
  really infer `Constraint[Literal[...]]` under the repo's `ty`? Does the concept's
  `Choice[*Opts] -> Constraint[Literal[*Opts]]` really fail? Then push further: single option,
  runtime-`str` widening, non-str options, mixed literals — does the story hold at the edges, and is
  it the cleanest available shape?
- **Re-run S2 (Outcome encoding)** from scratch. Confirm the union-alias fails to narrow `T` under
  `ty` **and** that the class-with-methods form types. Then interrogate the *response*: is a
  method-combinator class the most elegant result type, or does the finding actually undercut the
  case for a sum-type result at all? Try alternative encodings (single tagged class; `Result`-style;
  no result wrapper) and typecheck them. Consider whether a different checker (pyright/mypy, if you
  can get one) narrows the union — and whether the repo *should* change its checker rather than its
  design.
- **Re-run S3 (pydantic-ai surface)**: `NativeOutput` import, `output_type=str` vs `NativeOutput(M)`,
  `usage` as property. Then probe what the plan *didn't*: does `NativeOutput(M)` actually enforce
  `response_format` and validate+retry as assumed? What exactly is `raw.output` for each mode? Does
  the coupling surface hide anything the plan glosses.
- **Re-verify the wire shapes** (`Constraint.wire()` bodies for schema/regex/choice/grammar; the
  closed `response_format` body; the loopback + bounded-input guards) against VERIFICATION.md and the
  v2 source. Assert they'd reproduce byte-for-byte. Never accept a request shape on faith.
- **Typecheck the key DESIGN signatures**: build a throwaway module with the `Constraint`,
  `Outcome`, `AgentSpec`/`Backend`/`Agent`, `Authorizer`/`Effector`, and `fleet.execute` sketches
  and run `ty` — find the places the prose claims a type the checker won't grant (the plan already
  admits the `match`-narrowing gap; look for others, e.g. `then` re-typing non-Ok variants,
  `fleet.typed` re-narrowing, `Choice` widening, the `NativeOutput` layering).
- **Internal-consistency pass**: the user's decisions were rippled across 8 docs. Find every place
  the ripple missed or contradicts itself (variant counts, extras layout, `Violated` vs
  `ConstraintViolation`, `[agent]` residue, capture delivery, error hierarchy).

Keep all spike/verification code in a scratch dir; do not touch `src/` or `main`. Hold spike code to
the real bar: `ty` clean, `ruff` clean.

---

## Sharpest questions to press (non-exhaustive)

Use these as seeds, not a checklist — add your own.

- **Codec:** Is `parse()` "always runs" honest for `Schema` mode (identity) — or is calling it a
  no-op that pretends symmetry? Is `check()`-at-build the right time and place? Do `Nullable`/`OneOf`
  belong in the design at all given their wire shape is uncaptured (R3)?
- **Outcome:** Is the `Ok/Failed` + `Denied`-on-execute + `Violated`-in-`Failed` partition *one*
  coherent model or three special cases? Would callers actually reach for `.then/.map/.unwrap` or
  fall back to `match` (which isn't typed)? Is a monadic spine Pythonic here, or cosplay?
- **Authority:** Does `authorize(a) >> Effector` compose as cleanly as claimed once real effectors
  (async, fallible, fornix-shaped) exist? Is `Effect` distinct from `Outcome` justified, or
  redundant? Does "B1 impossible" truly hold, or just move the footgun?
- **Layering:** Is the `NativeOutput`-in-Layer-1 concession (R2) acceptable, or does it falsify the
  "single-importer / pydantic-ai confined to Layer 2" story? Is `closed`'s import-isolation
  test-enforceable and actually true end-to-end?
- **Closed & deps:** Given the user kept pydantic-ai in core, is the closed path's whole
  attack-surface argument now weaker than the plan admits? Should closed be the primary path?
- **Context axis:** Is shipping `Context`/`Reuse` now (with `CHUNK` latent) real value or speculative
  generality (YAGNI) ahead of any consumer? Is the cache-namespace bookkeeping the library's job?
- **Config/plugins:** Is the `allow_modules` allowlist + per-seam registry the minimum viable seam,
  or is it machinery for a plugin ecosystem that may never exist?
- **Fleet:** Is `Agent[Any]` + `fleet.typed()` re-narrowing honest, or a typing dodge? Does routing
  belong in the library?
- **Phases/Tests:** Does each phase truly stand alone and green? Are the acceptance criteria
  falsifiable, or aspirational? Is the codec round-trip property test as valuable as claimed?

---

## Deliverables of this review session

Produce these in `.scratch/projects/09-constraint-codec-rewrite/` (propose filenames, e.g.
`REVIEW.md` + `REVIEW_SPIKES/`):

1. **A findings report** — severity-ranked (blocker / major / minor), each finding: the claim being
   challenged (with a `file:section` pointer), why it's wrong/weak/unproven, the evidence (a re-run
   spike, a wire capture, a typecheck), and the recommended fix or alternative.
2. **A verdict on each "settled" item** — a table over the kickoff invariants, the user's two
   decisions, and the three spike conclusions: **upheld / weakened / overturned**, one line of
   evidence each. This is the heart of "interrogate what's settled."
3. **A "cleaner architecture" delta (if one exists)** — if, after breaking things, you see a more
   elegant overall shape (fewer concepts, better boundaries, a different central abstraction), lay it
   out explicitly and argue it against the current plan. If the current plan *is* the cleanest, say so
   and explain why the alternatives you tried are worse — a review that only nitpicks has under-tried.
4. **Independent spike results** — your from-scratch reproductions (S1–S3 + any new ones), with the
   raw `ty`/runtime output, and where they confirm vs contradict the plan.
5. **A must-fix-before-building list** — the subset of findings that gate starting Phase 1, ordered.

---

## Stance & method

- **Adversarial toward *both* the concept and the plan.** For every claim ("the codec dissolves four
  tensions," "every v2 finding is structurally impossible," "the lighter Outcome avoids the v2 wart,"
  "closed shares primitives with zero duplication"), try to falsify it. If it survives, say so; if it
  doesn't, propose the fix.
- **Ground every challenge.** Re-run spikes; capture wire shapes; typecheck alternatives. Opinion
  without evidence is a weak finding. "Never guess a request shape."
- **Design for reasoning, not for typing-around.** Prefer designs where illegal states are
  unrepresentable over designs that need runtime guards or casts — and call out anywhere the plan
  *claims* the former but delivers the latter.
- **Separate finding from response.** State the empirical fact, then judge the design reaction to it
  independently.
- **Record disagreements loudly, including with the user's decisions and this kickoff's framing.** If
  "elegance over everything" leads somewhere the user should veto, surface it as a decision with your
  recommendation — don't silently soften it. If a user decision is architecturally worse, make the
  case and let them re-decide; don't just comply and don't just revert.
- **Don't churn for its own sake.** The goal is the cleanest architecture, which sometimes means "this
  is already right." Upholding a decision with fresh evidence is as valuable as overturning one.

---

## Ground rules (repo)

- Run everything in devenv: `devenv shell -- <cmd>` from the repo root (`cd`-ing elsewhere breaks
  devenv resolution). Never bare `uv`/`python`/`pytest`/`ty`/`ruff`.
- Toolchain to verify against: Python 3.13, `ty` (the repo's sole checker — no pyright/mypy in
  devenv; if you want a second opinion you must add one deliberately), `pydantic` 2.13.x,
  `pydantic-ai` 2.11.0.
- Version control is **plain git** in this repo (NOT gitman). The plan lives on branch
  `plan/constraint-codec-v3`; do review work there or on a scratch branch, in `.scratch/`. **Do not
  modify `src/` or `main`.**
- No AI-authorship trailers/attributions in any commits or docs.
- Quality bar for any spike/verification code: `ty` clean, `ruff` clean — model the standard the real
  build will hold to.

---

## Definition of done for the review session

- Every "settled" item (the kickoff invariants, the user's two decisions, the three spike
  conclusions) has an explicit **upheld/weakened/overturned** verdict with evidence.
- The three spikes are independently reproduced; any contradiction with the plan is documented.
- Every v2 review finding's "structurally impossible" claim is checked and confirmed or refuted.
- Every wire/type claim is grounded in a re-run, not the transcript.
- If a cleaner overall architecture exists, it is laid out and argued; if not, that conclusion is
  defended against the alternatives actually tried.
- A ranked must-fix-before-building list exists, and the findings are concrete enough to act on
  without re-deriving them.
