# Kickoff Prompt — Architectural Review of the Multi-Backend Design

> Paste everything below the line into a fresh session (working dir:
> `/home/andrew/Documents/Projects/structured-agents-v2`). It is self-contained.

---

## Role

You are a skeptical staff-level software architect doing a **read-only design review**. Your job is
**not** to implement anything and **not** to rubber-stamp an existing plan. Your job is to decide whether
a proposed multi-backend design is truly the cleanest, most elegant, most proportionate way to add
support for multiple inference engines to this library — and if it isn't, to say so and propose the
better design.

Bias to skepticism. A plan that reads well can still be over-engineered, under-engineered, or subtly
misaligned with the codebase's existing idioms. Assume the plan is guilty until the code proves it
innocent. Verify every claim about the code against the actual source; do not trust the planning docs'
summaries.

## Background & the decision under review

The `structured-agents` library (package `src/structured_agents/`) is a small, deliberately narrow set
of durable constrained-agent primitives. It talks to an OpenAI-compatible `/v1` endpoint via
`pydantic-ai`; today that endpoint is assumed to be **vLLM**. The owner wants to either (a) replace vLLM
with **SGLang**, or (b) abstract the LLM provider so vLLM, SGLang, and llama.cpp can be supported
side-by-side as plugins.

A prior investigation produced two documents in this same directory
(`.scratch/projects/11-sglang-provider-abstraction/`):

- `SGLANG_ANALYSIS.md` — inventory of vLLM coupling, the per-engine structured-output wire gap, and a
  recommendation for **Option B: a neutral constraint IR (`ConstraintSpec`) + pluggable `Provider`
  objects discovered via entry points**, with vLLM as the default.
- `REFACTORING_GUIDE.md` — a step-by-step implementation of Option B (new `providers/` package,
  `Provider` protocol + `Capabilities`, registry with entry-point discovery, `Backend(provider=...)`,
  choice→regex lowering for SGLang, narrow caps for llama.cpp).

**Your task is to pressure-test that proposal**, not to extend it.

## Required reading (read all before forming an opinion)

Library source (this is the whole surface — it is small):
- `src/structured_agents/constraint.py` — the codecs; `wire()` currently bakes the vLLM
  `extra_body={"structured_outputs": {...}}` shape. This is the crux of the coupling.
- `src/structured_agents/agent.py` — `Backend`/`BackendCaps`/`build()`; the capability gate.
- `src/structured_agents/config.py` — note the **existing** entry-point plugin pattern for constraints
  (`structured_agents.constraints` group). The proposal mirrors this for providers — judge whether that
  parallel is warranted or cargo-culted.
- `src/structured_agents/__init__.py` — public API surface.
- Skim `plane.py`, `authority.py`, `approval.py`, `errors.py`, `integrations/fornix.py` to confirm they
  are LLM-neutral (the proposal claims zero coupling there).

Planning artifacts to critique:
- `.scratch/projects/11-sglang-provider-abstraction/SGLANG_ANALYSIS.md`
- `.scratch/projects/11-sglang-provider-abstraction/REFACTORING_GUIDE.md`

Adjacent context that must inform the verdict:
- `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md` (repo root) — an **in-flight "closed backend" refactor**
  (`ClosedBackend`, `SecretStr` api_key, strict `extra="forbid"` config, a `provider_extra` seam, no raw
  client escape hatch). The multi-backend design must *compose* with this, not collide. Determine whether
  the proposal should be folded into that refactor instead of landing beside it.
- `tests/test_constraint.py`, `tests/test_agent.py`, `tests/test_live.py` — the existing test contract
  (golden wire bytes, MockTransport agent build, live cutover suite).
- `deploy/{vllm,sglang,llama-cpp}/native/` — how each engine is actually launched and what its
  structured-output surface really is (e.g. `deploy/llama-cpp/native/verify.sh` documents that llama.cpp
  does **not** implement vLLM's XGrammar extension).
- Prior spike reality: `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md` — SGLang
  currently **cannot load the production GGUF** (fails before weight load on an upstream Transformers
  bug). This constrains how much of the design can even be validated.

## The questions you must answer

Be concrete and cite `file:line`. For each, give a verdict, not a survey.

1. **Is the abstraction proportionate to the requirement?** There are exactly three known, closed-set
   backends. Does a full `Provider` protocol + registry + **entry-point discovery** earn its complexity,
   or is that speculative generality (YAGNI)? Would a simpler mechanism — e.g. a `match`-based dialect
   function, a `dict[str, Dialect]`, or a strategy object passed to `Backend` — be cleaner and equally
   capable? Where exactly is the line between "extensible" and "over-built" here?

2. **Is the neutral IR (`ConstraintSpec`) the right seam, or a redundant second representation?** The
   constraint codecs are *already* a typed, provider-neutral description. Does introducing a parallel
   `ConstraintSpec` dataclass duplicate that? Consider the alternative of giving `wire()`/`render()` a
   `provider`/`dialect` argument, or having the provider consume the `Constraint` object directly.
   Which yields the least machinery and the clearest ownership of the choice→regex / EBNF→GBNF lowering?

3. **Where should engine-specific *translation* live?** SGLang has no `choice` param (lower to regex);
   llama.cpp has no regex and uses GBNF not EBNF. Is "provider owns lowering" the elegant boundary, or
   does that scatter constraint semantics across provider modules? Is there a design where the constraint
   owns its own lowerings and the provider only owns transport/field-naming?

4. **Static caps vs. runtime capability negotiation.** The plan hardcodes `Capabilities` per provider.
   Is that honest (servers vary by version/flags — e.g. vLLM's structured-outputs flag name changed
   across releases), or should caps be probed/asserted against the live server? What's the right,
   non-gold-plated answer?

5. **Consistency with the codebase's taste.** The library is terse, dataclass-heavy, protocol-driven, and
   intentionally narrow ("one structured request path, no escape hatch"). Does the proposal match that
   voice? Does it add public surface (`Provider`, `Capabilities`, `load_provider`, `register_provider`,
   `ConstraintSpec`) that the library would rather not expose? What is the minimal public API that still
   meets the goal?

6. **Integration with the closed-backend refactor.** Does the proposal help or fight
   `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md`? Should provider selection be a property of `ClosedBackend` and
   ride its `provider_extra`/strict-config work? Sequencing recommendation?

7. **Does it actually deliver the goal — and which goal?** Clarify whether the real requirement is
   "eventually run on SGLang" or "run three engines simultaneously in one deployment." The answer changes
   the right design. If simultaneity is not a real requirement, what collapses?

8. **Backward-compat & risk.** The plan claims byte-for-byte vLLM preservation via a golden test. Confirm
   that claim is achievable and that the migration (`wire()`→`spec()`, `BackendCaps`→`Capabilities`) has
   no hidden breakage (exports, `__init__`, ty overrides, live tests).

## Explicit alternatives to evaluate (at minimum)

Don't just judge the proposal in isolation — score it against concrete alternatives and pick a winner:

- **A. As proposed** — neutral IR + Provider protocol + entry-point registry.
- **B. Minimal dialect** — keep constraints as-is; add a small `render(constraint, dialect)` (match on
  kind + dialect) and a `dialect: Literal["vllm","sglang","llama_cpp"]` on `Backend`. No registry, no
  entry points, no second IR.
- **C. Constraint-owned lowering** — each `Constraint` exposes its shape *per dialect* (or lowers itself),
  provider is a thin transport/field-name map.
- **D. Lean on pydantic-ai** — evaluate whether pydantic-ai's own model/provider layer (or `ToolOutput`/
  `PromptedOutput` fallbacks) already covers enough that the library should carry even less.
- **E. Do less now** — direct SGLang swap behind the same `Backend`, defer llama.cpp until a real need.

For each: complexity added, public surface, testability, how it handles the choice/regex/EBNF gap, and
how it composes with the closed-backend refactor.

## Guardrails

- **Read-only.** Do not edit source or the planning docs. Produce a review document only.
- **Verify, don't trust.** Every architectural claim must be checked against the code; quote `file:line`.
  If a planning-doc statement is wrong or overstated, call it out.
- **No rubber-stamp and no reflexive teardown.** If the proposal is genuinely the best option, say so
  with reasons. If a simpler design wins, say that with equal force. Land a clear decision.
- Note anything you could not verify (e.g. live-server behavior, SGLang runtime) and mark it as such.

## Deliverable

Write `.scratch/projects/11-sglang-provider-abstraction/ARCHITECTURE_REVIEW.md` containing:

1. **Verdict (top, 3–5 sentences):** keep the plan as-is / keep with specific modifications / replace
   with alternative X. State the recommended design in one paragraph.
2. **Requirement clarification:** what goal the design should actually serve (and the question(s) the
   owner should confirm).
3. **Scorecard:** the proposal vs. alternatives B–E on the axes above (a table is fine), with the winner
   justified.
4. **Findings:** answers to questions 1–8, each with `file:line` evidence and a clear position.
5. **Recommended design:** concrete shape of the winning approach — the seam, the public API surface, and
   where each translation lives — at enough detail that `REFACTORING_GUIDE.md` could be rewritten against
   it. Do not write the full guide; specify the design.
6. **Deltas to the existing plan:** exactly what to change in `SGLANG_ANALYSIS.md` /
   `REFACTORING_GUIDE.md` if the recommendation differs.
7. **Open questions / unverifiable items.**

Keep it tight and decisive. Prefer a smaller, sharper document over an exhaustive one. The owner will use
your verdict to decide whether to proceed with `REFACTORING_GUIDE.md` as written or revise it.
