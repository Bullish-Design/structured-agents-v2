# Planning Kickoff — structured-agents v3 (the Constraint-Codec rewrite)

**Use this to start a *planning* session (Plan mode). The deliverable is a plan, not code.**

---

## Your role & the one metric

You are the architect for a **ground-up rewrite** of the `structured-agents-v2` library. A detailed
concept already exists (see "Read first"). Your job is to turn that concept into a **complete,
buildable architecture and phased plan** — and to make it *better* where you can.

**The only success metric is the quality of the resulting architecture: the cleanest, most elegant,
most internally-coherent design possible.** We explicitly **do not care about implementation
difficulty, effort, migration cost, or timeline.** Do not prune an elegant design because it is a
large build. Do not pick the pragmatic option over the clean one. When two designs trade
"simpler-to-build" against "more-coherent-to-reason-about," choose coherence every time. Token/agent
budget for the *planning* work is not a constraint either — be exhaustive.

Corollary: **the concept doc is the strongest current proposal, not a spec to implement verbatim.**
You are expected to pressure-test it, find its weak points, and improve it. Where you change it,
record *what* changed and *why* in a decision log. If you believe a different architecture is cleaner
than the concept, make that case explicitly rather than quietly complying.

---

## Read first (in this order)

1. **`.scratch/projects/09-constraint-codec-rewrite/CONCEPT.md`** — the authoritative v3 concept.
   The whole plan orbits this. Internalize the three axes (constraint/adapter/context), the
   `Constraint[T]` codec, the `Outcome[T]` spine, `Authorizer × Effector`, the layer stack, and the
   open questions (§18). Pay special attention to §20–23 (the three axes + context/cache axis) and the
   PIC-literature grounding (MEPIC/MiniPIC/KV Packet).
2. **`.scratch/projects/07-library-code-review/CODE_REVIEW.md`** — the deep review of the shipped v2.
   Every finding it raised (A1–A5, B1–B5, section C, D, F, G) is a real tension; v3 must dissolve each
   *structurally*, not re-fix it. This is your list of "problems the new design must make impossible."
3. **`.scratch/projects/02-library-wrapper/CONCEPT.md`** and its **`VERIFICATION.md`** — the v2 design
   and, crucially, the **captured wire shapes**. The decode-mode → wire mapping (`response_format` /
   `extra_body` `structured_outputs`) is empirically verified truth and must survive verbatim as the
   body of `Constraint.wire()`.
4. **`.scratch/projects/06-pydanticai-usage-compatibility/ISSUE.md`** — the downstream (Lodestar)
   report. Lodestar consumes the **`closed` path**, not the agent/fleet path — this is why the rewrite
   is low-risk despite being total, and why the `closed` guarantees are non-negotiable.
5. **The shipped code on `main` (v0.2.0)** — `src/structured_agents_v2/`. Read it to know exactly what
   you are replacing and what is worth salvaging (tests, the ASGI mock, `closed.py`'s guarantees, the
   httpx-capture technique, `deploy/vllm/verify.sh`).

---

## Context you need

- **What exists:** v0.2.0 (the v2 architecture, review-hardened) is shipped on `main`, tagged, 124
  tests green, `ty check src` + `ruff` clean. It works. v3 replaces its *core* (constraint, decoder,
  agent, fleet, executor) with the codec architecture.
- **Downstream:** one real consumer (Lodestar) pins the library and uses **`closed.py`** — a
  loopback-only, json-schema-only, no-retention, detail-free client that deliberately does **not**
  import pydantic-ai. v3 must preserve those guarantees exactly (as the `closed_backend` preset over
  the shared `wire/` layer). Blast radius of rewriting everything *else* is therefore low.
- **Backend reality:** a live vLLM on `tower` (XGrammar + per-agent LoRA verified) behind
  `deploy/vllm/verify.sh`; an in-process `httpx.ASGITransport` mock for tests. The KV-cache/PIC
  capabilities (§22) are a *future* deploy concern — the library cooperates with them, never
  implements them.
- **Environment/tooling:** Python 3.13, `pydantic`, `pydantic-ai-slim[openai]` 2.11.0, hatchling+uv,
  managed by **devenv**. All in-repo commands run inside `devenv shell -- <cmd>`. Version control is
  **plain git in this repo** (NOT gitman). Quality gates: `devenv shell -- pytest`,
  `devenv shell -- ty check src`, `devenv shell -- ruff check src tests` — all green is the bar.

---

## Non-negotiable invariants (the design must preserve or make *structural*)

These are settled. Challenge anything else, but not these:

1. **The constraint is one first-class value** — a `Constraint[T]` codec (`wire()` + `parse()`), the
   single source of truth for both the outbound wire shape and the inbound typed parse. No dunder
   ClassVars, no separate `DecoderSpec`/`apply`/`_guard`.
2. **Type-honesty end to end, no `cast` lies.** `T` flows from `Constraint[T]` → `AgentSpec[T]` →
   `Agent[T]` → `Outcome[T]`. A `Constraint[str]` truly yields `str`; a `Constraint[Literal[…]]` truly
   yields the literal. If a design needs a `cast` to assert what the checker can't verify, it's wrong.
3. **Three orthogonal axes, never conflated** — output (constraint) / weights (adapter) / input
   (context). A LoRA is not a `Constraint`; cache policy is not a prompt mode. Each cooperates with a
   server capability it never reimplements; each has its own plugin seam.
4. **Decisions are data; exceptions are for bugs.** One uniform way for any pipeline stage to decline
   (the `Outcome` variants). Exceptions reserved for programmer/config error.
5. **Explicit effects only.** Nothing runs implicitly; side effects happen only at an explicit
   `execute`/`run` the caller makes.
6. **Authority = decision × effect**, decomposed (`Authorizer` fail-closed by construction; `Effector`
   the side effect). Fornix is an `Effector`, DryRun a composition — not new executor subclasses.
7. **One-way layered dependencies** — `wire → constraint → agent → fleet`, with pydantic-ai confined
   to the agent layer (the sole importer of `pydantic_ai.models.openai`) and `closed` depending only
   on `wire` + `constraint` (never pydantic-ai).
8. **The `closed` guarantees** — loopback-only, one request, json-schema-only, no capture, no
   retention, detail-free errors — preserved exactly.
9. **The wire-grounded mode table** (from VERIFICATION.md) survives verbatim; the caching contract is
   modeled as a **capability** (boundaries + identity + context-dependence), never a specific PIC
   algorithm.
10. **Cache correctness bookkeeping is the library's** — chunk cache namespace folds in
    `base_model + adapter` so KV is never wrongly shared across LoRAs; fidelity is a coarse
    `EXACT|BLENDED` posture, never a numeric recompute ratio.

---

## Decisions you must resolve during planning

Resolve each with a recommendation + rationale + rejected alternatives. Start from the concept's §18
open questions, and add the repo-level ones:

**From the concept (§18):**
- **A. pydantic-ai coupling depth.** Keep pydantic-ai as the Layer-2 model loop (retries, message
  handling, `NativeOutput`), or own the loop directly on `wire/` and make `Constraint.parse` the sole
  parser for every mode (maximally uniform; re-implements retry/message machinery)? *Difficulty is not
  a factor — decide on cleanliness/coherence alone.*
- **B. The `Outcome[T]` sum-type spine.** Commit fully (Ok/Denied/Violated/Failed + `then`/`unwrap`
  everywhere), or a lighter variant (Ok/Failed for `run`, richer union only for the executed
  pipeline)? Weigh elegance vs Python idiom honestly.
- **C. Heterogeneous-fleet typing.** Is `Fleet` inherently `Agent[Any]`-valued (re-narrow via
  `fleet[name] -> Agent[T]`), or is a typed-router (`Router[Enum]`) worth the machinery?
- **D. Streaming.** Does `Outcome[T]` gain a streaming sibling, or is constrained decoding batch-shaped
  for the use case?
- **E. Tool/function-calling agents.** In scope, or firmly a different abstraction (keep `NativeOutput`
  only)?
- **F. `Choice` variadic generics.** Confirm `ty` synthesizes `Constraint[Literal[*opts]]`, or fall
  back to an explicit `Choice[L]` form. (Spike this against the actual `ty`.)
- **G. Multi-turn sessions & the context axis (§22.8).** Does `Context` grow a `Session` sibling that
  threads a growing history as `Reuse.PREFIX` segments? This is where PIC pays off most.

**Repo/strategy level:**
- **H. Repo & name.** Is v3 a **new repository** (fresh name — propose one that fits the fleet naming
  families) or a rewrite on a branch of `structured-agents-v2`? How do the two coexist while Lodestar
  stays on v0.2.0? Define the Lodestar migration path off `closed`.
- **I. Package/extras layout.** Core deps (pydantic + pydantic-ai-slim[openai]) and extras
  (`[grammar-check]`, `[observe]`, future cache/adapter tooling). What does the lean core install?
- **J. Naming finalization.** Lock the public vocabulary (`Constraint`, `AgentSpec`, `Agent`,
  `Outcome`, `Fleet`, `Authorizer`/`Effector`, `Context`/`Segment`/`Reuse`, `Adapter`,
  `ContextProvider`, `AdapterProvider`, …). Resolve the `Agent` name collision with `pydantic_ai.Agent`.
- **K. Config/plugin registration.** The exact `register_*` / entry-point mechanism per seam (§23) and
  the import-allowlist model at the config edge (§11).

---

## Deliverables of this planning session

Produce these as documents in `.scratch/projects/09-constraint-codec-rewrite/` (propose the filenames):

1. **DECISIONS.md** — every decision above (and any you surface), each: recommendation, rationale,
   rejected alternatives. This is the spine of the plan.
2. **Module-by-module design spec** — for each module in the layer stack (§14 is the starting layout,
   improve it): its responsibility, its public types and signatures (sketch the actual
   `Protocol`s/dataclasses/functions), its dependencies (must obey the one-way layering), and its
   invariants. This is the heart of the deliverable — someone should be able to implement from it.
3. **PHASES.md — a phased build plan.** Even though difficulty doesn't gate design choices, sequencing
   still matters so each phase leaves a **green, self-contained, demonstrable** state. Order by
   value-per-coherence (the concept suggests: Constraint+wire → Authorizer×Effector → config/code split
   → closed preset → Outcome spine → context axis → observe). Each phase: scope, the modules it lands,
   its acceptance criteria (tests + `ty` + a runnable demonstration).
4. **TESTS.md — the test architecture.** The codec round-trip property tests (the single most valuable
   new surface), wire-shape assertions (carried from VERIFICATION.md), `Outcome` algebra, authority
   fail-closed, `ty`-level `assert_type` regressions, the `closed` guarantees, and the in-process ASGI
   technique. Map v2's salvageable tests to their v3 homes.
5. **SALVAGE.md** — a ledger of what carries over verbatim vs is rewritten vs is dropped (tests, the
   ASGI mock, `closed`'s guarantees, the capture technique, `deploy/` verify scripts, VERIFICATION.md).
6. **RISKS.md / spike list** — the handful of things to *verify empirically* before committing (e.g.,
   `ty`'s handling of variadic `Choice`; whether pydantic-ai's `NativeOutput` + `output_type=str`
   split behaves as the codec assumes; the `Outcome` ergonomics against real call sites). Run small
   spikes where cheap and fold results into DECISIONS.md.
7. **A one-page executive summary** at the top of the plan: the shape of v3 in a page, the decisions
   taken, and where you *departed from or improved on* the concept doc.

---

## Stance & method

- **Be adversarial toward the concept.** For each of its claims ("the codec dissolves four tensions,"
  "the Outcome spine is worth it," "closed is a thin preset"), try to break it. If it survives, say so;
  if it doesn't, propose the fix. A planning session that merely restates the concept has failed.
- **Design for reasoning, not for typing-around.** Prefer designs where the *types* make illegal
  states unrepresentable over designs that need runtime guards or casts.
- **Spikes are allowed and encouraged** to de-risk a decision (especially the `ty`/generics and
  pydantic-ai-coupling questions) — but the deliverable is the *plan*, not the implementation. Keep
  spikes in a scratch dir; don't touch `src/` or `main`.
- **Ground every wire-facing claim** in VERIFICATION.md or a fresh capture against the ASGI mock /
  `deploy/vllm/verify.sh` — never guess a request shape.
- **Record disagreements loudly.** If you think the mandate ("elegance over everything") leads somewhere
  the user should veto, surface it as a decision with your recommendation — don't silently soften it.

---

## Definition of done for the planning session

- Every decision in the list above is resolved with rationale.
- A reader could implement each module from the design spec without re-deriving the architecture.
- The phased plan leaves a green, demonstrable state at every step and covers the whole concept
  (all three axes, the closed preset, observe).
- Every v2 review finding (07/CODE_REVIEW.md) is shown to be *structurally impossible* in the v3
  design, not merely re-fixed — with a one-line pointer to the mechanism that kills it.
- The plan states explicitly where it improved on or departed from CONCEPT.md, and why.

---

## Ground rules (repo)

- Run everything in devenv: `devenv shell -- <cmd>`. Never bare `uv`/`python`/`pytest`.
- Version control is **plain git** in this repo (not gitman). Do planning work on a scratch branch or
  in `.scratch/`; do not modify `src/` or `main` during planning.
- No AI-authorship trailers in any commits/docs.
- Quality bar for any spike code: `ty check` clean, `ruff` clean — model the standard the real build
  will hold to.
