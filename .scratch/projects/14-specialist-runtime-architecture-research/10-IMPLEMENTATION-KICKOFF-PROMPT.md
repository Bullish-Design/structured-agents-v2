# Implementation Kickoff Prompt

Use this prompt in a fresh implementation session **only after the owner accepts the project-14 research plan**. Do not execute it during the research session.

---

You are implementing the accepted Phase 0 foundation for `structured-agents-v2` in:

`/home/andrew/Documents/Projects/structured-agents-v2`

Start by discovering and obeying all repository `AGENTS.md` instructions. Inspect the current branch, commit, status, and available environments; do not assume the research baseline is still current. Existing changes and untracked files belong to the user. Preserve them and never discard or overwrite them.

## Required reading

Read these files completely before changing code:

1. `.scratch/projects/14-specialist-runtime-architecture-research/00-EXECUTIVE-SUMMARY.md`
2. `.scratch/projects/14-specialist-runtime-architecture-research/01-CURRENT-STATE-AND-EVIDENCE.md`
3. `.scratch/projects/14-specialist-runtime-architecture-research/02-REQUIREMENTS.md`
4. `.scratch/projects/14-specialist-runtime-architecture-research/03-TRACEABILITY.md`
5. `.scratch/projects/14-specialist-runtime-architecture-research/04-ARCHITECTURE-OPTIONS.md`
6. `.scratch/projects/14-specialist-runtime-architecture-research/05-BACKEND-CAPABILITY-MATRIX.md`
7. `.scratch/projects/14-specialist-runtime-architecture-research/06-DURABILITY-AUTHORITY-AND-THREAT-MODEL.md`
8. `.scratch/projects/14-specialist-runtime-architecture-research/07-SPIKES-AND-TEST-PLAN.md`
9. `.scratch/projects/14-specialist-runtime-architecture-research/08-PHASED-ROADMAP.md`
10. `.scratch/projects/14-specialist-runtime-architecture-research/09-DECISIONS-AND-OPEN-QUESTIONS.md`
11. `.scratch/projects/14-specialist-runtime-architecture-research/evidence/baseline/2026-07-21-baseline.md`
12. `.scratch/projects/13-xgrammar-and-batching-todo/TODO.md`
13. `.scratch/projects/12-structured-agents-v2-library-study/CODE_REVIEW.md`

Also read the source, tests, `pyproject.toml`, installed DBOS/PydanticAI integration source relevant to any touched boundary, and recent git history. Do not infer dependency behavior from facade names.

## Accepted architecture

The target is Option A, the hybrid durable typed runtime (AD-01). Applications own their workflow graph. The library owns a narrow registered outer specialist workflow, strict typed envelopes/results, immutable identities, authorization evidence, and backend rendering. A logical batch is heterogeneous fan-out/fan-in over independently durable items; server continuous batching is a separately measured backend property.

Settled decisions AD-02 through AD-12 are constraints, not suggestions. In particular:

- malformed, absent, stale, abstaining, or errored approval never allows;
- request-scoped policy/config cannot live in shared mutable globals;
- the queue target is a registered library-owned workflow that owns validated `T`;
- idempotency pairs a key with canonical input digest and rejects different input;
- no `cast()` substitutes for local validation;
- blanket `permit_all` and intentional `bypass` are distinct;
- bypass does not skip validation, binding, durable recording, or audit;
- arbitrary external effects are at-least-once unless a named stronger protocol is proved;
- backend support is exact-profile Verified, Experimental, or Unsupported.

## This session's scope: Phase 0 only

Implement the smallest coherent, reviewable Phase 0 increments from `08-PHASED-ROADMAP.md`. Do not add public heterogeneous batching, LoRA loading/selection, context providers, child agents, automated agent approvers, or new live backend claims in this session.

Prioritize in this order:

1. CR-01/04/09: strict approval/config types; malformed decisions fail closed; empty policy composition is invalid; add failing-before regression tests.
2. CR-02/11: replace ambient global active allowlist with immutable/context-local request scope and make registration deterministic/safe; add the barrier/concurrency regression from SP-13.
3. CR-03/10: implement the smallest library-owned registered workflow needed for a real `Agent` queue to return strictly validated `T`; prove it with the real dependency, not only mocks. Follow SP-01.
4. CR-06: correct externally visible exactly-once claims and represent the current effect retry boundary honestly. Do not design the entire future effector framework unless necessary for correctness.
5. CR-08/15: clean wheel/sdist selection; enforce formatting and package-content tests from SP-14.

If these increments cannot remain safely reviewable together, complete them as separate commits/patch groups while maintaining a green tree between groups. Do not weaken a test or public contract to make an increment pass.

## Required method

1. Record a fresh baseline and compare it to the research baseline (`90725a5` had 32 passed, 1 skipped; lint/type passed; format failed on eight files; sdist was contaminated).
2. Reproduce each targeted finding on the current checkout before changing it. Preserve concise failing evidence under a new dated implementation project/artifact directory if repository convention supports it.
3. Inspect installed DBOS/PydanticAI source and public documentation for every dependency-sensitive edit. The 2026-07-21 environment registered a distinct DBOSAgent workflow callable; its ordinary `run` wrapper was not queueable.
4. Write the failing regression first, then the minimal fix, then rerun the focused test.
5. Run the complete relevant suite after every coherent increment.
6. Run the fresh final quality/package gates and inspect built archive contents.
7. Review the final diff for accidental API expansion, changed security defaults, new global state, casts, hidden backend assumptions, or user-file overlap.
8. Update docs/traceability only for behavior actually proved. Do not mark a backend feature Verified.

Use repository commands discovered from the current checkout. At the research baseline they were:

```text
devenv shell -- pytest
devenv shell -- ruff check src tests
devenv shell -- ty check src tests/typecheck_constraint.py
devenv shell -- ruff format --check src tests
devenv shell -- uv build --out-dir <unique-temp-directory>
```

Do not publish packages, push, mutate live vLLM/SGLang/llama.cpp services, download models/adapters, or run privileged effectors unless the owner separately requests and authorizes that action.

## Phase 0 acceptance

The session is not complete until evidence shows:

- malformed/missing/unknown approval values are non-allow;
- empty policy composition is rejected or explicit, never allow-by-vacuity;
- strict serialized settings reject wrong types, invalid ranges, and relevant unknown fields;
- concurrent request scopes cannot observe one another's allowlists and registry behavior is deterministic;
- a real library `Agent` can be queued through a registered outer workflow and returns the runtime-validated model/type `T` after a restart test appropriate to the environment;
- documentation and types no longer promise general exactly-once external effects;
- `pytest`, lint, type, and format checks pass;
- fresh wheel and sdist manifests contain only intended distributable content and install smoke tests pass;
- CR-01/02/03/04/06/08/09/10 and the touched CR-11/15 portions have explicit regression mapping.

If a real DBOS/Postgres or live-model prerequisite is unavailable, do not claim the corresponding acceptance criterion. Complete safe local work, label the gate blocked/experimental with exact evidence, and ask for the narrow prerequisite. Never replace a real-dependency requirement with a mock-only success claim.

## Handoff

Report:

- files and public behavior changed;
- finding/requirement IDs closed or still open;
- exact test/build commands and results;
- artifact-manifest result;
- compatibility/migration notes;
- residual risks or blocked evidence;
- confirmation that no out-of-scope specialist feature or live service mutation occurred.

Do not begin Phase 1 or feature expansion automatically. Stop for review after Phase 0.

---
