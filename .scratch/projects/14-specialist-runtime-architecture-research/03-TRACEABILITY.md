# Traceability

Date: 2026-07-21

This document is the completeness index. Requirement IDs are normative in `02-REQUIREMENTS.md`; architecture decisions (AD) are in `04-ARCHITECTURE-OPTIONS.md`; spikes (SP) and permanent suites are in `07-SPIKES-AND-TEST-PLAN.md`; phases are in `08-PHASED-ROADMAP.md`.

## Code-review findings

| Finding | Requirements | Decisions | Planned proof | Phase / disposition |
|---|---|---|---|---|
| CR-01 malformed approval can allow | APR-01, CFG-01, AUTH-01/02 | AD-08/09 | UT-AUTH, IT-APPROVAL, SP-11 | Phase 0; release blocker |
| CR-02 allowlist leaks across threads | CFG-02, SEC-01 | AD-12 | CT-CONFIG, SP-13 | Phase 0; release blocker |
| CR-03 real public queue targets unregistered function | BAT-03, DUR-01 | AD-02 | IT-QUEUE, SP-01 | Phase 0/1; release blocker |
| CR-04 empty `all_of` authorizes | AUTH-01/03, APR-04 | AD-08/09 | UT-AUTH, SP-11 | Phase 0; release blocker |
| CR-05 approval binds workflow, not exact request | APR-02/04, DUR-02, AUTH-02 | AD-03/09 | UT-DIGEST, IT-IDEMPOTENCY, SP-11/12 | Phase 1/2; release blocker for mutating work |
| CR-06 exactly-once claim too broad | DUR-04/05 | AD-10 | RT-EFFECT, SP-10 | Phase 0 docs/API; Phase 2 mechanism |
| CR-07 backends advertised before verification | CFG-03, XGR-03/04, TEST-02/03 | AD-11 | LT-VLLM, LT-SGLANG, SP-03–07 | Phase 1 status model; Phase 4 qualification |
| CR-08 sdist contains unrelated repository content | PKG-01, TEST-03 | — | PT-ARTIFACT, SP-14 | Phase 0; release blocker |
| CR-09 serialized config not type-strict | CFG-01 | AD-12 | UT-CONFIG | Phase 0 |
| CR-10 schema parsing doesn't enforce local contract | XGR-02, BAT-03 | AD-02 | UT-CONSTRAINT, IT-QUEUE, SP-01/03/04 | Phase 0 |
| CR-11 factory discovery is unsynchronized global mutation | CFG-02, DUR-03 | AD-12 | CT-REGISTRY, SP-13 | Phase 0/1 |
| CR-12 lifecycle/client ownership incomplete | OPS-01 | AD-01 | IT-CLIENT, leak/close tests | Phase 1 |
| CR-13 process effectors lack resource bounds; loose decoding | OPS-03, CTX-02/03, CFG-01 | AD-07 | saturation/timeout tests, SP-09/10 | Phase 1/5 |
| CR-14 broad ranges/private imports conflict with compatibility | PKG-02, CFG-03 | AD-11 | PT-INSTALL, startup probes, SP-14 | Phase 1/6 |
| CR-15 tests/docs/release gates lag claims | TEST-01/02/03 | AD-11 | all suites; SP-01–14 evidence manifests | Phase 0 and every later gate |

## Project-13 ToDo workstreams

| ToDo section | Requirements | Decisions | Planned proof | Roadmap |
|---|---|---|---|---|
| P0 — Repair concurrency path used for verification | BAT-03/04, DUR-01, OPS-02/03 | AD-02/04 | SP-01/02; IT-QUEUE, CT-BATCH | Phases 0, 1, 3 |
| P0 — Make batched specialist dispatch first-class | BAT-01–06, LORA-02 | AD-04 | SP-02; CT-BATCH | Phase 3 |
| P1 — Verify SGLang XGrammar constrained outputs | XGR-01/02/04, CFG-03, TEST-02 | AD-05/11 | SP-04; LT-SGLANG | Phase 4 |
| P1 — Investigate/verify XGrammar with vLLM | XGR-01/02/03, CFG-03, TEST-02 | AD-05/11 | SP-03; LT-VLLM | Phase 4 |
| P1 — Per-agent LoRA and multi-LoRA batching | LORA-01–05, BAT-04, SEC-03 | AD-04/05 | SP-05–07 | Phases 3/4 |
| P1 — Durable context enrichment, hooks, chaining | CTX-01–03, AGC-01–03, DUR-05 | AD-06/07 | SP-08–10 | Phase 5 |
| P1 — Explicit bypass and automated approval | AUTH-01–04, APR-01–04, SEC-01/02 | AD-08/09 | SP-11/12; ST-INJECTION | Phase 2 |
| P1 — Concurrent requests and continuous batching | BAT-04/05, LORA-03, OPS-02/03 | AD-04/11 | SP-05–07; engine metrics | Phases 3/4 |
| P2 — Capability/documentation cleanup + live evidence checklist | CFG-03, TEST-02/03, PKG-01/02 | AD-11 | manifest checks, PT-DOCS, SP-14 | Phases 1/6 |

Project-13's architectural requirement—different agents with different LoRAs running concurrently and continuously batched where the engine supports it—is specifically BAT-01/04, LORA-02/03, AD-04/05, and SP-05–07. It is not considered satisfied by SP-02 alone.

## Owner decision coverage

| # | Owner decision | Requirements | Decisions | Proof / gate |
|---:|---|---|---|---|
| 1 | Batched specialist central | BAT-01–06 | AD-01/04 | SP-02; Phase 3 |
| 2 | Per-agent LoRA required | LORA-01/02/05 | AD-05 | SP-05/06; Phase 4 |
| 3 | Different agents/LoRAs concurrently batched where supported | BAT-04, LORA-03, XGR-05 | AD-04/05/11 | SP-05–07 + engine metrics |
| 4 | vLLM/SGLang priority | XGR-03/04, LORA-03, TEST-02 | AD-11 | SP-03–07; Phase 4 |
| 5 | XGrammar verified on both, never inferred | XGR-03/04, CFG-03 | AD-11 | LT-VLLM/LT-SGLANG manifests |
| 6 | Automatic enrichment and bounded chaining before final | CTX-01/03, AGC-01–03 | AD-06/07 | SP-08/09; Phase 5 |
| 7 | Functions/scripts for context or automated checks | CTX-02, APR-04, OPS-03 | AD-07/09 | SP-09/11 |
| 8 | Enforce/automated/blanket/bypass modes | AUTH-01–04 | AD-08 | SP-11/12; Phase 2 |
| 9 | Automated approvers may be constrained agents | APR-01–04, SEC-02 | AD-09 | SP-11; ST-INJECTION |
| 10 | Local-first specialized small models | LORA-02/03, CFG-03 | AD-01/05/11 | OQ-08 selection; SP-03–07 |
| 11 | Explicit Verified/Experimental/Unsupported evidence | CFG-03, TEST-02/03 | AD-11 | capability freshness/release gates |

“Blanket” is represented by the less ambiguous public name `permit_all`; it remains distinct from `bypass`.

## Requirement-category coverage

| Category | Current evidence/problem source | Architecture owner | Implementation phase | Primary verification |
|---|---|---|---|---|
| BAT | CR-03/13, project-13 batching/concurrency | AD-02/04 | 0, 1, 3, 4 | SP-01/02/05–07 |
| LORA | owner 2/3, project-13 multi-LoRA | AD-05/11 | 3, 4 | SP-05–07 |
| XGR | CR-07/10, project-10/13 evidence | AD-05/06/11 | 0, 3, 4 | SP-03/04/07 |
| CTX | owner 6/7, project-13 chaining | AD-06/07 | 5 | SP-08/09 |
| AGC | owner 6, PydanticAI composition | AD-06/07 | 5 | SP-09/10 |
| AUTH | CR-01/04/05, owner 8 | AD-08/09 | 0, 2 | SP-11/12 |
| APR | CR-01/04/05, owner 9 | AD-09 | 0, 2 | SP-11/12 |
| DUR | CR-03/05/06, DBOS behavior | AD-02/03/10 | 0–2 | SP-01/10/12 |
| CFG | CR-02/07/09/11 | AD-11/12 | 0, 1 | SP-13 + capability gates |
| OPS | CR-12/13, backend metrics | AD-01/04/11 | 1, 3–6 | SP-02/05–10 |
| SEC | authority/adapters/context/pickle boundaries | AD-05/07/08/09/10 | 1–6 | SP-07/09/11/12 + ST suites |
| PKG | CR-08/14 | AD-11 | 0, 6 | SP-14 |
| TEST | CR-15 and evidence directive | all | every phase | all tiers/manifests |

## Architecture decision verification

| AD | Requirements implementing it | Falsification target |
|---|---|---|
| AD-01 hybrid runtime | DUR-01, CTX-01, AGC-02, OPS-01 | SP-01/08/09 exposes unacceptable composition cost |
| AD-02 outer typed workflow | BAT-03, DUR-01 | SP-01 cannot queue/replay typed result |
| AD-03 envelope/digest | DUR-02/03, APR-02 | idempotency mutation/property tests fail |
| AD-04 logical heterogeneous batch | BAT-01–06 | SP-02 loses isolation/partial outcomes |
| AD-05 neutral adapter identity/rendering | LORA-01–05 | SP-05/06 require irreconcilable public semantics |
| AD-06 separate final constrained call | AGC-03, XGR-02 | SP-08/09 shows type/provenance cannot survive handoff |
| AD-07 read context vs mutating effects | CTX-01/02, DUR-04 | SP-08/10 cannot isolate replay/effect boundary |
| AD-08 four modes | AUTH-01–04 | SP-11/12 reveals ambiguous state/composition |
| AD-09 typed bound evidence | APR-01–04 | mutation/injection/quorum tests fail closed incorrectly |
| AD-10 at-least-once external effects | DUR-04/05 | only revisable with a specifically proved stronger protocol |
| AD-11 evidence-backed capability | CFG-03, TEST-02/03 | profile mismatch/staleness gate fails |
| AD-12 immutable/context-local configuration | CFG-02, DUR-03 | SP-13 observes cross-request/registry drift |

## Completeness audit

- CR-01 through CR-15: 15/15 mapped.
- Project-13 major P0/P1/P2 workstreams: 9/9 mapped.
- Owner decisions: 11/11 mapped.
- Required spike topics: 14/14 specified as SP-01 through SP-14.
- Requirement categories requested by kickoff: 13/13 used.
- Architecture options: two credible end-to-end options plus one deliberately minimal option scored.
- Evidence vocabulary: Verified locally, Verified upstream, Inferred, Unknown, Contradicted used; product capability states separately normalized to Verified, Experimental, Unsupported.
