# Phased Roadmap

Date: 2026-07-21

## Sequencing rule

No batching, LoRA, context, child-agent, or automated-approval feature may expand the release surface until Phase 0 closes. Each phase is a small series of reviewable changes, not one long-lived branch. A phase exits only on its evidence gate; a merged API without passing evidence is incomplete.

## Phase 0 — fail-closed and releasable foundation

**Goal:** remove confirmed unsafe/incorrect behavior before adding reach.

Prerequisite: architecture/requirements approval.

Reviewable increments:

1. Strict approval/config message types; malformed decisions fail closed; empty policy composition is invalid; explicit legacy behavior/deprecation.
2. Replace process-global active allowlist with immutable/context-local scope; synchronize/freeze registration and add barrier tests.
3. Correct the queue contract through the smallest library-owned registered workflow proof, without yet publishing full batching.
4. Replace raw schema cast with strict local validation into `T`.
5. Correct exactly-once documentation/API language and add explicit effect safety metadata at the boundary that exists today.
6. Restrict package discovery/content; enforce formatting, artifact manifests, and first real-dependency regression tests.

Requirements: APR-01/04, AUTH-01/03, CFG-01/02, BAT-03, XGR-02, DUR-01/04, PKG-01, TEST-01/03.  
Findings closed: CR-01, CR-02, CR-03, CR-04, CR-06, CR-08, CR-09, CR-10, material CR-11/15 portions.

Exit criteria:

- malformed/missing approval and empty policy tests fail closed;
- concurrent allowlist/registry tests have zero leakage;
- real `Agent` queue proof returns validated `T`;
- external-effect documentation/tests state the actual guarantee;
- pytest, lint, type, format, wheel and sdist allowlists pass from a clean checkout;
- no public support claim was broadened.

Required spikes: SP-01, SP-13, SP-14.  
Release posture at exit: safe patch release may be considered; specialist feature release remains Experimental.

## Phase 1 — durable typed invocation core

**Goal:** establish the target domain/durability contract before batch and backend-specific features.

Reviewable increments:

1. Immutable identity types and versioned canonical `InvocationEnvelope`.
2. Same-key/same-digest attach and same-key/different-digest conflict.
3. Library-owned outer workflow with typed outcomes for success, denial, validation failure, cancellation, timeout, and operational failure.
4. Explicit client/DBOS lifecycle ownership and startup/registration/freeze/shutdown state machine.
5. Correlated, redacted stage evidence and capability-record skeleton defaulting to Unsupported/Experimental.

Requirements: DUR-01–05, LORA-01, XGR-02, CFG-03, OPS-01/02, SEC-04.  
Findings closed: CR-03/05/06/07/10/11/12/14 core portions.

Exit criteria:

- SP-01 fully passes across restart;
- idempotency conflict and all typed outcomes pass real DBOS integration;
- identity/version mismatch cannot silently replay;
- lifecycle/leak tests pass;
- capability lookup is configuration-specific and has no unqualified support boolean.

Required spikes: SP-01 and initial SP-10 boundaries.

## Phase 2 — authority modes and effect contracts

**Goal:** make all requested authority modes safe before adding more autonomous composition.

Reviewable increments:

1. `enforce`, `automated`, `permit_all`, and `bypass` algebra plus strict decision evidence.
2. Canonical decision binding, expiry, quorum/precedence, abstention/error, and durable human escalation.
3. Scoped bypass with alertable audit and a separately named raw-effector boundary only if owners explicitly accept reduced guarantees.
4. Effector declaration for idempotency, retry, outbox/transaction/reconciliation/compensation, and last-mile binding check.
5. Automated constrained approver support after deterministic controls.

Requirements: AUTH-01–04, APR-01–04, DUR-04/05, SEC-01/02.  
Findings closed: CR-01/04/05/06 and project-13 approval/bypass workstream.

Exit criteria:

- SP-11 and SP-12 pass;
- one-field command mutations invalidate approval/bypass;
- injection and self-approval tests fail closed;
- crash ambiguity for the fake effector is handled by a named mechanism;
- documentation distinguishes permit-all, bypass, and raw effect access.

Required spikes: SP-10–12.

## Phase 3 — heterogeneous logical batching and adapter domain

**Goal:** publish the engine-neutral specialist batch contract without claiming unproved GPU behavior.

Reviewable increments:

1. `BatchItem`, per-item handle, and typed outcome; submission-index metadata and partial results.
2. Bounded queue admission, cancellation, timeouts, aggregate/fan-in utilities, retry eligibility.
3. Agent/base/adapter identities, control-plane adapter registry, and backend renderer interfaces.
4. Constraint identities and neutral renderer contract.
5. Observability sufficient to distinguish library batch, client concurrency, and backend batching.

Requirements: BAT-01/02/04–06, LORA-01/02/04/05, XGR-01, OPS-02/03, SEC-03.  
Owner intent advanced: batched specialist is now first-class and per-agent LoRA is representable.

Exit criteria:

- SP-02 passes mixed success/failure/cancellation/restart;
- adapter path/unregistered/wrong-base cases fail before dispatch;
- renderer contract tests cover both engines without engine syntax in persisted application identity;
- API/documentation state that engine continuous batching is not yet Verified.

Required spike: SP-02; backend live proof intentionally deferred to Phase 4.

## Phase 4 — vLLM then SGLang qualification

**Goal:** turn upstream hypotheses into exact-profile evidence.

Order:

1. vLLM non-GGUF XGrammar control (SP-03).
2. vLLM two-adapter continuous-batching profile (SP-05).
3. vLLM mixed adapter/constraint run (SP-07).
4. Select/pin viable SGLang version/model; prove base + XGrammar (SP-04).
5. SGLang OpenAI-compatible multi-LoRA (SP-06).
6. SGLang mixed adapter/constraint run (SP-07).

Requirements: LORA-02/03/05, XGR-03–06, BAT-04, CFG-03, TEST-02.  
Owner intent advanced: vLLM/SGLang priority, both XGrammar paths, concurrent different LoRAs.

Exit criteria per engine/profile:

- every claimed matrix cell passes its independent qualification gate;
- server metrics prove overlap/continuous batching rather than client-only concurrency;
- adapter/constraint identity cross-talk is zero in the declared test;
- limits, failure modes, hardware, model, quantization, plugin, versions, and evidence expiry are published;
- failed cells remain Experimental or Unsupported.

This phase may verify vLLM while SGLang remains Experimental; do not hold honest per-profile statuses hostage to a single combined label.

## Phase 5 — bounded context and child-agent composition

**Goal:** add automatic enrichment and bounded chaining on top of the proven core.

Reviewable increments:

1. Read-only typed `ContextProvider` with provenance/trust/freshness/sensitivity/budgets.
2. Allowlisted function/script provider contract with sandbox, resource, environment, network, cwd, and output policy.
3. Typed child-agent stage with parent lineage and usage/depth/time/cost limits.
4. Separate final constrained generation stage and explicit message-history sanitization.
5. Tool/effector integration only through Phase 2 authority/durability declarations.

Requirements: CTX-01–03, AGC-01–03, OPS-03, SEC-01/02.  
Owner intent advanced: automatic context enrichment, functions/scripts, bounded chaining before final answer.

Exit criteria:

- SP-08 and SP-09 pass, followed by applicable SP-10 crash boundaries;
- cyclic/oversized/stale/malicious context terminates or is rejected predictably;
- final `T` is produced by its own strict constrained stage;
- child/provider output cannot escalate authority or select identities.

## Phase 6 — production qualification and release

**Goal:** prove the complete supported profiles and freeze accurate claims.

Reviewable increments:

1. Full concurrency/saturation/cancellation and crash matrix.
2. Security/injection/adapter/secret/retention suite.
3. Supported dependency/environment install matrix and private-API probes.
4. Live evidence manifests, expiry enforcement, README/API examples, operational runbooks.
5. Release candidate artifact rebuild and independent evidence review.

Requirements: all P0/P1, especially TEST-01–03, PKG-01/02, OPS-02/03, SEC-01–04.

Exit criteria:

- every P0/P1 requirement has traceable passing evidence or an explicitly accepted, documented deferral that removes the affected public claim;
- CR-01–15 are closed with regression evidence;
- supported capability cells have current manifests; other cells say Experimental/Unsupported;
- package/CI/docs/install/live gates are green from a clean commit;
- migration and rollback guidance exists for 0.3.0 users.

## Migration and compatibility

1. Introduce strict new types beside the 0.3.0 facade, but route every compatible legacy call through the same outer workflow.
2. Emit targeted deprecations for calls missing identity/mode/effect semantics; never guess a security mode from legacy truthiness.
3. Preserve existing non-mutating single-call ergonomics through constructors/defaults only when the resulting envelope is unambiguous.
4. Version durable schemas and canonicalization before persisting new executions. Do not silently reinterpret old records.
5. Offer migration adapters for stored configuration/messages where safe; reject ambiguous approval/idempotency records.
6. Keep engine-specific flags in deployment/profile adapters rather than expanding the neutral public API.

## What intentionally waits

- A fully library-owned planning graph waits for repeated proven patterns.
- Offline provider batch APIs are not required.
- Arbitrary runtime LoRA loading remains outside the data plane.
- llama.cpp is not upgraded into the XGrammar/mixed-LoRA contract.
- Recursive open-ended agents, ambient global context, and universal exactly-once effects are rejected, not backlogged.
