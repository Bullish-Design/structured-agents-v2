# Requirements

Date: 2026-07-21  
Normative terms: **MUST**, **SHOULD**, and **MAY** have their RFC 2119 meanings.

## Requirement record and evidence tiers

Every row is normative. “Acceptance” is the observable condition, not an implementation prescription.

| Tier | Required evidence |
|---|---|
| T0 | Static analysis, schema/property checks, or deterministic unit test |
| T1 | Real dependency integration in an isolated local process/database |
| T2 | Concurrency, cancellation, restart, replay, or fault-injection test |
| T3 | Live pinned backend/model/adapter test with request/result/metrics artifact |
| T4 | Clean build/install, compatibility matrix, CI, security, and release artifact |

The first bold RFC 2119 term in each requirement is its normative priority: **MUST** or **SHOULD**. A **MAY** inside a MUST requirement describes an optional facility, not the priority of its safeguards. The `Pri` column adds delivery urgency: P0 release blocker, P1 required for the specialist-runtime release, or P2 hardening/later extension. “Related” names the applicable finding, owner/ToDo workstream, or a new target-architecture risk; the category-level traceability in `03-TRACEABILITY.md` supplies the complete prior-work mapping.

## Batch execution

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| BAT-01 | P1 | The API **MUST** accept a heterogeneous sequence of typed batch items, each carrying its own agent, adapter, constraint, input, and invocation identity; a homogeneous convenience wrapper **MAY** exist. This makes the requested specialist unit explicit. | Two items with different agents/adapters/constraints validate and execute without positional or identity ambiguity. | T1/T3 | DUR-01, LORA-01, XGR-01 | Owner 1/3; positional mix-ups |
| BAT-02 | P1 | Every item **MUST** receive an independent durable handle and terminal outcome (`succeeded`, `denied`, `failed`, `cancelled`, or `timed_out`). One item must not erase another's result. | Partial success and per-item lookup work after process restart. | T2 | BAT-01, DUR-02 | Project-13 batch; partial loss |
| BAT-03 | P0 | Queue submission **MUST** target a registered library-owned durable workflow whose persisted result is the locally validated output type. | A real `Agent` queues, restarts, and returns `T`; the registered-function check passes. | T1/T2 | DUR-01, XGR-02 | CR-03/10 |
| BAT-04 | P1 | Client concurrency **MUST** be flow-controlled and distinct from server continuous batching. A capability claim **MUST** include engine metrics proving overlapping running sequences or batching. | Load test records client in-flight count plus engine running/queued/token-step metrics. | T3 | OPS-02/03, CFG-03 | Owner 3; false batching claims |
| BAT-05 | P1 | The batch surface **MUST** define bounded admission, backpressure, per-item timeout/cancellation, aggregate timeout, and retry eligibility. | Saturation and cancellation tests remain within configured bounds and preserve other item outcomes. | T2 | OPS-03, DUR-04 | CR-13; retry storms |
| BAT-06 | P1 | Ordering **MUST** be explicit: submission order is retained as metadata, while completion order and execution order are not guaranteed. | Tests cover out-of-order completion and stable identity-based reconstruction. | T1 | BAT-02 | accidental positional correlation |

## Agent and LoRA identity

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| LORA-01 | P1 | `AgentIdentity`, `BaseModelIdentity`, and `AdapterIdentity` **MUST** be immutable/versioned values, not free-form paths. Adapter identity includes logical ID, content digest/revision, base-model identity, rank, and target-module metadata when known. | Identity serialization round-trips and rejects digest/base mismatches. | T0/T1 | CFG-01 | Owner 2; replay drift |
| LORA-02 | P1 | Every agent **MUST** bind exactly one default adapter identity or explicit base-model sentinel, with an allowed per-invocation override policy. | Agent A/B select distinct adapters and recorded outcomes retain their identities. | T1/T3 | LORA-01, DUR-03 | Owner 2 |
| LORA-03 | P1 | An engine profile claiming multi-LoRA **MUST** declare loaded-adapter limits, adapters-per-batch, rank limits, eviction/pinning behavior, base compatibility, and request rendering. | Qualification rejects an adapter outside the declared limits and proves two distinct adapters overlap in one measured run. | T3 | CFG-03, OPS-02 | Owner 3/4; memory/eviction |
| LORA-04 | P0 | Data-plane inputs **MUST NOT** accept arbitrary adapter filesystem paths or unrestricted runtime load/unload. Trusted control-plane registration verifies artifact origin/digest and maps logical IDs to backend selectors. | Traversal/unregistered IDs fail before HTTP; audit identifies the registry revision. | T1/T4 | SEC-03, LORA-01 | Upstream security warning; RCE/supply chain |
| LORA-05 | P1 | Backend rendering **MUST** remain adapter-specific: e.g. vLLM logical model selection and SGLang base/adapter selection are derived from one neutral identity. | Contract tests compare neutral request with exact backend payload and response identity. | T1/T3 | LORA-01, CFG-03 | backend leakage |

## Structured outputs and XGrammar

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| XGR-01 | P1 | The public constraint model **MUST** be engine-neutral and distinguish JSON Schema, regex, choice, EBNF/grammar, structural tags, and unconstrained output. Unsupported combinations fail before dispatch. | Exhaustive renderer tests cover supported and rejected kind/profile pairs. | T0/T1 | CFG-03 | Project-13 XGrammar; dialect drift |
| XGR-02 | P0 | Successful output **MUST** pass strict local parsing/validation into `T`; casts and backend “finished” status are insufficient. The constraint digest and validator version are recorded. | Raw dict-to-model conversion returns the model; malformed/extra-field output never succeeds. | T0/T1 | CFG-01, DUR-03 | CR-10 |
| XGR-03 | P1 | vLLM XGrammar support **MUST** be live-verified for each claimed version/model/quantization/plugin and constraint kind. | Schema, regex, choice, and grammar cases pass through the public library with evidence artifacts. | T3 | XGR-01/02, CFG-03 | Owner 5; historical evidence ages |
| XGR-04 | P1 | SGLang XGrammar support **MUST** be live-verified independently for each claimed exact profile. Upstream docs alone never set status `Verified`. | Pinned SGLang base chat plus schema, regex, and EBNF pass through public API. | T3 | XGR-01/02, CFG-03 | Owner 5; current profile absent |
| XGR-05 | P1 | A profile claiming heterogeneous constrained batching **MUST** prove simultaneous requests with distinct constraints and adapters and validate each result against its own contract. | Mixed load has zero cross-item validator/adapter identity errors and shows overlap metrics. | T3 | BAT-04, LORA-03, XGR-03/04 | cache/key contamination |
| XGR-06 | P2 | Grammar compilation/cache entries **SHOULD** key engine, tokenizer, constraint digest, compiler version, strictness flags, and serialization version. | Cache property tests reject all mismatched dimensions and survive safe restart. | T1/T2 | DUR-03 | stale masks/version mismatch |

## Context enrichment and agent composition

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| CTX-01 | P1 | A `ContextProvider` **MUST** return typed content plus provenance, trust class, freshness, sensitivity, and deterministic cache identity; providers are read-only by contract. | Provider results serialize durably and untrusted content cannot alter authority metadata. | T1/T2 | DUR-03, SEC-01/02 | Owner 6/7; prompt injection |
| CTX-02 | P1 | Any function or script used for context enrichment **MUST** run through a declared sandbox, input schema, output schema, timeout, resource budget, network policy, and provenance capture; offering this facility is optional. | Allowlisted function/script passes; undeclared I/O, timeout, and oversized output fail closed. | T1/T4 | CTX-01, OPS-03, SEC-01 | Owner 7; code execution |
| CTX-03 | P1 | Enrichment **MUST** be bounded by provider calls, bytes/tokens, latency, cost, and freshness, with deterministic truncation and secret redaction. | Boundary and redaction tests produce stable manifests and no secret-bearing prompt/artifact. | T1/T4 | OPS-02/03, SEC-04 | data exfiltration/context blowup |
| AGC-01 | P1 | Agent loops **MUST** declare maximum depth, turns, model/tool calls, tokens, time, and cost; exhaustion yields a typed non-success outcome. | Cyclic delegation terminates at each configured limit without orphan work. | T2 | CTX-01, OPS-03 | Owner 6; runaway spend |
| AGC-02 | P1 | Child-agent calls **MUST** use typed input/output and immutable identity and appear as child durable stages with parent correlation. | Restart retains parent/child lineage and exact output types. | T2 | DUR-01/03, AGC-01 | replay drift |
| AGC-03 | P1 | After enrichment/tool/child-agent work, final output **MUST** be produced by a distinct constrained generation stage unless a backend profile has live evidence for the combined loop/constraint contract. | Final call has no mutating tools, records final constraint, and rejects invalid output. | T1/T3 | XGR-02, AGC-01 | Owner 6; grammar/tool incompatibility |

## Authority and approval

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| AUTH-01 | P0 | Mutating requests **MUST** select an explicit `enforce`, `automated`, `permit_all`, or `bypass` mode; omission uses a safe configured default. | Exhaustive mode/state tests show no implicit truthy/falsy conversion. | T0/T1 | CFG-01 | Owner 8; CR-01/04 |
| AUTH-02 | P0 | Input/schema validation, identity resolution, request binding, durable outcome recording, and audit **MUST** occur in every mode. | Invalid/unregistered requests fail identically before policy or effect in all four modes. | T1/T2 | DUR-01/02, SEC-04 | “bypass” as validation bypass |
| AUTH-03 | P0 | `permit_all` **MUST** record an explicit allow decision; `bypass` **MUST** record that policy evaluation was intentionally skipped. Neither is encoded as an empty policy set. | Audit and result types distinguish the modes; empty `all_of` is invalid. | T0/T1 | AUTH-01, APR-01 | CR-04; Owner 8 |
| AUTH-04 | P1 | Bypass **MUST** be scope-, actor-, action-, environment-, reason-, and expiry-bound and disabled by default for untrusted callers. | Scope/expiry/role negative tests fail closed and every bypass emits an alertable event. | T1/T4 | AUTH-01, SEC-01 | privilege escalation |
| APR-01 | P0 | Approval decisions **MUST** be strict tagged values: `allow`, `deny`, `abstain`, `requires_human`, or `error`, with evidence. Missing/malformed/unknown values never allow. | Malformed strings, numbers, missing fields, timeouts, and exceptions all produce non-allow. | T0/T1 | CFG-01 | CR-01; Owner 9 |
| APR-02 | P0 | A decision **MUST** bind canonical request digest, subject, action, normalized arguments, agent/model/adapter/constraint identities, policy revision, approver identity, issued/expiry time, and nonce/correlation ID. | Any one-field mutation invalidates reuse; exact replay remains attributable. | T1/T2 | DUR-02/03, APR-01 | CR-05; TOCTOU |
| APR-03 | P1 | Automated approvers **MAY** be constrained agents, but **MUST** use immutable prompt/model/adapter/constraint identity and strict locally validated output; they cannot approve their own configuration changes. | Adversarial prompt/format tests deny on invalid evidence and expose all approver versions. | T1/T3/T4 | XGR-02, APR-01/02, SEC-02 | Owner 9; model compromise |
| APR-04 | P1 | Policy composition **MUST** specify empty-set behavior, quorum, deny precedence, abstention/error handling, human escalation, freshness, and re-approval after material change. | Truth-table/property tests cover all decision combinations and stale/request-mutated cases. | T0/T2 | APR-01/02 | CR-04/05 |

## Durability and replay

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| DUR-01 | P0 | A library-owned outer workflow **MUST** contain authorization, provider/model stages, local validation, and the terminal typed outcome; only registered workflows are queue targets. | Real queued execution, denial, restart, and validation failure all resolve from one handle. | T1/T2 | BAT-03, AUTH-02 | CR-03/05 |
| DUR-02 | P0 | `InvocationEnvelope` **MUST** have canonical serialization and digest. Same idempotency key/different digest is a typed conflict; same key/same digest attaches to the existing execution. | Property tests and DBOS integration prove both cases across restart. | T1/T2 | CFG-01 | DBOS workflow-ID no-op behavior |
| DUR-03 | P1 | Persisted records **MUST** contain versioned identities for code/schema, agent, base model, adapter, constraint, prompt/tools, policy, approver, context providers, backend profile, and serializer. | Replay refuses or explicitly migrates missing/mismatched required identities. | T1/T2 | LORA-01, XGR-02 | silent replay drift |
| DUR-04 | P0 | The API **MUST** label external effects as at-least-once unless protected by a named transactional/outbox/idempotent protocol; retry eligibility and compensation are declared per effect. | Crash-after-commit-before-checkpoint test does not duplicate an idempotent effect; unprotected effect is rejected or prominently typed. | T2 | DUR-01/02 | CR-06 |
| DUR-05 | P1 | Recovery **MUST** define behavior for crash before dispatch, during model call, after model response, during validation, before effect, after effect, and during result commit. | Fault injection at each boundary yields one documented terminal/recoverable state and complete audit lineage. | T2 | DUR-01/04 | partial/duplicate work |

## Configuration, operations, security, packaging, and testing

| ID | Pri | Requirement and rationale | Acceptance | Tier | Dependencies | Related / risks |
|---|---:|---|---|---:|---|---|
| CFG-01 | P0 | All public configuration and persisted messages **MUST** use strict validation, bounded numeric values, forbidden unknown fields where security-relevant, and explicit enums. | Invalid temperature/token limits/decision shapes fail construction. | T0 | — | CR-01/09 |
| CFG-02 | P0 | Request-scoped policy/configuration **MUST** use immutable passed values or context-local scope; mutable registries are synchronized and frozen after startup where possible. | Barrier-based concurrency tests show no cross-request leakage or lost registry update. | T2 | CFG-01 | CR-02/11 |
| CFG-03 | P1 | Capabilities **MUST** be configuration-specific records with status `Verified`, `Experimental`, or `Unsupported`, evidence date/artifact, version/model/quantization/plugin/GPU dimensions, and expiry. | Public lookup never reports an unqualified boolean and stale/mismatched evidence downgrades status. | T0/T3 | OPS-02 | CR-07; Owner 11 |
| OPS-01 | P1 | Client, queue, database, and background-task ownership **MUST** be explicit and support idempotent close plus async context management. | Leak/double-close/external-client ownership tests pass. | T1/T2 | — | CR-12 |
| OPS-02 | P1 | Correlated logs/traces/metrics **MUST** expose invocation/item/workflow IDs, backend/profile, adapter/constraint digests, queue/engine latency, retries, decisions, and status while redacting prompts, secrets, and paths by policy. | Observability test correlates a mixed run without leaking seeded secrets. | T3/T4 | SEC-04 | BAT-04; incident response |
| OPS-03 | P1 | Every queue, provider, child agent, model call, tool/effect, and aggregate batch **MUST** have explicit resource, timeout, cancellation, and admission behavior. | Saturation/fault suite shows bounded tasks, memory, descriptors, and queue depth. | T2/T4 | CFG-01 | CR-13 |
| SEC-01 | P0 | Inputs, context, tools, adapters, approvers, callers, and backend profiles **MUST** carry trust classification; privileges are derived from authenticated control-plane policy, never prompt text. | Trust-boundary tests prevent low-trust content from changing identities, mode, tool scope, or destination. | T1/T4 | AUTH-02 | confused deputy |
| SEC-02 | P0 | Untrusted model/context output **MUST** remain data. It cannot directly select authority mode, approve itself, register adapters/tools, or form unsanitized mutating arguments. | Injection corpus fails to cross each boundary and yields recorded denial/escalation. | T1/T4 | SEC-01, APR-03 | Owner 9; prompt injection |
| SEC-03 | P0 | Adapter/control-plane operations **MUST** authenticate, authorize, validate origin/digest/base compatibility, use allowlisted roots, and separate registration from invocation. | Tampered, traversal, wrong-base, and untrusted dynamic-load attempts fail closed. | T1/T4 | LORA-04 | supply chain/RCE |
| SEC-04 | P0 | Durable serialization and evidence **MUST NOT** treat pickle as an untrusted interchange format or persist secrets/raw sensitive context by default; encryption/redaction/retention are specified. | Malicious/untrusted payload and secret-canary tests pass; retention deletion is auditable. | T1/T4 | DUR-03 | DBOS pickle boundary |
| PKG-01 | P0 | Wheel and sdist **MUST** contain only intended distributable files; deploy environments, nested projects, vendor tests, secrets, models, and research artifacts are excluded. | Built-artifact allowlist test passes from a clean checkout. | T4 | — | CR-08 |
| PKG-02 | P1 | Supported dependency ranges **MUST** be tested as a compatibility matrix and public APIs preferred; any private import is isolated, pinned, probed, and documented. | Min/max/pinned jobs pass or unsupported combinations fail at startup with guidance. | T4 | CFG-03 | CR-14 |
| TEST-01 | P0 | CI **MUST** run deterministic unit, real-dependency integration, concurrency/replay, and artifact tests; mocks cannot be the sole evidence for public durability or queue behavior. | CR-01–15 each has a failing-before/passing-after test or explicit release gate. | T0–T4 | all P0 | CR-15 |
| TEST-02 | P1 | Live backend evidence **MUST** include a machine-readable manifest, sanitized request/response or digest, server config/version, model/adapter/constraint identities, metrics window, timestamps, and command/result logs. | Independent reviewer can classify every matrix cell from artifacts alone. | T3/T4 | CFG-03, OPS-02 | Owner 11 |
| TEST-03 | P0 | Formatting, lint, types, tests, documentation examples, build contents, and support-matrix freshness **MUST** be release gates. | Clean checkout passes all gates; stale/absent live evidence cannot be labeled Verified. | T4 | PKG-01, TEST-02 | CR-15 |

## Global acceptance rule

A requirement is not complete merely because code exists. It is complete only when its stated evidence tiers pass, the traceability matrix points to the artifacts, documentation uses the same semantics, and no narrower configuration is generalized into an engine-wide claim.
