# Architecture Options

Date: 2026-07-21

## Evaluation method

Scores are 1 (poor) through 5 (strong). Weights reflect owner priorities and the confirmed safety/durability findings. Scores are planning judgments—**inferred**, not benchmark results.

| Criterion | Weight |
|---|---:|
| Fail-closed authority and request binding | 12 |
| Durable/replay correctness | 12 |
| Heterogeneous batch and partial-result model | 10 |
| Per-agent/mixed LoRA identity | 10 |
| vLLM/SGLang portability | 9 |
| XGrammar portability and validation | 8 |
| Bounded context/child-agent composition | 8 |
| Testability and evidence clarity | 8 |
| Incremental migration | 7 |
| Operational visibility/backpressure | 6 |
| API simplicity | 4 |
| Backend feature reach | 3 |
| Maintenance cost (higher is cheaper) | 3 |

## Option A — hybrid durable typed runtime (recommended)

Applications own the workflow graph. The library owns one narrow outer durable specialist invocation, neutral domain contracts, backend adapters, and optional typed primitives for providers/children/authority.

```text
app workflow/graph
   |
   +--> ContextProvider / child stage(s) -- typed evidence -->+
   |                                                        |
   +--> authority stage ------------------------------------+
                                                            v
                                    DurableSpecialistInvocation[T]
                                      - canonical envelope/digest
                                      - immutable identities
                                      - strict decision/result validation
                                      - durable terminal outcome
                                                |
                                  +-------------+-------------+
                                  |                           |
                              vLLM adapter                SGLang adapter
                                  |                           |
                                  +------ engine scheduler ---+
```

Why it fits: it corrects the durability boundary without trying to replace DBOS or PydanticAI, makes backend differences explicit, and lets applications express domain-specific graphs. It supports a first-class batch facade by composing independently durable items.

Tradeoffs: applications must still design their workflow graph; durable identity/envelope migration requires careful versioning; true continuous batching remains an engine qualification rather than a library guarantee.

## Option B — library-owned full agent graph/runtime

The library owns planning, context retrieval, child agents, tool loops, policy, finalization, queues, and most workflow structure behind a large state machine.

Advantages: one batteries-included API, centralized budgets and observability, greater opportunity for cross-request scheduling.

Costs: duplicates or constrains DBOS/PydanticAI graph semantics, greatly expands trusted code and replay migrations, couples generic library releases to product-specific orchestration, and makes provider-specific tool/grammar combinations harder to reason about. This is an attractive future product only after the narrow contracts are proven.

## Option C — minimal queue repair plus batch convenience facade

Fix the registered callable, add `gather()` over queue handles, pass adapter strings through, and retain the current policy/config model.

Advantages: fastest short-term patch, smallest surface change, useful as an internal spike.

Costs: leaves request binding, global races, typed parsing, exact-once language, identity, context/agency, security, and capability evidence structurally unresolved. It can create the appearance of batching without proving server-side continuous batching. It is not an acceptable target architecture.

## Weighted comparison

Weighted score is the sum of `weight × score`, divided by the maximum, shown as a percentage.

| Criterion | A | B | C |
|---|---:|---:|---:|
| Fail-closed authority and request binding | 5 | 4 | 1 |
| Durable/replay correctness | 5 | 4 | 2 |
| Heterogeneous batch and partial-result model | 5 | 4 | 2 |
| Per-agent/mixed LoRA identity | 5 | 4 | 2 |
| vLLM/SGLang portability | 5 | 3 | 2 |
| XGrammar portability and validation | 5 | 4 | 2 |
| Bounded context/child-agent composition | 4 | 5 | 1 |
| Testability and evidence clarity | 5 | 3 | 2 |
| Incremental migration | 4 | 2 | 5 |
| Operational visibility/backpressure | 4 | 5 | 2 |
| API simplicity | 3 | 4 | 5 |
| Backend feature reach | 4 | 5 | 2 |
| Maintenance cost | 4 | 1 | 5 |
| **Weighted result** | **93.0%** | **75.4%** | **44.4%** |

Select Option A. Use Option C only as disposable test scaffolding; do not publish it as the architecture. Reconsider Option B only after Option A has production evidence and a concrete repeated graph pattern justifies promotion.

## Architecture decisions

| ID | Decision | Consequence |
|---|---|---|
| AD-01 | Select Option A, the hybrid durable typed runtime. | Application graphs remain first-class; library contract stays narrow. |
| AD-02 | A library-owned outer workflow is the sole queue target and owns the terminal typed outcome. | Fixes CR-03 and puts binding/validation inside the durable boundary. |
| AD-03 | Canonical `InvocationEnvelope` plus digest governs idempotency. | Same key/different input becomes an explicit conflict. |
| AD-04 | A library batch is logical heterogeneous fan-out/fan-in over independent durable items. | Partial results/cancellation are item-scoped; engine batching is measured separately. |
| AD-05 | Neutral immutable adapter identity is rendered by each backend adapter. | vLLM/SGLang syntax does not leak into application state. |
| AD-06 | Enrichment/child work precedes a separate final constrained generation stage. | Avoids assuming universal tool-loop-plus-grammar support. |
| AD-07 | Context providers are read-only; mutating tools are effect stages with explicit durability/authority policy. | Separates untrusted information from authority and side effects. |
| AD-08 | Authority is a four-value mode algebra: enforce, automated, permit-all, bypass. | Empty policies and booleans no longer encode security semantics. |
| AD-09 | Decisions are typed evidence records bound to the canonical request and approver versions. | Malformed/stale/model-generated responses fail closed and remain attributable. |
| AD-10 | General external effects are at-least-once; stronger guarantees name their mechanism. | Removes the CR-06 overclaim and forces idempotency/outbox/compensation. |
| AD-11 | Capability is a versioned configuration/evidence record with Verified/Experimental/Unsupported state. | Marketing/static sets cannot masquerade as runtime proof. |
| AD-12 | Runtime configuration is immutable/context-local; startup registries are synchronized and versioned. | Eliminates cross-request policy and registry races. |

## Core domain model

The names are conceptual and do not prescribe exact Python syntax.

```text
InvocationEnvelope[TInput]
  invocation_id / idempotency_key / canonical_digest
  AgentIdentity
  BaseModelIdentity + AdapterIdentity|BaseModelSentinel
  ConstraintIdentity + expected output schema
  prompt/tool/context/policy/backend profile identities
  authority mode + caller/tenant/trust context
  input + budgets + deadlines + trace metadata

BatchItem[TInput, TOutput]
  envelope
  item_id + submission_index

InvocationOutcome[TOutput]
  status
  validated value | denial | typed failure
  all resolved identities and evidence references
  timing/usage/retry/replay metadata

ApprovalEvidence
  decision enum + reason/evidence
  canonical request digest
  approver identity/model/adapter/prompt/constraint versions
  policy revision + issued/expires + nonce
```

## Durable execution sequence

```text
submit envelope
  -> validate + resolve immutable identities
  -> canonicalize; reserve (idempotency key, digest)
       -> same key, same digest: attach
       -> same key, different digest: conflict
  -> evaluate/record authority decision
  -> checkpoint provider/child stages as declared
  -> dispatch backend request in retry-aware model step
  -> strict local parse/validate T
  -> checkpoint typed terminal outcome
  -> optional separately declared mutating effect stage
```

The effect stage is not smuggled into context enrichment or model generation. Its declaration names retry safety and recovery strategy.

## Migration boundary

The current `Agent` facade may remain as a compatibility layer only if it constructs the same envelope and calls the same outer workflow. It must not retain an alternate direct policy path, raw cast, mutable global allowlist, or separate undocumented queue semantics. Legacy calls should emit deprecation guidance when they cannot provide required identity or durability metadata.
