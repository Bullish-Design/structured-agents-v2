# Current State and Evidence

Date: 2026-07-21  
Evidence snapshot: [`evidence/baseline/2026-07-21-baseline.md`](evidence/baseline/2026-07-21-baseline.md)  
Primary sources: [`evidence/upstream/2026-07-21-source-register.md`](evidence/upstream/2026-07-21-source-register.md)

## Evidence vocabulary

| Label | Meaning |
|---|---|
| **Verified locally** | Directly reproduced or inspected in this checkout, installed environment, or active local runtime on the stated date. |
| **Verified upstream** | A primary upstream source documents the behavior; this is not proof for the local configuration. |
| **Inferred** | A conclusion follows from inspected evidence but the complete end-to-end behavior was not executed. |
| **Unknown** | Evidence is absent or insufficient. |
| **Contradicted** | Direct evidence refutes the claim for the stated configuration. |

All capability statements must include configuration and date. “The engine supports X” is too broad.

## Fixed repository baseline

- Branch/commit: `main` at `90725a56f28c6a5a09c0a93a31afcb15f3dfa504`.
- Environment: Python 3.13.13, structured-agents 0.3.0, DBOS 2.23.0, pydantic-ai-slim 2.11.0, Pydantic 2.13.3, HTTPX 0.28.1; XGrammar is not installed in the library environment.
- Pre-existing untracked project-12 and project-13 research directories were preserved.
- No source, test, deployment, dependency, lock, CI, or README file was changed by this investigation.

Fresh checks:

| Check | Result | Evidence |
|---|---|---|
| Unit/integration suite | 32 passed, 1 skipped | **Verified locally** |
| Ruff lint | pass | **Verified locally** |
| `ty` type check | pass | **Verified locally** |
| Ruff format check | fail; 8 files would change | **Verified locally** |
| Wheel build/content | pass; package-only | **Verified locally** |
| Sdist build/content | build passes; deploy/vendor contamination | **Verified locally** |

## Current library flow

```text
caller
  -> Agent.run / Agent.run_sync
       -> optional policy check in Plane
       -> DBOSAgent.run wrapper
       -> Schema.parse(result.output)

caller
  -> Agent.enqueue
       -> Queue.enqueue_async(DBOSAgent.run)   X unregistered workflow

Plane policy state
  -> module-level config registry / active allowlist   X shared process state
  -> Policy.evaluate
  -> ApprovalDecision.from_json-ish response           X permissive truthiness
```

The dependency's actual durable path is different: `DBOSAgent` registers an internal named workflow and its public `run` wrapper calls that workflow. Queuing the wrapper therefore fails DBOS registration validation. Model and MCP calls can become DBOS steps, but arbitrary custom tools and event handlers do not automatically become durable steps. These facts are **verified locally** by installed-source inspection and the focused probe.

## Review findings, revalidated

| ID | Severity | Current disposition | Evidence |
|---|---|---|---|
| CR-01 | Critical | Malformed `{"allowed":"false"}` is truthy and authorizes. | **Verified locally**, focused probe |
| CR-02 | Critical | Active allowlist is shared global state; synchronized threads both observed one caller's allowlist. | **Verified locally**, focused probe |
| CR-03 | High | Real `Agent.enqueue` raises `DBOSWorkflowFunctionNotFoundError`. | **Verified locally**, real-object probe |
| CR-04 | High | Empty `all_of()` returns allow. | **Verified locally**, focused probe |
| CR-05 | High | Approval is not strongly/canonically bound to exact subject, action, arguments, identities, and expiry. | **Verified locally** by code inspection; adversarial test pending |
| CR-06 | High | “Exactly once” language exceeds arbitrary external-effect guarantees. | **Verified upstream** and **verified locally** by dependency/source inspection |
| CR-07 | High | Advertised backend support is static rather than evidence/configuration driven. | **Verified locally** by code/docs inspection |
| CR-08 | High | Fresh sdist includes deploy/vendor projects and vendored tests. | **Verified locally** by fresh build |
| CR-09 | Medium | Invalid temperature string and negative token count are accepted. | **Verified locally**, focused probe |
| CR-10 | Medium | `Schema(Plan).parse(dict)` returns a raw dict, not `Plan`. | **Verified locally**, focused probe |
| CR-11 | Medium | Shared registries permit lifecycle and identity races. | **Verified locally** by code inspection; concurrent mutation test pending |
| CR-12 | Medium | Client/session ownership and shutdown behavior are underspecified. | **Verified locally** by code inspection |
| CR-13 | Medium | Timeouts, cancellation, resource bounds, and backpressure are incomplete. | **Verified locally** by code inspection |
| CR-14 | Medium | Private DBOS import and broad dependency ranges create compatibility risk. | **Verified locally** by package/source inspection |
| CR-15 | Medium | Tests, CI, docs, and format gates do not cover the public claims. | **Verified locally** by suite/config inspection |

The prior review and this investigation use the same commit. “Revalidated” means the earlier findings have not been invalidated by source changes.

## Runtime snapshot

No active service was changed.

| Profile | State on 2026-07-21 | What is proved | What is not proved |
|---|---|---|---|
| vLLM 0.25.0, XGrammar 0.2.3, custom GGUF plugin, Gemma-4-12B QAT GGUF | Active on loopback; one `base` model; no adapters | Health/model state **verified locally**. Preserved project-10 runs verify schema, regex, choice, and grammar for this exact profile after environment recovery. | LoRA, mixed adapters, continuous batching under this launcher, per-item mixed constraints |
| SGLang 0.5.14 historical GGUF profile | Inactive; historical startup failed before tensor load | Failure is **verified locally** from artifacts. | Base chat, XGrammar, LoRA, batching for an exact usable profile |
| llama.cpp current local profile | Active, GBNF-oriented, one slot | Service/state **verified locally**. | XGrammar or heterogeneous LoRA batching |

Current vLLM documentation verifies upstream support for structured outputs and concurrent distinct adapters when configured with a sufficient `max_loras`. Current SGLang documentation verifies upstream XGrammar and same-batch multi-LoRA support. Those are qualification hypotheses, not local support claims.

## Durability boundary

The durable evidence supports this narrower model:

```text
workflow input + workflow ID
       |
       +-- completed workflow/step result -> replayed from DBOS
       +-- DB transaction             -> exactly-once commit semantics
       +-- arbitrary external effect  -> at-least-once attempt boundary
                                           needs idempotency/outbox/compensation
```

A preserved crash artifact proves that a completed model/effect stage survived worker restart without a second append. It does not test a crash after an external service committed but before DBOS recorded step completion. Therefore a general exactly-once claim is **contradicted**; the narrower recovery claim is **verified locally**.

DBOS also returns the existing execution when a workflow ID is reused. The runtime must store a canonical input digest and reject same-key/different-input reuse; DBOS does not infer that conflict for the library.

## Structural gaps

The present public surface lacks:

- an independently addressable heterogeneous batch item/result model;
- immutable base-model/adapter/constraint/prompt identities;
- a configuration-specific backend capability registry;
- bounded typed context-provider and child-agent contracts;
- explicit `enforce`, `automated`, `permit_all`, and `bypass` semantics;
- denial/abstention/error decision evidence and request binding;
- durable effect declarations and recovery policy;
- observable continuous-batching evidence;
- package-content and live-backend release gates.

These are architecture requirements, not requests to add isolated helper methods to the current facade.

## Evidence limitations

- No new generation request was sent to an active model service.
- No model, adapter, or dependency was downloaded.
- No live SGLang service existed to test.
- No active service configuration was mutated.
- Process command lines were not captured because doing so could expose credentials.
- SGLang upstream documentation is current rather than pinned to a proven local version.
- Multi-LoRA throughput, adapter eviction, mixed constraints, and crash-inside-effect behavior remain unknown until the planned spikes run.
