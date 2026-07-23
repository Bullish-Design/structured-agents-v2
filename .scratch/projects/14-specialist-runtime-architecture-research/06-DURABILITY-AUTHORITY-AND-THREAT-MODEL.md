# Durability, Authority, and Threat Model

Date: 2026-07-21

## Security objective

A caller may request specialist inference, context enrichment, or a mutating effect. The runtime must bind every decision and outcome to the exact validated request and immutable identities, recover without silently changing those identities, and never turn malformed, absent, stale, or untrusted model output into authority.

## Trust boundaries

```text
trusted control plane
  identities, registries, backend profiles, policy, adapter registration,
  approver configuration, code/schema versions
                         |
                         v
authenticated caller -> validation/canonical binding -> durable workflow
       |                         |                         |
       |                         |                         +-> model backend (semi-trusted)
       |                         |                         +-> context sources (mixed trust)
       |                         |                         +-> effectors (privileged)
       |                         |
       +-> request content ------+  untrusted until validated

model output, retrieved text, tool text, histories, and script output
are data; none may mutate the control plane or confer authority.
```

Trust classes should minimally distinguish control-plane trusted, authenticated tenant input, allowlisted external source, untrusted content, sensitive content, model-derived content, and privileged effector. Trust and sensitivity are separate axes.

## Durable state machine

```text
RECEIVED
  -> REJECTED_VALIDATION
  -> IDENTITY_RESOLVED
       -> CONFLICTED_IDEMPOTENCY
       -> BOUND
            -> DENIED | REQUIRES_HUMAN | APPROVED | POLICY_BYPASSED
                 -> CONTEXT_RUNNING
                      -> MODEL_RUNNING
                           -> OUTPUT_VALIDATING
                                -> SUCCEEDED
                                -> FAILED_VALIDATION
                      -> FAILED | TIMED_OUT | CANCELLED
                 -> EFFECT_PENDING
                      -> EFFECT_SUCCEEDED
                      -> EFFECT_FAILED | COMPENSATION_PENDING
```

Every transition records workflow/invocation/item ID, canonical digest, code/schema version, resolved identities, actor, previous/new state, time, attempt, and sanitized evidence reference. Terminal status does not erase intermediate evidence.

## Replay and idempotency contract

1. Validate the envelope and resolve all logical identities through the versioned control-plane snapshot.
2. Canonicalize the security-relevant request and calculate a versioned digest.
3. Reserve `(tenant/scope, idempotency_key, digest)` inside the durable workflow boundary.
4. If key and digest match an existing execution, attach and return its handle/outcome.
5. If the key matches but digest differs, return `IdempotencyConflict`; never rely on DBOS's existing-workflow-ID no-op as equivalence proof.
6. Persist decisions and typed stage results. Replay consumes the recorded versions/results, not whatever a mutable registry contains now.
7. Reject replay when a required identity/artifact is missing or incompatible unless an explicit, audited migration exists.

Canonical input includes caller/tenant scope, subject/action/normalized arguments, mode, agent/model/adapter/constraint, prompt/tool/context/policy profiles, deadlines/budgets relevant to semantics, and input. Trace labels that do not affect behavior may be excluded only by a documented canonicalization version.

## Crash semantics

| Crash point | Required recovery behavior |
|---|---|
| Before binding | Revalidate or deterministically resume; no model/effect dispatch occurred. |
| After binding, before decision | Resume with the bound identity/policy snapshot. |
| After decision checkpoint | Reuse the recorded decision if still valid for the exact digest; never ask a changed approver silently. |
| During model/provider call | Retry only under declared step policy; request IDs and attempts remain visible. Duplicate inference may occur. |
| After model response, before checkpoint | Model call may repeat; no external exactly-once claim is made. |
| During/after local validation | Validation is deterministic for the recorded validator version; invalid output is never success. |
| Before a mutating effect | Resume the declared effect protocol. |
| After remote effect commit, before local checkpoint | Duplicate attempt is possible; remote idempotency, transactional outbox, reconciliation, or compensation must make this safe. |
| After terminal outcome | Handle returns the same recorded outcome and identities. |

DBOS transactions may provide exactly-once database commit semantics. DBOS steps around arbitrary services do not turn those services into exactly-once effectors. The public documentation must name the narrower mechanism whenever it uses “exactly once.”

## Authority mode algebra

| Mode | Policy evaluation | Decision record | Validation/binding | Audit | Appropriate use |
|---|---|---|---|---|---|
| `enforce` | Deterministic configured policy/approver chain | Required | Always | Always | Default production mutating path |
| `automated` | Constrained automated approver composition | Required, including model evidence | Always | Always | Low/medium-risk actions under explicit policy |
| `permit_all` | Explicit policy whose result is allow | Required: `allow` | Always | Always | Trusted bounded environment; still a policy decision |
| `bypass` | Intentionally skipped | Required: `policy_bypassed` with actor/scope/reason/expiry | Always | Always plus alertable event | Emergency/operator path, disabled for untrusted callers |

An empty policy collection is invalid. No mode uses `None`, empty list, string truthiness, or a generic boolean as its security representation.

“Bypass” does not mean raw arbitrary execution. It skips normal policy evaluation only. It still resolves registered identities, validates schemas and destinations, binds the request, records the outcome, enforces caller scope and expiry, and applies effect safety. If the product needs a lower-level raw effector, it must be separately named, separately authorized, impossible to call from model output, and documented as carrying reduced guarantees.

## Approval evidence

```text
ApprovalEvidence
  decision: allow | deny | abstain | requires_human | error
  request_digest
  subject, action, normalized argument digest
  agent/base/adapter/constraint identities
  policy identity + revision
  approver identity
  approver model/adapter/prompt/constraint identities (when agentic)
  reason codes + sanitized evidence references
  issued_at, expires_at, nonce/correlation_id
  parent decisions/quorum result
```

Default composition is fail closed: explicit deny has precedence; error and abstain do not become allow; unsatisfied quorum is non-allow; `requires_human` is terminal or waits in a separately bounded human workflow. The exact composition policy is versioned because changing it changes security semantics.

Approval is consumed only for the bound digest and before expiry. Material request, identity, destination, policy, or relevant context changes require a new decision. Effectors verify the binding again at the last responsible moment to narrow time-of-check/time-of-use risk.

## Automated approver constraints

An automated approver is a constrained specialist, not a source of ambient authority.

- Its model, adapter, prompt, tools, output constraint, and policy identity are immutable/versioned.
- Its output is strict locally validated data; parsing failures become `error`.
- It receives the exact normalized proposal and carefully delimited evidence, with untrusted text labeled as such.
- It cannot modify its own prompt/model/policy, register tools/adapters, choose bypass, or approve control-plane changes affecting itself.
- High-risk actions require deterministic guards and/or an independent quorum/human path; a single model judgment is not universal authorization.
- Its reason/evidence is retained with redaction and retention controls.

## Context, scripts, tools, and child agents

Context providers are read-only and return typed evidence. Scripts/functions run only from a control-plane allowlist with declared sandbox, arguments, outputs, network/filesystem policy, timeout, resource budget, and version digest. Their output receives a trust label and cannot supply executable policy fields.

Mutating tools are effectors. Each declares:

- authenticated destination and action schema;
- required authority level/mode;
- idempotency protocol and key placement;
- retry-safe vs unsafe failure classes;
- transaction/outbox/reconciliation/compensation strategy;
- timeout, rate, and concurrency limits;
- audit/redaction requirements.

Child agents receive a least-privilege capability set, bounded usage, typed input/output, and parent correlation. Delegation never carries more authority than the parent has explicitly granted.

## Threat register

| ID | Threat / abuse case | Primary controls | Planned proof |
|---|---|---|---|
| TH-01 | Malformed approval (`"false"`, missing field, unknown enum) becomes allow | APR-01, CFG-01, fail-closed composition | Unit/property/adversarial decoder tests |
| TH-02 | Concurrent request observes another request's allowlist/policy | CFG-02, AD-12, immutable envelope | Barrier and high-contention tests |
| TH-03 | Approval replayed for changed args/adapter/destination | DUR-02, APR-02/04 | One-field mutation matrix, expiry/replay test |
| TH-04 | Prompt injection selects bypass or authorizes itself | SEC-01/02, AUTH-02/04 | Injection corpus across context/history/tool output |
| TH-05 | Agentic approver is manipulated or returns invalid JSON | APR-01/03, XGR-02 | constrained live and malformed transport tests |
| TH-06 | Adapter path traversal, malicious adapter, wrong base model | LORA-04, SEC-03 | digest/base/traversal/registration negative tests |
| TH-07 | Adapter or constraint cache cross-contaminates batch items | immutable keys, XGR-05/06 | mixed-item load plus per-item validation |
| TH-08 | Same workflow ID silently returns different request's result | DUR-02 | same-key/different-digest integration test |
| TH-09 | External effect duplicates after crash | DUR-04/05, effector protocol | crash-after-remote-commit fault injection |
| TH-10 | Pickled/untrusted durable state executes or leaks data | SEC-04, trusted serializer boundary | malicious payload, secret canary, retention tests |
| TH-11 | Unbounded child/tool loop exhausts spend/resources | AGC-01, OPS-03 | cyclic graph and saturation tests |
| TH-12 | Cancellation leaves orphan request/effect | BAT-05, OPS-03, DUR-05 | cancel at each state, reconcile backend/workflow |
| TH-13 | Capability claim is generalized across incompatible profile | CFG-03, TEST-02/03 | profile mismatch/staleness gates |
| TH-14 | Dynamic LoRA operation exposes server to untrusted control | LORA-04, SEC-03 | data-plane rejection/control-plane auth test |
| TH-15 | Logs/artifacts expose prompts, credentials, paths, sensitive context | OPS-02, SEC-04 | seeded secret and artifact scanning |
| TH-16 | Sdist/vendor content executes or ships unintended files | PKG-01 | clean artifact allowlist/SBOM scan |
| TH-17 | Registry mutation during replay changes resolved implementation | DUR-03, CFG-02 | freeze/version/restart/mismatch test |
| TH-18 | Automated approval and execution collude through shared agent/config | APR-03, separation of duties | self-change and same-identity policy tests |

## Residual risks

Even after the controls pass, model nondeterminism, backend scheduler changes, hardware faults, provider-side duplication, and malicious-but-valid adapters remain. The system reduces these through versioned evidence, isolation, least privilege, reconciliation, and conservative capability statuses; it cannot remove them. High-impact irreversible actions should retain a human or transactional boundary independent of model output.
