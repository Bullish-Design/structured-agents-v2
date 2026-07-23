# Decisions and Open Questions

Date: 2026-07-21

## Decisions made

These decisions implement the owner's stated direction; they should not be reopened during implementation without new contradictory evidence.

| Decision | Resolution and rationale |
|---|---|
| Architecture | Select hybrid durable typed runtime (AD-01). It composes DBOS/PydanticAI while keeping the library small and code-first. |
| Batch unit | A batch contains independent heterogeneous durable items with per-item handles/outcomes (AD-04), not one opaque provider batch job. |
| GPU batching claim | Concurrent client dispatch and engine continuous batching are distinct; only engine metrics qualify the latter. |
| Queue durability | Queue only a registered library-owned outer workflow that owns policy binding and validated `T` (AD-02). |
| Idempotency | Canonical digest is paired with the idempotency key; different input is a conflict (AD-03). |
| LoRA identity | Agents bind neutral immutable adapter identities; backend syntax is renderer output (AD-05). |
| Runtime adapter loading | Registration/loading is a trusted control-plane operation, never an arbitrary request path. |
| XGrammar | vLLM and SGLang are independently configuration-qualified; upstream documentation alone means Experimental. |
| Context/agency | Read-only typed enrichment and bounded child stages precede a separate final constrained call (AD-06/07). |
| Authority | Four explicit modes: enforce, automated, permit-all, bypass (AD-08). Empty policy is invalid. |
| Bypass | Skips policy evaluation only; validation, binding, scope/expiry, durable result, and audit remain. A raw effector, if needed, is separately named. |
| Approval | Strict decision enum and evidence chain bound to exact request and approver versions (AD-09). |
| Effects | Arbitrary external effects are at-least-once unless a named stronger mechanism applies (AD-10). |
| Capability | Status is Verified, Experimental, or Unsupported for an exact evidence-backed profile (AD-11). |
| Configuration | Request configuration is immutable/context-local; startup registries are synchronized/versioned/frozen (AD-12). |
| Backend priority | Qualify vLLM first, then SGLang. llama.cpp remains a boundary/comparison profile. |

## Owner decisions covered

| # | Owner direction | Resolution |
|---:|---|---|
| 1 | Batched specialist is central | BAT-01–06, AD-04, Phase 3 |
| 2 | Per-agent LoRA required | LORA-01/02/05, AD-05 |
| 3 | Different agents/LoRAs concurrent where supported | LORA-03, BAT-04, SP-05–07 |
| 4 | vLLM/SGLang priority | Phase 4 and capability matrix |
| 5 | XGrammar verified on both, not inferred | XGR-03/04 and SP-03/04 |
| 6 | Automatic enrichment/bounded chaining before final | CTX/AGC requirements, AD-06/07 |
| 7 | Functions/scripts context or automated checks | CTX-02 and SP-09/11 |
| 8 | Four mutating authority modes | AUTH-01–04 and AD-08 |
| 9 | Automated approvers may be constrained agents | APR-03, SP-11 |
| 10 | Local-first small specialists | backend order/profile selection; no mandatory remote provider |
| 11 | Evidence explicitly verified/experimental/unsupported | CFG-03, AD-11, matrix and manifests |

## Remaining owner decisions

These choices do not alter the settled architecture, but implementation needs an owner selection at the named gate.

| ID | Decision needed | Recommended default | Needed by | Evidence/input |
|---|---|---|---|---|
| OQ-01 | Public batch return shape: eager aggregate, async iterator, collection of handles, or all three | Primary collection of typed handles plus explicit `gather` and completion iterator | Phase 3 API review | SP-02 ergonomics/failure evidence |
| OQ-02 | Control-plane source for agent/adapter/profile registries | Code-authored immutable registry at startup; pluggable signed persistent registry later | Phase 1/3 | deployment ownership and SEC-03 review |
| OQ-03 | Durable serializer/interchange policy | Treat DBOS pickle as trusted internal storage only; define versioned JSON-safe envelopes/evidence | Phase 1 | compatibility and payload-size tests |
| OQ-04 | Default authority mode for mutating calls and whether bypass ships enabled | `enforce`; bypass disabled unless explicit operator configuration | Phase 2 | SP-12 and risk review |
| OQ-05 | Automated approval quorum/escalation by risk class | deterministic deny precedence; agent approval only for enumerated lower-risk actions; human for high risk | Phase 2 | SP-11, threat review |
| OQ-06 | Whether a separately named raw-effector API is truly required | Do not ship until a concrete trusted use case exists | Phase 2 | product use case and security sign-off |
| OQ-07 | Default context/child budgets and truncation strategy | Explicit per-agent budgets; conservative global ceilings; deterministic provenance-aware truncation | Phase 5 | SP-09 quality/cost results |
| OQ-08 | First native model and two LoRA artifacts for cross-engine comparison | Smallest license-compatible model supported by both exact pinned engines and fitting hardware with headroom | Before SP-03–07 | model/license/hardware review; downloads need approval |
| OQ-09 | Pinned SGLang version/profile | Current stable version that passes native base-model smoke test; do not reuse failed 0.5.14 GGUF assumption | Before SP-04 | compatibility preflight |
| OQ-10 | Minimum supported DBOS/PydanticAI/Python matrix | Narrow around currently proved versions, widen only through PT-INSTALL | Phase 1/release | PKG-02 matrix cost |
| OQ-11 | Capability evidence expiry | 30 days for active development, mandatory rerun on any material dimension change | Phase 4 | CI/GPU availability |
| OQ-12 | Whether llama.cpp stays a documented fallback | Keep documented only for its exact verified GBNF/GGUF scope, outside combined contract | Documentation review | owner product positioning |
| OQ-13 | Retention/redaction policy for prompts, approval reasons, and context evidence | Digests/metadata by default; raw content opt-in, encrypted, short retention | Before production evidence | privacy/compliance requirements |

## Unknowns assigned to spikes

| Unknown | Assigned evidence |
|---|---|
| Exact outer-workflow shape compatible with real DBOSAgent | SP-01 |
| Best partial-result/cancellation API | SP-02, OQ-01 |
| vLLM native XGrammar mapping at pinned version | SP-03 |
| Viable exact SGLang version/model and constraint mapping | SP-04, OQ-08/09 |
| vLLM/SGLang actual different-LoRA overlap and capacity | SP-05/06 |
| Adapter/constraint cache correctness under mixed load | SP-07 |
| PydanticAI custom tool/toolset durable boundary | SP-08 |
| Context/child composition quality, budgets, and injection resistance | SP-09, OQ-07 |
| Recovery in ambiguous external-effect window | SP-10 |
| Heterogeneous approval truth table and durable human wait | SP-11, OQ-05 |
| Scoped bypass usability and binding | SP-12, OQ-04/06 |
| Context-local/thread/task registry correctness | SP-13 |
| Clean/reproducible artifacts and supported install matrix | SP-14, OQ-10 |

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Keep global active allowlist with locks | A lock can serialize mutation but does not give nested/task-local request semantics and remains ambient authority. |
| Treat `[]`, `None`, or truthy values as authorization modes | Ambiguous and already produces fail-open behavior. |
| Queue `DBOSAgent.run` directly | It is not the registered workflow in the installed integration and does not define the library's typed terminal contract. |
| Return `cast(T, raw_output)` | Static appearance does not perform runtime validation and CR-10 reproduces the mismatch. |
| Call concurrent HTTP requests “continuous batching” | It proves only client concurrency, not engine scheduling. |
| Store adapter filesystem paths in durable invocations | Paths are deployment-specific, mutable, and unsafe at the data-plane boundary. |
| Dynamic LoRA load/unload available to ordinary callers | Upstream itself warns of security risk; it expands trusted control-plane mutation into request handling. |
| One universal `supports_xgrammar` or `supports_lora` boolean | Capability depends on exact engine/version/model/format/plugin/settings and feature combination. |
| One call that freely mixes tool loop and final grammar on every backend | Compatibility is unproved; the two-stage final call is portable and testable. |
| Blanket approval as bypass | It erases the important difference between a policy decision and intentionally skipped evaluation. |
| Automated approver output directly triggers an effector | Model output remains untrusted data and must cross validation, binding, deterministic policy, and durable audit boundaries. |
| General “exactly once” external effects | Contradicted by the crash-after-remote-commit-before-checkpoint window. |
| Full library-owned agent graph now | Too much trusted/replay surface before the narrow primitive and repeated application pattern are proven. |
| Require native provider offline batch APIs | Not needed for durable logical batches or continuous server scheduling and reduces portability. |

## Decision-change rule

Changing AD-02/03/08/09/10 or the trust boundaries is a security/durability architecture change, not an implementation detail. It requires an updated threat model, traceability, migration analysis, and new failing-before/passing-after evidence. Backend/profile decisions may change through capability records without changing the neutral domain contract.
