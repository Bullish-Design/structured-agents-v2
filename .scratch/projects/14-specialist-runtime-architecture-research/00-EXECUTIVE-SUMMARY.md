# Executive Summary

Date: 2026-07-21  
Baseline: `main` at `90725a56f28c6a5a09c0a93a31afcb15f3dfa504`  
Scope: research and implementation planning only

## Decision

Adopt a **hybrid durable specialist runtime**: applications author the durable workflow graph, while this library provides one narrow, typed, queueable specialist-invocation primitive plus explicit context, adapter, constraint, authorization, and evidence contracts.

Do **not** begin feature expansion yet. The current release is no-go until the first safety gate closes:

- malformed approval responses can authorize actions (CR-01);
- policy allowlists are process-global and can cross-contaminate calls (CR-02);
- the real agent queue path targets an unregistered callable (CR-03);
- an empty `all_of()` policy authorizes (CR-04);
- approvals are not durably and strongly bound to the exact request (CR-05/06);
- source distributions include deployment/vendor material (CR-08).

All six behaviors are either freshly reproduced or artifact-inspected on the fixed baseline. Passing unit tests do not negate them.

## Recommended shape

```text
Application workflow
    |
    +-- context providers / bounded child-agent stages
    |      (typed, budgeted, provenance-preserving)
    |
    +-- authority decision stage
    |      (enforce | automated | permit_all | bypass)
    |
    +-- durable specialist invocation(s)
    |      InvocationEnvelope + immutable identities
    |      one durable handle/outcome per logical item
    |              |
    |              +--> vLLM adapter (first qualification target)
    |              +--> SGLang adapter (required, exact profile unverified)
    |
    +-- distinct final constrained generation stage
           (strict local validation before typed result commits)
```

The library batch API is heterogeneous at the logical level. Each item may select a different agent, adapter, constraint, and idempotency key. Queue concurrency creates overlapping requests; vLLM or SGLang must prove that it continuously batches compatible sequences. The API must never claim that submitting a Python list itself produces engine batching.

## Non-negotiable contracts

1. Every agent, base model, adapter, constraint, prompt/tool bundle, approver, and context provider has immutable or versioned identity.
2. The durable workflow owns the validated typed result. The existing raw `DBOSAgent.run` queue target is not the public contract.
3. Reusing an idempotency key with different canonical input is a conflict, not silent attachment to an earlier result.
4. Arbitrary external effects are not advertised as exactly once. They require an idempotency key, transaction/outbox, or compensation plan.
5. `permit_all` and `bypass` are different. Bypass may skip policy evaluation, but it does not silently skip validation, binding, recording, or audit.
6. Automated approvers return a strict decision enum and evidence chain. Missing, malformed, stale, abstaining, or failed decisions never become allow.
7. Context/tool/child-agent loops are bounded by calls, tokens, time, depth, and cost. Untrusted context is data, never authority.
8. Backend capability claims are configuration-specific and reported only as **Verified**, **Experimental**, or **Unsupported**.

## Backend conclusion

| Target | Current evidence | Planning status |
|---|---|---|
| vLLM 0.25.0 + active GGUF plugin | Historical live XGrammar success; current health/model verified; launcher excludes LoRA | First qualification target; structured output verified for exact profile, LoRA unsupported in current profile |
| vLLM 0.25.0 upstream native profile | XGrammar and multi-LoRA documented upstream | Experimental until exact local model/adapter/concurrency evidence exists |
| SGLang current upstream | XGrammar and mixed multi-LoRA documented upstream | Experimental; version and exact model profile must be pinned and proved |
| SGLang 0.5.14 historical GGUF profile | Failed before tensor loading | Contradicted as a usable current profile |
| llama.cpp | GBNF and LoRA documented; different LoRA configurations are not batched together | Comparison/fallback only; unsupported for the required combined contract |

## Roadmap gates

- **Gate 0 — safe foundation:** close CR-01/02/03/04/05/06/08 and add strict configuration, formatting, and package gates.
- **Gate 1 — durable typed core:** canonical envelope, stable registries, registered outer workflow, typed validation, effect semantics.
- **Gate 2 — authority and logical batching:** four modes, evidence decisions, heterogeneous item/handle/result API, cancellation/backpressure.
- **Gate 3 — backend qualification:** vLLM first, then SGLang, with mixed adapter/constraint load and metrics evidence.
- **Gate 4 — bounded context and agency:** providers, scripts/functions, child agents, and separate final constrained call.
- **Gate 5 — release evidence:** concurrency, crash/replay, injection, packaging, documentation, and support-matrix gates.

No phase advances on documentation alone. The exit conditions are executable evidence artifacts defined in `07-SPIKES-AND-TEST-PLAN.md`.

## Immediate next action

Approve the architecture and Gate 0 scope. Then hand `10-IMPLEMENTATION-KICKOFF-PROMPT.md` to an implementation agent. The implementation agent should not start backend or agentic features until Gate 0 is green.
