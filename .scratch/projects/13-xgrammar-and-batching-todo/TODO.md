# Repository ToDo — XGrammar and Concurrent Inference

**Created:** 2026-07-21  
**Repository version:** `structured-agents` 0.3.0 at `90725a5`  
**Related review:** [`../12-structured-agents-v2-library-study/CODE_REVIEW.md`](../12-structured-agents-v2-library-study/CODE_REVIEW.md)

## Goal

Establish with reproducible runtime evidence which constrained-output, per-agent LoRA, and concurrent-processing capabilities work with vLLM and SGLang, then make the library advertise only capabilities that have passed those tests.

## Architectural requirement

Batched specialist inference is a core product requirement, not an optional optimization. The target system keeps one capable small base model resident, applies a task-specialized LoRA per agent, constrains each agent to its narrow output/tool contract, and continuously batches independent agent requests on the GPU.

The implementation must preserve these invariants:

- One logical agent has an explicit base-model identity, adapter identity, constraint, instructions, and settings.
- Different agents may select different LoRAs on the same resident base model.
- Requests using the base model and several LoRAs can be active together and share server batches where the engine supports it.
- Every request retains its own adapter and constraint; neither may contaminate another sequence in a mixed batch.
- Durable queueing provides backpressure, identity, replay, and result isolation without disabling inference-server continuous batching.
- Agents can durably enrich their context through deterministic providers, authorized scripts/tools, and bounded calls to other specialized agents before producing a final constrained output.
- Trusted applications can explicitly select enforced authorization, automated approval, blanket approval, or scoped bypass while retaining durable command binding and audit evidence.
- The library measures throughput and latency under realistic mixed-agent workloads, not only homogeneous synthetic prompts.
- Engine adapter limits, GPU memory, rank, target modules, and scheduling behavior are explicit deployment configuration.

## Current baseline

- vLLM is the library's reference wire dialect.
- SGLang constraint rendering exists but is explicitly marked unverified.
- The library sends constraint declarations to the inference server; it does not run constrained decoding itself.
- Installing the `grammar-check` extra only enables local XGrammar compilation checks.
- `Queue.submit_batch()` intends to enqueue independent concurrent workflows; it is not a provider batch endpoint.
- The current queue implementation cannot submit a real library `Agent`, so it cannot yet prove production batching behavior.
- The repository's isolated SGLang deployment is configured with one request slot, which serializes inference in that specific deployment.
- `AgentSpec.adapter` selects a vLLM adapter by using its registered adapter name as the OpenAI `model` value.
- SGLang's OpenAI-compatible interface requires `base-model:adapter-name`; the library currently forwards `adapter` unchanged and does not construct that engine-specific identity.
- Both built-in engine capability sets advertise LoRA, but the repository's active native vLLM/GGUF profile has no LoRAs and SGLang LoRA remains unverified.
- The public `Agent.run()` currently accepts only a prompt. It exposes no hooks, context providers, tools, dependencies, message history, or nested-agent interface.
- PydanticAI's underlying DBOS adapter supports registered durable toolsets and nested DBOS workflows, but `structured-agents` does not expose or configure that functionality.
- Authorization can currently be bypassed informally by calling `Effector.run()` directly, and blanket approval can be implemented with an authorizer that always returns `Decision(True)`; neither is represented as a first-class audited policy mode.
- `Authorizer.decide()` is synchronous, so it cannot naturally perform durable scripts or nested LLM/agent approval calls.

## P0 — Repair the concurrency path used for verification

- [ ] Fix `Queue.submit()` so it enqueues a registered workflow from a real `structured_agents.Agent`.
- [ ] Ensure `WorkflowHandleAsync.get_result()` returns the declared output type `T`, not PydanticAI's `AgentRunResult[T]` hidden behind a cast.
- [ ] Ensure queued regex and choice outputs still pass through `Constraint.parse()`.
- [ ] Replace the structurally inaccurate `QueueAgent` test double with a real `Backend` and mocked OpenAI-compatible transport.
- [ ] Add a real-agent `submit_batch()` regression test covering success, one isolated failure, concurrency limits, rate limits, and stable workflow keys.
- [ ] Add a fresh-process recovery test for queued agent work.

### P0 acceptance criteria

- A real agent can be submitted through `Queue` before and after a worker restart.
- The queue handle returns `T` at runtime and under static type checking.
- Reusing a key does not issue another model request.
- One failed batch member does not lose the handles or results of successful siblings.

## P0 — Make batched specialist dispatch a first-class API contract

- [ ] Write the public batch contract before implementation: independent inputs, independent workflow IDs, independent typed outcomes, explicit partial failure, bounded concurrency, and stable ordering.
- [ ] Decide whether `submit_batch()` should return handles immediately for every accepted item or return a structured submission result containing per-item acceptance failures.
- [ ] Ensure queue concurrency limits control application backpressure without unnecessarily limiting the number of requests the inference server can continuously batch.
- [ ] Support heterogeneous batches containing different agents, adapters, constraints, prompts, and output types; do not restrict the central concept to one agent over many prompts.
- [ ] Add a specialist-dispatch primitive that can submit `[(agent, prompt, key), ...]` while preserving each result's type/identity as far as Python's heterogeneous typing permits.
- [ ] Define fairness and priority behavior so one high-volume specialist cannot starve other agents.
- [ ] Add cancellation behavior for one item and for an entire logical batch.
- [ ] Add observability that relates logical batch ID, workflow ID, agent name, adapter identity, engine request ID, and timing without logging sensitive prompts by default.
- [ ] Add mixed-agent load tests using realistic coding roles such as planning, code editing, test generation, review, and repository search.

### Batched specialist acceptance criteria

- A single call can dispatch work to several specialized agents and return isolated handles/outcomes.
- Different agents in the same logical batch may use different LoRAs and different constraints.
- Partial failure and cancellation never discard successful sibling handles.
- Server telemetry proves concurrent requests reach continuous batching rather than being serialized by the client or queue.
- Mixed-specialist throughput materially exceeds serial execution on the target hardware without unacceptable tail latency or output degradation.

## P1 — Verify SGLang XGrammar constrained outputs

- [ ] Select and record the exact SGLang version, XGrammar version, model, tokenizer, launch command, GPU, and request endpoint.
- [ ] Launch SGLang with its XGrammar grammar backend explicitly enabled; do not rely on an implicit default.
- [ ] Confirm the selected model can complete an unconstrained control request before testing constraints.
- [ ] Capture the exact HTTP request and response for `Schema`, `Regex`, `Choice`, and `Grammar`.
- [ ] Verify the library's SGLang wire fields against the live server:
  - [ ] JSON Schema through `response_format`/PydanticAI `NativeOutput`.
  - [ ] Regex through `extra_body.regex`.
  - [ ] Choice lowered to an escaped regex alternation.
  - [ ] Grammar through `extra_body.ebnf`.
- [ ] Use adversarial prompts that ask for invalid output and prove decoding still cannot leave the constraint.
- [ ] Test choice values containing regex metacharacters, whitespace, Unicode, quotes, backslashes, prefixes, and an empty string if empty choices remain supported.
- [ ] Determine whether SGLang regex matching is full-match or substring/search based; add anchors if required for true `Choice` semantics.
- [ ] Test malformed and unsupported grammars and require clear build-time or server errors.
- [ ] Verify schema strictness and additional-property behavior.
- [ ] Run the same tests through the public `Backend(engine="sglang")`, not only raw `curl` requests.
- [ ] Add opt-in live pytest markers that select SGLang explicitly and cannot accidentally run against vLLM.
- [ ] Store timestamped requests, responses, server logs, versions, and commands under `artifacts/`.
- [ ] Update the SGLang engine capability set and documentation based only on passing evidence.

### SGLang acceptance criteria

- Every advertised constraint kind passes both a raw HTTP contract test and a public-library end-to-end test.
- Invalid output is prevented by server-side decoding rather than rejected only after generation.
- Tests identify the exact supported SGLang/XGrammar version range.
- Unsupported constraint kinds fail during `Backend.build()` with `BackendCapabilityError`.

## P1 — Investigate and verify XGrammar with vLLM

- [ ] Record the exact vLLM, XGrammar, model, tokenizer, quantization/plugin, and launch versions in the active deployment.
- [ ] Confirm which vLLM launch option selects XGrammar for the installed version, including whether it is `--structured-outputs-config.backend xgrammar` or a version-specific predecessor.
- [ ] Verify that XGrammar is actually loaded by the server rather than silently falling back to another structured-output backend.
- [ ] Establish an unconstrained control request before constrained testing.
- [ ] Capture raw `Schema`, `Regex`, `Choice`, and `Grammar` requests using the library's current `structured_outputs` wire format.
- [ ] Repeat the same requests through `Backend(engine="vllm")` and a real durable `Agent`.
- [ ] Use adversarial prompts and assert that invalid outputs cannot be generated.
- [ ] Test schema recursion, enums, unions, optional fields, numeric bounds, and strict additional-property handling within XGrammar's supported subset.
- [ ] Test regex anchors, Unicode, escapes, long alternations, and pathological patterns.
- [ ] Test EBNF compilation failures and runtime grammar errors separately.
- [ ] Compare behavior with and without the local `grammar-check` extra installed.
- [ ] Confirm that local `Constraint.check()` and server-side XGrammar accept the same schema/grammar corpus.
- [ ] Capture server error details for the previously observed constrained-output HTTP 500 path and determine whether the cause is vLLM, XGrammar, the model, quantization, or the GGUF plugin.
- [ ] Add a minimal known-good model/config control to separate model/plugin incompatibility from general vLLM XGrammar support.
- [ ] Add version-pinned opt-in vLLM/XGrammar live tests and preserve timestamped evidence.
- [ ] Document the verified support matrix and any model- or quantization-specific exclusions.

### vLLM acceptance criteria

- All four advertised constraints pass raw and public-library tests against a pinned deployment.
- Server configuration proves XGrammar, rather than another backend, enforced the constraints.
- The known structured-output HTTP 500 is either fixed with a demonstrated cause or recorded as an explicit unsupported configuration.
- The supported matrix distinguishes core vLLM behavior from GGUF/plugin-specific behavior.

## P1 — Verify per-agent LoRA selection and multi-LoRA batching

- [ ] Replace the ambiguous adapter string contract with an engine-neutral adapter identity, or precisely document the required value for each engine.
- [ ] Decide whether an adapter specification contains `name`, `base_model`, optional version/hash, rank, and expected target modules.
- [ ] Keep adapter selection separate from arbitrary `Settings.extra_body` so it remains observable, capability-gated, and testable.
- [ ] Validate that every agent's adapter belongs to the configured base model before accepting work.
- [ ] Record the adapter identity and immutable weight/version digest in durable workflow metadata so replay cannot silently use different weights under the same name.
- [ ] Reject a reused workflow key if the agent, base model, adapter, constraint, or prompt identity conflicts with the recorded invocation.
- [ ] Test base-only, one-adapter, same-adapter concurrent, and heterogeneous multi-adapter workloads.
- [ ] Combine LoRA selection with every supported constraint kind and prove both are enforced in the same request.
- [ ] Use distinguishable test adapters or deterministic fixtures to detect adapter cross-contamination between batched sequences.
- [ ] Measure adapter-switching overhead, GPU memory, throughput, TTFT, inter-token latency, and tail latency as adapter cardinality increases.
- [ ] Determine the optimal LoRA rank and number of resident adapters for the target small model and GPU.
- [ ] Test adapter hot loading/unloading separately from the steady-state pinned/resident-adapter path.
- [ ] Treat runtime adapter loading endpoints as privileged deployment operations; do not expose arbitrary filesystem or remote adapter paths through ordinary agent requests.

### vLLM multi-LoRA work

- [ ] Launch a LoRA-compatible safetensors base model with `--enable-lora` and named `--lora-modules`.
- [ ] Set and record `--max-loras`, `--max-lora-rank`, and `--max-cpu-loras`; do not depend on defaults.
- [ ] Verify `AgentSpec.adapter` maps to the registered vLLM adapter model ID.
- [ ] Prove base-model and multiple distinct-adapter requests execute in parallel and can occupy one server batch when `max_loras` permits.
- [ ] Test behavior when distinct active adapters exceed `max_loras`: queueing/fairness must be observable and correct.
- [ ] Confirm the chosen base model, quantization, and vLLM model implementation support LoRA; the current native GGUF profile is not sufficient evidence.
- [ ] Verify mixed XGrammar-constrained, multi-LoRA requests rather than testing LoRA and constraints in isolation.

### SGLang multi-LoRA work

- [ ] Translate the engine-neutral adapter identity into SGLang's OpenAI `base-model:adapter-name` value.
- [ ] Test the native `/generate` `lora_path` batch interface only as a separate optional path; do not confuse it with the library's OpenAI-compatible transport.
- [ ] Launch with `--enable-lora`, explicit `--lora-paths`, `--max-loras-per-batch`, `--max-lora-rank`, and the selected LoRA backend.
- [ ] Evaluate SGLang's `csgmv` LoRA backend for the intended high-concurrency workload.
- [ ] Prove different adapters are applied to different sequences within the same batch.
- [ ] Test pinning the hottest specialist adapters and measure the effect on fairness, eviction, and memory.
- [ ] Test adapter cardinality above `max-loras-per-batch` and verify queued adapters make progress.
- [ ] Evaluate overlap loading only after establishing the resident-adapter baseline; measure its effect on prefill batching and TTFT.
- [ ] Verify mixed XGrammar-constrained, multi-LoRA requests with the public OpenAI-compatible `Backend(engine="sglang")` path.

### Multi-LoRA acceptance criteria

- Each logical agent deterministically selects its intended adapter on both engines.
- At least two different LoRAs plus the base model can process concurrent requests without output or constraint cross-contamination.
- Engine telemetry proves multi-LoRA batch residency/scheduling according to the configured adapter-slot limit.
- Durable metadata identifies the exact base model and adapter version used for every result.
- The supported target model/quantization/rank/hardware matrix is measured and documented.

## P1 — Add durable context enrichment, hooks, and agent chaining

- [ ] Separate three concepts in the public design instead of calling all of them hooks:
  - [ ] **Context providers:** run automatically before generation and return read-only context fragments.
  - [ ] **Tools/actions:** selected during reasoning and guarded by explicit authorization before side effects.
  - [ ] **Agent calls:** invoke another typed specialist and incorporate its result into the caller's context.
- [ ] Define an immutable `AgentRequest` carrying prompt, workflow identity, agent/model/adapter identity, budgets, and caller metadata.
- [ ] Define a typed `ContextFragment` carrying content, provenance, sensitivity classification, freshness/version, and token-cost estimate.
- [ ] Define an async `ContextProvider` protocol for repository search, file reads, symbol indexes, git state, test results, and other deterministic enrichment.
- [ ] Define explicit `before_run`, `after_context`, `after_result`, and `on_error` extension points only where providers/workflows do not express the requirement more safely.
- [ ] Specify hook ordering, parallelism, failure policy, timeouts, cancellation, and whether each hook runs during durable replay.
- [ ] Run script-based context providers as DBOS steps so a recovered workflow reuses recorded output rather than rerunning an uncontrolled script.
- [ ] Route any script capable of mutation through `Authorizer`/`Effector`; never let an ordinary context hook silently become an authority bypass.
- [ ] Add a first-class typed agent-to-agent call primitive implemented as a nested DBOS workflow.
- [ ] Preserve the child agent's workflow ID, output type, model/adapter identity, usage, and provenance in the parent trace.
- [ ] Detect and reject agent-call cycles unless an explicit bounded recursive workflow opts in.
- [ ] Enforce budgets for recursion depth, child calls, model requests, tool calls, wall time, tokens, and accumulated context size.
- [ ] Allow independent context providers and child-agent calls to run concurrently where dependencies permit.
- [ ] Define deterministic context assembly, ordering, deduplication, truncation, and conflict resolution.
- [ ] Prevent untrusted retrieved text from becoming system-level instructions; preserve a clear boundary between instructions and context data.
- [ ] Redact secrets and enforce workspace/path allowlists before context enters a model prompt or durable record.
- [ ] Make cache identity include provider version, relevant repository revision, agent/model/adapter identity, and other inputs that affect enriched context.

### Reasoning loop and final-output design

- [ ] Decide whether the agent uses one provider-native tool loop or a two-stage design:
  - [ ] Stage 1 produces bounded tool/agent-call decisions and accumulates context.
  - [ ] Stage 2 performs one final XGrammar-constrained generation over the assembled context.
- [ ] Investigate whether vLLM and SGLang allow tool calling and each structured-output mode in the same request without weakening either contract.
- [ ] Prefer a separate final constrained call if request-wide grammar constraints prevent or distort intermediate tool calls.
- [ ] Require every intermediate agent/tool decision to have its own typed schema rather than parsing free-form action text.
- [ ] Ensure only the final stage returns the declared `Constraint[T]`; intermediate outputs must not be mistaken for final results.
- [ ] Support iterative enrichment when a provider or child agent reveals that more context is needed, subject to the shared budgets.
- [ ] Record why each provider, tool, or child agent was invoked and how its result contributed to the final call.

### Coding-assistant context providers

- [ ] Repository metadata and current revision.
- [ ] File and directory reads with explicit workspace boundaries.
- [ ] Fast text/symbol search.
- [ ] Language-server or static-analysis queries.
- [ ] Git diff/status/history context.
- [ ] Build, test, lint, and type-check execution through authorized effectors.
- [ ] Previous durable agent results relevant to the same task.
- [ ] Specialized planner, implementation, test, review, and debugging agent calls.

### Context/chaining acceptance criteria

- An agent can automatically gather repository context before its first model call.
- An agent can make bounded typed calls to other specialized agents and use their outputs in its final prompt.
- Read-only enrichment and mutating actions have distinct, enforced authority paths.
- Worker recovery does not unintentionally repeat scripts, child calls, or side effects.
- The final result still satisfies the agent's declared constraint after any number of permitted enrichment steps.
- Cycles, runaway recursion, token growth, and tool-call explosions fail predictably within configured budgets.
- Traces identify every context source, child workflow, model/adapter, decision, and effect without exposing secrets by default.
- Multi-stage agent calls remain batchable across concurrent top-level tasks; enrichment does not serialize the whole agent plane.

## P1 — Add explicit bypass and automated approval policies

- [ ] Preserve direct advanced effector access for trusted application code, but document that it bypasses library policy enforcement.
- [ ] Add an explicit `PermitAll`/blanket authorizer instead of requiring applications to write an anonymous always-true rule.
- [ ] Add a first-class bypass mode that is distinguishable from an ordinary approval in durable records and observability.
- [ ] Do not model bypass as an ambiguous boolean. Use explicit strategies such as `Enforce`, `AutomatedApproval`, `PermitAll`, and `Bypass`.
- [ ] Define which trusted bootstrap/configuration boundary is allowed to select bypass or blanket approval.
- [ ] Allow bypass to be scoped by agent, command type, effector, workflow, environment, repository, path, operation class, caller, and time window.
- [ ] Support optional expiry, maximum uses, and revocation for delegated bypass authority.
- [ ] Require a non-empty reason and actor/authority identity for explicit bypass unless the trusted application deliberately configures a named permanent bypass policy.
- [ ] Ensure bypass skips policy enforcement only; it must not skip durable workflow identity, command validation, exact command binding, result recording, or audit events.
- [ ] Make `PermitAll` a normal positive authorization decision with named policy evidence, distinct from bypass.

### Automated approver model

- [ ] Introduce an asynchronous durable approver protocol so approval can call functions, scripts, services, or agents without blocking the event loop.
- [ ] Retain a simple synchronous adapter for cheap deterministic predicate checks.
- [ ] Add a `FunctionApprover` for typed in-process checks.
- [ ] Add a `ScriptApprover` that runs a validated argv as a DBOS step and strictly parses a typed approval envelope.
- [ ] Add an `AgentApprover` that invokes a named constrained agent and accepts only a strict schema such as `{allowed: bool, reason: str, evidence: [...]}`.
- [ ] Add a `PermitAll` automated approver for explicitly configured blanket approval.
- [ ] Allow an automated approver to return `allow`, `deny`, `abstain`, or `requires_human`; do not overload malformed output as approval.
- [ ] Compose automated checks with `all_of`, `any_of`, threshold/quorum, ordered fallback, and human-escalation policies.
- [ ] Permit policies such as "deterministic checks must pass and either the reviewer agent or a human must approve."
- [ ] Support automated pre-approval followed by human approval only when risk or confidence thresholds require it.
- [ ] Store structured evidence from every automated check, not only a free-form reason.

### Decision integrity and replay

- [ ] Canonicalize the exact command and bind every decision to its digest, command type, effector identity, workflow ID, and business key.
- [ ] Bind agent-based decisions to the approver's model, adapter, constraint, prompt/policy version, and relevant context digest.
- [ ] Execute exactly the command that was approved; any mutation after approval must invalidate the decision.
- [ ] Define whether a decision is replayed, revalidated, or expired after worker recovery based on policy freshness rules.
- [ ] Reject reuse of an approval or bypass grant for a different command, agent, effector, or environment.
- [ ] Record the complete decision chain: deterministic checks, scripts, agents, humans, blanket policies, and bypass grants.
- [ ] Make malformed script/agent output fail closed even when other approval mechanisms exist, unless an explicit composition rule treats it as `abstain`.
- [ ] Use strict booleans throughout; values such as `"false"`, `1`, or non-empty containers must never become approval through truthiness.

### Approval/bypass acceptance criteria

- Applications can explicitly choose ordinary enforcement, automated approval, named blanket approval, or scoped bypass.
- A function, durable script, or constrained specialist agent can approve when typed requirements are satisfied.
- Blanket approval is a named observable policy, not an accidental empty policy collection.
- Bypass remains visible and is bound to the exact validated command and durable workflow.
- Automated decisions can escalate to a human without losing prior evidence.
- Replays cannot apply a decision or bypass grant to changed inputs or changed policy/model identities.
- Malformed, timed-out, failed, or ambiguous automated checks never accidentally approve.

## P1 — Verify concurrent request processing and continuous batching

- [ ] Define terminology in the README:
  - **Library batch:** submitting several independent prompts and receiving separate handles.
  - **Concurrent requests:** several HTTP generations in flight at once.
  - **Continuous batching:** the inference server combines active sequences internally.
  - **Provider batch API:** an offline/asynchronous bulk API, which this library does not currently expose.
- [ ] Add a direct `asyncio.gather()` test for multiple `Agent.run()` calls without `Queue`.
- [ ] Add real-agent `Queue.submit_batch()` tests after the P0 queue repair.
- [ ] Record request start/end times, time to first token where available, total latency, throughput, and server batch/concurrency telemetry.
- [ ] Test concurrency levels 1, 2, 4, 8, and the configured server maximum.
- [ ] Test homogeneous constraints and a mixed workload of schema, regex, choice, and grammar requests.
- [ ] Verify that one request's constraint cannot contaminate another concurrent request.
- [ ] Verify independent workflow IDs, outputs, failures, and retries under concurrency.
- [ ] Verify queue backpressure and rate-limit behavior rather than merely peak task counts.
- [ ] Run the same benchmark protocol against both engines with the same model or document why a like-for-like model is impossible.

### vLLM concurrency checks

- [ ] Confirm the deployment admits more than one concurrent request.
- [ ] Confirm vLLM continuous batching through server metrics/logs, not timing inference alone.
- [ ] Measure whether constrained decoding changes batching efficiency or throughput.
- [ ] Test mixed constrained and unconstrained requests.
- [ ] Verify LoRA adapter requests can coexist safely if LoRA is enabled.

### SGLang concurrency checks

- [ ] Increase the isolated deployment from its current one-request slot only after the single-request constraint suite passes.
- [ ] Record the exact SGLang scheduling and request-slot configuration.
- [ ] Confirm multiple requests are simultaneously active through server telemetry.
- [ ] Measure XGrammar overhead and batching behavior at increasing concurrency.
- [ ] Check whether different grammar/regex states can share a batch safely.
- [ ] Verify failure isolation when one constraint fails compilation.

### Concurrency acceptance criteria

- The library can issue multiple independent requests concurrently through both direct agents and the repaired queue.
- Server telemetry proves whether each engine continuously batches those requests.
- Results remain associated with the correct prompt, workflow ID, constraint, and model/adapter.
- Concurrency limits and rate limits are enforced and documented.
- Benchmarks publish enough configuration and raw data to be reproduced.

## P2 — Capability and documentation cleanup

- [ ] Introduce a support-status distinction such as `verified`, `experimental`, and `unsupported` instead of treating every item in `Engine.supports` equally.
- [ ] Prevent experimental backend capabilities from looking production-verified without an explicit caller opt-in.
- [ ] Add a backend capability matrix to the README with server versions and evidence links.
- [ ] Add a base-model/LoRA compatibility matrix covering adapter selection syntax, maximum active adapters per batch, rank, quantization, and verified hardware.
- [ ] Document that `grammar-check` validates compilation locally but does not perform decoding.
- [ ] Document that XGrammar runs inside vLLM/SGLang, not inside `structured-agents`.
- [ ] Document that llama.cpp uses GBNF and is not an XGrammar backend.
- [ ] Explain that `submit_batch()` submits separate durable requests and relies on the server for opportunistic continuous batching.
- [ ] Document context-provider, tool/action, and nested-agent semantics separately, including durability, budgets, and authority.
- [ ] Document enforcement, automated approval, blanket approval, and bypass as distinct modes with examples and durable-record semantics.
- [ ] Add a troubleshooting guide for constraint compiler errors, server 4xx/5xx responses, tokenizer incompatibility, and model/plugin incompatibility.

## Evidence checklist for every live run

- [ ] UTC timestamp and git commit.
- [ ] Exact command and environment variables with secrets redacted.
- [ ] Engine, XGrammar, model, tokenizer, quantization, and plugin versions.
- [ ] GPU and relevant server scheduling configuration.
- [ ] Raw request and response.
- [ ] Relevant server logs and metrics.
- [ ] Test stdout/stderr and exit status.
- [ ] Unconstrained control result.
- [ ] Constrained results and adversarial assertions.
- [ ] Concurrency measurements and raw benchmark data where applicable.
- [ ] A concise conclusion distinguishing verified facts from inference.

## Definition of done

This project is complete when the repository has a reproducible, version-pinned support matrix for vLLM and SGLang XGrammar constraints; heterogeneous specialized agents can select different LoRAs and execute through a repaired durable batch path; agents can safely and durably enrich context through providers, tools, and bounded child-agent calls; trusted applications can explicitly use enforced, automated, blanket, or bypass authorization with command-bound audit evidence; server telemetry proves mixed-adapter continuous batching; and public documentation advertises only behavior supported by preserved runtime evidence.
