# Clean-Session Kickoff Prompt — Research, Analysis, and Requirements Planning

You are beginning a new research and architecture-planning session for the `structured-agents-v2` repository.

Work from:

```text
/home/andrew/Documents/Projects/structured-agents-v2
```

Your mission is to independently understand the current library, verify or challenge the prior code-review findings, research the external runtime capabilities on which the concept depends, convert the owner’s product intent into precise requirements, and recommend a coherent target architecture plus phased implementation/evidence plan.

This is a research, analysis, and requirements-planning project. Do not implement the library changes in this session.

## 1. Product intent

The target is high-quality local coding assistance built from specialized, fine-tuned small-model agents. The intended system keeps a capable base model resident, applies different task-specific LoRA adapters per logical agent, constrains outputs to narrow typed contracts, and batches many independent requests efficiently on local GPUs.

Agents must be able to enrich their own context before returning a final result. Context enrichment may include deterministic functions, repository searches, scripts, tools, and calls to other specialized agents. The system must support bounded multi-stage or iterative reasoning while still returning a final constrained output.

Authorization must be flexible rather than mandatory in one fixed form. Trusted applications must be able to choose among:

- Enforced policy checks.
- Automated approval by functions, scripts, services, or agents.
- Named blanket approval.
- Explicit, potentially scoped authorization bypass.
- Human approval or escalation where desired.

Bypass may skip policy enforcement, but the design should distinguish whether command validation, durable identity, exact input binding, result recording, and audit evidence still apply. Do not silently assume that bypass means unobservable execution; analyze the alternatives and recommend explicit semantics.

## 2. Owner decisions that are already settled

Treat the following as requirements, not optional ideas:

1. Batched specialist processing is central to the concept.
2. Per-agent LoRA selection is required.
3. Different agents using different LoRAs must be usable with concurrent/continuously batched inference where the engine supports it.
4. vLLM and SGLang are the priority inference engines.
5. XGrammar constrained-output support must be verified against both vLLM and SGLang rather than inferred from request dictionaries.
6. Agents need automatic context enrichment and bounded agent-to-agent/model-call chaining before final output.
7. Functions and scripts must be usable as context providers or automated checks.
8. Mutating actions must have explicit authority semantics, including enforce, automated approve, blanket approve, and bypass options.
9. Automated approvers may themselves be constrained agents.
10. The system remains local-first and should exploit specialized small models rather than depending on one very large general model.
11. The final design must be evidence-driven and explicit about what is verified, experimental, or unsupported.

Do not reopen these product decisions unless two requirements are technically incompatible. If that occurs, document the conflict and present the narrowest viable alternatives.

## 3. Required local inputs

Read these completely before forming conclusions:

1. [`README.md`](../../../README.md)
2. [`pyproject.toml`](../../../pyproject.toml)
3. Every file under [`src/structured_agents`](../../../src/structured_agents/)
4. Every first-party test under [`tests`](../../../tests/)
5. [`../12-structured-agents-v2-library-study/CONCEPTUAL_OVERVIEW.md`](../12-structured-agents-v2-library-study/CONCEPTUAL_OVERVIEW.md)
6. [`../12-structured-agents-v2-library-study/CODE_REVIEW.md`](../12-structured-agents-v2-library-study/CODE_REVIEW.md)
7. [`TODO.md`](TODO.md)
8. Relevant deployment material under `deploy/vllm`, `deploy/sglang`, and `deploy/llama-cpp`
9. Relevant planning/research history under:
   - `.scratch/projects/10-durable-agent-plane-build`
   - `.scratch/projects/11-sglang-provider-abstraction`
   - `.scratch/projects/08-unsloth-gemma4-gguf-compatibility`
10. Relevant runtime artifacts under `artifacts/`

The prior documents are inputs, not authority. Verify their claims against the current checkout and installed dependency versions. Clearly distinguish current code from superseded designs and historical artifacts.

Inspect installed PydanticAI and DBOS source where necessary to establish real wrapper, workflow, toolset, serialization, queue, recovery, and lifecycle contracts. Do not rely on type casts or comments as proof of dependency behavior.

## 4. Workspace and output rules

1. Check for repository guidance such as `AGENTS.md` and follow it.
2. Record the current branch, commit, remotes, and dirty worktree before beginning.
3. Preserve all existing user changes. Do not modify or delete prior project documents.
4. Determine the next available numbered directory under `.scratch/projects` and create a new directory for this research project. At the time this prompt was written, project 13 was current, so the expected next number is 14; verify rather than assume.
5. Save all research deliverables in that new directory.
6. Do not modify `src/`, `tests/`, deployment code, dependency files, locks, CI, or README during this planning session.
7. Read-only probes, tests, builds into `/tmp`, dependency-source inspection, and non-mutating runtime inspection are allowed when relevant.
8. Do not restart, stop, reconfigure, install into, or otherwise mutate active inference services without explicit user authorization.
9. Do not download models, adapters, or large dependencies without explicit user authorization.
10. Do not describe a live capability as verified unless this session produces or locates reproducible runtime evidence for the exact relevant configuration.

## 5. Research standards

Current vLLM, SGLang, XGrammar, PydanticAI, DBOS, LoRA, and engine-version behavior is temporally unstable. Research current primary sources:

- Official vLLM documentation and source.
- Official SGLang documentation and source.
- Official XGrammar documentation and source.
- Official PydanticAI documentation and source.
- Official DBOS documentation and source.
- Primary model/adapter documentation where compatibility depends on a specific architecture.

Use secondary sources only to find primary evidence or to identify a question. Cite direct links next to every external capability claim. Record document/version dates when available. If official documentation and the installed version disagree, report both and privilege the behavior of the installed/pinned version for current-repository conclusions.

Separate all conclusions into:

- **Verified locally:** directly established against this checkout or an exact preserved artifact.
- **Verified upstream:** stated or tested in authoritative upstream material, but not reproduced locally.
- **Inferred:** follows from evidence but was not directly tested.
- **Unknown:** insufficient evidence.
- **Contradicted:** local and upstream evidence disagree.

## 6. Establish the baseline

Before proposing a design:

1. Map the current public API and internal dependency graph.
2. Trace these execution paths line by line:
   - `AgentSpec` to `Backend.build()` to `Agent.run()`.
   - Constraint declaration to engine rendering to PydanticAI model settings.
   - DBOS agent registration and nested workflow execution.
   - Queue submission and handle result retrieval.
   - Direct and batched agent calls.
   - `execute()` authorization and effect execution.
   - Human approval request, pending discovery, response, timeout, and replay.
   - Serialized config and schema resolution.
3. Run the repository’s documented baseline tests, lint, types, format check, and package build if safe and available.
4. Preserve exact commands, exit codes, versions, and concise outputs.
5. Reproduce the most consequential prior findings with minimal probes where needed, especially:
   - Malformed approval truthiness.
   - Cross-thread allowlist contamination.
   - Real `Agent` queue failure and result-type mismatch.
   - Empty `all_of()` authorization.
   - Source-distribution contamination.
6. Note whether the checkout has changed since commit `90725a5`; reclassify fixed or stale findings instead of copying them forward.

## 7. Research track A — Revalidate the code review

For CR-01 through CR-15 in the earlier review:

1. Reproduce or inspect the finding against the current checkout.
2. Classify it as confirmed, partially confirmed, fixed, stale, disputed, or superseded.
3. Reassess severity using concrete impact.
4. Identify which new product requirements interact with it.
5. Recommend the architectural resolution, not merely a local patch.
6. Specify regression and acceptance tests.

Pay particular attention to interactions that can amplify risk:

- Authorization bypass plus malformed approval parsing.
- Agent-driven automated approval plus recursive agent calls.
- Durable replay plus changing LoRA/model/policy identity.
- Batch keys plus heterogeneous commands/prompts.
- Context scripts plus effect authorization.
- Mutable adapter names plus idempotent workflow replay.

## 8. Research track B — Batched specialist processing

Determine the correct meaning and API for batching at four distinct layers:

1. Logical application batch: several agent/prompt operations submitted together.
2. Durable queue concurrency and backpressure.
3. Multiple concurrent HTTP inference requests.
4. Inference-server continuous batching.

Do not conflate these with an offline provider batch API.

Answer at least these questions:

- What should a heterogeneous batch accept and return when agents have different output types?
- Should submission return all accepted handles immediately, a structured per-item submission result, or a durable batch handle?
- How are partial submission failures, inference failures, cancellation, retry, ordering, fairness, priority, and timeouts represented?
- How should logical batch IDs relate to DBOS workflow IDs?
- How can queue backpressure avoid accidentally serializing the inference server?
- How can multi-stage agents remain batchable across context, planning, tool, and final-generation stages?
- Should scheduling group work by base model, adapter, grammar/constraint, sequence length, or urgency—or should that remain entirely server-owned?
- What observability proves that the server actually formed batches?
- What target concurrency levels and workload mixes matter on the available GPUs?
- What throughput, TTFT, inter-token latency, tail latency, memory, and quality thresholds constitute success?

Research DBOS queue registration and result semantics carefully. The current real-agent queue defect must be resolved at the architecture level rather than hidden behind another cast or inaccurate test double.

## 9. Research track C — Per-agent LoRA and multi-LoRA batching

Establish current vLLM and SGLang behavior for:

- Static adapter registration.
- Dynamic loading and unloading.
- OpenAI-compatible adapter selection.
- Native API adapter selection.
- Multiple adapters in one batch.
- Base-model and adapter requests in one batch.
- Maximum distinct adapters per batch.
- Maximum loaded/resident adapters.
- GPU pinning and eviction.
- Adapter rank and target-module constraints.
- Quantization compatibility.
- CUDA graph and LoRA backend implications.
- Adapter loading overlap and its effect on prefill batching.

Resolve the library abstraction:

- Is `adapter: str` sufficient?
- Should identity include base model, adapter name, immutable digest/version, rank, and target modules?
- How should vLLM’s adapter-as-model-ID convention and SGLang’s `base-model:adapter-name` convention map from one neutral specification?
- Should native SGLang `lora_path` batching be supported, or should the library remain OpenAI-compatible only?
- How is exact adapter identity persisted in durable workflow metadata?
- What happens when a workflow key is replayed after adapter weights change under the same name?
- How are arbitrary runtime adapter paths prevented from becoming a code/weight loading authority vulnerability?

Explicitly analyze the current GGUF deployment. Determine whether the active model/quantization/plugin path can support LoRA at all. Separate core-engine capability from model-loader/plugin limitations. Recommend at least one known-good small-model plus LoRA test configuration that isolates the engine feature from GGUF-specific problems.

## 10. Research track D — XGrammar on vLLM and SGLang

Build a capability matrix for every engine and constraint:

- Pydantic JSON Schema.
- Regex.
- Finite choice.
- Grammar.
- Tool/action decision schemas.
- Final constrained output after tool use.

For each combination, establish:

- Exact client request shape.
- Exact server launch/configuration requirement.
- Grammar backend actually selected.
- Supported syntax/dialect.
- Compile-time and runtime validation.
- Version/model/tokenizer limitations.
- Interaction with LoRA.
- Interaction with concurrent heterogeneous batches.
- Interaction with tool calling.
- Error behavior and observability.

Investigate the known constrained-output HTTP 500 evidence in the repository. Determine which hypotheses remain plausible: vLLM version, XGrammar version, model architecture, tokenizer, GGUF quantization, vendored plugin, deployment drift, or request format.

Do not treat a 200 response alone as proof. Adversarial prompts must demonstrate that invalid outputs cannot escape the constraint. Static dictionary tests prove only rendering, not constrained decoding.

Resolve the EBNF/GBNF issue explicitly. Do not describe llama.cpp pass-through as XGrammar support.

## 11. Research track E — Context enrichment, hooks, tools, and agent chaining

Distinguish and design:

1. Automatic context providers that run before generation.
2. Lifecycle hooks or middleware.
3. Model-selected read-only tools.
4. Model-selected mutating actions.
5. Typed nested calls to other agents.
6. Iterative or recursive agent workflows.
7. A final constrained-output stage.

Research what current PydanticAI and its DBOS integration already provide:

- Constructor-time tools and toolsets.
- Runtime function toolsets.
- MCP toolsets and registration requirements.
- Dependencies and run context.
- Dynamic instructions.
- Event/capability hooks.
- Nested agent calls.
- Durable model/tool steps.
- Serialization and recovery constraints.
- Unsupported dynamic behavior under DBOS.

Compare at least these architectural options:

- One provider-native tool loop with final structured output.
- A two-stage loop: typed context/action decisions followed by a separate final XGrammar-constrained generation.
- An application-authored durable workflow graph around narrow single-call agents.
- A hybrid that offers context-provider and child-agent primitives but keeps arbitrary orchestration in application workflows.

Evaluate:

- Whether final request-wide constraints interfere with intermediate tool calls.
- Determinism and replay.
- Context provenance and freshness.
- Prompt injection across retrieved context.
- Secret handling.
- Workspace/path restrictions.
- Token/context budgeting and truncation.
- Cycles and recursion depth.
- Model/tool/call/time budgets.
- Parallel context collection.
- Batchability of each stage.
- Exact typing of intermediate and final results.

Recommend a minimal public abstraction consistent with the library’s existing small, code-first character. Avoid recreating all of PydanticAI behind a second abstraction unless evidence justifies it.

## 12. Research track F — Authorization, automated approval, blanket approval, and bypass

The design must support explicit policy modes rather than assuming every action follows one human approval path.

Research and specify:

- Synchronous deterministic authorizers.
- Asynchronous/durable function approvers.
- Script approvers.
- Agent approvers with strict constrained decisions.
- Human approvers.
- Named blanket approval (`PermitAll`).
- Explicit bypass.
- Composition: all, any, threshold/quorum, fallback, escalation, and abstention.

At minimum, define decision outcomes such as:

- Allow.
- Deny.
- Abstain.
- Requires human.
- Error/invalid evidence.

Analyze whether bypass should be:

- A direct advanced call to an effector.
- A first-class execution mode.
- A scoped capability/grant.
- A named authorizer.
- Some combination with different audit semantics.

Define what bypass may skip and what it must preserve. Consider scope by command type, effector, agent, workflow, environment, repository/path, operation class, actor, expiry, and use count.

Every decision should be evaluated for binding to:

- Canonical command digest.
- Command type/schema.
- Effector identity.
- Workflow/business key.
- Environment/repository revision.
- Policy version.
- Context/evidence digest.
- Approver identity.
- Agent approver model, adapter, constraint, and prompt version.

Resolve approval-message correlation, stale pending events, strict boolean parsing, policy changes during replay, decision freshness/expiry, and TOCTOU between approval and execution.

Do not assume an LLM approver is inherently trusted or untrusted. Make trust a deliberate application policy and specify how constrained output, evidence, confidence, quorum, and human escalation affect that policy.

## 13. Research track G — Durability and idempotency semantics

Define the precise guarantees for:

- Model calls.
- Context-provider scripts.
- Tool calls.
- Nested agents.
- Approval checks.
- Human messages.
- External effects.
- Batch submission.
- Workflow replay after model/adapter/policy/config changes.

Replace vague "exactly once" language with guarantees that can be defended across crash boundaries. Determine where external idempotency keys, transactional outboxes, command digests, or compensating actions are required.

Specify conflict behavior when the same workflow/business key is reused with different:

- Prompt.
- Agent.
- Base model.
- LoRA/version.
- Constraint.
- Context-provider inputs.
- Command.
- Policy/bypass mode.

## 14. Research track H — Configuration, lifecycle, packaging, and qualification

Include the remaining code-review concerns in the target plan:

- Strict serialized settings and unknown-field rejection.
- Thread/task-safe config context.
- Deterministic plugin/factory registration.
- Local schema validation.
- DBOS configure/register/launch/shutdown state machine.
- HTTP client ownership.
- Process timeouts, output limits, cancellation, cwd/environment policy.
- Public versus private dependency APIs and supported version ranges.
- Clean wheel/sdist manifests.
- CI and backend-specific live test tiers.
- Accurate README and support-status matrix.

Do not allow the exciting runtime features to leave the confirmed authority and packaging defects unplanned.

## 15. Requirements methodology

Produce atomic, testable requirements with stable IDs. At minimum use categories such as:

- `BAT-*` — batch and scheduling.
- `LORA-*` — adapter identity and multi-LoRA.
- `XGR-*` — constrained decoding.
- `CTX-*` — context providers and hooks.
- `AGC-*` — agent chaining and reasoning loops.
- `AUTH-*` — authorization and bypass.
- `APR-*` — automated/human approval.
- `DUR-*` — durability and idempotency.
- `CFG-*` — configuration.
- `OPS-*` — lifecycle and observability.
- `SEC-*` — trust boundaries and data handling.
- `PKG-*` — packaging/release.
- `TEST-*` — evidence and qualification.

For every requirement include:

1. Priority: MUST, SHOULD, or COULD.
2. Rationale.
3. Concrete acceptance criteria.
4. Required evidence/test tier.
5. Dependencies.
6. Related prior finding(s) and ToDo section(s).
7. Risks or unresolved decisions.

Create a traceability matrix from every CR-01 through CR-15 and every major project-13 ToDo section to one or more requirements, architectural decisions, and planned tests. No prior finding or owner requirement should disappear silently.

## 16. Architecture evaluation

Develop at least two credible end-to-end architecture options and one deliberately minimal option. Compare them on:

- Fit to the product intent.
- Public API size and conceptual clarity.
- DBOS durability/recovery compatibility.
- PydanticAI composition rather than duplication.
- Batchability and GPU efficiency.
- Multi-LoRA portability.
- XGrammar/tool compatibility.
- Type safety.
- Authority and bypass clarity.
- Testability.
- Migration cost from 0.3.0.
- Operational complexity.
- Failure isolation.

Recommend one option. State why it wins, what it intentionally does not solve, and which assumptions require spikes before implementation.

Use diagrams where they materially clarify:

- Build-time object relationships.
- A multi-stage agent call.
- Heterogeneous batching and server scheduling.
- Approval/bypass decision flow.
- Durable workflow/replay boundaries.

## 17. Required spikes and evidence plan

Specify small, falsifiable spikes before committing to uncertain API design. Each spike must include:

- Question/hypothesis.
- Exact environment and version.
- Minimal implementation/probe.
- Success and failure criteria.
- Evidence to capture.
- Cleanup and safety boundary.
- Decision unblocked by the result.

At minimum plan spikes for:

1. A real library `Agent` through a corrected DBOS queue workflow.
2. Heterogeneous typed batch handles/results.
3. vLLM XGrammar with a known-good non-GGUF small model.
4. SGLang XGrammar with the same or comparable model.
5. vLLM multi-LoRA continuous batching.
6. SGLang multi-LoRA batching using its OpenAI-compatible API.
7. Mixed LoRA plus distinct constraints in one active server batch.
8. PydanticAI/DBOS tools plus final constrained output.
9. Two-stage child-agent/context enrichment plus final constrained output.
10. Crash/recovery at every boundary of a multi-stage call.
11. Function, script, agent, and human approval composition.
12. Scoped bypass with durable command binding.
13. Concurrent config allowlist isolation.
14. Clean package artifact manifests.

## 18. Required deliverables

Create the following documents in the new numbered research directory. You may combine two only when the result is clearer, but do not omit content.

1. `00-EXECUTIVE-SUMMARY.md`
   - Overall conclusions.
   - Release blockers.
   - Recommended target architecture.
   - Highest-value next decisions.

2. `01-CURRENT-STATE-AND-EVIDENCE.md`
   - Current architecture and execution flows.
   - Baseline commands/results.
   - Installed versions.
   - Revalidated CR-01 through CR-15.
   - Verified/inferred/unknown classification.

3. `02-REQUIREMENTS.md`
   - Numbered MUST/SHOULD/COULD requirements.
   - Acceptance criteria and dependencies.
   - Non-functional requirements.

4. `03-TRACEABILITY.md`
   - Code-review finding to requirement/decision/test mapping.
   - Project-13 ToDo section to requirement/decision/test mapping.
   - Owner decision coverage.

5. `04-ARCHITECTURE-OPTIONS.md`
   - Options and scorecard.
   - Recommended architecture.
   - Public API concepts and lifecycle.
   - Diagrams.

6. `05-BACKEND-CAPABILITY-MATRIX.md`
   - vLLM, SGLang, and relevant llama.cpp boundaries.
   - XGrammar, tools, LoRA, multi-LoRA batching, quantization, and evidence status.
   - Versioned primary-source links.

7. `06-DURABILITY-AUTHORITY-AND-THREAT-MODEL.md`
   - Workflow/step boundaries.
   - Idempotency semantics.
   - Approval and bypass modes.
   - Command binding and replay.
   - Trust boundaries, prompt injection, secrets, and runtime adapter loading.

8. `07-SPIKES-AND-TEST-PLAN.md`
   - Required spikes.
   - Unit/integration/concurrency/crash/live/package test tiers.
   - Evidence capture format.
   - Backend qualification gates.

9. `08-PHASED-ROADMAP.md`
   - Ordered phases with prerequisites.
   - Explicit exit criteria.
   - Suggested reviewable implementation increments.
   - What must land before feature expansion.

10. `09-DECISIONS-AND-OPEN-QUESTIONS.md`
    - Decisions made with rationale.
    - Remaining owner decisions.
    - Unknowns assigned to spikes.
    - Rejected alternatives.

11. `10-IMPLEMENTATION-KICKOFF-PROMPT.md`
    - A self-contained prompt for a later clean session to begin implementation only after the research plan is accepted.
    - Do not implement from it during this session.

## 19. Planning constraints and taste

- Preserve the library’s code-first, typed, relatively small character.
- Prefer explicit composition over hidden ambient behavior.
- Prefer engine-neutral domain concepts with narrow engine-specific rendering.
- Do not hide mismatches behind `cast()`.
- Avoid global mutable request context.
- Make experimental versus verified support visible.
- Use strict typed envelopes at authority and durability boundaries.
- Keep read-only context acquisition distinct from mutating effects.
- Keep blanket approval distinct from bypass.
- Make trusted escape hatches explicit rather than pretending they do not exist.
- Do not require all applications to use authorization, human approval, recursive agents, dynamic LoRA loading, or native provider batch APIs.
- Do not overfit the public API to one current server flag or one deployment experiment.
- Do not claim arbitrary external effects are transactionally exactly-once.

## 20. Quality bar

The final research set must be detailed enough that a separate implementation agent can work without rediscovering basic architecture or making product decisions implicitly.

Every major recommendation must have:

- Direct code evidence.
- Dependency or primary-source evidence where applicable.
- Explicit tradeoffs.
- Testable acceptance criteria.
- A migration story.
- A failure/recovery story.
- A security/authority story.
- An operational/observability story.

Avoid vague conclusions such as "support batching," "add hooks," or "use LoRA." Specify which layer, API, identity, durability boundary, failure behavior, limits, and evidence qualify each capability.

## 21. Final response for the research session

When the research work is complete, report:

1. The new numbered project directory.
2. Links to every deliverable.
3. The recommended architecture in a short paragraph.
4. The top release blockers.
5. The most important unresolved questions or required live spikes.
6. Baseline verification results.
7. Confirmation that no implementation or service mutation was performed.

Do not mark the project complete merely because documents exist. Cross-check every owner decision, project-13 ToDo area, and CR-01 through CR-15 against the traceability matrix before finishing.
