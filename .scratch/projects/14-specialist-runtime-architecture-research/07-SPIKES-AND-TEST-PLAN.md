# Spikes and Test Plan

Date: 2026-07-21

## Execution rules

Spikes are disposable experiments, not production implementations. Run them in isolated worktrees/environments, never against privileged production destinations, and preserve exact version/configuration evidence. A spike that partially works remains failed for its stated hypothesis; narrow the claim rather than broadening the interpretation.

Before any GPU spike, record GPU model/memory, driver/CUDA, competing processes, free disk, expected model/adapter bytes, listener, and cleanup procedure. Do not download or replace models without explicit approval.

## Required falsifiable spikes

### SP-01 — corrected real DBOS queue workflow

- **Question:** Can a library-owned registered outer workflow queue a real `Agent`, validate `T`, and resolve the same durable typed outcome after restart?
- **Environment:** baseline Python 3.13.13, DBOS 2.23.0, pydantic-ai-slim 2.11.0, isolated Postgres/DBOS schema, deterministic test model; then one live model control.
- **Probe:** minimally register the proposed outer workflow; enqueue allow/deny/valid/invalid cases; kill the worker after model completion; resolve handles in a fresh process.
- **Pass:** no registration error; `T` rather than raw dict/`AgentRunResult`; stable denial/failure types; replay gives the same identities/outcome.
- **Fail:** wrapper is unregistered, authorization occurs outside the workflow, casts survive, or restart changes/reruns a completed stage unexpectedly.
- **Evidence:** source diff, function registry, workflow IDs/status/events, sanitized inputs/digests/results, DBOS logs, kill timeline.
- **Safety/cleanup:** isolated database namespace and fake model first; remove only the named namespace/container.
- **Unblocks:** AD-02, BAT-03, DUR-01; Gate 1 design.

### SP-02 — heterogeneous typed batch handles/results

- **Question:** Does fan-out/fan-in over independent workflows provide stable partial results, identity correlation, cancellation, and bounded concurrency?
- **Environment:** SP-01 environment, deterministic delayed/failing models, queue concurrency 2–4.
- **Probe:** submit mixed `T`, agents, adapters-as-identities, delays, denial, invalid output, failure, and cancellation; restart during execution.
- **Pass:** one handle/outcome per item; completion may be out of order but identity/submission index are stable; no fail-fast loss; bounds hold.
- **Fail:** positional cross-talk, aggregate exception discards results, orphan work, or concurrency exceeds configuration.
- **Evidence:** batch manifest, item state timeline, queue metrics, task/process counts, restart result.
- **Safety/cleanup:** no live GPU initially; bounded item count and timeouts.
- **Unblocks:** BAT-01/02/05/06 and public result-shape decision.

### SP-03 — vLLM XGrammar on known-good non-GGUF model

- **Question:** Does pinned vLLM 0.25.0 with its normal model path honor every claimed constraint through the public library?
- **Environment:** vLLM 0.25.0, XGrammar 0.2.3, a pre-approved small native model fitting the target GPU, recorded tokenizer/revision; no custom GGUF plugin.
- **Probe:** raw server and public-library schema/regex/choice/EBNF requests, including invalid/impossible and strict extra-field cases.
- **Pass:** exact payload mapping; valid outputs pass strict local validation; invalid combinations fail before dispatch or explicitly; repeat runs stable.
- **Fail:** silent fallback, schema-only cast, transport field mismatch, or unsupported kind advertised.
- **Evidence:** environment freeze, server args, model digest, request/response digests, compiler/backend logs, validation results.
- **Safety/cleanup:** isolated port/GPU, no mutation of active service, bounded tokens; remove only spike assets after approval.
- **Unblocks:** XGR-03 and vLLM constraint profile.

### SP-04 — SGLang XGrammar on comparable model

- **Question:** Can a pinned current SGLang version serve a comparable native model and honor its documented XGrammar API through the library?
- **Environment:** select/pin one version after compatibility check; same model family/size as SP-03 where supported; recorded XGrammar/CUDA/GPU.
- **Probe:** base chat first, then JSON Schema, regex, EBNF, invalid dual-constraint request, and strict local validation.
- **Pass:** stable base startup plus all claimed constraint kinds; one-constraint rule rendered correctly; public result is `T`.
- **Fail:** startup/model incompatibility, field mismatch, silent unconstrained output, or only upstream docs available.
- **Evidence:** full startup log, environment/server args, raw and library artifacts, memory/latency metrics.
- **Safety/cleanup:** isolated service and native model; do not mutate inactive historical deployment until accepted.
- **Unblocks:** XGR-04 and exact SGLang support floor.

### SP-05 — vLLM multi-LoRA continuous batching

- **Question:** Can different adapters for the same base model overlap in engine batches under vLLM, and what limits are safe?
- **Environment:** SP-03 base, two compatible pre-approved LoRAs, vLLM 0.25.0 with explicit `max_loras`, `max_cpu_loras`, rank, sequence/token limits, metrics enabled.
- **Probe:** A/B/base requests with synchronized starts at increasing concurrency; compare `max_loras=1` negative control and sufficient setting.
- **Pass:** per-item adapter behavior/identity is correct; metrics show overlapping running sequences and multi-LoRA activity; limits/backpressure are observable; no cross-talk.
- **Fail:** only sequential completion, wrong adapter output/identity, OOM/preemption outside declared bounds, or no metrics proof.
- **Evidence:** adapter digests/config, payloads, concurrency timeline, running/waiting/LoRA/token-step/KV metrics, GPU memory, outputs.
- **Safety/cleanup:** trusted preloaded adapters only; no runtime load endpoint exposed; rate/token/memory guardrails.
- **Unblocks:** LORA-03, BAT-04, vLLM multi-LoRA claim.

### SP-06 — SGLang multi-LoRA through OpenAI-compatible API

- **Question:** Does the chosen SGLang profile correctly render and batch distinct adapters through the OpenAI-compatible path used by the library?
- **Environment:** SP-04 server/base, two compatible adapters, explicit loaded/per-batch limits and metrics.
- **Probe:** base/A/B synchronized requests, negative unknown/wrong-base adapter, eviction/pinning only if claimed.
- **Pass:** distinct adapter behavior and response identity; engine metrics prove overlap; errors are typed; limits match capability record.
- **Fail:** native-only path is required but public claim says OpenAI compatible, cross-talk, sequential-only behavior, or unstable loading.
- **Evidence:** same classes as SP-05 plus exact SGLang selector rendering and scheduler metrics.
- **Safety/cleanup:** pre-register adapters in trusted control plane; dynamic loading disabled unless separately tested in isolation.
- **Unblocks:** LORA-03/05 and SGLang capability status.

### SP-07 — mixed LoRA plus distinct constraints in one active batch

- **Question:** Do adapter and grammar/compiler caches remain item-correct when both dimensions differ concurrently?
- **Environment:** each profile that passed its XGrammar and multi-LoRA spikes; adapter A/schema A, adapter B/regex or schema B, base/grammar C.
- **Probe:** synchronized repeated shuffled batches, adversarial schemas with mutually exclusive markers, cache warm/cold and restart cases.
- **Pass:** every result matches its own adapter oracle and validator; metrics prove overlap; zero cross-item identity/cache errors over declared sample.
- **Fail:** any swapped constraint/adapter, silent fallback, only client concurrency proof, or unsupported combination.
- **Evidence:** randomized seed, item manifests/digests, validator output, backend metrics, cache logs if exposed, failure corpus.
- **Safety/cleanup:** bounded sample/tokens; stop on first identity violation and preserve state.
- **Unblocks:** XGR-05 and combined specialist contract.

### SP-08 — PydanticAI/DBOS tools plus final constrained output

- **Question:** Which tool/toolset/event-handler calls are durably checkpointed, and can tool work feed a separate strict final call without replaying unsafe I/O?
- **Environment:** baseline PydanticAI/DBOS, deterministic model, one pure function tool, one I/O tool declared as DBOS step, one custom-toolset negative control.
- **Probe:** crash before/during/after each tool and before/after final generation; inspect durable events.
- **Pass:** declared I/O boundaries behave as documented; unsupported runtime toolsets fail early; final `T` is strict and durably owned.
- **Fail:** I/O tool executes inline unknowingly, duplicate effect lacks mitigation, or final constraint is lost after tool use.
- **Evidence:** registration graph, workflow event timeline, effect ledger/idempotency keys, prompts/messages digests, typed results.
- **Safety/cleanup:** effect is an isolated append/idempotent fake service; no privileged tools.
- **Unblocks:** AD-06/07, DUR-04/05, tool API boundaries.

### SP-09 — two-stage context/child enrichment and final constraint

- **Question:** Can typed provider and child-agent evidence be durably composed with budgets/provenance and a separate constrained final call?
- **Environment:** SP-01 core, deterministic providers, two child identities/models or fakes, strict final schema.
- **Probe:** successful enrichment, stale/untrusted/oversized context, cyclic child request, timeout, restart, and malicious instruction content.
- **Pass:** provenance/trust/budgets survive; limits terminate cycles; injection cannot change mode/identity; final output validates.
- **Fail:** ambient mutable history, unbounded loop, provenance loss, or child authority exceeds grant.
- **Evidence:** stage DAG/timeline, provider/child identities, budgets/usage, sanitized context manifest, final validation.
- **Safety/cleanup:** read-only providers and no network by default; synthetic injection corpus.
- **Unblocks:** CTX-01–03, AGC-01–03.

### SP-10 — crash/recovery at every multi-stage boundary

- **Question:** Does the proposed state machine recover deterministically at all boundaries, including the true ambiguous external-effect window?
- **Environment:** isolated DBOS database, killable worker, deterministic model/provider, fake remote effector with idempotency and commit ledger.
- **Probe:** inject hard exit before/after binding, decision, context, model response/checkpoint, validation, effect commit/checkpoint, and terminal commit.
- **Pass:** each case matches the documented state; completed checkpoints are reused; ambiguous effect is reconciled/idempotent; audit is complete.
- **Fail:** duplicate unprotected effect, identity changes, stuck/orphan workflow, lost denial/result, or “exactly once” asserted without mechanism.
- **Evidence:** reproducible fault schedule, DBOS events, remote ledger, process logs, final outcomes, attempt counts.
- **Safety/cleanup:** isolated fake destination and database namespace; explicit kill targets only.
- **Unblocks:** DUR-04/05 and release durability language.

### SP-11 — function, script, agent, and human approval composition

- **Question:** Does typed decision composition fail closed across heterogeneous approvers and escalation?
- **Environment:** isolated policy engine, allowlisted pure function, sandboxed script, constrained fake/live approver agent, deterministic human callback fixture.
- **Probe:** all decision enum combinations, malformed/timeouts, quorum/deny precedence, stale evidence, self-approval/config-change attempt, human escalation/resume.
- **Pass:** truth table matches versioned policy; malformed/error/abstain never become allow; lineage and separation of duties are preserved.
- **Fail:** boolean coercion, empty-set allow, model self-authorization, or lost durable human wait.
- **Evidence:** generated truth table, policy/approver identities, transcripts/digests, workflow waits/resumes, audit graph.
- **Safety/cleanup:** synthetic actions only; agent has no mutating tool access.
- **Unblocks:** APR-01–04 and automated-approver contract.

### SP-12 — scoped bypass with durable command binding

- **Question:** Can bypass skip policy evaluation while preserving validation, exact binding, scope/expiry, audit, and replay safety?
- **Environment:** isolated workflow/effect fixture with operator and untrusted caller principals.
- **Probe:** exact valid bypass; mutated action/arg/destination/adapter; expired/wrong tenant/wrong actor; replay; injection attempt; compare permit-all.
- **Pass:** only exact in-scope request proceeds; all other cases fail; audit differentiates bypass and permit-all; no reapproval ambiguity.
- **Fail:** bypass skips validation/binding, is selectable from content, or its token/evidence is reusable broadly.
- **Evidence:** canonical digests, auth claims, decision/audit records, mutation matrix, effect ledger.
- **Safety/cleanup:** fake effector; short expiry; no reusable real credential in artifacts.
- **Unblocks:** AUTH-01–04 and owner mode semantics.

### SP-13 — concurrent configuration allowlist isolation

- **Question:** Does the replacement for global request state isolate tasks/threads and keep registry lifecycle deterministic?
- **Environment:** baseline Python, threaded and asyncio barrier harness, proposed immutable/context-local configuration and synchronized startup registry.
- **Probe:** thousands of shuffled two-tenant reads, nested contexts, exception cleanup, task propagation, concurrent registration/freeze/read.
- **Pass:** zero cross-tenant observations; context restores after exceptions; deterministic duplicate/freeze behavior; race detector/property suite clean.
- **Fail:** any leak/lost update, shared default mutation, or nondeterministic factory resolution.
- **Evidence:** seeds/counts, failure trace, registry version snapshots, thread/task IDs, type/static results.
- **Safety/cleanup:** process-local only.
- **Unblocks:** CFG-02, AD-12, CR-02/11 closure.

### SP-14 — clean package artifact manifests

- **Question:** Can a clean checkout build reproducible wheel/sdist artifacts containing only intended files and install them in supported environments?
- **Environment:** clean checkout at implementation commit, locked builder, fresh Python environments at declared compatibility points.
- **Probe:** build twice, enumerate/hash/archive-diff, deny nested pyprojects/deploy/vendor/tests/secrets/research/model extensions, install/import/run smoke tests.
- **Pass:** allowlist passes for both formats, reproducible or explained metadata-only delta, smoke tests and metadata/dependency checks pass.
- **Fail:** current deploy/vendor contamination remains, sensitive/large files appear, or min/max environment cannot install/start clearly.
- **Evidence:** manifests/hashes/diffs, build logs, SBOM/secret scan, install smoke logs.
- **Safety/cleanup:** build to a unique temp directory; no publish/upload.
- **Unblocks:** PKG-01/02 and Gate 0 release eligibility.

## Permanent test tiers

| Suite | Purpose | Representative IDs | Required cadence |
|---|---|---|---|
| Unit/static (T0) | Strict schemas, canonicalization, policy truth tables, renderers, cache keys, types | UT-AUTH, UT-DIGEST, UT-CONSTRAINT, UT-CONFIG | Every PR |
| Dependency integration (T1) | Real DBOS registration/database, HTTP adapter, local validation, lifecycle | IT-QUEUE, IT-IDEMPOTENCY, IT-CLIENT, IT-APPROVAL | Every PR or protected integration job |
| Concurrency (T2) | Task/thread isolation, queue bounds, partial results, cancellation | CT-CONFIG, CT-BATCH, CT-REGISTRY | Every PR where practical; required merge gate for core changes |
| Crash/replay (T2) | Boundary fault injection, effect ambiguity, human wait, restart | RT-BOUNDARIES, RT-EFFECT, RT-CHILD | Nightly and release; core durability PRs |
| Live backend (T3) | Exact XGrammar/LoRA/mixed batching/capacity profile | LT-VLLM, LT-SGLANG, LT-MIXED | Scheduled, profile change, and release qualification |
| Security/adversarial (T1/T4) | Injection, decision parsing, path/digest, secrets, privilege boundaries | ST-INJECTION, ST-ADAPTER, ST-REDACTION | Every PR for deterministic corpus; full release run |
| Package/release (T4) | Format/lint/type/docs, artifact contents, install matrix, capability freshness | PT-ARTIFACT, PT-INSTALL, PT-DOCS | Every PR/build; release |

## Evidence artifact format

Each spike/live run writes a dated directory containing:

```text
manifest.json
  evidence_schema_version
  spike/test ID, git commit, start/end timestamps, operator
  environment and dependency lock/freeze
  backend/profile/model/adapter/constraint identities and digests
  hardware/driver/CUDA and server configuration digest
  commands and exit codes
  safety/cleanup declaration
  result: pass | fail | blocked
  limitations and capability cells affected

commands.log                 sanitized, ordered
server.log / client.log      sanitized
requests.jsonl               IDs/digests; raw content only when approved
outcomes.jsonl               typed per-item outcomes and validation
metrics.prom                 bounded timestamped scrape window
timeline.jsonl               workflow/stage/attempt/correlation events
artifact-manifest.sha256     evidence-file hashes
```

Secret scanning and path/prompt redaction run before evidence is retained. Raw sensitive prompts are opt-in, encrypted, access controlled, and never necessary to classify a capability cell.

## Backend qualification gate

A profile becomes **Verified** only when:

1. base startup/chat and exact client mapping pass;
2. each claimed constraint kind independently passes strict local validation;
3. each claimed adapter mode passes identity and wrong-adapter negative controls;
4. mixed-adapter/mixed-constraint claims pass SP-07;
5. client concurrency and engine batching metrics are correlated;
6. timeout, cancellation, saturation, and backend error behavior are typed;
7. restart/replay behavior passes the applicable durability suite;
8. the evidence manifest is complete, sanitized, and current;
9. documentation names exact versions/profile/limitations;
10. a profile change or evidence expiry automatically downgrades the claim.

## Stop conditions

Stop a live run and preserve evidence on cross-item identity/constraint mismatch, unexplained privilege crossing, secret leakage, uncontrolled adapter loading, repeated OOM, service collision, or an effect outside the isolated target. A safety stop is a failed spike until root cause and a safer rerun are reviewed.
