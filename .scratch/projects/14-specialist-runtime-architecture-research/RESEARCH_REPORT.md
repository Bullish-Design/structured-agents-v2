# Research Report

Status: complete  
Started: 2026-07-21T18:45:59Z  
Scope: research, analysis, requirements, architecture, and evidence planning only; no library implementation or active-service mutation.

## Fixed baseline

- Repository: `/home/andrew/Documents/Projects/structured-agents-v2`
- Branch: `main`
- Commit: `90725a56f28c6a5a09c0a93a31afcb15f3dfa504`
- Remote: `origin https://github.com/Bullish-Design/structured-agents-v2.git`
- Initial worktree: untracked project-12 and project-13 research directories; preserved as user work.
- New project directory: `.scratch/projects/14-specialist-runtime-architecture-research`

## Acceptance boundary

The session is complete only when the eleven kickoff deliverables exist, every owner decision, CR-01 through CR-15, and every major project-13 ToDo area is traceable to requirements, architecture decisions, and planned evidence, and all claims are marked verified locally, verified upstream, inferred, unknown, or contradicted.

## Safety boundaries

- No changes to library, tests, deployments, dependencies, locks, CI, or README.
- No model or large-dependency downloads.
- No active inference-service mutation.
- Read-only probes and artifacts remain under this project directory or `/tmp`.

## Investigation log

### 2026-07-21 — kickoff and repository baseline

The kickoff was read completely. The checkout commit matches the prior review baseline exactly; later evidence will determine whether uncommitted research material describes a different runtime state, but it is not treated as code authority.

### 2026-07-21 — local baseline and focused reproductions

The documented test, lint, type, format, and build checks were run through the repository's locked devenv environment. Tests, lint, types, wheel build, and sdist build pass; format check fails on eight files. A fresh sdist reconfirms deployment/vendor contamination.

Minimal real-object probes reconfirm malformed approval truthiness, cross-thread config allowlist contamination, the real-agent queue registration failure, empty `all_of()` authorization, weak settings validation, and schema cast behavior. The current local runtime snapshot shows vLLM and llama.cpp active on separate loopback ports/GPUs, SGLang inactive, and only the GGUF-backed `base` model served by vLLM; no LoRA is registered.

### 2026-07-21 — dependency, runtime, and upstream investigation

Installed PydanticAI/DBOS source was inspected rather than inferred from the library facade. `DBOSAgent` registers a distinct workflow callable; the library currently queues its ordinary `run` wrapper, which explains the reproduced registration failure. PydanticAI durably wraps model and MCP calls, but not arbitrary custom tools or event handlers. DBOS steps are at-least-once at the external-effect boundary; only DBOS transactions carry the narrower exactly-once commit guarantee.

Read-only runtime inspection established a configuration-specific boundary: the active vLLM 0.25.0, XGrammar 0.2.3, custom-GGUF profile has historical constrained-decoding evidence but its launcher explicitly excludes LoRA. The historical SGLang 0.5.14 GGUF profile failed before tensor loading and is not active. Current upstream vLLM and SGLang both document structured output and multi-LoRA facilities, but neither upstream claim proves this repository's exact model, quantization, launcher, GPU, or client mapping.

Primary sources and their claim boundaries are recorded in [`evidence/upstream/2026-07-21-source-register.md`](evidence/upstream/2026-07-21-source-register.md).

### 2026-07-21 — synthesis

The recommended design is a hybrid runtime: application-authored durable workflows compose a narrow, typed, queueable library invocation stage. The stage owns immutable model/adapter/constraint identity, canonical input digests, explicit capability checks, strict result validation, and durable typed results. A heterogeneous library batch is a set of independently identified durable invocations; the backend—not the Python client—owns continuous token-level batching.

Context enrichment and child-agent work are bounded typed stages. They finish before a distinct final constrained generation call, avoiding an unproved assumption that every provider can combine an open-ended tool loop with a request-wide output grammar. Authorization has four explicit modes (`enforce`, `automated`, `permit_all`, `bypass`), but every mode preserves input validation, identity binding, durable outcome recording, and audit unless the caller deliberately chooses a separately named raw-effector escape hatch.

## Bottom line

The architecture is feasible, but implementation is **no-go for feature expansion** until the current fail-open approval behavior, process-global policy races, incorrect queue target, idempotency overclaim, and sdist contamination are fixed and proved. Upstream support makes vLLM the first live qualification target. SGLang remains a required priority backend, but its exact local model/quantization profile is unverified and requires an isolated spike before any support claim.

The eleven requested planning deliverables are `00-EXECUTIVE-SUMMARY.md` through `10-IMPLEMENTATION-KICKOFF-PROMPT.md`. They trace all fifteen review findings, every major project-13 workstream, and all eleven owner decisions to stable requirements, architecture decisions, phases, and evidence gates.
