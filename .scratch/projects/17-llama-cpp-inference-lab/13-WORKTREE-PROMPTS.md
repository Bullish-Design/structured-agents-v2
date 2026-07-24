# Project 17 — independent worktree prompts

Use one prompt per clean worktree. Each prompt has exclusive file ownership;
do not edit files owned by another workstream. Preserve Project 16, existing
sglang work, and any unrelated dirty worktree changes. Use `apply_patch` for
edits, preserve ignored raw evidence locally, run the stated checks, and commit
only the owned changes.

## 1. Grammar soak and repeat-run reliability

```text
You are continuing Project 17 in structured-agents-v2.

Objective: prove the owned llama.cpp + xgrammar JSON path is repeat-run safe
and measure its real local overhead. This is Phase 1 hardening only; do not
start prefix-KV cache or multi-LoRA work.

Read in full first:
- .scratch/projects/17-llama-cpp-inference-lab/PLAN.md
- .scratch/projects/17-llama-cpp-inference-lab/02-DECISIONS.md
- .scratch/projects/17-llama-cpp-inference-lab/07-GATE1-DOUBLE-ACCEPT.md
- .scratch/projects/17-llama-cpp-inference-lab/09-GATE2-TOKENIZER-EQUIV.md
- .scratch/projects/17-llama-cpp-inference-lab/12-XGRAMMAR-API-FINDINGS.md
- .scratch/projects/17-llama-cpp-inference-lab/RESEARCH_REPORT.md

Established evidence:
- Ornith runs coherently in llama.cpp.
- tokenizer equivalence remains 26/26 probes + 600/600 fuzz under the pinned
  transformers environment.
- model vocab is 248320, larger than tokenizer vocab; masks cover model width.
- apply mask -> sampler apply -> exactly one sampler accept -> exactly one
  matcher accept is the required contract.
- Three sequential fresh-matcher requests with one compiled grammar completed
  successfully; an earlier apparent missing third record was only early command
  output, not a runtime failure.

Exclusive file ownership:
- examples/soak_grammar.py
- tests/test_grammar_soak.py
- .scratch/projects/17-llama-cpp-inference-lab/RESEARCH_REPORT.md (append only)
- .scratch/projects/17-llama-cpp-inference-lab/PLAN.md (append-only progress)

You may read but must not edit llama_core/decode.py, grammar.py, benchmark.py,
pyproject.toml, uv.lock, or other examples/tests. If the existing decode API
cannot support a correct soak, document the exact needed boundary instead of
editing another workstream's files.

Implement a reproducible CLI soak harness that:
1. accepts model path, tokenizer id, request count, schema/prompt settings,
   artifact directory, and deterministic seed where relevant;
2. compiles a grammar once, creates a fresh matcher for every request, and
   treats any finish_reason other than clean stop as a failure before Pydantic
   validation;
3. validates every result, emits one benchmark record per request plus an
   aggregate JSON summary (valid count, invalid count, cutoff count, token
   counts, p50/p95 phase timings, mask overhead); and
4. returns nonzero on any malformed, rejected, or truncated result.

Add GPU-free unit tests for aggregation and failure accounting. Run a bounded
CPU Ornith smoke (at least 10 requests if practical; otherwise record the
exact resource/time limitation). Preserve raw logs/artifacts only under an
ignored artifacts/project17-* directory.

Run focused tests, ruff, format, and then the relevant full suite. Update the
report with exact commands, environment tuple, result boundary, and what the
run does not prove. Commit and push with Git CLI. Final response must list
files, test/smoke evidence, aggregate results, remaining limitations, commit,
and push status.
```

## 2. Prefix-KV shared interfaces and strict compatibility tests

```text
You are continuing Project 17 in structured-agents-v2.

Objective: begin Phase 2 only at the shared-interface/test layer. Define the
teaching-first persistent exact-prefix KV cache contracts and prove their
compatibility checks without loading a model or writing a real persistence
backend. Do not implement state capture/restore yet.

Read in full first:
- .scratch/projects/17-llama-cpp-inference-lab/PLAN.md
- .scratch/projects/17-llama-cpp-inference-lab/02-DECISIONS.md
- .scratch/projects/17-llama-cpp-inference-lab/03-SPIKE-FINDINGS.md
- .scratch/projects/17-llama-cpp-inference-lab/05-FUTURE-PARKED.md
- .scratch/projects/17-llama-cpp-inference-lab/08-GATE3-ORNITH-RESTORE.md
- .scratch/projects/17-llama-cpp-inference-lab/RESEARCH_REPORT.md

Established facts:
- Ornith state save/restore is bit-exact, including cross-instance.
- Correct restore flow is restore prefix state -> decode uncached suffix token
  -> consume fresh logits. Logits immediately after restore are stale.
- Cache compatibility must use the frozen LlamaEngineFingerprint and exact
  token IDs, never a prompt hash alone.
- This is a persistent exact-prefix snapshot cache, not LMCache integration.
- Whole-state snapshots are large; do not claim break-even before measurement.

Exclusive file ownership:
- src/structured_agents/llama_core/prefix_cache.py
- tests/test_prefix_cache_contracts.py
- tests/typecheck_prefix_cache.py

You may import the existing fingerprint/boundary modules but must not edit
them. Do not edit grammar, decode, benchmark, pyproject, uv.lock, examples,
or project docs in this worktree.

Implement typed, hot-path-light contracts only:
- a cache key/entry containing namespace, engine fingerprint key, exact prefix
  token IDs, checkpoint token count, state size/checksum, and format version;
- deterministic key construction that cannot collide across fingerprints or
  token sequences;
- an explicit restore plan separating cached prefix and uncached suffix, with
  validation that at least one suffix token is decoded after restore;
- compatibility/rejection helpers with clear failure reasons; and
- an abstract blob-store/index protocol sufficient for a later filesystem
  implementation, but no database, SQLite, distributed system, or radix tree.

Write focused tests for deterministic keys, token-vs-prompt distinctions,
fingerprint mismatch, corrupted checksum, partial-prefix/suffix restore plans,
and the stale-logit prevention rule. Keep Pydantic at public boundaries only.

Run focused tests, ruff, format, and the relevant full suite. Commit and push
with Git CLI. Final response must state precise interfaces, what is deliberately
not implemented, test results, commit, and push status.
```

## 3. Prefix-KV local filesystem persistence implementation

```text
You are continuing Project 17 in structured-agents-v2.

Objective: implement the smallest durable local filesystem backend for the
already-defined exact-prefix cache contracts. This worktree depends on the
interface workstream being merged first; rebase onto it before editing. Do not
implement GPU measurement, model state capture, multi-LoRA, a DB, or LMCache.

Read in full first:
- .scratch/projects/17-llama-cpp-inference-lab/PLAN.md
- .scratch/projects/17-llama-cpp-inference-lab/05-FUTURE-PARKED.md
- .scratch/projects/17-llama-cpp-inference-lab/08-GATE3-ORNITH-RESTORE.md
- .scratch/projects/17-llama-cpp-inference-lab/RESEARCH_REPORT.md
- the merged src/structured_agents/llama_core/prefix_cache.py and its tests

Exclusive file ownership:
- src/structured_agents/llama_core/prefix_cache_store.py
- tests/test_prefix_cache_store.py
- examples/prefix_cache_store_demo.py

Implement a local single-user store only:
- fixed configured root; no user-controlled paths;
- atomic blob writes (temporary file then replace), checksum verification, and
  atomic metadata/index updates;
- lookup by the strict cache key from the contracts workstream;
- corruption/missing blobs are cache misses with an explicit diagnostic, never
  inference failures;
- safe namespace layout and deterministic paths.

Use JSON metadata and files only. Do not add SQLite/SQLModel, background
eviction, compression, distributed tiers, or a real LMCache connector. Add
tests for restart persistence, atomic-write behavior, corruption, missing blob,
namespace isolation, and no path traversal. The example must run without GPU
or a model using synthetic bytes.

Run tests, ruff, format, and the relevant full suite. Commit and push with Git
CLI. Report changed files, evidence, intentionally deferred scope, commit, and
push status.
```

## 4. Stateful llama.cpp capture/restore adapter and CPU correctness smoke

```text
You are continuing Project 17 in structured-agents-v2.

Objective: connect the already-merged prefix-cache contracts/store to
llama-cpp-python state capture and restore, then prove the strict suffix-decode
contract on CPU Ornith. Do not attempt GPU break-even, cache scheduling, or
multi-LoRA work.

Read in full first:
- .scratch/projects/17-llama-cpp-inference-lab/PLAN.md
- .scratch/projects/17-llama-cpp-inference-lab/06-LLAMACPP-BUILD-WORKFLOW.md
- .scratch/projects/17-llama-cpp-inference-lab/08-GATE3-ORNITH-RESTORE.md
- .scratch/projects/17-llama-cpp-inference-lab/10-CUDA-BUILD-FINDINGS.md
- .scratch/projects/17-llama-cpp-inference-lab/RESEARCH_REPORT.md
- merged prefix-cache contract/store code and tests

Exclusive file ownership:
- src/structured_agents/llama_core/llama_state.py
- tests/test_llama_state_contract.py
- examples/ornith_prefix_state_smoke.py

Use llama-cpp-python low-level/high-level state APIs only as necessary. Keep
the adapter narrow: capture state bytes for an exact checkpoint, restore bytes,
and explicitly decode one or more uncached suffix tokens before reading logits.
Never read logits immediately after a restore as if they were fresh.

Add pure/fake-native tests that enforce operation order, checksum/error
handling, and no stale-logit access. Then run a fresh ignored CPU Ornith smoke:
baseline greedy continuation versus save/restart-or-new-instance/restore plus
suffix decode must match token-for-token. Record exact model/runtime inputs and
raw artifacts under ignored artifacts/project17-*.

If the chosen state API cannot meet the contract, stop at that boundary and
document the exact API behavior with a minimal reproducer. Do not substitute a
high-level completion fallback. Run tests, ruff, format, commit, and push.
```

## 5. GPU prefix-cache break-even benchmark

```text
You are continuing Project 17 in structured-agents-v2.

Objective: measure, not assume, whether persistent prefix-state restore beats
re-prefill on the established 2x RTX 3060 CUDA build. This worktree depends on
the merged state adapter and local store. It is a benchmark/research task; do
not expand cache product scope.

Read in full first:
- .scratch/projects/17-llama-cpp-inference-lab/PLAN.md
- .scratch/projects/17-llama-cpp-inference-lab/06-LLAMACPP-BUILD-WORKFLOW.md
- .scratch/projects/17-llama-cpp-inference-lab/08-GATE3-ORNITH-RESTORE.md
- .scratch/projects/17-llama-cpp-inference-lab/10-CUDA-BUILD-FINDINGS.md
- .scratch/projects/17-llama-cpp-inference-lab/11-BUILD-SPEED.md
- .scratch/projects/17-llama-cpp-inference-lab/RESEARCH_REPORT.md

Exclusive file ownership:
- examples/ornith_prefix_cache_break_even.py
- tests/test_prefix_cache_benchmark_math.py
- .scratch/projects/17-llama-cpp-inference-lab/14-PREFIX-CACHE-BENCHMARK.md

Use the established cuda-shell.nix/build artifact and recorded real host driver
path. Do not alter root environment configuration. Create fresh ignored raw
artifact directories per runtime attempt. Fixed model/backend/context inputs
must be recorded.

Measure at several prefix lengths (at minimum 128, 256, 512, 1024 if capacity
allows), with repeated trials:
- ordinary prefill time;
- capture size/time;
- durable write/read time;
- restore time;
- mandatory suffix decode time;
- end-to-end restore-route time; and
- continuation token equality versus a no-cache baseline.

Emit structured benchmark records using the shared harness and calculate the
break-even curve honestly. A negative result is a valid outcome. Do not claim
that restore is beneficial based on a single run, CPU timings, or a stale-logit
comparison. Update the benchmark report with raw paths, environment tuple,
median/p95 results, limits, and explicit conclusion. Run static tests, commit,
and push.
```
