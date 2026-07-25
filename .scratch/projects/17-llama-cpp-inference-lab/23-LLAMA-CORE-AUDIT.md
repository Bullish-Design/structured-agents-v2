# `llama_core` audit — library vs research (2026-07-25)

> What's shipped as library-quality code in `src/structured_agents/llama_core/`
> vs what exists only as research runners in `benchmarks/project17/`. Drives the
> fold-the-flagship-into-the-library work.

## Library-quality, implemented (with tests)

| Module | What it is | Maturity |
| --- | --- | --- |
| `models.py` | Boundary Pydantic: `EngineConfig`, `GenerationRequest`, `GenerationResult` (extra="forbid") | solid; the boundary discipline (Pydantic at edges, plain/numpy on hot path) |
| `fingerprint.py` | `LlamaEngineFingerprint` (frozen), `ArtifactIdentity`, `file_identity`, `register_artifact`, `cache_key()` | solid; shared key for grammar + prefix cache + adapters |
| `decode.py` | `OwnedLlamaDecoder` — SINGLE-seq owned loop; `LogitsHook`/`TokenHook`/`SynchronizeHook` protocols; sampler single-accept (Gate 1); benchmark-instrumented; `DecodeOutcome`/`DecodeText` with explicit finish_reason | solid; the teaching reference. **Single sequence only.** |
| `grammar.py` | `JsonSchemaGrammar` (per-seq matcher, `logits_hook`/`token_hook` that plug into `decode.py`), `apply_packed_bitmask_inplace` (torch-free), `GrammarCompilerCache` (fingerprint-keyed) | solid; **Pillar 2 done**, composes with decode via hooks |
| `benchmark.py` | `BenchmarkRecord`/`BenchmarkTimer`, ns breakdown, `write_benchmark_record` | solid (Phase 0.c) |
| `diagnostics.py` | `RuntimeDiagnostics`, `collect_runtime_diagnostics` (version tuple) | solid (Phase 0.a) |
| `prefix_cache.py` | POLICY/persistence: `PrefixCacheKey` (fingerprint), `PrefixCacheEntry`, `CacheCompatibility`/rejection, `RestorePlan` ("restore then decode fresh suffix"), blob-store/index Protocols, `PersistentPrefixCache` | policy layer solid, but **NOT wired to live `llama_state_seq_*`** — 0 state-capture calls; the actual restore mechanism is still in benchmarks |

## Legacy — retire (Decision D1)

`engine/{sglang,vllm,llama_cpp}.py` — the old provider abstraction that renders
constraints to an OpenAI-API wire spec for an external server. Not the owned-decode
path; superseded by `llama_core`. Slated for removal once the flagship is folded in.

## Proven in research, NOT yet in the library

All in `benchmarks/project17/` (merged, GPU-validated) but not exposed as typed
library API:

- **Multi-sequence batched decode** — own-batch, `seq_id == row`, per-seq
  `llama_get_logits_ith`. (`decode.py` is single-seq only.)
- **Multi-LoRA context-pool router** — shared base, per-adapter pinned contexts,
  batched-within + multiplexed-across (`context_pool_router.py`).
- **P2 mixed-batch fork** — `enable_seq_routing`/`run_seq_routed` + the C++ patch.
- **Cached-prefix restore (live)** — `cache_prefix`/`run_cached` using
  `llama_state_seq_get/set_data`. The library's `prefix_cache.py` policy layer is
  ready to wrap this but isn't connected.
- **Grammar-constrained ROUTING in a batch** — grammar per-seq in a multi-seq wave
  (the library `grammar.py` hooks only target the single-seq decoder today).

## Pillar status

| Pillar | State |
| --- | --- |
| 1. Pydantic surface | boundary models done; **flagship (router/adapters) not yet surfaced** |
| 2. XGrammar | **done** (`grammar.py` + cache), composes with decode via hooks |
| 3. Inference-as-a-workflow | **embryonic** — only the 2 decode hooks; no general middleware pipeline / KV+admission events |
| 4. Custom batching | mechanics proven in research; **no library batching/admission layer** |

## Fold plan (this is what we start now)

1. `llama_core/router.py`: Pydantic surface (`AdapterSpec`, `RouterConfig`,
   `RouteRequest`, `RouteResult`) + `MultiLoRARouter` — shared base, per-adapter
   contexts, batched multi-seq routing, optional grammar via `grammar.py`. Reuses
   `EngineConfig`/`fingerprint`/`grammar`. (Fork `run_seq_routed` optional behind a flag.)
2. A runnable demo: `examples/multi_lora_router.py`.
3. Later increments: wire `prefix_cache.py` policy to the live restore; generalize
   the hook set into a middleware pipeline (Pillar 3); a batching/admission layer
   (Pillar 4); retire `engine/{sglang,vllm}.py`.
