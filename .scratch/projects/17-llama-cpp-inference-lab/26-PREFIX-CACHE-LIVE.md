# 26 — Prefix cache: policy layer wired to live per-seq KV state

## What changed

The `prefix_cache.py` policy layer (identity keys, compatibility, `RestorePlan`)
was solid but had **0 state-capture calls** — never connected to
`llama_state_seq_*` (see 23-LLAMA-CORE-AUDIT.md). This adds the bridge.

New module `src/structured_agents/llama_core/prefix_cache_live.py`:

- `LlamaSeqStateBridge` — ctypes wrapper over the proven primitives from
  `run_seq_reuse.py` / `context_pool_router.py`: `capture_seq_state`
  (`llama_state_seq_get_data`), `restore_seq_state` (`llama_state_seq_set_data`,
  returns 0 on reject), and an own-batch `decode_tokens` with explicit positions
  + logits only on the final token. `llama_cpp` imported lazily (like `decode.py`).
- `capture_prefix(...)` — decode a shared prefix once, capture its per-seq KV
  blob, and `publish` a `PrefixCacheEntry` keyed by `PrefixCacheKey.from_fingerprint`.
  Records the capture `n_seq_max` as a runtime fact.
- `restore_and_continue(...)` — look up the key, gate on `n_seq_max` match,
  run `plan_restore`, then drive the mandatory lifecycle through the existing
  `restore_then_decode_suffix` (load state, decode fresh suffix). All failure
  modes are returned as `LiveRestoreResult` data, never raised.
- `InMemoryPrefixCache` — dependency-free cache with the same publish/lookup
  contract as `PersistentPrefixCache`, reusing `check_compatibility` /
  `check_state_integrity` for the fakes.

`prefix_cache.py`: added two `CacheRejectionReason`s — `N_SEQ_MAX_MISMATCH`
(portability gate, checked before touching llama.cpp) and
`STATE_SET_DATA_REJECTED` (the fail-closed 0-return path).

## Correctness-critical rules enforced

1. **Matched `n_seq_max`** — a blob only loads into a context with the same
   `n_seq_max`. Checked explicitly from the recorded runtime fact *and*
   fail-closed via the `set_data` 0 return (`SeqRestoreRejected`).
2. **Decode fresh suffix after restore** — saved state excludes the logits
   buffer, so a restore is only valid followed by ≥1 suffix decode, in an
   own-batch at explicit positions after the restored prefix. This is delegated
   to `prefix_cache.restore_then_decode_suffix`, so the bridge cannot skip it.

## Tests — `tests/test_prefix_cache_live.py`

Unit (no GPU, token-list fake bridge): round-trip through `InMemoryPrefixCache`
and `PersistentPrefixCache`, restored==cold continuation, `n_seq_max` reject
(never decodes), set_data-0 fail-closed, request-does-not-extend-prefix, miss.

GPU-gated integration (`test_gpu_restored_continuation_matches_cold_prefill`):
real Ornith-1.0-9B, n_ctx=2048, n_seq_max=2, 200-token shared prefix + 1 suffix
token, capture through `PersistentPrefixCache` (disk round-trip), restore into a
fresh context, decode suffix. **Bar: restored greedy token == cold-prefill
greedy token, exact.** Skipped without `LLAMA_CPP_LIB_PATH`/`CUDA_VISIBLE_DEVICES`/model.

## Results

- CPU suites: `test_prefix_cache_live` + `test_prefix_cache_contracts` +
  `test_persistent_prefix_cache` → all pass (GPU test skipped). Full `tests/`
  suite green (no regressions).
- **GPU exact-match: PASSED** (1 passed, ~15s) on GPU 0 (3060), postfix2 CUDA
  build, spike venv. Restored continuation token == cold-prefill token, exact;
  `set_data` return > 0.

## Deferred

- No high-level router integration (`ContextPoolRouter` still owns its own
  in-memory `PrefixCache`); this bridge is the reusable primitive it could adopt.
- Break-even (~192 tokens, per 25-guide) is a policy decision left to callers;
  the bridge always restores when compatible.
