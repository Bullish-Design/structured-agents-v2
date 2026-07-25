# 29 — ContextPoolRouter adopts the library `prefix_cache_live` bridge

Date: 2026-07-25
Branch: `project17-cpr-adopt-prefix-cache`

## Goal

Dogfood the merged library primitive: have the research
`benchmarks/project17/context_pool_router.py` build its prefix-cache path on
`structured_agents.llama_core.prefix_cache_live` (the `LlamaSeqStateBridge`
capture/restore + suffix-decode discipline) instead of duplicating the ctypes
per-seq KV logic inline. Behaviour must stay token-identical — the cached==cold
exact-match runners are the acceptance gate.

## What changed

### Router (`benchmarks/project17/context_pool_router.py`)
- Added an `src`-path shim + `from structured_agents.llama_core.prefix_cache_live
  import LlamaSeqStateBridge` (imports standalone with only `benchmarks/project17`
  on `PYTHONPATH`; verified).
- Construct one `LlamaSeqStateBridge(n_batch, n_vocab, native=llama_cpp)` in
  `__init__`. The bridge now owns the correctness-critical per-seq discipline.
- `_decode_prefill` → delegates to `bridge.decode_tokens` (own-batch, explicit
  positions, logits only on last) + `bridge.last_token` (greedy argmax). Deleted
  the ~20-line inline batch loop.
- `_get_seq` → `bridge.capture_seq_state`. Deleted inline
  `llama_state_seq_get_size/get_data` ctypes.
- Deleted `_set_seq` entirely. Both cached waves
  (`_generate_wave_cached`, `_generate_wave_constrained_cached`) now call the new
  `bridge.restore_blob_into_seq(ctx, blob, seq_id)` which fail-closes on
  `set_data == 0`, replacing the duplicated `if _set_seq(...) == 0: raise` checks.
- Dropped the module-level `import ctypes` (only the P2-fork `enable_seq_routing`
  still uses it, via its own local import).

The router keeps its own multi-seq batched wave (`_batched_step`, the
`seq_id == row` invariant, grammar loop) — the bridge is single-seq oriented, so
only the capture / restore-reject / suffix-decode pieces are routed through it.

### Library extension (`src/structured_agents/llama_core/prefix_cache_live.py`)
- Added `LlamaSeqStateBridge.restore_blob_into_seq(ctx, blob, seq_id) -> int`:
  restores a captured blob into one seq slot and raises `SeqRestoreRejected` on
  the `set_data == 0` reject (n_seq_max mismatch / incompatible blob), returning
  bytes read on success. This is the general primitive for the router's
  multi-slot case (same blob → multiple seq slots of one context), centralizing
  the fail-closed rule that was previously duplicated in the router.
- Minimal, general, no policy-layer coupling. No importlib (allowlist-clean).

### Tests (`tests/test_prefix_cache_live.py`)
- `test_restore_blob_into_seq_loads_same_blob_into_multiple_slots` — same blob
  into slots 0,1,2 via a fake native.
- `test_restore_blob_into_seq_fails_closed_on_zero` — `SeqRestoreRejected` with
  the offending `seq_id`.

## Verification

CPU: `pytest tests/test_prefix_cache_live.py` → 9 passed, 1 GPU-skipped;
full `tests/` suite green (no allowlist/import-hygiene regressions).

GPU (3060, CUDA_VISIBLE_DEVICES=0, out-cuda-3060-postfix2, n_ctx 4096):
- `run_context_pool_router_cached.py` → **verdict GO**, 24/24 match flags true,
  0 false (cached==cold==baseline token-exact across base/probe-a/probe-b, every
  sequence). Artifact: `.../cpr-adopt/context_pool_router_cached.json`.
- `run_router_grammar_cached.py` → **verdict GO** (schema-valid + cached==cold).
  Artifact: `.../cpr-adopt/router_grammar_cached.json`.

## Deferred / notes
- The batched multi-seq wave itself is intentionally NOT pushed into the bridge
  (the bridge is single-context/single-seq by design). If a future need arises
  for a batched multi-slot restore helper in the library, `restore_blob_into_seq`
  is the composable building block to loop over.
</content>
