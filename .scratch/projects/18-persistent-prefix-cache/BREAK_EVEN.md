# Whole-state cache lesson — RTX 3060

Source: completed GPU-only run
`artifacts/project17-prefix-cache-20260724T175312Z/summary.json`.
All values are synchronized wall time; `llama_decode` submission timing is not
throughput.  Each cache request includes lookup, disk read/checksum, state load,
and one suffix-token prefill before reading logits.

| Exact prefix tokens | State blob | Cold prefill mean | Cache end-to-end mean | Break-even |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 69.1 MB | 92.3 ms | 832.3 ms | -740.0 ms |
| 32 | 85.5 MB | 83.1 ms | 1472.7 ms | -1390.0 ms |
| 64 | 118.4 MB | 113.4 ms | 1683.4 ms | -1570.0 ms |
| 96 | 151.2 MB | 145.7 ms | 1741.6 ms | -1595.9 ms |

```
prefix tokens  16        32        64        96
cold (ms)      ███       ███       ████      █████
cache (ms)     ██████████████████████████████████████████████████████+
```

Teaching conclusion: persistence moves increasingly large whole-context state
blobs while CUDA prefill remains cheap.  This cache is correct, but it is not a
speedup on this model/GPU through 96 tokens.  The longer sweep is deliberately
separate evidence, not extrapolation.

## Longer-prefix confirmation

`artifacts/project17-prefix-cache-20260724T191154Z/` exited 0 at 128/192/256
tokens (two repetitions). Cold/cache synchronized means were 210.4/1952.1,
271.8/2162.0, and 373.5/2293.6 ms; state blobs were 184.0/186.1/188.2 MB.
There is no crossover through 256 tokens.

## Per-sequence API decision

`artifacts/project17-seq-state-20260724T191450Z/` copied and accepted
53,740,972 bytes from `llama_state_seq_get_data`/`set_data`, but the suffix
continuation diverged (baseline token 21059, restored token 364). This API is
therefore **not** used by the MVP. Whole-context `save_state`/`load_state`
remains the only proven state codec; partial-prefix/per-sequence reuse is
unproven and blocked on a correct upstream/runtime contract.

## Teaching summary — three lessons (2026-07-24)

1. **Correct ≠ faster.** The cache restores the exact continuation at every
   length, yet it is 5–9× slower than just recomputing the prefill. On a fast
   GPU, prefill is cheap (~1.2 ms/token); the cache instead moves a 69–188 MB
   blob through disk read, SHA-256, pickle, and a host→device state set. Moving
   a big blob is slower than recomputing a short prefix — and the gap *widens*
   with length, so "use a longer prefix" makes it worse, not better.

2. **The blob is mostly dead weight.** ~0.99 MB/token of the blob is
   `LlamaState.scores` — prefill logits (`n_vocab=248320`, fp32, capped at
   `n_batch=128` rows) that the restore lifecycle never uses, because we always
   re-decode a suffix for fresh logits. The actual model state (recurrent
   GatedDeltaNet + KV) is a nearly flat ~53–61 MB. A native codec
   (`llama_state_get_data`/`set_data`, same C call minus the wrapper) would
   drop the growth term entirely. That is the smallest justified next experiment
   (`run_native_state_decompose.py`): it is only worth promoting if native
   restore actually beats cold prefill with an exact continuation match.

3. **A successful byte-copy is not a successful restore.** The per-sequence API
   copied and re-loaded all 53,740,972 bytes and returned success, but the
   continuation diverged (21059 vs 364). The pinned `llama.h` documents that
   recurrent caches are a "partial state" needing the `_ext` + `PARTIAL_ONLY`
   path; the probe used the plain path. Never trust byte counts, non-zero
   returns, or absence of exceptions as proof of semantic correctness — only a
   matching deterministic continuation counts.

## Correction — the per-sequence divergence was our bug (2026-07-24)

A follow-up GPU sweep (`artifacts/project17-seq-state-sweep-20260724T194742Z`)
**overturns the divergence result above.** The plain non-ext
`llama_state_seq_get_data`/`set_data` path matches the greedy baseline in all
six tested configs (K∈{1,16,31} × suffix∈{1,4}) once the caller restores the
Python n_past bookkeeping (`n_tokens`, `input_ids`) after `set_data`. The
original probe left `n_tokens=0`, so `Llama.eval()` ran `kv_cache_seq_rm` and
**wiped the restored KV cache** before decoding the suffix — that, not the API,
caused 21059→364. Lesson 3 still holds (byte counts prove nothing), but the
cause here was the harness, not a broken recurrent-state API. The `_ext`
`PARTIAL_ONLY` path is *expected* to mismatch: it serializes only the partial
recurrent sub-state, not the full sequence.

A companion decomposition run
(`artifacts/project17-native-decompose-20260724T194604Z`) confirms lesson 2 with
numbers: a native codec (no score buffer) restores in a flat ~180 ms vs the
`LlamaState` path's 239–394 ms, and an **in-memory** native checkpoint crosses
cold prefill at ≈256 tokens. Persistent disk still loses at n_ctx=512.
