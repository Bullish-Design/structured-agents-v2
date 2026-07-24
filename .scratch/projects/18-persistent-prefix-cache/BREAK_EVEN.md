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
