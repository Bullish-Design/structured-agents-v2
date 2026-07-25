# P2 — Mixed-Batch Multi-LoRA Fork: **GO** (exact-match validated)

> 2026-07-25. True single-`llama_decode` multi-LoRA on our private fork of the
> pinned build (`c588c4f47`): different sequences in ONE decode use different
> adapters — the vLLM/Punica capability. Design in `19-P2-FORK-DESIGN.md`.

## Result (`artifacts/project17-p2-mixed-batch-20260725T024343Z/`)

One context (`n_seq_max=4`), one mixed batch of 4 sequences with adapters
`[probe-a, probe-b, base(none), probe-a]`, greedy 32 tokens each:

| gate | result |
| --- | --- |
| routed == isolated single-adapter baseline, token-exact (per seq) | ✅ 4/4 |
| outputs distinct across adapters | ✅ true |

Baselines use the UNMODIFIED uniform-LoRA path (`llama_set_adapters_lora` on a
single-seq context); the routed path exercises the forked `build_lora_mm`. Exact
match on every sequence — including the `-1` "no adapter" sentinel (seq 2 = base)
— proves per-sequence routing with correct isolation.

## What was built (private fork; NOT upstreamed)

Patch: `patches/p2-mixed-batch-lora.patch` (5 files, base `c588c4f47`). Fork lib:
`.llamacpp-builds/out-cuda-3060-p2fork/lib` (fresh `libllama.so` + unchanged ggml).

- **`build_lora_mm` routes per-sequence** (`llama-graph.cpp`): when a seq-adapter
  pool is set, for each pool adapter compute `scale·B(Ax)` over all tokens and mask
  it to the tokens whose sequence selected it (`ggml_mul` broadcast of a per-token
  0/1 column), then sum. Correctness-first (unfused): N× the LoRA FLOPs, no stacked
  tensors, no `mul_mat_id`. Centralized — all 18 projection sites funnel through
  this one method, so zero call-site edits.
- **`llm_graph_input_seq_lora_mask`**: F32 `[n_tokens, n_adapters]` input; `set_input`
  fills column k = 1.0 where a token's primary `seq_id` maps to adapter k. Built
  lazily and cached on a `mutable` member so the `const build_lora_mm` can read it.
- **Context API**: `llama_set_seq_adapters(ctx, adapters, n)` registers the ordered
  pool; `llama_set_seq_adapter(ctx, seq_id, idx)` assigns a sequence (`-1` = none).
  Stored on `llama_context` (`seq_loras`, `seq_adapter_map`), threaded through
  `llm_graph_params` into the graph context. No `llama_batch` ABI change.
- **Bindings**: bound via ctypes on the loaded lib in
  `benchmarks/project17/context_pool_router.py` (`enable_seq_routing`/`run_seq_routed`);
  no wheel change needed.

## Build

Incremental in the existing Ninja dir (`cmake --build ... --target llama`) inside
the CUDA nix-shell; touching headers recompiled 164 CXX objects (~mins, CUDA
kernels untouched/cached), linked clean. Env per `llama-cpp-gpu-driver-stub-fix`.

## Fork vs no-fork (context-pool router, P1)

Same capability shape, different tradeoff: the P1 router runs N contexts (one per
adapter) and multiplexes; the fork runs ONE context and routes within a single
decode — closer to vLLM batching, and it removes the per-adapter context VRAM /
n_seq_max split. Correctness now proven; a throughput comparison (fork one-decode
vs P1 N-context multiplex vs sequential) is the next measurement.

## Next (P2b, deferred)

Replace the N-adapter masked loop with the stacked `mul_mat_id` path (guide §7.1-7.3:
`A_stack[in,r_max,N]`, `B_stack[r_max,out,N]`, `ids` from seq_id) once N or rank grow
enough that N× LoRA FLOPs matter. The masked path is the 90%-simpler change that
proves the capability; optimize only if a benchmark demands it.
