# P1 — Context-Pool Multi-LoRA Router (no fork): **GO**

> 2026-07-24. Delivers the guide-14 §10 / Decision-D2 path-(a) router with ZERO
> llama.cpp changes, reusing the proven per-sequence / multi-seq primitives. The
> pragmatic teaching MVP; contrast with the §7 mixed-batch fork (P2).

## What it is

`benchmarks/project17/context_pool_router.py` — `ContextPoolRouter`:
- Load the **base model once**; every context shares `llm._model.model`.
- Load each LoRA via `llama_adapter_lora_init` against the shared model.
- **One llama_context per adapter (+ a base context)**, each pinned with
  `llama_set_adapters_lora` (via the `llama_set_adapter_lora` shim). Within a
  single `llama_decode` every sequence uses that context's one adapter — this is
  NOT mixed-batch; different adapters live in different contexts.
- `run(requests)`: route each request to its adapter's context, **batch concurrent
  requests within the context** (own-batch multi-seq decode, `seq_id == batch row`,
  per-seq `llama_get_logits_ith`), **multiplex across contexts**, waves of `n_seq_max`.
- `baseline(request)`: ground truth — the request alone on a fresh `n_seq_max=1`
  context pinned to the same adapter.

Driver + gates: `benchmarks/project17/run_context_pool_router.py`.

## Result (`artifacts/project17-context-pool-router-20260724T212025Z/`)

| gate | result |
| --- | --- |
| equivalence: routed == isolated baseline, token-exact greedy (9 reqs) | ✅ true |
| routing distinct: base / probe-a / probe-b full sequences differ | ✅ true |

Throughput (Ornith Q4 9B, 1×3060, n_seq_max=8, 9 reqs across 3 adapters → batch-3):
- routed **78.7 tok/s** vs sequential baseline **44.5 tok/s** = **1.77×**.
- Prior multi-seq work scales to ~4× at S=16; packing more concurrent requests
  per adapter climbs toward that ceiling.

## Findings

- **Exact equivalence holds.** Batching + multiplexing introduces no greedy drift
  vs single-seq — answers guide §13's "exact vs approximate" in the affirmative
  for this path. Per-seq attention masking by `seq_id` keeps sequences independent.
- **Routing distinctiveness is real but tail-weighted** with the synthetic probes:
  they only perturb layer-3 attention, so a strong prompt ("…is Paris.") shares
  the first ~8 tokens across adapters and diverges later. Gate compares FULL
  sequences. Real fine-tuned router adapters would diverge earlier.
- **KV-position discipline (bug fixed):** feed each freshly-sampled token at
  `pos == len(prompt)` for the first one, and advance `pos` only AFTER the decode
  step — advancing before triggers `find_slot: non-consecutive token position`.

## Cached shared-prefix restore composed in (2026-07-24)

`context_pool_router.py` `cache_prefix()` / `run_cached()`, driver
`run_context_pool_router_cached.py`, artifact
`artifacts/project17-context-pool-router-cached-20260724T212851Z/`.

Design: a shared prefix is prefilled ONCE per adapter in seq 0; its per-sequence
KV blob (`llama_state_seq_get_data`) is restored into every seq slot
(`llama_state_seq_set_data`); each request then decodes only its suffix. The blob
is **adapter-specific** (layer-3 K/V are adapter-perturbed) and restore is valid
only into a context with the **same n_seq_max** — both guaranteed by the pool.

- **Correctness: exact.** All 12 sequences (3 adapters x 4 requests, shared
  223-token prefix) are token-exact vs the isolated full-prompt baseline, for BOTH
  the cached-restore and cold re-prefill paths. Neighbour isolation across restored
  slots holds.
- **Throughput: ~1.03x cached-vs-cold at a 223-token prefix.** NOT a meaningful
  win in this regime: GPU prefill of a few hundred tokens is already cheap and the
  24-token batched generation dominates (~1.6 s total). The restore win grows with
  much longer shared prefixes; refines the earlier "break-even ~192 tokens" note,
  which measured restore-vs-prefill in isolation (generation excluded).
- **Constraint surfaced:** the per-sequence budget is `n_ctx_seq = n_ctx /
  n_seq_max`, NOT n_ctx. A ~400-token prefix under n_ctx=2048/n_seq_max=8 (256/seq)
  → `llama_decode rc=1`. Guard added in the driver; use larger n_ctx for long prefixes.

## Long-prefix sweep — where restore wins (2026-07-25)

`run_prefix_restore_sweep.py`, artifact `artifacts/project17-prefix-restore-sweep-*/`.
Isolates prefill-only vs restore-only (generation excluded) over prefix length P,
4 seqs sharing the prefix, n_ctx=8192/n_seq_max=4 (per-seq budget 2048), gen=8,
best-of-3. Ornith 9B, 1x3060.

| P (tok) | cold prefill | restore | prefill/restore | e2e cold | e2e cached | e2e |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 1432ms | 1087ms | 1.32x | 2412 | 2002 | 1.20x |
| 256 | 2496ms | 1145ms | 2.18x | 3577 | 2073 | 1.73x |
| 512 | 4674ms | 1209ms | 3.87x | 5626 | 2193 | 2.57x |
| 1024 | 8598ms | 1310ms | 6.56x | 9519 | 2219 | 4.29x |
| 1536 | 7917ms | 1250ms | 6.33x | 8561 | 1802 | 4.75x |

**Restore is flat (~1.1-1.3s) while cold prefill scales linearly** — the break-even
shape, demonstrated end-to-end through the router. At a 1536-token shared prefix
the batch is served **4.75x faster** than cold; raw prefill-vs-restore hits 6.6x.
Restore wins from the smallest tested prefix (128). Resolves the earlier ~1.03x
result, which used a short 223-tok prefix + long 24-tok generation that masked the
prefill savings. Realistic router economics (long shared system prompt, short
decision output) => large win. Blob grows 57MB->103MB with P (mostly the constant
recurrent SSM state + KV) but restore stays cheap vs recompute.

(Nit: the JSON field `prefill_beats_restore_at_prefix` is really
"restore-beats-prefill-at" — it reports the first P where prefill/restore > 1.)

## Scope / next

- Adapters are still the **synthetic** probes (application proven, not behaviour).
  Swapping in real fine-tuned Ornith router LoRAs is the remaining P0 asset work.
- Not yet added: EOS early-stop (currently fixed-length greedy to preserve the
  batched `seq_id==row` invariant), cross-context scheduling policy, per-sequence
  cached-prefix restore composed into the router (proven separately; wiring it in
  is the natural next increment), xgrammar-constrained routing.
- P2 (fork, §7 `build_lora_mm_routed`) remains the flagship stretch for true
  vLLM-style mixed-batch multi-LoRA.

Related: `15-ORNITH-LORA-P0-GO.md`, memory `ornith-lora-runtime-go`,
`llama-cpp-gpu-driver-stub-fix`.
