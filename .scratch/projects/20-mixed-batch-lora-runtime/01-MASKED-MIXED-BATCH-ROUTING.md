# Masked mixed-batch multi-LoRA routing (teaching note)

> Project 20 productionizes the Project-17 P2 fork. This note explains the one idea
> the fork adds and how the library exposes it. Evidence lives in the 17-… pillar
> docs (`19-P2-FORK-DESIGN`, `20-P2-MIXED-BATCH-GO`, `21-P2-THROUGHPUT`,
> `22-P2B-FUSION-TRADEOFF`); this is the "what shipped and why" summary.

## The problem

The shipping router (`MultiLoRARouter`, `backend="context_pool"`) serves K adapters
by holding **one `llama_context` per adapter**. A mixed workload is grouped by
adapter and each group multiplexed on its own context — K decodes of ~S/K
sequences. Correct and always available, but every adapter costs a context (VRAM +
an `n_seq_max` slice), and a single decode never mixes adapters.

## The fork's one idea

The P2 fork makes **one** decode carry a mix of adapters. In `build_lora_mm` — the
single method every projection funnels through — when a per-sequence adapter pool is
registered it computes each pool adapter's LoRA delta over all tokens and **masks**
it to the tokens whose sequence selected that adapter:

```
res = W·x                              # base, once, all tokens
for k in pool:                         # every adapter, every token
    delta_k = scale_k · B_k(A_k·x)
    res += mask_k · delta_k            # mask_k ∈ {0,1} per token (seq→adapter)
```

Masks are exactly 0.0/1.0, so `+0·delta` changes nothing — the routing is
mathematically exact. Cost is `K× the LoRA FLOPs` (each adapter computed for every
token), but no stacked tensors, no `mul_mat_id`, no rank padding, no new GPU weights.
`overhead ≈ 2·K·r/d`: ~1.5% at K=2/r=16, crossing "this matters" (~10–15%) only
around K ≈ a dozen. The stacked-`mul_mat_id` fusion (P2b) that removes the K× is
**parked** until a measured workload needs it.

## Two C entry points (the fork ABI)

- `llama_set_seq_adapters(ctx, adapters, n)` — register the ordered pool once; the
  pool index is the routing id.
- `llama_set_seq_adapter(ctx, seq_id, idx)` — route a sequence to a pool slot;
  `-1` = no adapter / raw base.

No `llama_batch` ABI change, so the build stays ABI-compatible with the pinned
`llama-cpp-python 0.3.34` bindings — we bind these two symbols by hand via ctypes.

## How the library exposes it (fail-closed)

- **`seq_routing.py`** — `SeqRoutingBinding` binds the two symbols behind a
  `hasattr` capability guard; a stock lib raises `SeqRoutingUnavailable` at bind
  time, never mid-inference.
- **`fingerprint.py` / `diagnostics.py`** — `seq_adapter_routing` distinguishes a
  fork engine from a stock one in the cache key and surfaces the capability in
  diagnostics, so fork and stock never share cache state.
- **`router.py`** — `RouterConfig.backend`:
  - `context_pool` — the shipping path (always available).
  - `seq_routed` — the fork path; **explicit opt-in**, raises if the lib is stock.
  - `auto` (default) — `seq_routed` when the loaded lib reports the capability, else
    `context_pool`. A stock lib silently uses the context-pool path — missing
    capability is never an inference failure.
  The public `RouteRequest` / `RouteResult` surface is identical on both backends.
- **`batching.py`** — `ContinuousBatchScheduler` admits a mixed-adapter wave: a
  `BatchRequest` carries an `adapter`, and a fork-capable backend gets its slot
  routed at admission (`set_slot_adapter`) so one in-flight decode spans adapters.

## Correctness discipline

Ground truth = each sequence decoded **alone** on a single-adapter context (the
proven uniform-LoRA path). One mixed fork decode must be token-exact greedy per
sequence. Residual token flips at larger batch are **base-GEMM FP nondeterminism**
(present in the no-fork path too), classified as benign by the divergence classifier
— never reported as a routing bug. See `tests/test_seq_routing_gpu.py`.

## What is intentionally NOT here

P2b stacked-`mul_mat_id` fusion, upstreaming the fork, mixed-rank/rank-padding
beyond the masked path, speculative decode. The fork is a **private** build of the
pinned commit; the maintenance recommendation (ship `auto` / opt-in / park) is
Workstream F's deliverable, decided by measurement, not assumed.
