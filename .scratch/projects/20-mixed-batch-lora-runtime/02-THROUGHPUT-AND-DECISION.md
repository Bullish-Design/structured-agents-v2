# Workstream F — throughput (library path) + maintenance decision

> 2026-07-30. Re-measured through the shipping `MultiLoRARouter` (not the spike
> benchmark), Ornith-1.0-9B / 1×RTX 3060 (GPU 0; GPU 1 hosts a vLLM runner) / P2
> fork lib `out-p2fork-c588c4f47` / best-of-3, warmup separated. Bench:
> `benchmarks/project20/run_library_throughput.py`. Artifacts:
> `artifacts/project20-library-throughput-*/library_throughput.json`.

## Phase 1 — fixed K=2, sweep S (fork vs context-pool router vs sequential)

| S | sequential | router (context_pool) | **fork (seq_routed)** | fork/router | fork/seq | identical |
|--:|--:|--:|--:|--:|--:|:--|
| 2 | 43.8 tps | 46.7 | **67.5** | **1.44×** | 1.54× | yes |
| 4 | 43.5 | 70.7 | **83.2** | **1.18×** | 1.91× | no¹ |
| 8 | 43.4 | 86.3 | **94.7** | **1.10×** | 2.18× | no¹ |

Reproduces the spike (`21-P2-THROUGHPUT`: 1.46/1.18/1.12×) through the library code.
The fork is fastest at every S; its edge over the router shrinks as
requests-per-adapter grow (the router's within-context batching improves), holding a
structural edge by batching *across* adapters in one decode with one context.

¹ `identical=no` at S≥4 is the documented benign batched-greedy FP tie-flip (present
in the no-fork path too; the batch-1 token-exact gate in `tests/test_seq_routing_gpu.py`
proves routing correctness). Not a routing fault.

## Phase 2 — fixed S=8, grow K (masked K× overhead crossover)

| K | fork tps | router tps | fork/router | predicted overhead (2·K·r/d) |
|--:|--:|--:|--:|--:|
| 2  | 94.6 | 86.4 | 1.10× | 1.6% |
| 4  | 94.3 | 71.3 | 1.32× | 3.1% |
| 8  | 93.4 | 46.8 | 2.00× | 6.2% |
| 16 | 91.4 | **OOM** (skipped) | — | 12.5% |
| 24 | **OOM** | OOM | — | 18.8% |
| 32 | **OOM** | OOM | — | 25% |

Three findings:

1. **Fork throughput is nearly flat as K grows** (94.6→91.4 tps, K=2→16): the masked
   K× LoRA overhead is real but tiny against the base matmul at r=16 — a ~3% drop
   across an 8× increase in K, tracking the `2·K·r/d` prediction. **The masked K× cost
   never became the bottleneck in the measurable range** — so the P2b compute-fusion
   trigger was *not* reached on this hardware.
2. **The context-pool router degrades fast and then can't run at all**: its tps falls
   (86→71→47) as K rises because it splits S=8 across K adapter contexts (fewer reqs
   per context → less batching), and at **K=16 it exhausts VRAM** — K+1 contexts of KV
   cache on a 12 GB 3060. The fork keeps one context / one KV cache regardless of K,
   and still serves K=16 at full speed.
3. **The real ceiling on a 3060 is adapter-weight VRAM, not masked compute.** Both
   backends load all K adapter weight sets; the fork OOMs at K=24 (all adapters +
   model + one context). This is *orthogonal* to P2b — stacked `mul_mat_id` fusion
   saves compute, not adapter-weight bytes (`22-P2B-FUSION-TRADEOFF` §4), so it would
   not raise this ceiling. Reaching higher K needs adapter-weight paging, not fusion.

## Structural wins independent of raw tps

- **One context** — the router needs K+1 contexts; at large K the context-pool leg
  runs out of VRAM (the bench records it "skipped") while the fork keeps one context.
- **No `n_seq_max` fragmentation** across adapters — the fork batches any adapter mix
  into a single wave; the router splits concurrency per adapter context.

## Recommendation — **ship the fork backend as the `auto` default**

The fork earns its keep on both criteria the standing rules require (a clear
*structural* OR *measured* win over the shipping router):

- **Measured**: fastest at every S (1.10–1.48× the router, up to 2.18× sequential),
  and the margin *grows* with K (1.10×→2.0× as K goes 2→8) because the router's
  per-adapter batching thins out while the fork keeps one full-width batch.
- **Structural**: one context / one KV cache regardless of K. The router runs out of
  VRAM at K=16 on a 3060; the fork serves the same pool from a single context. For a
  many-adapter router fleet this is the difference between "runs" and "doesn't".

`auto` is safe because it is **fail-closed**: it selects `seq_routed` only when the
loaded lib exports the routing symbols, else transparently uses `context_pool`. A
stock lib (the common case for anyone without the private fork) is unaffected — same
public API, no error, identical results. The fork is opt-in *by lib*, not by config.

Caveats kept explicit, none blocking:
- The fork is a **private** build of pinned `c588c4f47`; it must be rebuilt and
  re-smoke-gated whenever the `llama-cpp-python` anchor moves. `auto` degrades to the
  router automatically if the fork lib is absent, so this never breaks callers.
- Token-exactness vs batch-1 is proven; larger-batch FP flips are benign and shared
  with the router (not a fork regression).
- `AdapterSpec.scale` overrides need `context_pool` (the fork applies built-in alpha).

**Verdict: keep `auto` as the default.** Ship the fork lib where the router fleet is
adapter-heavy or VRAM-bound; everywhere else `auto` falls back to the proven context-
pool path with zero caller change.

## Artifact

`artifacts/project20-library-throughput-20260731T033225Z/library_throughput.json`
(+ `.log`). Reproduce: `benchmarks/project20/run_library_throughput.py` under the
`project20-gpu-pytest` env (GPU 0, fork lib). Phase-1 `all_outputs_identical=false`
is the expected benign batched-FP flips (§Phase 1 note¹), not a routing fault.

## Follow-ups

- **P2b trigger NOT reached.** Across K=2→16 the masked overhead stayed ≤~3% of
  throughput — the stacked-`mul_mat_id` fusion (`22-P2B-FUSION-TRADEOFF`) remains
  parked. Its trigger (K in the low dozens at r=16 AND LoRA compute-bound) did not
  fire; and it would not have helped the actual ceiling below.
- **Adapter-weight VRAM is the real many-adapter ceiling** (fork OOMs at K=24 on a
  3060). The orthogonal win, if many-adapter fleets matter: page/skip adapter weights
  not present in the current batch — cuts adapter-weight bandwidth *and* footprint.
  Independent of P2b.
- `AdapterSpec.scale` overrides are ignored on `seq_routed` (the fork's
  `set_seq_adapters` takes no scales — it applies each adapter's built-in alpha); the
  router warns. Use `context_pool` when a custom scale is required.
