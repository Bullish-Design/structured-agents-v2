# P2 Throughput: fork mixed-batch vs no-fork router vs sequential

> 2026-07-25. Same S-request mixed-adapter workload served three ways.
> `run_p2_throughput.py` + `run_p2_correctness_check.py`. Ornith 9B, 1x3060,
> K=2 adapters (probe-a/probe-b), max_tokens=32, best-of-3.

## Throughput (`artifacts/project17-p2-throughput-20260725T025513Z/`)

| S | sequential | router (no-fork) | **fork** | fork/router | fork/seq | router/seq |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 44.9 tps | 48.6 | **70.8** | 1.46x | 1.57x | 1.08x |
| 4 | 44.9 | 74.4 | **87.8** | 1.18x | 1.95x | 1.66x |
| 8 | 44.7 | 89.9 | **100.3** | 1.12x | 2.24x | 2.01x |

- **Fork is fastest at every batch size** — it batches all S sequences into ONE
  decode; the router runs K decodes of ~S/K (one per adapter context); sequential
  is S decodes of 1.
- **fork/router shrinks with S** (1.46 -> 1.12x): as requests-per-adapter grows the
  router's within-context batching improves, narrowing the gap. The fork keeps an
  edge by batching across adapters in a single decode (and needs only one context,
  no per-adapter VRAM / n_seq_max split).
- At the small end (S=2, 1 req/adapter) the router degenerates to batch-1 per
  context (~= sequential); the fork still batches both -> 1.46x.

## Correctness: the cross-mode mismatch is batch-size FP, not a fork bug

`run_p2_correctness_check.py` compares each mode to the single-seq baseline
(batch-1, unmodified uniform-LoRA path = ground truth):

| S | fork vs baseline | router vs baseline | sequential vs baseline |
|--:|--:|--:|--:|
| 4 | 3/4 (r3 @tok25) | **4/4** | 4/4 |
| 8 | 7/8 (r5 @tok17) | **5/8** (r3@25, r5@17, r6@2) | 8/8 |

Conclusive: the divergences are **base-GEMM floating-point nondeterminism** at
larger batch sizes, flipping rare near-tied greedy argmaxes — NOT a routing error:
- sequential (batch-1) matches baseline 100% (it *is* batch-1, bit-identical).
- BOTH fork and router diverge as batch grows; at S=8 the **no-fork router diverges
  more (3/8) than the fork (1/8)**. A fork routing bug could not make the router
  diverge. r5@tok17 flips for fork AND router (same near-tie).
- The fork's masked routing is mathematically exact (masks are exactly 0.0/1.0, so
  `+0.0*delta` changes nothing); the differences come only from the batched base
  matmul accumulating in a different order than batch-1.

This is the guide-14 §13 "exact vs approximate" caveat and applies to ALL batched
greedy inference (fork and router alike). The earlier §7.6 gate
(`run_p2_mixed_batch`) was token-exact at batch-4 because those specific prompts had
no near-tie at the flip point; it is prompt-dependent, not mode-dependent.

## Verdict

The fork delivers the best throughput (up to 2.24x sequential, 1.12-1.46x the
no-fork router) with correct per-sequence routing; residual token flips are benign
batched-greedy FP, present in the no-fork path too. Whether the 1.12-1.46x over the
already-shipping context-pool router justifies maintaining a private fork is the
D2/P4 call. The fork's structural wins beyond raw tps: one context (no per-adapter
VRAM), no n_seq_max fragmentation across adapters, and scaling to many adapters
without a context each.

## Next (P2b)

The masked path computes all N adapters' deltas per token (N x LoRA FLOPs). At
K=2 that's negligible vs the base matmul (fork already wins). The stacked
`mul_mat_id` fusion (guide §7.1-7.3) matters only when K or rank grow large; measure
the N x cost as K rises before investing in it.
