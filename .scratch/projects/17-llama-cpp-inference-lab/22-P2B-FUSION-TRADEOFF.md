# P2a masked vs P2b stacked-`mul_mat_id`: when the fusion earns its keep

> 2026-07-25. Decision report on the mixed-batch multi-LoRA LoRA-application path.
> Context: the P2 fork ships the **masked** path (`20-P2-MIXED-BATCH-GO.md`,
> `21-P2-THROUGHPUT.md`). This documents exactly when to invest in the **stacked
> `mul_mat_id`** fusion (guide-14 §7.1-7.3). We keep the fork on the masked path for now.

## 1. What the two paths do

Base projection weight `W: [out, in]`; adapter `k`: `A_k: [r, in]`, `B_k: [out, r]`,
rank `r`, pool size `K`. Applied delta for one adapter: `scale·B_k(A_k·x)`.

**P2a — masked (current fork, `build_lora_mm`):**
```
res = W·x                              # base, once
for k in 0..K-1:
    delta_k = scale_k · B_k(A_k·x)     # full delta for ALL tokens
    res += mask_k · delta_k            # mask_k in {0,1} per token (seq->adapter)
```
Every token computes ALL K deltas; `(K-1)/K` is masked to exactly 0.0 and discarded.
LoRA cost = **K × single-adapter cost**. Simple: reuses each adapter's existing
`a`/`b` tensors, no new GPU weights, no `mul_mat_id`.

**P2b — stacked `mul_mat_id` (deferred):**
```
A_stack: [in,  r_max, K]   B_stack: [r_max, out, K]   ids: [1, n_tokens] (token->adapter)
delta = mul_mat_id(B_stack, mul_mat_id(A_stack, x, ids), ids)
res   = W·x + scale·delta
```
Each token gathers ONLY its adapter's slice and computes ONE delta. LoRA cost =
**single-adapter cost, independent of K** (O(1) in K). Same op family as MoE
routing; the analog of Punica SGMV / S-LoRA BGMV.

## 2. Cost model (MACs per token, one targeted projection)

- Base `W·x`:            `out · in`
- One adapter `B(Ax)`:   `r · (in + out)`
- Masked path (K):       `K · r · (in + out)`
- Fused path (any K):    `~ r · (in + out)` (+ padded-rank waste, + gather overhead)

**LoRA overhead vs base (masked):**
```
overhead ≈ K · r · (in + out) / (out · in)      # square proj (in≈out≈d):  ≈ 2·K·r / d
```
Linear in K and r; base term `d` fixed.

## 3. The numbers (Ornith, d=4096, probe r=16)

`o_proj` in=out=4096: base = 16.8M MACs; one adapter = 0.26M MACs.

| K | r | masked LoRA MACs | overhead ≈ 2Kr/d |
|--:|--:|--:|--:|
| 2  | 16 | 0.52M | **~1.6%**  (negligible — where we are) |
| 8  | 16 | 2.1M  | ~6%   |
| 16 | 16 | 4.2M  | ~13%  |
| 64 | 16 | 16.8M | ~50%  |
| 64 | 64 | 67M   | ~200% (LoRA = 2× the base matmul) |

Crossover into "this matters" (>~10-15%): **K ≈ a dozen at r=16**, sooner at higher
rank. This is why the fork already wins at K=2 despite the "wasteful" masked path —
the waste is ~1.5% of the base matmul, i.e. in the noise.

## 4. Costs of the fusion (why not just do it)

1. **Building `A_stack`/`B_stack` on the GPU** is the hard part: allocate backend
   buffers, copy+pad each adapter's `a`/`b` into its expert slice (mirror
   `llama-adapter.cpp` tensor loading). The masked path avoids this entirely.
2. **Rank padding waste**: pad to `r_max`; mixed ranks burn FLOPs on padded rows,
   clawing back some of the K× win.
3. **Gather overhead**: `mul_mat_id` indexing + less regular memory access than a
   dense matmul.
4. **Bandwidth nuance (important):** decode is weight-memory-bandwidth bound
   (throughput scaled ~4× S=1->16 as base weights amortized). BOTH paths still read
   all K adapters' small A/B matrices from VRAM; the fusion saves **compute**, not
   adapter-weight **bytes**. So it pays off specifically when LoRA becomes
   **compute-bound** — large K and/or rank and/or batch — not merely when K > 2.
   (A further win, only if implemented: skip loading adapters absent from the current
   batch to cut adapter-weight bandwidth too — orthogonal to the compute fusion.)

## 5. Decision rule

- **Now (K=2, r=16):** keep the masked path. ~1.5% overhead; fork already beats the
  no-fork router (`21-P2-THROUGHPUT.md`). P2b would add stacked-tensor allocation +
  padding complexity to optimize a non-bottleneck.
- **Trigger to implement P2b:** as adapters are added, measure the LoRA fraction of
  decode time. Invest when the K× masked overhead actually shows in throughput —
  roughly **K in the low dozens at r=16**, lower at higher rank, and specifically
  when the LoRA step is compute- (not bandwidth-) bound. Sanity-check against
  `overhead ≈ 2·K·r/d` before building anything.
- Same fail-closed gate as P2a (`run_p2_mixed_batch.py`): token-exact vs isolated
  single-adapter baseline, swept over K, ranks (incl. mixed → padding), the -1
  sentinel. Reuse the correctness harness (`run_p2_correctness_check.py`) which
  already separates real divergence from batched-greedy FP tie-flips.

## 6. References

- `19-P2-FORK-DESIGN.md` (design), `20-P2-MIXED-BATCH-GO.md` (masked path GO),
  `21-P2-THROUGHPUT.md` (fork vs router vs sequential + FP-nondeterminism analysis).
- Guide-14 §7.1-7.3 (stacked layout), §14 Punica (SGMV) / S-LoRA (BGMV).
- Fork patch: `patches/p2-mixed-batch-lora.patch`.
