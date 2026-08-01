# SPIKE-DISTILL-TRACEWEIGHT — TRACE up-weighted re-distill (v3)

Date: 2026-07-31 · Adapter: `spike_context_distill_adapter_v3.pt` (rank-16 LoRA, 400 KL steps,
TRACE queries repeated ×3 in the train pool = 40-item pool, 24 TRACE / 8 WARN / 8 FATAL,
fixed-seed shuffle) · Same teacher-fixed CONTEXT as v2 (5 exemplars)
Results: `spike_distill_result_v3.json` (plain path, builtin 6-query eval) ·
`spike_distill_template_v3_result.json` (template path, 24 queries) · Log: `traceweight_v3_run.log`

## Results — template path, same 24 queries, all three versions (format-agnostic scoring)

| version | student total | student TRACE | student WARN+FATAL | student format | student KL | ceiling total | gap_recovered |
|---|---|---|---|---|---|---|---|
| v1 (old CONTEXT, balanced) | 15/24 | 0/8 | 15/16 | 1.0 | 0.095 | 15/24 | 1.0 |
| v2 (teacher exemplars, balanced) | 15/24 | 0/8 | 15/16 | 1.0 | 0.125 | 20/24 | 0.75 |
| **v3 (exemplars + TRACE ×3)** | **23/24** | **7/8** | **16/16** | 1.0 | **0.092** | 20/24 | **1.15** |

TRACE per-item (v3 student): 7/8 correct, all in the context-specified format
(`LOG(level="TRACE", code=1000/1001, msg="...")`). The single miss remains the familiar pattern:
"Heartbeat received from every worker" → `LOG(level="WARN", code=3000, ...)` (TRACE expected).

## The finding: the TRACE gap was a training-mix problem, and 3:1:1 fixes it

The teacher-fix round (v2) proved the teacher could be corrected in-context (ceiling 15→20/24, TRACE
0→7/8) but the student did not inherit it (stayed 15/24, TRACE 0/8) — an in-context improvement
that failed to become parametric. This round shows the missing ingredient was **gradient on the
low-severity class**, not teacher signal:

- TRACE oversampling (×3) moved the student from 0/8 → 7/8 on TRACE — the negative-class
  under-learning hypothesis is confirmed.
- WARN/FATAL did not regress — it *improved* to 16/16 (from 15/16). No trade-off; the larger,
  severity-balanced pool helped everywhere.
- The student now **beats the context-fed teacher**: 23/24 vs 20/24 (gap 1.15), format/code 1.0,
  KL 0.092 (closer to the improved ceiling than v2 was).
- Net journey: v1 15/24 → v2 15/24 (teacher-fix alone: no student gain) → v3 23/24
  (teacher-fix + up-weighting: +8).

The distilled student, with **no context at inference**, now matches the teacher's format exactly,
gets the severity mapping right on 23/24 held-out incidents (the strongest held-out result this
session), and does so via a 24 MB rank-16 adapter.

## Interpretation

Combined with all prior runs, the picture is now complete:
1. Distillation transfers teacher behavior with high fidelity (WARN/FATAL 15-16/16 across every
   adapter).
2. Low-severity (TRACE) under-learning was a **class-imbalance / negative-class** problem, not a
   capacity or teacher-signal problem: exemplars alone (v2) did nothing for the student; a 3:1:1
   training mix (v3) fixed it in one run.
3. The production recipe is now known: teacher few-shot exemplars (in the KV/context side for the
   teacher) + class-weighted distillation mix (in the adapter) — and the resulting adapter needs
   neither context nor KV at inference.

## Caveats

- n=24, greedy, bf16; the TRACE 7/8 and WARN+FATAL 16/16 splits are the strongest yet but still
  coarse-grained.
- 400 steps at weight 3 means ~10 epochs of the 40-item pool (vs ~16.7 of the 24-item pool before);
  the win is attributable to the mix, with step count roughly comparable.
- The single TRACE miss ("Heartbeat...") is the same over-escalation pattern; if it matters, a
  slightly higher weight (×4) or one more exemplar for "routine recurring signals" would likely
  close it.

## Artifacts

- `spike_context_distill_adapter_v3.pt` — the production candidate (24 MB)
- `spike_distill_template_v3_result.json` — full 24-query template-path results
- `spike_distill_result_v3.json` — v3 plain-path builtin eval
- `traceweight_v3_run.log` — full log (pool=40, TRACE=24; KL 1.25 → 0.017)

## Verdict

**GO — and the best result of the session.** Teacher exemplars + TRACE up-weighting produce a
distilled adapter that scores 23/24 held-out level accuracy with no context, no KV, no thinking
overhead, beating its own context-fed teacher. The context-distillation premise is now validated
end-to-end with a concrete production recipe. The natural next step is S1 (nanbeige-P2 build) —
the science side of the expert-bundle mechanism has converged.
