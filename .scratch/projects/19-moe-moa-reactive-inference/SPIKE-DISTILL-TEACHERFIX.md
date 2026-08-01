# SPIKE-DISTILL-TEACHERFIX — Teacher few-shot fix + re-distill + re-eval

Date: 2026-07-31 · Adapter: `spike_context_distill_adapter_v2.pt` (rank-16 LoRA, 400 KL steps, trained
against the teacher-fixed CONTEXT) · v1 baseline: `spike_context_distill_adapter_v1.pt`
Results: `spike_distill_result_v2.json` (plain path, builtin 6-query eval) ·
`spike_distill_template_v2_result.json` (template path, 24 queries) · Report: this file

## What changed (the teacher fix)

CONTEXT gained 5 few-shot exemplars (3 TRACE / 1 WARN / 1 FATAL), e.g.
`"A user logged out of the console." -> LOG(level="TRACE", code=1001, msg="User logged out")`.
Incidents chosen to not overlap the train/held-out query sets. Everything else (training procedure,
adapter config, eval scripts) unchanged. The old adapter was snapshotted as v1.

## Results — template path, same 24 queries, v1 vs v2 (scored with the format-agnostic regex)

| config | v1 total | v2 total | v1 TRACE | v2 TRACE | v1 WARN+FATAL | v2 WARN+FATAL | v1 KL | v2 KL |
|---|---|---|---|---|---|---|---|---|
| floor | 0/24 | 0/24 | 0/8 | 0/8 | 0/16 | 0/16 | 1.75 | 1.84 |
| ceiling (teacher) | 15/24 | **20/24** | 0/8 | **7/8** | 15/16 | 13/16 | 0.0 | 0.0 |
| student (LoRA, no ctx) | 15/24 | 15/24 | 0/8 | 0/8 | 15/16 | 15/16 | 0.095 | 0.125 |

## The finding: the teacher fix worked — and did NOT transfer to the student

**Teacher side: fixed.** The exemplars in the system-role CONTEXT cured the template-path teacher's
TRACE over-escalation: TRACE 0/8 → 7/8, total 15/24 → 20/24, format 0.958 → 1.0. The v2 CONTEXT is
demonstrably active (this improvement only comes from the exemplars). Cost: WARN/FATAL dipped
15/16 → 13/16 — a small trade, net +5.

**Student side: completely unchanged.** Same 15/24, same 0/8 TRACE (all eight still
`LOG(level="WARN", code=3000, ...)`), same 15/16 WARN+FATAL. The student's behavior is *bit-for-bit
identical* to v1's profile. Its KL vs the (now better) ceiling even worsened slightly (0.095 → 0.125).

This is a clean negative result with a sharp implication: **in-context teacher improvements do not
automatically become parametric in the distilled student.** The v2 teacher provided correct TRACE
targets during training (the plain-path builtin eval shows the v2 teacher at 6/6 level_correct), yet
the rank-16 student still emits WARN for every routine event. The student's TRACE failure is a
**learning-side limitation, not a teacher-signal problem** — the signal was there and was not
absorbed.

## Why TRACE specifically fails to distill

Consistent with every run this session, the student nails WARN/FATAL (15/16, three different
teachers/adapters) but never learns TRACE (0/8 on the template path in v1 and v2; 4/8 was the best,
on the v1 plain path). Working hypothesis: TRACE is the "negative class" (don't escalate) and the
base model's severity prior is high; KL distillation with a balanced mix and 400 steps cannot flip
that prior for the low-severity class, while the high-severity classes get learned easily. The
exemplars prove the *teacher* can be corrected in-context; the *student* needs a training-side fix.

## Candidate fixes (ranked)

1. **TRACE up-weighting in the training mix** (the original SPIKE-DISTILL-CONFIRM recommendation):
   oversample TRACE queries in the distill loop (e.g., 3:1:1) so the low-severity class gets more
   gradient. Directly tests the "negative-class under-learning" hypothesis. Cheap: same script with a
   weighting tweak.
2. **More steps / higher rank** (rank 16 → 32) if under-learning is capacity-driven rather than
   class-driven; weaker hypothesis since WARN/FATAL fit fine in rank 16.
3. **Accept the split**: the student is already excellent on the classes that matter (WARN/FATAL
   15/16); TRACE over-escalation is a bounded, well-characterised 8/24 failure. In production,
   TRACE detection could stay on the context/KV side (few-shot exemplars are free at inference via
   the KV blob) while the adapter carries the procedural high-severity skill — a concrete instance
   of the concept doc's "stable → weights, volatile → KV" split.

## Artifacts

- `spike_context_distill_adapter_v1.pt` — original adapter (pre-fix), 24 MB
- `spike_context_distill_adapter_v2.pt` — re-distilled against teacher-fixed CONTEXT, 24 MB
- `spike_distill_result_v2.json` — v2 plain-path builtin eval (teacher 6/6, student 2/6)
- `spike_distill_template_v2_result.json` — v2 template-path 24-query eval (this report's table)
- `teacherfix_v2_run.log` — full run log (KL convergence 0.04 → 0.02)

## Verdict

Teacher fix: **GO** (20/24, TRACE 7/8, format 1.0 — the exemplar approach works in-context).
Distillation transfer of the fix: **NO-GO** (student unchanged at 15/24, TRACE 0/8) — the few-shot
benefit does not become parametric under the current training setup. The distillation premise itself
remains validated (high-fidelity transfer of WARN/FATAL); this run bounds its reach: it transfers
what the model can learn, not merely what the teacher demonstrates. Next step: TRACE up-weighting
(fix 1) or the production split (fix 3).
