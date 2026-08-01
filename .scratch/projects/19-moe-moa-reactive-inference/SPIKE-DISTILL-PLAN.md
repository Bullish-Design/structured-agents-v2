# Spike plan — context distillation for a zero-KV expert bundle (Project 19)

Tests CONCEPT.md §6.5.2 option (2): can a small LoRA **internalize a fixed
context** so that at inference the *adapter-only* model (context removed from the
prompt) behaves like the *base* model *with* that context in its prompt? If yes,
an "expert bundle" hat can carry its knowledge in **weights — zero KV at
inference**, dissolving both KV-consistency traps of option (1) and the P4
"splice must beat re-prefill" burden for that class of bundle (§6.5.2/§6.5.3).

Script: `spike_context_distill.py` (self-contained; pure-torch LoRA, no `peft`).

## Design

Method follows Snell et al. 2022, "Learning by Distilling Context"
(arXiv:2209.15189), with the compression framing of Mu et al. 2023, "Learning to
Compress Prompts with Gist Tokens" (arXiv:2304.08467).

- **Teacher:** base model, LoRA delta OFF, **CONTEXT in the prompt**. Greedy
  (do_sample=False) continuation gives the target token ids per training query.
- **Student:** same model instance, LoRA delta ON, **CONTEXT removed** from the
  prompt. One model in VRAM; a per-module `scaling_enabled` flag toggles the
  low-rank delta so teacher and student share the frozen base.
- **Objective:** token-level `KL(teacher || student)` on next-token logits at
  exactly the positions that predict each teacher-forced target token. Teacher
  and student have different prefixes (context present vs absent) but predict the
  same target tokens, so alignment is on the output positions only; teacher logits
  are detached (fixed target). `reduction="batchmean"`, temperature 1.0.
- **LoRA:** rank 16, alpha 32, dropout 0.05, injected into
  `q_proj,v_proj,gate_proj,up_proj,down_proj` (84 linears, ~5.95M trainable
  params on this model). Base frozen; AdamW over LoRA params only, lr 2e-4, grad
  clip 1.0. Adapters kept fp32 for stable optimization; B init zero so delta=0 at
  start.

### Synthetic context chosen

A compact (~180-token) self-contained fictional logging protocol — the "ACME
event logger" — deliberately **non-standard** so the base cannot know it a priori
(genuinely low floor). Valid levels are exactly `TRACE / WARN / FATAL` (no
INFO/ERROR/DEBUG), with arbitrary code ranges (TRACE 1000-1999, WARN 3000-3999,
FATAL 9000-9999) and a fixed one-line output format
`LOG(level="<LEVEL>", code=<CODE>, msg="<=8 words>")`. This is exactly the
style/procedure/small-facts regime §6.5.3 says distills well, and success is
crisply measurable (regex format + level accuracy). No external download.

### Query set

30 in-code `(incident, expected_level)` pairs spanning the three levels. Split:
first 24 = **train** (teacher continuations distilled), last 6 = **held-out**
(never seen in training; all eval numbers are on these). Smoke uses train[:6],
held[:2].

### Metrics (per config, on held-out)

Three configs evaluated against a common teacher reference (base+ctx greedy):
- **floor** = base, NO context (what you get for free)
- **ceiling** = base, WITH context (what distillation tries to reach; KL 0 by def.)
- **student** = LoRA, NO context (the thing under test)

Reported per config: task dict `{format_ok, level_valid, level_correct,
code_prefix_ok}` (regex on the emitted LOG line) and `mean_token_kl_vs_ceiling`
(per-token KL of that config's logits vs the base+ctx teacher, on the teacher's
own continuation). Verdict adds `*_gap_recovered = (student-floor)/(ceiling-floor)`
and `student_kl_below_floor_kl`.

## Smoke-test result (2026-07-31, GPU 1)

Ran clean end-to-end on GPU 1 (CUDA_VISIBLE_DEVICES=1; GPU 0 untouched). Imports
OK, base loads on GPU in ~1.7s as `Qwen3_5ForCausalLM`, LoRA injected into 84
linears (5.95M params), 4 distill steps run (KL 1.25 -> 0.68), eval harness
produced the three numbers:

| config  | mean_token_kl_vs_ceiling | task |
|---------|--------------------------|------|
| floor (base, no-ctx)    | **1.677** | all 0.0 |
| ceiling (base, +ctx)    | **0.000** | all 0.0 |
| student (LoRA, no-ctx)  | **0.266** | all 0.0 |

`student_kl_below_floor_kl = true` — even after only 4 steps the student's logit
distribution has moved most of the way from floor (1.68) toward the ceiling (0.0),
i.e. the KL objective is wired correctly and pushing in the right direction. This
is the primary smoke signal and it is green.

Task-accuracy fields are all 0.0 at smoke scale — expected and NOT a blocker:
only 4 steps, `max_new_tokens=16` (truncates the LOG line), and, notably, the
tiny model's natural emission is `LOG(TRACE, 1000, "...")` while the scoring regex
requires the context-specified `LOG(level="TRACE", code=1000, msg="...")`. So even
the ceiling scores format_ok=0 here. Watch this in the full run (see risks).

### Fixes made (minimal; no redesign)

1. **Wrong venv in the docstring.** The reused `.venv-spike` ships transformers
   4.57.6, which does not recognize this checkpoint's `qwen3_5` architecture (hard
   `ValueError` on load). The working env is the project-19 **`.venv-distill`**
   (transformers 5.13.1 + torch 2.13.0/cu130), created by the prior agent
   alongside the script. Updated the docstring run commands to use `.venv-distill`.
2. **triton C-compiler for the rotary op.** torch 2.13's `qwen3_5` rotary path
   JIT-compiles a triton kernel (`bmm_outer_product`) and needs a C compiler;
   NixOS has no bare `cc` on PATH. Added CC/CXX auto-detection to the script's
   bootstrap (picks the newest `/nix/store/*gcc-wrapper-*/bin/gcc`). The canonical
   command below now runs with no manual `CC=` needed.

No changes to the distillation logic, objective, context, queries, or metrics.

## Full run — canonical command

Run on the free GPU (GPU 1). GPU 0 is left alone. From the 17-lab dir so
`.cuda_runtime_ld` resolves:

```
cd /home/andrew/Documents/Projects/structured-agents-v2/.scratch/projects/17-llama-cpp-inference-lab
LD_LIBRARY_PATH="$(cat .cuda_runtime_ld)" CUDA_VISIBLE_DEVICES=1 \
  ../19-moe-moa-reactive-inference/.venv-distill/bin/python \
  ../19-moe-moa-reactive-inference/spike_context_distill.py \
  --steps 400 --rank 16 --alpha 32 --lr 2e-4 --max-new-tokens 48
```

Writes `spike_context_distill_result.json` (full report + verdict) and
`spike_context_distill_adapter.pt` (the trained adapter) in the project-19 dir.
Single tiny model + rank-16 adapter on one 3060 — comfortably within 12 GB.

## Go / no-go criteria

Judged on the **held-out** queries after the full run:

- **GO** if `level_correct_gap_recovered >= 0.5` (student closes at least half the
  floor->ceiling task gap) AND `student_kl_below_floor_kl` is true (student logits
  are closer to the base+ctx teacher than the no-context floor is). A strong GO is
  gap_recovered >= 0.8 with student mean_token_kl within ~0.1 of ceiling.
- **NO-GO** if the student stays near the floor (gap_recovered < ~0.2) or KL does
  not drop below floor — rank-16 could not internalize this context; revisit rank,
  target modules, step count, or fall back to the KV-prefix bundle (option 1).

## Risks / watch-items for the full run

- **Format-metric vs emitted format.** The scoring regex expects
  `LOG(level="...", code=..., msg="...")` but the model naturally emits
  `LOG(TRACE, 1000, ...)`. If the ceiling still scores format_ok=0 at 48 tokens,
  the task metrics are uninformative and the **KL numbers become the sole signal**
  (they are the more fundamental measure anyway). Optionally tighten the CONTEXT's
  format instruction or relax the regex — a scoring tweak, not a redesign.
- **Verdict divisions.** `gap_recovered` returns null when ceiling==floor on a
  metric (as in smoke). Rely on the KL-based verdict if task metrics collapse.
- **Capacity ceiling (§6.5.3).** rank-16 fits style/procedure/small-facts; if it
  underfits, that is itself an informative result about bundle capacity.
