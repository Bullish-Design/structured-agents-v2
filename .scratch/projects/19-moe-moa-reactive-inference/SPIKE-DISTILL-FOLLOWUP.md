# SPIKE-DISTILL-FOLLOWUP — Eval with think-stripped scoring + 192-token budget

Date: 2026-07-31 · Script: `spike_distill_followup_eval.py --max-new-tokens 192` (unmodified, run exactly as specified)
Adapter: `spike_context_distill_adapter.pt` (rank-16 LoRA, 400 KL steps) · Base: Project-17 snapshot, bf16, GPU 1 (CUDA_VISIBLE_DEVICES=1 honoured)
Result: `spike_distill_followup_result.json` · n = 6 held-out queries (indices 24–29) · greedy decoding

The run completed cleanly in ~60 s (no crash, no retry needed). Peak VRAM stayed well under the ~5 GB estimate.

---

## 1. Task metrics per config (think-stripped scoring, 192 new tokens)

| config | format_ok | level_valid | level_correct | code_prefix_ok | mean_token_kl_vs_ceiling | think_in_output |
|---|---|---|---|---|---|---|
| floor_base_noctx | 0.0 | 0.0 | 0.0 | 0.0 | 0.2553 | 0 |
| ceiling_base_ctx | 0.833 | 0.833 | 0.333 | 0.833 | 0.0 | 2 |
| student_lora_noctx | **1.0** | **1.0** | **0.667** | **1.0** | **0.0191** | 6 |

`floor` = base model, no context, no LoRA · `ceiling` = base model + context, LoRA off (the teacher) · `student` = base + trained LoRA, no context.

Notable: the student **exceeds the ceiling on every task metric**. Ceiling level_correct is 2/6; student is 4/6.

## 2. Verdict fields (from JSON)

| field | value | meaning |
|---|---|---|
| level_correct_gap_recovered | **2.003** | (0.667−0)/(0.333−0) — >1.0 because the student *beats* the ceiling, not just approaches it |
| format_ok_gap_recovered | **1.2** | (1.0−0)/(0.833−0) |
| code_prefix_ok_gap_recovered | **1.2** | (1.0−0)/(0.833−0) |
| student_kl_below_floor_kl | **true** | student KL 0.0191 < floor KL 0.2553 |

All three gap metrics clear the GO bar (≥0.5) by a wide margin; `student_kl_below_floor_kl` is true.

## 3. Representative outputs (raw vs think-stripped)

**floor_base_noctx** — never produces a LOG line; echoes the incident or spams digits.
- *"A user updated their profile picture."* (TRACE) →
  raw = stripped = `"User profile picture updated successfully."` repeated ~18×, no LOG anywhere.
- *"The message broker crashed..."* (FATAL) → raw = stripped = `1000000000000…` (digit spam, no LOG).
- *"Total network partition..."* (FATAL) → raw = stripped = `"Total network partition isolated the datacenter"\nSeverity: Critical\nDescription: …` (prose, no LOG).

**ceiling_base_ctx** — emits real LOG lines; 2/6 start with an empty think block; 2/6 levels correct; 1/6 is a literal template.
- *"A user updated their profile picture."* (TRACE, had_think) →
  raw: `<think>\n\n</think>\n\nLOG(WARN, 3000, "User profile picture updated")\nLOG(WARN, 3000, …` (repeats)
  stripped: `LOG(WARN, 3000, "User profile picture updated")` — wrong level (WARN, expected TRACE).
- *"Garbage collection ran for 40 milliseconds."* (TRACE) →
  raw = stripped: `LOG(TRACE, 1000, "Garbage collection ran for 40 milliseconds.")` — **correct**.
- *"Total network partition..."* (FATAL) →
  raw = stripped: `LOG(level="<LEVEL>", code=<CODE>, msg="<short summary>")` repeated — literal template, format fail.

**student_lora_noctx** — the whole point: 6/6 emit exactly one clean LOG line; all 6 carry an empty think block (stripped below); 4/6 levels correct.
- *"Garbage collection ran for 40 milliseconds."* (TRACE, had_think) →
  raw: `<think>\n\n</think>\n\nLOG(WARN, 3000, "Garbage collection ran for 40 milliseconds")`
  stripped: `LOG(WARN, 3000, "Garbage collection ran for 40 milliseconds")` — wrong level (WARN, expected TRACE).
- *"The message broker crashed and will not restart."* (FATAL, had_think) →
  raw: `<think>\n\n</think>\n\nLOG(FATAL, 9000, "Message broker crashed and will not restart")`
  stripped: `LOG(FATAL, 9000, "Message broker crashed and will not restart")` — **correct; teacher got this one wrong (TRACE)**.
- *"Total network partition isolated the datacenter."* (FATAL, had_think) →
  raw: `assistant\n<think>\n\n</think>\n\nLOG(FATAL, 9000, "Network partition isolated datacenter")`
  stripped: `assistant\n\n\nLOG(FATAL, 9000, "Network partition isolated datacenter")` — **correct; teacher emitted the literal `<LEVEL>` template here**.

Student error pattern (2 misses): both held-out TRACE incidents map to WARN. The student is WARN-biased on mundane events, yet it *fixed* two of the teacher's four held-out errors (broker → real FATAL, partition → real FATAL instead of template) while regressing one the teacher had right (GC → WARN). The student is not copying teacher noise — it is a genuinely different, better-on-average mapping. 2/6 student outputs also carry a stray `assistant` role prefix before the LOG line (harmless to the regex search, unclean for production).

## 4. Interpretation — does the student reach/near ceiling with think stripped?

**Confirmed: the "distillation works, eval was confounded" hypothesis is correct.** Compare runs:

| metric | full run (48 tok, first-line, raw) | followup (192 tok, think-stripped) |
|---|---|---|
| student format_ok | 0.0 | **1.0** |
| student level_correct | 0.0 | **0.667** |
| student KL | 0.040 | **0.019** |
| ceiling format_ok | 0.5 | 0.833 |
| ceiling level_correct | 0.167 | 0.333 |

At 48 tokens the student's output was truncated inside the generation preamble (`"assistant"`, `"<think>"`) — the distilled LOG behaviour was there all along but sat past the budget. With 192 tokens and think-stripped scoring, the student hits format_ok / level_valid / code_prefix_ok = 1.0 and level_correct 0.667 — i.e. it **matches or exceeds the teacher** while receiving no context at inference. Gap recovery of 2.0× on level_correct is the single strongest signal: rank-16 LoRA on 5.9M params internalised the protocol from logits alone.

Caveat on the KL column: the followup regenerated teacher targets at 192 tokens (full repeated LOG lines), so both KL values shifted relative to the 48-token run (floor 0.906→0.255, student 0.040→0.019). Cross-run KL is not apples-to-apples; the valid comparison is within-run, and there the ordering student (0.019) << floor (0.255) holds decisively.

## 5. Caveats

1. **enable_thinking was NOT honoured.** The script tries `generation_config.enable_thinking = False`, but no `[gen]` line appears in the log — the attribute does not exist on this Qwen3.5 generation config (thinking is a per-`generate()` kwarg in the Qwen3 family). All 6 student outputs and 2/6 teacher outputs still contain `<think>\n\n</think>`; scoring strips them, so the *metrics* are clean, but the student still wastes ~3 tokens per call at inference. A real deployment must pass `enable_thinking=False` to `generate()`, not rely on config or post-hoc stripping.
2. **Budget: fixed, not masked.** 192 tokens is more than enough (the student's LOG line lands within the first ~30 tokens). But the improved ceiling numbers (0.5→0.833) come partly from the *scoring* change (regex search over full stripped text vs first line) plus the budget. Residual ceiling misses are semantic (wrong level, literal template), not truncation — the teacher's ceiling is real and weak.
3. **The teacher itself fails.** Base + full context gets only 2/6 held-out levels right, emits the literal `LOG(level="<LEVEL>", code=<CODE>, …)` template on one incident, and outputs TRACE for a FATAL crash. A weak teacher limits what distillation can transfer — and yet the student still outperforms it, which cuts both ways: encouraging for the premise, but it means "ceiling" is a low bar.
4. **n=6, greedy, bf16.** The entire verdict rests on 6 queries. 4/6 vs 2/6 (student vs teacher) is two flips away from parity; `gap_recovered > 1.0` values saturate the GO bar by exceeding the ceiling, not by approaching it. Variance is real.

---

## Verdict

**GO** for the context-distillation premise. With the two confounds removed — think-token truncation at 48 tokens and first-line-only scoring — a rank-16 LoRA student with *no context at inference* reaches format_ok 1.0 / level_correct 0.667 and KL 0.019, exceeding the context-fed teacher (0.833 / 0.333 / 0.0) on every task metric, with gap recovery 1.2–2.0× and student KL well below floor. The 48-token all-zero student result was an evaluation artifact, not a distillation failure. The single most informative next step: re-run the eval on a larger held-out set (n ≥ 24) with `generate(enable_thinking=False)` passed per-call — this both confirms the 4/6 level_correct signal with real confidence intervals and removes the hidden think-token cost that strip-at-score currently masks; if the TRACE→WARN bias persists at scale, follow with a training mix that up-weights low-severity incidents.
