#!/usr/bin/env python
"""Follow-up eval for the context-distillation spike (Project 19).

Tests whether the distilled student produces the *behaviour* (a valid LOG
line) once the two confounds from the full 400-step run are removed:
  1. thinking tokens ("<think>...</think>") are stripped / disabled, and
  2. the generation budget is raised past the reason-then-log preamble.

Reuses spike_context_distill's model loading, LoRA injection, prompts,
scoring regex and KL metrics. Loads the trained adapter
(spike_context_distill_adapter.pt) into a freshly injected rank-16 LoRA and
evaluates floor / ceiling / student on the held-out queries.

Usage (from the 17-lab dir so .cuda_runtime_ld resolves):
  LD_LIBRARY_PATH="$(cat .cuda_runtime_ld)" CUDA_VISIBLE_DEVICES=1 \
    .venv-distill/bin/python -u spike_distill_followup_eval.py --max-new-tokens 192
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

import spike_context_distill as scd

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)
THINK_ALT_RE = re.compile(r"<\|think\|>.*?<\|/think\|>", flags=re.S)


def strip_think(text: str) -> str:
    return THINK_ALT_RE.sub("", THINK_RE.sub("", text))


def generate_full(model, tok, prompt, device, max_new_tokens):
    """Greedy continuation; returns the FULL decoded text (not first line)."""
    enc = scd.encode(tok, prompt, device)
    prompt_len = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    new_ids = gen[0, prompt_len:]
    return tok.decode(new_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--adapter", default=str(
        Path(__file__).parent / "spike_context_distill_adapter.pt"))
    ap.add_argument("--out", default=str(
        Path(__file__).parent / "spike_distill_followup_result.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, "bfloat16")
    t0 = time.time()
    model, tok = scd.load_model_and_tokenizer(dtype)
    model.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s on {device}")

    # Try to disable thinking at the generation-config level (guarded; the
    # strip at scoring time covers the case where it is not honoured).
    try:
        gc = getattr(model, "generation_config", None)
        if gc is not None and hasattr(gc, "enable_thinking"):
            gc.enable_thinking = False
            print("[gen] enable_thinking=False set (config-supported)")
    except Exception as e:  # noqa: BLE001
        print(f"[gen] could not set enable_thinking: {e!r}")

    replaced = scd.inject_lora(
        model, ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        16, 32, 0.05)
    print(f"[lora] injected into {len(replaced)} linears")

    sd = torch.load(args.adapter, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[adapter] loaded {len(sd)} tensors; missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {list(unexpected)[:5]}")

    held = scd._QUERIES[scd.HELD_OUT_START:]
    print(f"[eval] {len(held)} held-out queries, "
          f"max_new_tokens={args.max_new_tokens}, think-stripped scoring")

    # Teacher reference continuations (base + ctx, LoRA off).
    scd.set_lora_enabled(model, False)
    teacher_targets = []
    for incident, _lvl in held:
        _, ids = scd.generate(
            model, tok, scd.build_prompt(tok, incident, True),
            device, args.max_new_tokens)
        teacher_targets.append(ids)

    configs = {
        "floor_base_noctx":   dict(with_context=False, lora_on=False),
        "ceiling_base_ctx":   dict(with_context=True,  lora_on=False),
        "student_lora_noctx": dict(with_context=False, lora_on=True),
    }
    report = {}
    for name, cfg in configs.items():
        scd.set_lora_enabled(model, cfg["lora_on"])
        scores, kls, samples, think_count = [], [], [], 0
        for (incident, lvl), tgt in zip(held, teacher_targets):
            prompt = scd.build_prompt(tok, incident, cfg["with_context"])
            raw = generate_full(model, tok, prompt, device, args.max_new_tokens)
            had_think = bool(THINK_RE.search(raw) or THINK_ALT_RE.search(raw))
            think_count += int(had_think)
            scored_text = strip_think(raw)
            scores.append(scd.score_line(scored_text, lvl))
            kl = scd.token_kl(model, tok, incident, tgt, device,
                              cfg["with_context"], cfg["lora_on"])
            if kl is not None:
                kls.append(kl)
            samples.append({
                "incident": incident, "expected": lvl, "had_think": had_think,
                "out_stripped": scored_text.strip(), "out_raw": raw.strip(),
            })
        report[name] = {
            "task": scd.aggregate(scores),
            "mean_token_kl_vs_ceiling": round(sum(kls) / len(kls), 4) if kls else None,
            "think_in_output": think_count,
            "samples": samples[:6],
        }
        print(f"  {name}: task={report[name]['task']} "
              f"kl={report[name]['mean_token_kl_vs_ceiling']} "
              f"think={think_count}")

    floor = report["floor_base_noctx"]
    ceil = report["ceiling_base_ctx"]
    stud = report["student_lora_noctx"]

    def gap(metric):
        f = floor["task"].get(metric)
        c = ceil["task"].get(metric)
        s = stud["task"].get(metric)
        if None in (f, c, s) or c == f:
            return None
        return round((s - f) / (c - f), 3)

    verdict = {
        "level_correct_gap_recovered": gap("level_correct"),
        "format_ok_gap_recovered": gap("format_ok"),
        "code_prefix_ok_gap_recovered": gap("code_prefix_ok"),
        "student_kl_below_floor_kl": (
            stud["mean_token_kl_vs_ceiling"] is not None
            and floor["mean_token_kl_vs_ceiling"] is not None
            and stud["mean_token_kl_vs_ceiling"] < floor["mean_token_kl_vs_ceiling"]
        ),
        "max_new_tokens": args.max_new_tokens,
    }
    out = {"config": vars(args), "report": report, "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}")
    print(f"[verdict] {json.dumps(verdict, indent=2)}")


if __name__ == "__main__":
    main()
