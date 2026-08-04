#!/usr/bin/env python
"""Template-path verification eval for the context-distillation spike (P19).

Answer to: does the distilled student work through the *production-style*
chat-template path, with thinking suppressed by the mechanism the model
actually supports (apply_chat_template(enable_thinking=False) — generate()
rejects the kwarg)?

Same 24 held-out queries as SPIKE-DISTILL-CONFIRM for direct comparison.
All prompts go through the tokenizer chat template:
  - system role carries CONTEXT (ceiling) or is absent (floor/student)
  - enable_thinking=False -> template renders the empty-think idiom
    ("<think>\\n\\n</think>"), which suppresses the model's own think emission
Budget is 384 new tokens, and every output is checked against the cap so a
truncation can never masquerade as a failure (reports hit_budget + max length).

Usage (from the 17-lab dir so .cuda_runtime_ld resolves):
  LD_LIBRARY_PATH="$(cat .cuda_runtime_ld)" CUDA_VISIBLE_DEVICES=1 \
    .venv-distill/bin/python -u spike_distill_template_eval.py [--smoke 2]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import spike_context_distill as scd
from spike_distill_confirm_eval import _CONFIRM_QUERIES

THINK_RE = __import__("re").compile(r"<think>.*?</think>", flags=__import__("re").S)
THINK_ALT_RE = __import__("re").compile(r"<\|think\|>.*?<\|/think\|>", flags=__import__("re").S)


def strip_think(text: str) -> str:
    return THINK_ALT_RE.sub("", THINK_RE.sub("", text))


def template_prompt(tok, incident, with_context, enable_thinking=False):
    """Chat-template prompt. Context goes in the system role when requested."""
    body = scd.INSTRUCTION.format(q=incident)
    conv = []
    if with_context:
        conv.append({"role": "system", "content": scd.CONTEXT})
    conv.append({"role": "user", "content": body})
    return tok.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False,
        enable_thinking=enable_thinking)


def generate_ids(model, tok, prompt, device, max_new_tokens):
    enc = scd.encode(tok, prompt, device)
    prompt_len = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        num_beams=1, pad_token_id=tok.pad_token_id)
    new_ids = gen[0, prompt_len:]
    return tok.decode(new_ids, skip_special_tokens=True), new_ids


@torch.no_grad()
def token_kl_template(model, tok, incident, target_ids, device,
                      with_context, lora_on, enable_thinking=False):
    """token_kl (from scd) but with chat-template prompts."""
    tgt = target_ids.to(device)
    if tgt.numel() == 0:
        return None
    scd.set_lora_enabled(model, False)
    t_prompt = template_prompt(tok, incident, True, enable_thinking)
    t_ids = scd.encode(tok, t_prompt, device)["input_ids"]
    t_full = torch.cat([t_ids, tgt.unsqueeze(0)], dim=1)
    t_logits = scd._logits(model, t_full, torch.ones_like(t_full))
    tp = t_ids.shape[1]
    t_pred = t_logits[0, tp - 1: tp - 1 + tgt.numel(), :].float()

    scd.set_lora_enabled(model, lora_on)
    c_prompt = template_prompt(tok, incident, with_context, enable_thinking)
    c_ids = scd.encode(tok, c_prompt, device)["input_ids"]
    c_full = torch.cat([c_ids, tgt.unsqueeze(0)], dim=1)
    c_logits = scd._logits(model, c_full, torch.ones_like(c_full))
    cp = c_ids.shape[1]
    c_pred = c_logits[0, cp - 1: cp - 1 + tgt.numel(), :].float()

    t_logp = torch.log_softmax(t_pred, dim=-1)
    c_logp = torch.log_softmax(c_pred, dim=-1)
    kl = (t_logp.exp() * (t_logp - c_logp)).sum(-1).mean()
    return float(kl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit held-out to N queries (local verification)")
    ap.add_argument("--adapter", default=str(
        Path(__file__).parent / "spike_context_distill_adapter.pt"))
    ap.add_argument("--out", default=str(
        Path(__file__).parent / "spike_distill_template_result.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, tok = scd.load_model_and_tokenizer(torch.bfloat16)
    model.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s on {device}")

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

    held = _CONFIRM_QUERIES if not args.smoke else _CONFIRM_QUERIES[:args.smoke]
    print(f"[eval] {len(held)} held-out queries via chat template "
          f"(enable_thinking=False), max_new_tokens={args.max_new_tokens}")

    scd.set_lora_enabled(model, False)
    teacher_targets = []
    for incident, _lvl in held:
        _text, ids = generate_ids(
            model, tok, template_prompt(tok, incident, True), device,
            args.max_new_tokens)
        teacher_targets.append(ids)

    configs = {
        "floor_base_noctx":   dict(with_context=False, lora_on=False),
        "ceiling_base_ctx":   dict(with_context=True,  lora_on=False),
        "student_lora_noctx": dict(with_context=False, lora_on=True),
    }
    report = {}
    for name, cfg in configs.items():
        scd.set_lora_enabled(model, cfg["lora_on"])
        scores, kls, samples = [], [], []
        think_count, hit_budget, max_out = 0, 0, 0
        for (incident, lvl), tgt in zip(held, teacher_targets):
            prompt = template_prompt(tok, incident, cfg["with_context"])
            raw, ids = generate_ids(model, tok, prompt, device, args.max_new_tokens)
            max_out = max(max_out, len(ids))
            hit_budget += int(len(ids) >= args.max_new_tokens)
            think_count += int(bool(THINK_RE.search(raw) or THINK_ALT_RE.search(raw)))
            scored = strip_think(raw)
            scores.append(scd.score_line(scored, lvl))
            kl = token_kl_template(model, tok, incident, tgt, device,
                                   cfg["with_context"], cfg["lora_on"])
            if kl is not None:
                kls.append(kl)
            samples.append({
                "incident": incident, "expected": lvl,
                "had_think": bool(THINK_RE.search(raw) or THINK_ALT_RE.search(raw)),
                "out_tokens": len(ids),
                "out_stripped": scored.strip(), "out_raw": raw.strip(),
            })
        report[name] = {
            "task": scd.aggregate(scores),
            "mean_token_kl_vs_ceiling": round(sum(kls) / len(kls), 4) if kls else None,
            "think_in_output": think_count,
            "hit_budget": hit_budget,
            "max_out_tokens": max_out,
            "samples": samples if args.smoke else samples[:8],
        }
        print(f"  {name}: task={report[name]['task']} "
              f"kl={report[name]['mean_token_kl_vs_ceiling']} "
              f"think={think_count} hit_budget={hit_budget} max_out={max_out}")

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
        "n_held_out": len(held),
        "max_new_tokens": args.max_new_tokens,
        "any_hit_budget": any(r["hit_budget"] for r in report.values()),
        "level_correct_gap_recovered": gap("level_correct"),
        "format_ok_gap_recovered": gap("format_ok"),
        "code_prefix_ok_gap_recovered": gap("code_prefix_ok"),
        "student_kl_below_floor_kl": (
            stud["mean_token_kl_vs_ceiling"] is not None
            and floor["mean_token_kl_vs_ceiling"] is not None
            and stud["mean_token_kl_vs_ceiling"] < floor["mean_token_kl_vs_ceiling"]
        ),
    }
    out = {"config": vars(args), "report": report, "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}")
    print(f"[verdict] {json.dumps(verdict, indent=2)}")


if __name__ == "__main__":
    main()
