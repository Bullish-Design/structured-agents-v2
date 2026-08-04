#!/usr/bin/env python
"""Confirmation eval for the context-distillation spike (Project 19).

Follow-up to SPIKE-DISTILL-FOLLOWUP. Two changes per the report's
recommendation:
  1. held-out set enlarged from 6 to 24 fresh incidents (8 TRACE / 8 WARN /
     8 FATAL, all distinct from the 30 used in training + follow-up);
  2. thinking disabled per generate() call (enable_thinking=False) instead of
     stripped after the fact — Qwen3-family models take it as a per-generate
     kwarg, not a generation_config attribute.

If the kwarg is not accepted (TypeError/ValueError — smoke-tested: this model
raises ValueError, `enable_thinking` is unused by Qwen3.5 in transformers
5.13) it falls back to plain generation and strips think blocks at scoring
time; the report flags which path applied and quantifies think-token overhead.

Usage (from the 17-lab dir so .cuda_runtime_ld resolves):
  LD_LIBRARY_PATH="$(cat .cuda_runtime_ld)" CUDA_VISIBLE_DEVICES=1 \
    .venv-distill/bin/python -u spike_distill_confirm_eval.py [--smoke 2]
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

# 24 fresh held-out incidents (never seen in training). Balanced 8/8/8.
_CONFIRM_QUERIES = [
    # TRACE (routine / informational)
    ("The log rotation job completed on schedule.", "TRACE"),
    ("A new user account was created.", "TRACE"),
    ("The nightly report was generated successfully.", "TRACE"),
    ("Cache hit ratio stayed above 99 percent.", "TRACE"),
    ("A heartbeat was received from every worker.", "TRACE"),
    ("The configuration file was validated successfully.", "TRACE"),
    ("A fresh backup snapshot was taken.", "TRACE"),
    ("The service began listening on its primary port.", "TRACE"),
    # WARN (recoverable / degraded)
    ("DNS resolution for a downstream service started timing out.", "WARN"),
    ("The connection pool reached 90 percent utilization.", "WARN"),
    ("An upstream payment service returned errors for a few requests.", "WARN"),
    ("Disk latency spiked above the warning threshold.", "WARN"),
    ("A client secret is due to rotate in three days.", "WARN"),
    ("Memory usage is trending upward across the fleet.", "WARN"),
    ("A worker reported slightly elevated packet loss.", "WARN"),
    ("Retry attempts exceeded the soft threshold for one batch.", "WARN"),
    # FATAL (unrecoverable / outage / data loss)
    ("The master key for the database was lost.", "FATAL"),
    ("The cluster lost quorum and went read-only.", "FATAL"),
    ("A hardware failure destroyed the RAID array.", "FATAL"),
    ("The core service crashed and cannot be restarted.", "FATAL"),
    ("The TLS private key is corrupted and cannot be loaded.", "FATAL"),
    ("The replication log is unreadable after a crash.", "FATAL"),
    ("A power surge fried the storage controller.", "FATAL"),
    ("The audit trail is missing and cannot be recovered.", "FATAL"),
]


def strip_think(text: str) -> str:
    return THINK_ALT_RE.sub("", THINK_RE.sub("", text))


def think_blocks(text: str) -> list:
    """All think-block strings found in text (both marker styles)."""
    return (THINK_RE.findall(text) + THINK_ALT_RE.findall(text))


def think_token_count(tok, text: str) -> int:
    """Total tokenizer tokens inside think blocks of text."""
    blocks = think_blocks(text)
    if not blocks:
        return 0
    joined = " ".join(blocks)
    return len(tok(joined)["input_ids"]) if joined.strip() else 0


def generate_no_think(model, tok, prompt, device, max_new_tokens):
    """Greedy continuation with thinking disabled per call. Falls back to
    plain generation if the kwarg is unsupported. Returns (text, ids, ok)."""
    enc = scd.encode(tok, prompt, device)
    prompt_len = enc["input_ids"].shape[1]
    try:
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            enable_thinking=False,
        )
        no_think_ok = True
    except (TypeError, ValueError):
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
        )
        no_think_ok = False
    new_ids = gen[0, prompt_len:]
    return tok.decode(new_ids, skip_special_tokens=True), new_ids, no_think_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit held-out to N queries (local verification)")
    ap.add_argument("--adapter", default=str(
        Path(__file__).parent / "spike_context_distill_adapter.pt"))
    ap.add_argument("--out", default=str(
        Path(__file__).parent / "spike_distill_confirm_result.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, "bfloat16")
    t0 = time.time()
    model, tok = scd.load_model_and_tokenizer(dtype)
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
    print(f"[eval] {len(held)} held-out queries, "
          f"max_new_tokens={args.max_new_tokens}, no-think generation")

    # Teacher reference (base + ctx, LoRA off), no thinking.
    scd.set_lora_enabled(model, False)
    teacher_targets = []
    for incident, _lvl in held:
        _text, ids, _ok = generate_no_think(
            model, tok, scd.build_prompt(tok, incident, True),
            device, args.max_new_tokens)
        teacher_targets.append(ids)

    configs = {
        "floor_base_noctx":   dict(with_context=False, lora_on=False),
        "ceiling_base_ctx":   dict(with_context=True,  lora_on=False),
        "student_lora_noctx": dict(with_context=False, lora_on=True),
    }
    report = {}
    no_think_ok_all = True
    for name, cfg in configs.items():
        scd.set_lora_enabled(model, cfg["lora_on"])
        scores, kls, samples, think_count, think_tok_sum = [], [], [], 0, 0
        for (incident, lvl), tgt in zip(held, teacher_targets):
            prompt = scd.build_prompt(tok, incident, cfg["with_context"])
            raw, _ids, no_think_ok = generate_no_think(
                model, tok, prompt, device, args.max_new_tokens)
            no_think_ok_all &= no_think_ok
            had_think = bool(think_blocks(raw))
            think_count += int(had_think)
            think_tok_sum += think_token_count(tok, raw)
            scored_text = strip_think(raw)
            scores.append(scd.score_line(scored_text, lvl))
            kl = scd.token_kl(model, tok, incident, tgt, device,
                              cfg["with_context"], cfg["lora_on"])
            if kl is not None:
                kls.append(kl)
            samples.append({
                "incident": incident, "expected": lvl, "had_think": had_think,
                "think_tokens": think_token_count(tok, raw),
                "out_stripped": scored_text.strip(), "out_raw": raw.strip(),
            })
        report[name] = {
            "task": scd.aggregate(scores),
            "mean_token_kl_vs_ceiling": round(sum(kls) / len(kls), 4) if kls else None,
            "think_in_output": think_count,
            "think_tokens_total": think_tok_sum,
            "samples": samples if args.smoke else samples[:8],
        }
        print(f"  {name}: task={report[name]['task']} "
              f"kl={report[name]['mean_token_kl_vs_ceiling']} "
              f"think={think_count} think_toks={think_tok_sum}")

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
        "no_think_kwarg_supported": no_think_ok_all,
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
