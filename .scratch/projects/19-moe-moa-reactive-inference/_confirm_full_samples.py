#!/usr/bin/env python
"""Full-sample eval for a given adapter: identical code path to
spike_distill_confirm_eval.py but persists ALL 24 samples (the official
script truncates persisted samples to 8 via samples[:8]).

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv-distill/bin/python -u _confirm_full_samples.py \
      [--adapter spike_context_distill_adapter.pt] \
      [--out spike_distill_confirm_samples_full.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import spike_context_distill as scd
from spike_distill_confirm_eval import (
    _CONFIRM_QUERIES, generate_no_think, strip_think,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="spike_context_distill_adapter.pt")
    ap.add_argument("--out", default="spike_distill_confirm_samples_full.json")
    ap.add_argument("--check", default="spike_distill_confirm_result.json",
                    help="official JSON to cross-check TRACE level parity against")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    t0 = time.time()
    model, tok = scd.load_model_and_tokenizer(dtype)
    model.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s; adapter={args.adapter}")

    scd.inject_lora(
        model, ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        16, 32, 0.05)
    sd = torch.load(args.adapter, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {list(unexpected)[:5]}")

    held = _CONFIRM_QUERIES
    print(f"[eval] {len(held)} held-out queries (full sample persistence)")

    # Teacher reference (base + ctx, LoRA off), no thinking.
    scd.set_lora_enabled(model, False)
    teacher_targets = []
    for incident, _lvl in held:
        _text, ids, _ok = generate_no_think(
            model, tok, scd.build_prompt(tok, incident, True),
            device, 192)
        teacher_targets.append(ids)

    configs = {
        "floor_base_noctx":   dict(with_context=False, lora_on=False),
        "ceiling_base_ctx":   dict(with_context=True,  lora_on=False),
        "student_lora_noctx": dict(with_context=False, lora_on=True),
    }
    report = {}
    for name, cfg in configs.items():
        scd.set_lora_enabled(model, cfg["lora_on"])
        samples = []
        for (incident, lvl), tgt in zip(held, teacher_targets):
            prompt = scd.build_prompt(tok, incident, cfg["with_context"])
            raw, _ids, _ok = generate_no_think(
                model, tok, prompt, device, 192)
            scored_text = strip_think(raw)
            samples.append({
                "incident": incident, "expected": lvl,
                "out_stripped": scored_text.strip(), "out_raw": raw.strip(),
            })
        report[name] = {"samples": samples}
        print(f"  {name}: done ({len(samples)} samples)")

    # Cross-check TRACE level-parity vs an official JSON, if it exists.
    check_path = Path(args.check)
    if check_path.exists():
        official = json.loads(check_path.read_text())
        LEVEL_RE = scd.LEVEL_RE
        def lvl(t):
            m = LEVEL_RE.search(t["out_stripped"])
            return m.group(1) if m else "NO_MATCH"
        parity = all(
            lvl(o) == lvl(n)
            for cfg in configs
            for o, n in zip(official["report"][cfg]["samples"],
                            report[cfg]["samples"])
        )
        print(f"[check] TRACE level-parity vs {check_path.name}: {parity}")
    else:
        print(f"[check] {check_path.name} not found; skipped parity check")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
