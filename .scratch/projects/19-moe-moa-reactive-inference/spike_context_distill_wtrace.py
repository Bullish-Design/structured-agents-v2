#!/usr/bin/env python
"""TRACE-weighted re-distillation for the context-distillation spike (Project 19).

Same training procedure as spike_context_distill.py (400 steps, rank 16,
lr 2e-4, bf16, batch-1 round-robin over the 24 train queries) plus:
  * --trace-weight W: multiply the KL distillation loss by W for TRACE-class
    training examples (W > 1 up-weights benign-event targets; the follow-up
    found the student inflates TRACE -> WARN, 4/8 at n=24);
  * --seed S: torch.manual_seed for reproducible init/optimisation so a
    control run (W=1) and treatment run (W>1) differ ONLY in the weighting.

Usage (GPU 1 only):
  CUDA_VISIBLE_DEVICES=1 .venv-distill/bin/python -u \
    spike_context_distill_wtrace.py --trace-weight 3.0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import spike_context_distill as scd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--trace-weight", type=float, default=1.0,
                    help="KL-loss multiplier for TRACE-class train examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", default="q_proj,v_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--out", default=str(
        Path(__file__).parent / "spike_context_distill_wtrace_result.json"))
    ap.add_argument("--save-adapter", default=str(
        Path(__file__).parent / "spike_context_distill_adapter_wtrace.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    print(f"[cfg] device={device} dtype={args.dtype} steps={args.steps} "
          f"rank={args.rank} lr={args.lr} trace_weight={args.trace_weight} "
          f"seed={args.seed} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    t0 = time.time()
    model, tok = scd.load_model_and_tokenizer(dtype)
    model.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")

    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    replaced = scd.inject_lora(model, targets, args.rank, args.alpha, args.dropout)
    n_lora = sum(p.numel() for p in scd.lora_parameters(model))
    print(f"[lora] injected into {len(replaced)} linears; "
          f"{n_lora/1e6:.2f}M trainable params")

    train = scd._QUERIES[:scd.HELD_OUT_START]
    n_tr = sum(1 for _, l in train if l == "TRACE")
    print(f"[data] {len(train)} train queries (TRACE={n_tr}, "
          f"WARN={sum(1 for _, l in train if l=='WARN')}, "
          f"FATAL={sum(1 for _, l in train if l=='FATAL')})")

    # ---- precompute teacher targets (base + ctx, LoRA off) ----
    scd.set_lora_enabled(model, False)
    print("[teacher] generating target continuations (base + context)...")
    train_targets = []
    for incident, lvl in train:
        _line, ids = scd.generate(model, tok,
                                  scd.build_prompt(tok, incident, True),
                                  device, args.max_new_tokens)
        train_targets.append(ids)

    # ---- optimiser over LoRA params only ----
    opt = torch.optim.AdamW(scd.lora_parameters(model), lr=args.lr,
                            weight_decay=0.0)

    print(f"[train] {args.steps} steps (KL context distillation, "
          f"trace_weight={args.trace_weight})...")
    model.train()
    n_trace_steps = 0
    for step in range(args.steps):
        incident, lvl = train[step % len(train)]
        tgt = train_targets[step % len(train)]
        loss = scd.distill_loss_one(model, tok, incident, tgt, device)
        if loss is None:
            continue
        if lvl == "TRACE" and args.trace_weight != 1.0:
            loss = loss * args.trace_weight
            n_trace_steps += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scd.lora_parameters(model), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0:
            print(f"  step {step:4d}  KL={loss.item():.4f}")

    model.eval()
    print(f"[train] done; {n_trace_steps} weighted TRACE steps of "
          f"{args.steps} ({100.0*n_trace_steps/args.steps:.1f}%)")

    print("[eval] held-out: floor(base,noctx) / ceiling(base,ctx) / student(lora,noctx)")
    report = scd.run_eval(model, tok, device,
                          scd._QUERIES[scd.HELD_OUT_START:], args.max_new_tokens)
    for name, r in report.items():
        print(f"  {name}: task={r['task']} "
              f"kl_vs_ceiling={r['mean_token_kl_vs_ceiling']}")

    out = {
        "config": vars(args),
        "n_lora_params": int(n_lora),
        "n_trace_weighted_steps": n_trace_steps,
        "report": report,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    torch.save(scd.lora_state_dict(model), args.save_adapter)
    print(f"[done] wrote {args.out}")
    print(f"[done] wrote {args.save_adapter}")


if __name__ == "__main__":
    main()
