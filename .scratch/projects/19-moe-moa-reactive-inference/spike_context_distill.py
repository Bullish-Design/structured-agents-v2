#!/usr/bin/env python3
"""Project 19 -- Context-distillation spike (CONCEPT.md 6.5.2 option 2).

Question under test
-------------------
Can a small LoRA adapter INTERNALIZE a fixed context via context distillation,
so that at inference the *adapter-only* model (context REMOVED from the prompt)
matches the behaviour of the *base* model WITH that context in its prompt?

If yes, an "expert bundle" hat (CONCEPT 6.5.2) can carry its knowledge in
weights -- zero KV at inference -- dissolving both KV-consistency traps and the
P4 "splice must beat re-prefill" burden for that class of bundle.

Method (Snell et al. 2022 "Learning by Distilling Context" arXiv:2209.15189;
Mu et al. 2023 "Gisting" arXiv:2304.08467)
-------------------------------------------------------------------------------
  Teacher : base + CONTEXT in prompt  ->  greedy target continuation.
  Student : base + rank-r LoRA, CONTEXT REMOVED from prompt.
  Objective: token-level KL(teacher || student) over the teacher's own
             (teacher-forced) continuation tokens.  We align on the *output*
             token positions -- teacher and student have different prefixes
             (context present vs absent) but predict the SAME target tokens.
  Eval    : on HELD-OUT queries, does student-no-ctx match base+ctx?
            Reported against two baselines:
              floor   = base, NO context   (what you get for free)
              ceiling = base, WITH context (what distillation tries to reach)

Reused stack (per task brief -- Project 17 tiny-LoRA eval, do NOT invent a new
one): the on-disk HF snapshot of Qwen3.5-0.8B and the `.venv-spike` (torch +
transformers) under 17-llama-cpp-inference-lab/. LoRA is implemented here in
pure torch so the spike is self-contained and needs no `peft` install.

The CONTEXT is a small, self-contained, *synthetic* protocol (a fictional
logging API with a deliberately non-standard level set) so that:
  * the base model cannot know it a priori  -> floor is genuinely low,
  * it is procedure + a few small facts      -> the regime distillation suits
                                                (CONCEPT 6.5.3), and
  * success is crisply measurable            -> regex format + level accuracy.

No external dataset download: queries are generated in-code.

Run
---
NOTE: the reused Project-17 `.venv-spike` ships transformers 4.57.6, which does
NOT recognise this model's `qwen3_5` architecture. Use the project-19
`.venv-distill` (transformers 5.13.1 + torch 2.13/cu130) instead. That torch
JIT-compiles a triton kernel for the qwen3_5 rotary op, so a C compiler must be
on CC (auto-detected below from the nix store if unset).

Smoke test (a few steps, proves it runs) on the FREE gpu (GPU 1):
    cd .scratch/projects/17-llama-cpp-inference-lab
    LD_LIBRARY_PATH="$(cat .cuda_runtime_ld)" CUDA_VISIBLE_DEVICES=1 \
      ../19-moe-moa-reactive-inference/.venv-distill/bin/python \
      ../19-moe-moa-reactive-inference/spike_context_distill.py --smoke

Full distillation (see SPIKE-DISTILL-PLAN.md for the canonical command):
    ... .venv-distill/bin/python .../spike_context_distill.py --steps 400

This script also *self-bootstraps* LD_LIBRARY_PATH (torch needs the nix
libstdc++ / CUDA runtime dirs recorded in 17-lab/.cuda_runtime_ld); if the env
is missing it re-execs itself once with the right LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# 0. Locate the Project-17 tiny-model stack and bootstrap the loader env.
# --------------------------------------------------------------------------- #
LAB = Path(
    "/home/andrew/Documents/Projects/structured-agents-v2/"
    ".scratch/projects/17-llama-cpp-inference-lab"
)
BASE_MODEL_DIR = LAB / "runtime" / "qwen35-lora-eval" / "base-Qwen3.5-0.8B"
LD_FILE = LAB / ".cuda_runtime_ld"


def _bootstrap_ld_and_reexec() -> None:
    """torch's shared libs need the nix libstdc++/CUDA dirs on LD_LIBRARY_PATH.
    If they are absent, prepend them (from 17-lab/.cuda_runtime_ld) and re-exec
    this process once so the dynamic loader picks them up before torch imports.
    """
    # triton (pulled in by the linear-attn layers) calls /sbin/ldconfig to find
    # libcuda, which does not exist on NixOS. Point it at the driver dir instead.
    if not os.environ.get("TRITON_LIBCUDA_PATH"):
        for cand in ("/run/opengl-driver/lib",
                     *(LD_FILE.read_text().strip().split(":") if LD_FILE.exists() else ())):
            if os.path.exists(os.path.join(cand, "libcuda.so.1")):
                os.environ["TRITON_LIBCUDA_PATH"] = cand
                break
    # torch 2.13's qwen3_5 rotary op JIT-compiles a triton kernel, which needs a
    # C compiler. NixOS has no bare `cc` on PATH; point CC/CXX at a nix gcc.
    if not os.environ.get("CC"):
        import glob
        for gcc in sorted(glob.glob("/nix/store/*gcc-wrapper-*/bin/gcc"), reverse=True):
            gpp = gcc[:-3] + "g++"
            if os.path.exists(gpp):
                os.environ["CC"], os.environ["CXX"] = gcc, gpp
                break
    if os.environ.get("_SPIKE_LD_BOOTSTRAPPED"):
        return
    if not LD_FILE.exists():
        return  # nothing we can do; let the import error speak for itself
    needed = LD_FILE.read_text().strip()
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if needed.split(":")[0] in cur:
        return  # already present
    os.environ["LD_LIBRARY_PATH"] = needed + (":" + cur if cur else "")
    os.environ["_SPIKE_LD_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_ld_and_reexec()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. The CONTEXT to internalize (a compact, self-contained protocol).
# --------------------------------------------------------------------------- #
# ~230 tokens. Deliberately non-standard so the base can't guess it: the valid
# levels are TRACE/WARN/FATAL (no INFO/ERROR/DEBUG), and the code ranges are
# arbitrary. This is exactly the "style/procedure/small-facts" regime that
# CONCEPT 6.5.3 says distills well.
# v2 (2026-07-31): added 5 few-shot exemplars (3 TRACE / 1 WARN / 1 FATAL) to
# fix the template-path TRACE over-escalation (both teacher and student mapped
# all 8 TRACE held-out incidents to WARN). Example incidents deliberately do
# not overlap the train/held-out query sets.
CONTEXT = """\
You are the ACME event logger. Convert each incident description into ONE log \
line and output nothing else. The format is exactly:
  LOG(level="<LEVEL>", code=<CODE>, msg="<short summary>")
Rules:
- <LEVEL> is one of exactly: TRACE, WARN, FATAL. No other levels exist.
- Routine or purely informational events use TRACE.
- Recoverable problems or degraded states use WARN.
- Unrecoverable failures, data loss, or outages use FATAL.
- <CODE> is a 4-digit integer. TRACE codes start with 1 (1000-1999), WARN \
codes start with 3 (3000-3999), FATAL codes start with 9 (9000-9999).
- <msg> is a short summary in double quotes, at most 8 words.
Examples:
- "A report was generated on demand." -> LOG(level="TRACE", code=1000, msg="Report generated on demand")
- "A user logged out of the console." -> LOG(level="TRACE", code=1001, msg="User logged out")
- "The scheduler tick fired." -> LOG(level="TRACE", code=1002, msg="Scheduler tick fired")
- "A replica is lagging slightly behind." -> LOG(level="WARN", code=3000, msg="Replica lagging behind")
- "The backup volume is offline." -> LOG(level="FATAL", code=9000, msg="Backup volume offline")
Output only the single LOG(...) line."""

INSTRUCTION = "Incident: {q}\nLog line:"

# --------------------------------------------------------------------------- #
# 2. Query set (generated in-code; split into train / held-out).
# --------------------------------------------------------------------------- #
# (incident text, expected level) -- the level lets us score task success.
_QUERIES = [
    ("The nightly backup completed successfully.", "TRACE"),
    ("User alice logged in.", "TRACE"),
    ("Cache warmed with 2000 entries.", "TRACE"),
    ("Health check returned OK.", "TRACE"),
    ("Configuration reloaded from disk.", "TRACE"),
    ("Scheduled job started on time.", "TRACE"),
    ("Metrics flushed to the collector.", "TRACE"),
    ("Session token refreshed for a client.", "TRACE"),
    ("Disk usage crossed 80 percent on node 4.", "WARN"),
    ("A request retried after a transient timeout.", "WARN"),
    ("Replica lag climbed to 12 seconds.", "WARN"),
    ("Memory pressure forced a cache eviction.", "WARN"),
    ("TLS certificate expires in 5 days.", "WARN"),
    ("Rate limiter throttled a noisy client.", "WARN"),
    ("A deprecated API endpoint was called.", "WARN"),
    ("Queue depth exceeded the soft limit.", "WARN"),
    ("The primary database is unreachable.", "FATAL"),
    ("Data corruption detected in the ledger.", "FATAL"),
    ("The whole cluster lost power.", "FATAL"),
    ("Irrecoverable write failure on the volume.", "FATAL"),
    ("All replicas are down; service is offline.", "FATAL"),
    ("The filesystem is full and writes are failing.", "FATAL"),
    ("Kernel panic on the storage host.", "FATAL"),
    ("Permanent loss of the last backup snapshot.", "FATAL"),
    # held-out region (indices >= HELD_OUT_START)
    ("A user updated their profile picture.", "TRACE"),      # 24
    ("Garbage collection ran for 40 milliseconds.", "TRACE"),
    ("CPU temperature is slightly above nominal.", "WARN"),
    ("A background sync fell behind by one batch.", "WARN"),
    ("The message broker crashed and will not restart.", "FATAL"),
    ("Total network partition isolated the datacenter.", "FATAL"),
]
HELD_OUT_START = 24  # last 6 are held-out; never seen in training

# Accept both the context-specified format LOG(level="TRACE", code=1000, msg="...")
# and the model's natural emission LOG(TRACE, 1000, "..."). Full-run amendment
# (2026-07-31): the strict form scored even the ceiling at format_ok=0, which
# nulled the task-metric verdicts; this relaxes the scoring only, not the design.
LEVEL_RE = re.compile(
    r'LOG\(\s*(?:level=")?([A-Z]+)(?:\s*",?\s*code=|\s*,\s*)(\d{4})'
    r'(?:,\s*msg="[^"]*"|\s*,\s*"[^"]*")?\s*\)'
)
VALID_LEVELS = {"TRACE", "WARN", "FATAL"}
_CODE_PREFIX = {"TRACE": "1", "WARN": "3", "FATAL": "9"}


# --------------------------------------------------------------------------- #
# 3. Minimal, self-contained LoRA (no peft dependency).
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with an additive low-rank delta B @ A.

    y = base(x) + scaling * (dropout(x) @ A^T) @ B^T
    A: (r, in), B: (out, r). Only A, B are trainable.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        dev, dt = base.weight.device, base.weight.dtype
        # LoRA matrices kept in fp32 for stable optimisation.
        self.lora_A = nn.Parameter(torch.zeros(r, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # B stays zero -> delta=0 at init
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # noqa: D401
        out = self.base(x)
        xd = self.drop(x).to(torch.float32)
        delta = (xd @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * delta.to(out.dtype)


def inject_lora(model, target_suffixes, r, alpha, dropout):
    """Replace every nn.Linear whose qualified name ends in one of
    `target_suffixes` with a LoRALinear. Returns list of (name, module)."""
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(
                full.endswith(s) for s in target_suffixes
            ):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(full)
    return replaced


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def lora_state_dict(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


# --------------------------------------------------------------------------- #
# 4. Model loading (reuse the on-disk Project-17 snapshot).
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(dtype):
    """Load the tiny base. It is a hybrid VL model (Qwen3_5ForConditional-
    Generation); we run it text-only (input_ids, no pixel_values). Try the
    generic causal-LM loader first, then the image-text-to-text class."""
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    last_err = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                        "AutoModel"):
        try:
            import transformers
            Loader = getattr(transformers, loader_name)
            try:  # transformers >=5 renamed torch_dtype -> dtype
                model = Loader.from_pretrained(
                    str(BASE_MODEL_DIR), dtype=dtype,
                    trust_remote_code=True, low_cpu_mem_usage=True)
            except TypeError:
                model = Loader.from_pretrained(
                    str(BASE_MODEL_DIR), torch_dtype=dtype,
                    trust_remote_code=True, low_cpu_mem_usage=True)
            print(f"[load] loaded via {loader_name}: {model.__class__.__name__}")
            return model, tok
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[load] {loader_name} failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"could not load base model: {last_err}")


def _logits(model, input_ids, attn_mask):
    """Text-only forward returning logits (B, T, V), robust to wrapper shapes."""
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, (tuple, list)):
        logits = out[0]
    return logits


# --------------------------------------------------------------------------- #
# 5. Prompt building.
# --------------------------------------------------------------------------- #
def build_prompt(tok, incident, with_context):
    """Return a plain (non-chat) prompt string. We keep it template-free so the
    teacher/student differ ONLY by the presence of CONTEXT."""
    body = INSTRUCTION.format(q=incident)
    if with_context:
        return CONTEXT + "\n\n" + body
    return body


def encode(tok, text, device):
    ids = tok(text, return_tensors="pt")
    return {k: v.to(device) for k, v in ids.items()}


# --------------------------------------------------------------------------- #
# 6. Teacher generation (greedy continuation from base + context).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, tok, prompt, device, max_new_tokens=48):
    enc = encode(tok, prompt, device)
    prompt_len = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    new_ids = gen[0, prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    # keep only the first LOG(...) line for stable scoring / short targets
    line = text.strip().splitlines()[0] if text.strip() else text
    return line, new_ids


# --------------------------------------------------------------------------- #
# 7. KL context-distillation loss for one example.
# --------------------------------------------------------------------------- #
def distill_loss_one(model, tok, incident, target_ids, device, temperature=1.0):
    """KL(teacher || student) over the target (teacher) continuation tokens.

    Teacher logits: base (LoRA delta OFF) on [CONTEXT + body + target].
    Student logits: base (LoRA delta ON)  on [body + target]  (no context).
    We compare next-token distributions at exactly the positions that PREDICT
    each target token, so the two prefixes need not be the same length.
    """
    tgt = target_ids.to(device)
    if tgt.numel() == 0:
        return None

    # ---- teacher pass: LoRA OFF, context IN ----
    set_lora_enabled(model, False)
    t_prompt = build_prompt(tok, incident, with_context=True)
    t_ids = encode(tok, t_prompt, device)["input_ids"]
    t_full = torch.cat([t_ids, tgt.unsqueeze(0)], dim=1)
    with torch.no_grad():
        t_logits = _logits(model, t_full, torch.ones_like(t_full))
    t_plen = t_ids.shape[1]
    # positions predicting the target tokens: [t_plen-1 .. t_plen+L-2]
    t_pred = t_logits[0, t_plen - 1 : t_plen - 1 + tgt.numel(), :].float()

    # ---- student pass: LoRA ON, context OUT ----
    set_lora_enabled(model, True)
    s_prompt = build_prompt(tok, incident, with_context=False)
    s_ids = encode(tok, s_prompt, device)["input_ids"]
    s_full = torch.cat([s_ids, tgt.unsqueeze(0)], dim=1)
    s_logits = _logits(model, s_full, torch.ones_like(s_full))
    s_plen = s_ids.shape[1]
    s_pred = s_logits[0, s_plen - 1 : s_plen - 1 + tgt.numel(), :].float()

    # KL(teacher || student), teacher detached (it is the fixed target).
    t_logp = F.log_softmax(t_pred / temperature, dim=-1).detach()
    s_logp = F.log_softmax(s_pred / temperature, dim=-1)
    kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")
    return kl


def set_lora_enabled(model, enabled: bool):
    """Toggle the LoRA delta on/off so the SAME model instance serves as both
    teacher (off) and student (on) -- no second copy of the weights in VRAM."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scaling_enabled = enabled
    # patch forward behaviour via a flag read in LoRALinear.forward:
    # (implemented by overriding scaling to 0 when disabled)


# Re-define LoRALinear.forward to honour an enable flag (kept simple & explicit)
def _lora_forward(self, x):
    out = self.base(x)
    if not getattr(self, "scaling_enabled", True):
        return out
    xd = self.drop(x).to(torch.float32)
    delta = (xd @ self.lora_A.t()) @ self.lora_B.t()
    return out + self.scaling * delta.to(out.dtype)


LoRALinear.forward = _lora_forward


# --------------------------------------------------------------------------- #
# 8. Metrics.
# --------------------------------------------------------------------------- #
def score_line(line, expected_level):
    """Return dict of task-success booleans for one generated LOG line."""
    m = LEVEL_RE.search(line)
    format_ok = m is not None
    level = m.group(1) if m else None
    code = m.group(2) if m else None
    level_valid = level in VALID_LEVELS if level else False
    level_correct = level == expected_level
    code_ok = bool(code and level and code[0] == _CODE_PREFIX.get(level, "?"))
    return {
        "format_ok": format_ok,
        "level_valid": level_valid,
        "level_correct": level_correct,
        "code_prefix_ok": code_ok,
    }


def aggregate(scores):
    keys = ["format_ok", "level_valid", "level_correct", "code_prefix_ok"]
    n = len(scores)
    return {k: round(sum(s[k] for s in scores) / n, 3) for k in keys} if n else {}


# --------------------------------------------------------------------------- #
# 9. Full held-out evaluation (3 configs + KL floor/ceiling).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def token_kl(model, tok, incident, target_ids, device, with_context, lora_on):
    """Per-token mean KL(teacher || this-config) where teacher = base+ctx,
    LoRA off. `this-config` is set by (with_context, lora_on)."""
    tgt = target_ids.to(device)
    if tgt.numel() == 0:
        return None
    # teacher reference (base + ctx, LoRA off)
    set_lora_enabled(model, False)
    t_prompt = build_prompt(tok, incident, with_context=True)
    t_ids = encode(tok, t_prompt, device)["input_ids"]
    t_full = torch.cat([t_ids, tgt.unsqueeze(0)], dim=1)
    t_logits = _logits(model, t_full, torch.ones_like(t_full))
    tp = t_ids.shape[1]
    t_pred = t_logits[0, tp - 1: tp - 1 + tgt.numel(), :].float()

    set_lora_enabled(model, lora_on)
    c_prompt = build_prompt(tok, incident, with_context=with_context)
    c_ids = encode(tok, c_prompt, device)["input_ids"]
    c_full = torch.cat([c_ids, tgt.unsqueeze(0)], dim=1)
    c_logits = _logits(model, c_full, torch.ones_like(c_full))
    cp = c_ids.shape[1]
    c_pred = c_logits[0, cp - 1: cp - 1 + tgt.numel(), :].float()

    t_logp = F.log_softmax(t_pred, dim=-1)
    c_logp = F.log_softmax(c_pred, dim=-1)
    # per-token KL then mean
    kl = F.kl_div(c_logp, t_logp, log_target=True, reduction="none").sum(-1).mean()
    return float(kl)


def run_eval(model, tok, device, held, max_new_tokens):
    """Evaluate the three configs on held-out queries. Returns a report dict."""
    # 1) teacher reference continuations (base + ctx) for KL alignment
    set_lora_enabled(model, False)
    teacher_targets = []
    for incident, _lvl in held:
        _, ids = generate(model, tok, build_prompt(tok, incident, True), device,
                          max_new_tokens)
        teacher_targets.append(ids)

    configs = {
        "floor_base_noctx":   dict(with_context=False, lora_on=False),
        "ceiling_base_ctx":   dict(with_context=True,  lora_on=False),
        "student_lora_noctx": dict(with_context=False, lora_on=True),
    }
    report = {}
    for name, cfg in configs.items():
        set_lora_enabled(model, cfg["lora_on"])
        scores, kls = [], []
        samples = []
        for (incident, lvl), tgt in zip(held, teacher_targets):
            prompt = build_prompt(tok, incident, cfg["with_context"])
            line, _ = generate(model, tok, prompt, device, max_new_tokens)
            scores.append(score_line(line, lvl))
            kl = token_kl(model, tok, incident, tgt, device,
                          cfg["with_context"], cfg["lora_on"])
            if kl is not None:
                kls.append(kl)
            samples.append({"incident": incident, "expected": lvl, "out": line})
        report[name] = {
            "task": aggregate(scores),
            "mean_token_kl_vs_ceiling": round(sum(kls) / len(kls), 4) if kls else None,
            "samples": samples[:3],
        }
    return report


# --------------------------------------------------------------------------- #
# 10. Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: proves it executes (few steps, small budget)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--trace-weight", type=int, default=1,
                    help="repeat TRACE queries this many times in the train pool "
                         "(e.g. 3 => TRACE:WARN:FATAL ~ 3:1:1)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--targets", default="q_proj,v_proj,gate_proj,up_proj,down_proj",
                    help="comma-separated nn.Linear name suffixes to LoRA-ify")
    ap.add_argument("--out", default=str(
        Path(__file__).parent / "spike_context_distill_result.json"))
    ap.add_argument("--save-adapter", default=str(
        Path(__file__).parent / "spike_context_distill_adapter.pt"))
    args = ap.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 4)
        args.max_new_tokens = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    print(f"[cfg] device={device} dtype={args.dtype} smoke={args.smoke} "
          f"steps={args.steps} rank={args.rank} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    t0 = time.time()
    model, tok = load_model_and_tokenizer(dtype)
    model.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")

    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    replaced = inject_lora(model, targets, args.rank, args.alpha, args.dropout)
    n_lora = sum(p.numel() for p in lora_parameters(model))
    print(f"[lora] injected into {len(replaced)} linears; "
          f"{n_lora/1e6:.2f}M trainable params; e.g. {replaced[:3]}")
    if not replaced:
        raise RuntimeError("no target linears matched; check --targets")

    # ---- split queries ----
    train = _QUERIES[:HELD_OUT_START]
    held = _QUERIES[HELD_OUT_START:]
    if args.smoke:
        train = train[:6]
        held = held[:2]

    # ---- precompute teacher targets (base + ctx, LoRA off) ----
    set_lora_enabled(model, False)
    print("[teacher] generating target continuations (base + context)...")
    train_targets = []
    for incident, lvl in train:
        line, ids = generate(model, tok, build_prompt(tok, incident, True),
                             device, args.max_new_tokens)
        train_targets.append(ids)
        if args.smoke:
            print(f"  [{lvl}] {incident[:40]!r} -> {line!r}")

    # ---- optimiser over LoRA params only ----
    opt = torch.optim.AdamW(lora_parameters(model), lr=args.lr, weight_decay=0.0)

    # ---- up-weighted train pool (TRACE oversampling; SPIKE-DISTILL-TEACHERFIX) ----
    trace_weight = args.trace_weight
    pool = []
    for i, (_inc, lvl) in enumerate(train):
        pool.extend([i] * (trace_weight if lvl == "TRACE" else 1))
    if trace_weight > 1:
        random.Random(42).shuffle(pool)
    print(f"[train] pool={len(pool)} (trace_weight={trace_weight}, "
          f"TRACE={sum(1 for i in pool if train[i][1] == 'TRACE')})")

    print(f"[train] {args.steps} steps (KL context distillation)...")
    model.train()
    for step in range(args.steps):
        idx = pool[step % len(pool)]
        incident, _lvl = train[idx]
        tgt = train_targets[idx]
        loss = distill_loss_one(model, tok, incident, tgt, device)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or args.smoke:
            print(f"  step {step:4d}  KL={loss.item():.4f}")

    model.eval()
    print("[eval] held-out: floor(base,noctx) / ceiling(base,ctx) / student(lora,noctx)")
    report = run_eval(model, tok, device, held, args.max_new_tokens)
    for name, r in report.items():
        print(f"  {name}: task={r['task']} "
              f"kl_vs_ceiling={r['mean_token_kl_vs_ceiling']}")

    # ---- go/no-go verdict (context distillation succeeded?) ----
    floor = report["floor_base_noctx"]
    ceil = report["ceiling_base_ctx"]
    stud = report["student_lora_noctx"]

    def gap_recovered(metric):
        f, c, s = (floor["task"].get(metric), ceil["task"].get(metric),
                   stud["task"].get(metric))
        if None in (f, c, s) or c == f:
            return None
        return round((s - f) / (c - f), 3)

    verdict = {
        "level_correct_gap_recovered": gap_recovered("level_correct"),
        "format_ok_gap_recovered": gap_recovered("format_ok"),
        "student_kl_below_floor_kl": (
            stud["mean_token_kl_vs_ceiling"] is not None
            and floor["mean_token_kl_vs_ceiling"] is not None
            and stud["mean_token_kl_vs_ceiling"] < floor["mean_token_kl_vs_ceiling"]
        ),
        "note": "GO if gap_recovered >= 0.5 on level_correct AND "
                "student_kl_below_floor_kl is True (see SPIKE-DISTILL-PLAN.md).",
        "smoke": args.smoke,
    }

    out = {
        "config": vars(args),
        "context_tokens": len(tok(CONTEXT)["input_ids"]),
        "n_lora_params": int(n_lora),
        "lora_targets_matched": len(replaced),
        "report": report,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    torch.save(lora_state_dict(model), args.save_adapter)
    print(f"[done] wrote {args.out}")
    print(f"[done] wrote {args.save_adapter}")
    print(f"[verdict] {json.dumps(verdict, indent=2)}")


if __name__ == "__main__":
    main()
