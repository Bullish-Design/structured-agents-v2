"""Create a synthetic LARGE-perturbation LoRA on the Ornith-1.0-9B base.

Purpose: §9 first go/no-go for 14-PER-SEQUENCE-LORA-GUIDE.md — does llama.cpp
apply a LoRA delta to Ornith's hybrid (Qwen3_5 / GatedDeltaNet) arch at runtime?
This adapter is intentionally synthetic and NOT a quality claim. It targets the
standard attention projections of ONE full_attention layer (layer 3) with a
scale large enough that greedy decoding must visibly diverge from the base if
(and only if) the delta is actually applied.

HF PEFT layout for the ForConditionalGeneration wrapper: the text stack lives
under model.language_model.layers.N. convert_lora_to_gguf strips
"base_model.model." then base.py strips "language_model." -> standard attn_{q,k,v,o}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# Ornith text_config full-attention shapes.
HIDDEN = 4096
N_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 256
Q_OUT = N_HEADS * HEAD_DIM     # 4096
KV_OUT = KV_HEADS * HEAD_DIM   # 1024
O_IN = N_HEADS * HEAD_DIM      # 4096
FULL_ATTN_LAYER = 3

# (out, in) per projection at the chosen full-attention layer.
# NB: Ornith's full-attention q_proj is GATE-FUSED to width 8192 in the GGUF
# (out=8192, verified via GGUFReader). We deliberately skip q_proj — fabricating
# a single-matrix q LoRA cannot represent the HF->GGUF gate fusion cleanly.
# k/v/o_proj have unambiguous dims that match the base GGUF exactly, and a large
# perturbation across all three is more than enough to move greedy decoding.
PROJECTIONS = {
    "k_proj": (KV_OUT, HIDDEN),   # base attn_k   (in=4096, out=1024)
    "v_proj": (KV_OUT, HIDDEN),   # base attn_v   (in=4096, out=1024)
    "o_proj": (HIDDEN, O_IN),     # base attn_output (in=4096, out=4096)
}


def create_adapter(root: Path, name: str, seed: int, rank: int, sigma: float, alpha: float) -> dict:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for proj, (out_size, in_size) in PROJECTIONS.items():
        prefix = (
            f"base_model.model.model.language_model.layers.{FULL_ATTN_LAYER}"
            f".self_attn.{proj}"
        )
        tensors[f"{prefix}.lora_A.weight"] = (
            torch.randn(rank, in_size, generator=g, dtype=torch.float32) * sigma
        ).to(torch.float16)
        tensors[f"{prefix}.lora_B.weight"] = (
            torch.randn(out_size, rank, generator=g, dtype=torch.float32) * sigma
        ).to(torch.float16)
    config = {
        "base_model_name_or_path": "deepreinforce-ai/Ornith-1.0-9B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(PROJECTIONS),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    cfg_path = target / "adapter_config.json"
    w_path = target / "adapter_model.safetensors"
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    save_file(tensors, w_path, metadata={"format": "pt", "purpose": "ornith-lora-probe"})
    return {
        "name": name,
        "seed": seed,
        "rank": rank,
        "sigma": sigma,
        "alpha": alpha,
        "scale": alpha / rank,
        "layer": FULL_ATTN_LAYER,
        "target_modules": list(PROJECTIONS),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(w_path.read_bytes()).hexdigest(),
        "weights_bytes": w_path.stat().st_size,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=64.0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "synthetic large-perturbation LoRA probe for Ornith hybrid-arch go/no-go",
        "quality_claim": False,
        "base_model": "deepreinforce-ai/Ornith-1.0-9B",
        "adapters": [
            create_adapter(args.output, "probe-a", 101, args.rank, args.sigma, args.alpha),
            create_adapter(args.output, "probe-b", 202, args.rank, args.sigma, args.alpha),
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
