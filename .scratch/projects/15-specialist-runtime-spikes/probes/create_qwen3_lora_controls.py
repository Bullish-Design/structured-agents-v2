"""Create deterministic rank-8 Qwen3 LoRA controls for scheduler qualification.

These adapters are intentionally synthetic and affect only layer 0 attention
projections. They prove loading, identity isolation, and scheduling—not model
quality.

SGLang's csgmv LoRA backend fuses the QKV projection and requires every adapter
to carry q_proj, k_proj, and v_proj weights together (a q_proj-only adapter that
vLLM tolerates fails to load). We therefore emit all three projections with the
correct GQA output dims (Qwen3-4B: 32 q heads, 8 kv heads, head_dim 128).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# Qwen3-4B attention shapes.
HIDDEN_SIZE = 2560
Q_SIZE = 4096  # 32 heads * 128
KV_SIZE = 1024  # 8 kv heads * 128
PROJECTIONS = {"q_proj": Q_SIZE, "k_proj": KV_SIZE, "v_proj": KV_SIZE}


def create_adapter(root: Path, name: str, seed: int) -> dict[str, object]:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rank = 8
    tensors: dict[str, torch.Tensor] = {}
    for proj, out_size in PROJECTIONS.items():
        prefix = f"base_model.model.model.layers.0.self_attn.{proj}"
        tensors[f"{prefix}.lora_A.weight"] = (
            torch.randn(rank, HIDDEN_SIZE, generator=generator, dtype=torch.float16) * 0.0005
        )
        tensors[f"{prefix}.lora_B.weight"] = (
            torch.randn(out_size, rank, generator=generator, dtype=torch.float16) * 0.0005
        )
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-4B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(PROJECTIONS),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config_path = target / "adapter_config.json"
    weights_path = target / "adapter_model.safetensors"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    save_file(tensors, weights_path, metadata={"format": "pt", "purpose": "scheduler-control"})
    return {
        "name": name,
        "seed": seed,
        "rank": rank,
        "target_modules": [f"model.layers.0.self_attn.{proj}" for proj in PROJECTIONS],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "weights_bytes": weights_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "synthetic LoRA controls for local multi-adapter scheduler qualification",
        "quality_claim": False,
        "base_model": "Qwen/Qwen3-4B-AWQ",
        "adapters": [
            create_adapter(args.output, "control-a", 101),
            create_adapter(args.output, "control-b", 202),
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
