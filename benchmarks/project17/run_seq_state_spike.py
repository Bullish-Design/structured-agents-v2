"""GPU-only probe of llama_state_seq_get/set_data on Ornith sequence 0."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    import torch
    from llama_cpp import Llama, llama_cpp

    def mk():
        return Llama(model_path=str(args.model), n_ctx=512, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False)

    def sync():
        torch.cuda.synchronize()

    def next_token(llm):
        return int(np.argmax(np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm._n_vocab,))))

    source = mk()
    tokens = tuple(source.tokenize(b"Sequence state API probe. " * 16, add_bos=False, special=True))
    prefix, suffix = tokens[:32], (tokens[32],)
    source.eval(list(prefix))
    sync()
    ctx = source._ctx.ctx
    size = int(llama_cpp.llama_state_seq_get_size(ctx, 0))
    buf = (ctypes.c_uint8 * size)()
    copied = int(llama_cpp.llama_state_seq_get_data(ctx, buf, size, 0))
    source.eval(list(suffix))
    sync()
    baseline = next_token(source)
    source.close()
    target = mk()
    loaded = int(llama_cpp.llama_state_seq_set_data(target._ctx.ctx, buf, copied, 0))
    target.eval(list(suffix))
    sync()
    restored = next_token(target)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], capture_output=True, text=True
    )
    result = {
        "prefix_tokens": len(prefix),
        "suffix_tokens": len(suffix),
        "seq_state_size": size,
        "copied_bytes": copied,
        "set_return": loaded,
        "baseline_token": baseline,
        "restored_token": restored,
        "match": baseline == restored,
        "gpu": gpu.stdout.strip(),
    }
    (args.artifacts / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["match"] and copied == size and loaded > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
