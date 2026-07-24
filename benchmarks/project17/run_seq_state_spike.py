"""GPU-only probe of llama_state_seq_* correctness for Ornith sequence state.

The original single probe copied and re-loaded the advertised byte count with
``llama_state_seq_get_data`` / ``set_data`` yet the greedy continuation diverged
(baseline 21059 vs restored 364).  Byte-count success is not semantic success.

The pinned llama.h documents recurrent/partial sequence state explicitly:

    // work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
    #define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

Ornith is a hybrid attention + GatedDeltaNet (linear-attention / recurrent)
model, so the plain non-ext ``llama_state_seq_*`` path is the prime suspect.
This runner sweeps checkpoint position x suffix length x flag variant so a
single synchronized run maps exactly where per-sequence restore first diverges
and whether the recurrent-aware ``_ext`` + PARTIAL_ONLY path restores it.  A
matching continuation is the only success signal; copied byte counts are
recorded but never treated as a pass.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--n-ctx", default=512, type=int)
    parser.add_argument("--positions", default="1,16,31", help="checkpoint token counts K to probe")
    parser.add_argument("--suffix-lens", default="1,4")
    parser.add_argument("--dest-seq-id", default=0, type=int)
    parser.add_argument(
        "--flags",
        default="none,partial_only",
        help="comma list of: none, partial_only (uses the _ext API when not 'none')",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    import torch
    from llama_cpp import Llama, llama_cpp

    flag_values = {
        "none": int(getattr(llama_cpp, "LLAMA_STATE_SEQ_FLAGS_NONE", 0)),
        "partial_only": int(getattr(llama_cpp, "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY", 1)),
    }
    has_ext = hasattr(llama_cpp, "llama_state_seq_get_data_ext")

    def mk() -> Any:
        return Llama(
            model_path=str(args.model), n_ctx=args.n_ctx, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False
        )

    def sync() -> None:
        torch.cuda.synchronize()

    def next_token(llm: Any) -> int:
        return int(np.argmax(np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm._n_vocab,))))

    def get_seq(ctx: Any, seq_id: int, flag: int) -> tuple[int, bytes]:
        if flag == 0 or not has_ext:
            size = int(llama_cpp.llama_state_seq_get_size(ctx, seq_id))
            buf = (ctypes.c_uint8 * size)()
            copied = int(llama_cpp.llama_state_seq_get_data(ctx, buf, size, seq_id))
            return copied, bytes(buf[:copied])
        size = int(llama_cpp.llama_state_seq_get_size_ext(ctx, seq_id, flag))
        buf = (ctypes.c_uint8 * size)()
        copied = int(llama_cpp.llama_state_seq_get_data_ext(ctx, buf, size, seq_id, flag))
        return copied, bytes(buf[:copied])

    def set_seq(ctx: Any, blob: bytes, dest: int, flag: int) -> int:
        arr = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        if flag == 0 or not has_ext:
            return int(llama_cpp.llama_state_seq_set_data(ctx, arr, len(blob), dest))
        return int(llama_cpp.llama_state_seq_set_data_ext(ctx, arr, len(blob), dest, flag))

    base_tokens = tuple(mk().tokenize(b"Sequence state API probe. " * 16, add_bos=False, special=True))
    positions = [int(v) for v in args.positions.split(",")]
    suffix_lens = [int(v) for v in args.suffix_lens.split(",")]
    flags = [f.strip() for f in args.flags.split(",")]

    results: list[dict[str, Any]] = []
    for k in positions:
        for slen in suffix_lens:
            if k + slen >= len(base_tokens):
                continue
            prefix, suffix = base_tokens[:k], base_tokens[k : k + slen]
            for flag_name in flags:
                flag = flag_values[flag_name]
                source = mk()
                source.eval(list(prefix))
                sync()
                copied, blob = get_seq(source._ctx.ctx, 0, flag)
                source.eval(list(suffix))
                sync()
                baseline = next_token(source)
                blob_sha = hashlib.sha256(blob).hexdigest()
                source.close()

                target = mk()
                loaded = set_seq(target._ctx.ctx, blob, args.dest_seq_id, flag)
                # Replay Python n_past bookkeeping so eval does not wipe the KV.
                target.n_tokens = len(prefix)
                target.input_ids[: len(prefix)] = np.array(prefix, dtype=np.intc)
                target.eval(list(suffix))
                sync()
                restored = next_token(target)
                target.close()

                results.append(
                    {
                        "checkpoint_tokens": k,
                        "suffix_tokens": slen,
                        "flag": flag_name,
                        "used_ext": bool(flag != 0 and has_ext),
                        "dest_seq_id": args.dest_seq_id,
                        "seq_state_size": len(blob),
                        "copied_bytes": copied,
                        "set_return": loaded,
                        "blob_sha256": blob_sha,
                        "baseline_token": baseline,
                        "restored_token": restored,
                        "match": baseline == restored,
                    }
                )
                print(json.dumps(results[-1], sort_keys=True), flush=True)

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], capture_output=True, text=True
    )
    summary = {
        "has_ext_api": has_ext,
        "results": results,
        "any_match": any(r["match"] for r in results),
        "gpu": gpu.stdout.strip(),
    }
    (args.artifacts / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["any_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
