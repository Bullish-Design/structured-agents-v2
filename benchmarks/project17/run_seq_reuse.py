"""GPU-only per-sequence state-reuse experiment for Ornith (router KV-reuse).

The Phase-2 sweep proved plain ``llama_state_seq_get_data``/``set_data`` restores
a single sequence correctly once n_past bookkeeping is replayed.  A multi-LoRA /
shared-prefix router needs more: restore a cached prefix into a *nonzero*
sequence slot of a *live multi-sequence* context without disturbing its
neighbours, and know exactly when a cached blob is portable.

This runner establishes, with synchronized GPU evidence:

  * multi-sequence decode correctness (control) — independent sequences in one
    context match their isolated single-sequence baselines;
  * the n_seq_max portability matrix — a blob captured under one ``n_seq_max``
    only loads into a context with the *same* ``n_seq_max`` (``set_data`` returns
    0 = "failed to load" otherwise, per llama.h — a safe, detectable reject);
  * the router path — restore into seq 1 of a context already holding seq 0,
    verifying both the restored continuation and neighbour isolation, swept over
    checkpoint length, with CUDA-synchronized restore timing;
  * cross-process restart — capture to disk in one process, restore in a fresh
    process/context and match the same continuation.

Success is deterministic greedy continuation equality only; ``set_data`` return
values and byte counts are recorded but never treated as proof.  GPU-only:
pin CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, verify GPU 1 idle.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np


def gpu() -> dict[str, Any]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}


def require_gpu0_only(snapshot: dict[str, Any]) -> None:
    values = [line.split(", ") for line in snapshot["stdout"].splitlines()]
    if len(values) < 2 or int(values[0][-1].split()[0]) < 4000 or int(values[1][-1].split()[0]) > 200:
        raise RuntimeError(f"GPU isolation rejected: {snapshot['stdout']}")


class Harness:
    def __init__(self, model_path: str, n_ctx: int, n_batch: int, seed: int) -> None:
        from llama_cpp import Llama, llama_cpp

        self.llama_cpp = llama_cpp
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=-1, seed=seed, verbose=False)
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self.n_ctx = n_ctx
        self.n_batch = n_batch

    def tokenize(self, text: bytes) -> list[int]:
        return list(self.llm.tokenize(text, add_bos=False, special=True))

    def make_ctx(self, n_seq_max: int) -> Any:
        p = self.llama_cpp.llama_context_default_params()
        p.n_ctx = self.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = n_seq_max
        return self.llama_cpp.llama_new_context_with_model(self.model, p)

    def free(self, ctx: Any) -> None:
        self.llama_cpp.llama_free(ctx)

    def sync(self) -> None:
        import torch

        torch.cuda.synchronize()

    def decode(self, ctx: Any, tokens: list[int], seq_id: int, start_pos: int) -> None:
        # Chunk into n_batch-sized pieces; a single llama_decode must satisfy
        # n_tokens_all <= n_batch. Logits are requested only on the final token.
        n = len(tokens)
        for off in range(0, n, self.n_batch):
            chunk = tokens[off : off + self.n_batch]
            m = len(chunk)
            batch = self.llama_cpp.llama_batch_init(m, 0, 1)
            batch.n_tokens = m
            for i, t in enumerate(chunk):
                batch.token[i] = t
                batch.pos[i] = start_pos + off + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 0
            if off + m == n:
                batch.logits[m - 1] = 1
            rc = self.llama_cpp.llama_decode(ctx, batch)
            self.llama_cpp.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode rc={rc}")

    def last_token(self, ctx: Any) -> int:
        ptr = self.llama_cpp.llama_get_logits_ith(ctx, -1)
        return int(np.argmax(np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))))

    def get_seq(self, ctx: Any, seq_id: int) -> bytes:
        size = int(self.llama_cpp.llama_state_seq_get_size(ctx, seq_id))
        buf = (ctypes.c_uint8 * size)()
        copied = int(self.llama_cpp.llama_state_seq_get_data(ctx, buf, size, seq_id))
        return bytes(buf[:copied])

    def set_seq(self, ctx: Any, blob: bytes, seq_id: int) -> int:
        arr = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        return int(self.llama_cpp.llama_state_seq_set_data(ctx, arr, len(blob), seq_id))

    def baseline(self, n_seq_max: int, prefix: list[int], suffix: list[int]) -> int:
        ctx = self.make_ctx(n_seq_max)
        self.decode(ctx, prefix, 0, 0)
        self.decode(ctx, suffix, 0, len(prefix))
        self.sync()
        tok = self.last_token(ctx)
        self.free(ctx)
        return tok


def restore_child(h: Harness, blob_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Second-process half of the cross-process restart test."""
    blob = blob_path.read_bytes()
    ctx = h.make_ctx(meta["n_seq_max"])
    set_return = h.set_seq(ctx, blob, meta["seq_id"])
    # Own-batch decode replays position explicitly, so no Llama.n_tokens replay
    # is needed (that is only required when using the high-level Llama.eval path).
    h.decode(ctx, meta["suffix"], meta["seq_id"], meta["prefix_len"])
    h.sync()
    restored = h.last_token(ctx)
    h.free(ctx)
    return {
        "set_return": set_return,
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "restored_token": restored,
        "match": restored == meta["baseline_token"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    # n_ctx is divided across sequences (n_ctx/n_seq_max cells each), so keep it
    # large enough that the longest prefix fits under every swept n_seq_max.
    parser.add_argument("--n-ctx", default=2048, type=int)
    parser.add_argument("--n-batch", default=128, type=int)
    parser.add_argument("--seed", default=17018, type=int)
    parser.add_argument("--prefix-lengths", default="32,64,128,256")
    parser.add_argument("--repetitions", default=3, type=int)
    # Internal: cross-process restore half.
    parser.add_argument("--restore-meta", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")

    if args.restore_meta is not None:
        meta = json.loads(args.restore_meta.read_text())
        h = Harness(str(args.model), args.n_ctx, args.n_batch, args.seed)
        result = restore_child(h, Path(meta["blob_path"]), meta)
        print(json.dumps({"crossprocess": result}, sort_keys=True))
        (args.restore_meta.parent / "crossprocess-child.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["match"] else 1

    if args.artifacts.resolve().parent.name != "artifacts" or not args.artifacts.name.startswith("project17-"):
        raise ValueError("artifacts must be under artifacts/project17-*")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    before = gpu()
    (args.artifacts / "gpu-before.json").write_text(json.dumps(before, indent=2) + "\n")

    h = Harness(str(args.model), args.n_ctx, args.n_batch, args.seed)
    after_load = gpu()
    require_gpu0_only(after_load)
    (args.artifacts / "gpu-after-load.json").write_text(json.dumps(after_load, indent=2) + "\n")

    toks = h.tokenize(b"Multi sequence router reuse workload. " * 60)
    pa, sa = toks[:24], [toks[24]]  # sequence A: a live neighbour
    lengths = [int(v) for v in args.prefix_lengths.split(",")]

    report: dict[str, Any] = {"n_ctx": args.n_ctx, "n_batch": args.n_batch}

    # 1. Multi-sequence decode control (independent sequences in one context).
    pb0, sb0 = toks[40:72], [toks[72]]
    base_a = h.baseline(1, pa, sa)
    base_b = h.baseline(1, pb0, sb0)
    ctx = h.make_ctx(2)
    h.decode(ctx, pa, 0, 0)
    h.decode(ctx, pb0, 1, 0)
    h.decode(ctx, sa, 0, len(pa))
    ma = h.last_token(ctx)
    h.decode(ctx, sb0, 1, len(pb0))
    mb = h.last_token(ctx)
    h.free(ctx)
    report["multiseq_control"] = {
        "baseline_A": base_a,
        "baseline_B": base_b,
        "multiseq_A": ma,
        "multiseq_B": mb,
        "A_ok": ma == base_a,
        "B_ok": mb == base_b,
    }

    # 2. n_seq_max portability matrix (capture nsm -> restore nsm).
    matrix = []
    for cap in (1, 2, 4):
        src = h.make_ctx(cap)
        h.decode(src, pb0, min(1, cap - 1), 0)
        blob = h.get_seq(src, min(1, cap - 1))
        h.free(src)
        for tgt_nsm in (1, 2, 4):
            tgt = h.make_ctx(tgt_nsm)
            seq = min(1, tgt_nsm - 1)
            ret = h.set_seq(tgt, blob, seq)
            match = None
            if ret > 0:
                h.decode(tgt, sb0, seq, len(pb0))
                h.sync()
                match = h.last_token(tgt) == base_b
            h.free(tgt)
            matrix.append(
                {
                    "capture_nsm": cap,
                    "restore_nsm": tgt_nsm,
                    "blob_bytes": len(blob),
                    "set_return": ret,
                    "loaded": ret > 0,
                    "match": match,
                }
            )
    report["nsm_portability_matrix"] = matrix

    # 3. Router path: restore B into seq 1 of a live n_seq_max=2 context holding
    #    A on seq 0, swept over checkpoint length, with synchronized restore time.
    router = []
    for length in lengths:
        pb = toks[40 : 40 + length]
        if 40 + length >= len(toks):
            continue
        sb = [toks[40 + length]]
        base = h.baseline(2, pb, sb)
        # capture B under matched n_seq_max=2 from seq 1
        src = h.make_ctx(2)
        h.decode(src, pb, 1, 0)
        h.sync()
        blob = h.get_seq(src, 1)
        h.free(src)
        set_returns, restore_ns, restored_ok, isolation_ok = [], [], [], []
        for _ in range(args.repetitions):
            tgt = h.make_ctx(2)
            h.decode(tgt, pa, 0, 0)  # live neighbour
            t = perf_counter_ns()
            ret = h.set_seq(tgt, blob, 1)
            h.sync()
            restore_ns.append(perf_counter_ns() - t)
            set_returns.append(ret)
            h.decode(tgt, sb, 1, len(pb))
            h.sync()
            restored_ok.append(h.last_token(tgt) == base)
            h.decode(tgt, sa, 0, len(pa))
            h.sync()
            isolation_ok.append(h.last_token(tgt) == base_a)
            h.free(tgt)
        router.append(
            {
                "checkpoint_tokens": length,
                "seq_blob_bytes": len(blob),
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "baseline_token": base,
                "set_return": set_returns[0],
                "restore_ns_mean": sum(restore_ns) / len(restore_ns),
                "restore_ns_p50": sorted(restore_ns)[(len(restore_ns) - 1) // 2],
                "restored_ok_all": all(restored_ok),
                "isolation_ok_all": all(isolation_ok),
            }
        )
    report["router_path"] = router

    # 4. Cross-process restart: capture to disk here, restore in a fresh process.
    pb = toks[40 : 40 + 64]
    sb = [toks[40 + 64]]
    base = h.baseline(2, pb, sb)
    src = h.make_ctx(2)
    h.decode(src, pb, 1, 0)
    h.sync()
    blob = h.get_seq(src, 1)
    h.free(src)
    blob_path = args.artifacts / "crossprocess-seq.bin"
    blob_path.write_bytes(blob)
    meta = {
        "blob_path": str(blob_path),
        "n_seq_max": 2,
        "seq_id": 1,
        "prefix_len": len(pb),
        "suffix": sb,
        "baseline_token": base,
    }
    meta_path = args.artifacts / "crossprocess-meta.json"
    meta_path.write_text(json.dumps(meta))
    child = subprocess.run(
        [
            sys.executable,
            __file__,
            "--model",
            str(args.model),
            "--artifacts",
            str(args.artifacts),
            "--n-ctx",
            str(args.n_ctx),
            "--n-batch",
            str(args.n_batch),
            "--restore-meta",
            str(meta_path),
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    report["crossprocess"] = {
        "child_exit": child.returncode,
        "child_stdout": child.stdout.strip(),
        "captured_baseline": base,
    }

    summary = {
        "gpu_before": before,
        "gpu_after_load": after_load,
        "gpu_after": gpu(),
        "report": report,
        "note": "Success = greedy continuation match. set_data return values are recorded, not trusted.",
    }
    (args.artifacts / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
