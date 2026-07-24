"""GPU-only decomposition of whole-state restore cost: LlamaState vs native codec.

This is the decisive follow-up experiment for Project 18 / Project 17 Phase 2.
The proven ``run_prefix_cache.py`` benchmark measured a single
``state_restore_wall`` that bundles pickle deserialization, the ``LlamaState``
score-array apply, and the native ``llama_state_set_data`` host->device restore.
This runner separates them and also measures a native-bytes codec that bypasses
the ``LlamaState`` wrapper entirely, so a single synchronized run answers:

  * how much of restore is Python pickle/score bookkeeping vs native state set;
  * whether removing the score buffer (native codec) meaningfully shrinks the
    blob and the restore, and whether that is enough to beat cold prefill;
  * how much of the cache path is disk read (persistent) vs pure restore
    (in-memory checkpoint), i.e. Project 18 Option 2 vs Option 3.

Both codecs restore the SAME native llama.cpp context state (the native path is
the exact C entry point ``Llama.load_state`` already wraps, minus the score
buffer), so continuation correctness is enforced as a hard, fail-closed gate:
any restored token that differs from the cold baseline raises.

GPU-only policy: pin CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, and verify GPU 1
stays idle before trusting any number.  Enqueue timing is not throughput; every
measured phase is CUDA-synchronized wall time.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structured_agents.llama_core.diagnostics import collect_runtime_diagnostics
from structured_agents.llama_core.fingerprint import LlamaEngineFingerprint, register_artifact


def gpu() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def require_gpu0_only(snapshot: dict[str, Any]) -> None:
    values = [line.split(", ") for line in snapshot["stdout"].splitlines()]
    if len(values) < 2 or int(values[0][-1].split()[0]) < 4000 or int(values[1][-1].split()[0]) > 200:
        raise RuntimeError(f"GPU isolation rejected: {snapshot['stdout']}")


def stats(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean_ns": sum(values) / len(values),
        "p50_ns": ordered[(len(values) - 1) // 2],
        "p95_ns": ordered[min(len(values) - 1, int(np.ceil(len(values) * 0.95)) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--prefix-lengths", default="16,64,128,256")
    parser.add_argument("--repetitions", default=3, type=int)
    parser.add_argument("--n-ctx", default=512, type=int)
    parser.add_argument("--n-batch", default=128, type=int)
    parser.add_argument("--seed", default=17018, type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES=0 and LLAMA_CPP_LIB_PATH")
    if args.artifacts.resolve().parent.name != "artifacts" or not args.artifacts.name.startswith("project17-"):
        raise ValueError("artifacts must be under artifacts/project17-*")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    before = gpu()
    (args.artifacts / "gpu-before.json").write_text(json.dumps(before, indent=2) + "\n")

    from llama_cpp import Llama, llama_cpp

    def mk() -> Any:
        return Llama(
            model_path=str(args.model),
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_gpu_layers=-1,
            seed=args.seed,
            verbose=False,
        )

    def sync() -> None:
        import torch

        torch.cuda.synchronize()

    def token(model: Any) -> int:
        return int(np.argmax(np.ctypeslib.as_array(model._ctx.get_logits(), shape=(model._n_vocab,))))

    def native_get(model: Any) -> bytes:
        ctx = model._ctx.ctx
        size = int(llama_cpp.llama_state_get_size(ctx))
        buf = (ctypes.c_uint8 * size)()
        copied = int(llama_cpp.llama_state_get_data(ctx, buf, size))
        return bytes(buf[:copied])

    def native_set(model: Any, blob: bytes, prefix: tuple[int, ...]) -> None:
        # Restore native context state, then replay the Python-side n_past/input
        # bookkeeping that Llama.load_state normally performs.  Without this,
        # Llama.eval() would kv_cache_seq_rm from position 0 and wipe the restore.
        arr = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        loaded = int(llama_cpp.llama_state_set_data(model._ctx.ctx, arr, len(blob)))
        if loaded != len(blob):
            raise RuntimeError(f"native llama_state_set_data returned {loaded}, expected {len(blob)}")
        model.n_tokens = len(prefix)
        model.input_ids[: len(prefix)] = np.array(prefix, dtype=np.intc)

    llm = mk()
    after_load = gpu()
    require_gpu0_only(after_load)
    (args.artifacts / "gpu-after-load.json").write_text(json.dumps(after_load, indent=2) + "\n")
    runtime = collect_runtime_diagnostics()
    identity = register_artifact(args.model, metadata={"role": "model-and-gguf-tokenizer"})
    fingerprint = LlamaEngineFingerprint(
        model=identity,
        tokenizer=identity,
        llama_cpp_python_version=runtime.llama_cpp_python_version or "unknown",
        llama_cpp_commit=runtime.llama_cpp_commit,
        llama_cpp_build_id=runtime.llama_cpp_build_id,
        backend="cuda",
        n_ctx=args.n_ctx,
    )
    tokens = tuple(llm.tokenize(b"Persistent exact prefix cache workload. " * 100, add_bos=False, special=True))
    records: list[dict[str, Any]] = []
    for length in (int(v) for v in args.prefix_lengths.split(",")):
        prefix, suffix = tokens[:length], (tokens[length],)

        # Capture both codecs from the same checkpoint.
        llm.reset()
        llm.eval(list(prefix))
        sync()
        pickle_blob = pickle.dumps(llm.save_state(), protocol=pickle.HIGHEST_PROTOCOL)
        native_blob = native_get(llm)
        pickle_sha = hashlib.sha256(pickle_blob).hexdigest()
        native_sha = hashlib.sha256(native_blob).hexdigest()

        # Cold baseline (fresh prefill of prefix+suffix) establishes truth.
        llm.reset()
        llm.eval(list(prefix + suffix))
        sync()
        cold_token = token(llm)

        timings: dict[str, list[int]] = {
            "cold_prefill_wall": [],
            # LlamaState (wrapper) path, from in-memory bytes (no disk):
            "ls_pickle_loads_wall": [],
            "ls_load_state_wall": [],
            "ls_suffix_wall": [],
            "ls_restore_total_wall": [],
            # Native codec path, from in-memory bytes (no disk):
            "nat_set_data_wall": [],
            "nat_suffix_wall": [],
            "nat_restore_total_wall": [],
        }
        for _ in range(args.repetitions):
            llm.reset()
            t = perf_counter_ns()
            llm.eval(list(prefix + suffix))
            sync()
            timings["cold_prefill_wall"].append(perf_counter_ns() - t)
            if token(llm) != cold_token:
                raise RuntimeError("cold prefill nondeterministic")

            # --- LlamaState wrapper path ---
            llm.reset()
            r0 = perf_counter_ns()
            t = perf_counter_ns()
            state = pickle.loads(pickle_blob)
            timings["ls_pickle_loads_wall"].append(perf_counter_ns() - t)
            t = perf_counter_ns()
            llm.load_state(state)
            sync()
            timings["ls_load_state_wall"].append(perf_counter_ns() - t)
            t = perf_counter_ns()
            llm.eval(list(suffix))
            sync()
            timings["ls_suffix_wall"].append(perf_counter_ns() - t)
            timings["ls_restore_total_wall"].append(perf_counter_ns() - r0)
            if token(llm) != cold_token:
                raise RuntimeError("LlamaState restore continuation diverged from cold baseline")

            # --- Native codec path ---
            llm.reset()
            r0 = perf_counter_ns()
            t = perf_counter_ns()
            native_set(llm, native_blob, prefix)
            sync()
            timings["nat_set_data_wall"].append(perf_counter_ns() - t)
            t = perf_counter_ns()
            llm.eval(list(suffix))
            sync()
            timings["nat_suffix_wall"].append(perf_counter_ns() - t)
            timings["nat_restore_total_wall"].append(perf_counter_ns() - r0)
            if token(llm) != cold_token:
                raise RuntimeError("native codec restore continuation diverged from cold baseline")

        record = {
            "prefix_tokens": length,
            "suffix_tokens": 1,
            "continuation_token": cold_token,
            "pickle_blob_bytes": len(pickle_blob),
            "native_blob_bytes": len(native_blob),
            "wrapper_overhead_bytes": len(pickle_blob) - len(native_blob),
            "pickle_sha256": pickle_sha,
            "native_sha256": native_sha,
            "phases": {name: stats(values) for name, values in timings.items()},
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "runtime": runtime.model_dump(),
        "fingerprint": fingerprint.model_dump(mode="json"),
        "gpu_before": before,
        "gpu_after": gpu(),
        "records": records,
        "note": (
            "In-memory restore isolates codec cost from disk I/O (Option 2). "
            "native_* phases bypass the LlamaState score buffer (Option 3). "
            "Enqueue timing is not throughput; all phases are CUDA-synchronized."
        ),
    }
    (args.artifacts / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
