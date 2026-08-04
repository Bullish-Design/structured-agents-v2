"""Project 20 — throughput re-measure through the shipping library path.

Unlike ``benchmarks/project17/run_p2_throughput.py`` (which drove the spike's
``ContextPoolRouter``), this measures the actual ``MultiLoRARouter`` the library
ships, comparing its two backends plus a sequential baseline over one mixed-adapter
workload:

  sequential : each request alone (batch-1), seq_routed backend one at a time
  router     : ``backend="context_pool"`` — batched per adapter context, multiplexed
  fork       : ``backend="seq_routed"`` — one context, one mixed-batch decode

Phase 1 (``--batch-sizes``): fixed pool K=2, sweep S. Phase 2 (``--k-sweep``):
fixed S, grow the adapter pool K to the low dozens to find where the masked K× LoRA
overhead (``overhead ≈ 2·K·r/d``) starts to erode the fork's structural win. Extra
pool adapters reuse the two probe GGUFs under distinct names — the masked path
computes every pool adapter's delta per token regardless of content, so pool *size*
is what drives the overhead being measured.

Requires the P2 fork lib (LLAMA_CPP_LIB_PATH -> a p2fork lib dir). GPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROMPTS = [
    "The capital of France is",
    "In a distant galaxy,",
    "def add(a, b):",
    "Once upon a time",
    "The three primary colors are",
    "Water boils at",
    "The opposite of hot is",
    "To make tea you first",
]


def gpu_snapshot() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _sync() -> None:
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass  # torch optional; llama_decode is synchronous, so timing stays sound


def _pool(k: int, lora_a: Path, lora_b: Path):
    from inferference.router import AdapterSpec

    paths = [str(lora_a), str(lora_b)]
    return tuple(AdapterSpec(name=f"probe-{i}", gguf_path=paths[i % 2]) for i in range(k))


def _make_router(model: Path, adapters, *, backend: str, n_ctx: int, n_seq_max: int, seed: int):
    from inferference.models import EngineConfig
    from inferference.router import MultiLoRARouter, RouterConfig

    return MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(model), n_ctx=n_ctx, n_gpu_layers=-1, n_batch=128, seed=seed, backend="cuda"
            ),
            adapters=adapters,
            n_seq_max=n_seq_max,
            include_base=False,
            backend=backend,
        )
    )


def _requests(s: int, pool_names: list[str], max_tokens: int):
    from inferference.router import RouteRequest

    return [
        RouteRequest(
            prompt=PROMPTS[i % len(PROMPTS)],
            adapter=pool_names[i % len(pool_names)],
            max_tokens=max_tokens,
            request_id=f"r{i}",
        )
        for i in range(s)
    ]


def _timed(fn, reps: int):
    fn()  # warmup (graph build / cache)
    best = None
    out = None
    for _ in range(reps):
        _sync()
        t0 = perf_counter_ns()
        out = fn()
        _sync()
        dt = perf_counter_ns() - t0
        best = dt if best is None else min(best, dt)
    return best / 1e6, out  # ms


def _toks(results):
    return [tuple(r.token_ids) for r in sorted(results, key=lambda r: r.request_id)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--n-seq-max", default=16, type=int)
    p.add_argument("--max-tokens", default=32, type=int)
    p.add_argument("--batch-sizes", default="2,4,8")
    p.add_argument("--k-sweep", default="2,4,8,16,24")
    p.add_argument("--k-sweep-s", default=16, type=int, help="fixed S for the K sweep")
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("GPU-only policy requires CUDA_VISIBLE_DEVICES in {0,1} and LLAMA_CPP_LIB_PATH")

    gpu_before = gpu_snapshot()

    # Each MultiLoRARouter loads its OWN full copy of the model, so two backends can't
    # coexist on a 12 GB 3060. Open exactly one router at a time; measure fork-side and
    # router-side in separate passes and merge by S / K.
    sizes = [int(x) for x in args.batch_sizes.split(",")]

    # ---- Phase 1: fixed K=2, sweep S ----
    pool2 = _pool(2, args.lora_a, args.lora_b)
    names2 = [a.name for a in pool2]

    fork_side: dict[int, tuple] = {}  # S -> (seq_ms, seq_out, fk_ms, fk_out)
    fork = _make_router(
        args.model, pool2, backend="seq_routed", n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed
    )
    try:
        for s in sizes:
            if s > args.n_seq_max:
                continue
            reqs = _requests(s, names2, args.max_tokens)
            seq_ms, seq_out = _timed(lambda rs=reqs: [g for r in rs for g in fork.run([r])], args.reps)
            fk_ms, fk_out = _timed(lambda rs=reqs: fork.run(rs), args.reps)
            fork_side[s] = (seq_ms, _toks(seq_out), fk_ms, _toks(fk_out))
    finally:
        fork.close()

    router = _make_router(
        args.model, pool2, backend="context_pool", n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed
    )
    router_side: dict[int, tuple] = {}
    try:
        for s in sizes:
            if s > args.n_seq_max:
                continue
            reqs = _requests(s, names2, args.max_tokens)
            rt_ms, rt_out = _timed(lambda rs=reqs: router.run(rs), args.reps)
            router_side[s] = (rt_ms, _toks(rt_out))
    finally:
        router.close()

    phase1 = []
    for s in sizes:
        if s > args.n_seq_max:
            phase1.append({"S": s, "skipped": f"S>{args.n_seq_max}"})
            continue
        seq_ms, seq_toks, fk_ms, fk_toks = fork_side[s]
        rt_ms, rt_toks = router_side[s]
        gen_tokens = s * args.max_tokens
        identical = seq_toks == rt_toks == fk_toks
        row = {
            "S": s,
            "gen_tokens": gen_tokens,
            "outputs_identical": identical,
            "sequential_ms": round(seq_ms, 1),
            "sequential_tok_s": round(gen_tokens / (seq_ms / 1e3), 1),
            "router_ms": round(rt_ms, 1),
            "router_tok_s": round(gen_tokens / (rt_ms / 1e3), 1),
            "fork_ms": round(fk_ms, 1),
            "fork_tok_s": round(gen_tokens / (fk_ms / 1e3), 1),
            "fork_vs_router": round(rt_ms / fk_ms, 2),
            "fork_vs_sequential": round(seq_ms / fk_ms, 2),
        }
        phase1.append(row)
        print(
            f"[thru] S={s}: seq={seq_ms:.0f}ms/{row['sequential_tok_s']}tps "
            f"router={rt_ms:.0f}ms/{row['router_tok_s']}tps "
            f"fork={fk_ms:.0f}ms/{row['fork_tok_s']}tps "
            f"(fork/router x{row['fork_vs_router']}) identical={identical}",
            flush=True,
        )

    # ---- Phase 2: fixed S, grow K to find the masked-overhead crossover ----
    phase2 = []
    s = min(args.k_sweep_s, args.n_seq_max)
    d = 4096  # Ornith hidden dim (o_proj in=out=d); probe rank r=16
    r_rank = 16
    for k in (int(x) for x in args.k_sweep.split(",")):
        poolk = _pool(k, args.lora_a, args.lora_b)
        namesk = [a.name for a in poolk]
        reqs = _requests(s, namesk, args.max_tokens)
        gen_tokens = s * args.max_tokens

        # Both legs can exhaust VRAM as K grows: context_pool allocates K+1 contexts,
        # and even the fork loads K adapter weight sets (its per-adapter cost). Guard
        # each independently and record "skipped" rather than crashing the whole run.
        fk_ms = None
        try:
            fk = _make_router(
                args.model, poolk, backend="seq_routed", n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed
            )
            try:
                fk_ms, _ = _timed(lambda rs=reqs, rt=fk: rt.run(rs), args.reps)
            finally:
                fk.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[ksweep] K={k}: fork skipped ({type(exc).__name__}: VRAM?)", flush=True)

        rk_ms = None
        try:
            rk = _make_router(
                args.model, poolk, backend="context_pool", n_ctx=args.n_ctx, n_seq_max=args.n_seq_max, seed=args.seed
            )
            try:
                rk_ms, _ = _timed(lambda rs=reqs, rt=rk: rt.run(rs), args.reps)
            finally:
                rk.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[ksweep] K={k}: context_pool skipped ({type(exc).__name__}: VRAM?)", flush=True)

        row = {
            "K": k,
            "S": s,
            "gen_tokens": gen_tokens,
            "fork_ms": round(fk_ms, 1) if fk_ms else None,
            "fork_tok_s": round(gen_tokens / (fk_ms / 1e3), 1) if fk_ms else None,
            "router_ms": round(rk_ms, 1) if rk_ms else None,
            "router_tok_s": round(gen_tokens / (rk_ms / 1e3), 1) if rk_ms else None,
            "fork_vs_router": round(rk_ms / fk_ms, 2) if (rk_ms and fk_ms) else None,
            "predicted_masked_overhead": round(2 * k * r_rank / d, 3),  # 2·K·r/d
        }
        phase2.append(row)
        ftxt = f"{fk_ms:.0f}ms/{row['fork_tok_s']}tps" if fk_ms else "skipped"
        rtxt = f"{rk_ms:.0f}ms/{row['router_tok_s']}tps (x{row['fork_vs_router']})" if rk_ms else "skipped"
        print(
            f"[ksweep] K={k} S={s}: fork={ftxt} router={rtxt} pred_overhead={row['predicted_masked_overhead']}",
            flush=True,
        )

    result = {
        "config": {
            "n_ctx": args.n_ctx,
            "n_seq_max": args.n_seq_max,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
            "seed": args.seed,
            "k_sweep_s": s,
        },
        "phase1_s_sweep_k2": phase1,
        "phase2_k_sweep": phase2,
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
        "all_outputs_identical": all(r.get("outputs_identical", True) for r in phase1 if "skipped" not in r),
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "library_throughput.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phase1": phase1, "phase2": phase2}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
