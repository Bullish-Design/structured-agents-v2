"""Spike B proxy benchmark: prefill wall-clock scaling on available hardware.

We cannot run the true blend_by_reanchor kernel (blocked on the stable llama.cpp
C API -- see node_blend_live.py). The decision-relevant quantity we CAN measure
is per-token prefill cost: a splice that reuses M cached tokens and heals H of
them saves ~(M-H) tokens of prefill, and is only worth building if that saving
dominates the RoPE-shift + selective-recompute overhead.

This measures, for a range of context lengths N:
  * full re-prefill of N fresh tokens (submit N tokens, read logits)
so we get ms/token and the fixed decode overhead, on the GPU that is free today.
"""
import os, time, statistics, json, sys

from llama_cpp import Llama

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL"]
N_GPU_LAYERS = int(os.environ.get("NGL", "-1"))
LENGTHS = [32, 64, 128, 256, 512]
REPS = 5

t0 = time.perf_counter()
llm = Llama(model_path=MODEL, n_ctx=1024, n_batch=512,
            n_gpu_layers=N_GPU_LAYERS, seed=17018, verbose=False)
load_ms = (time.perf_counter() - t0) * 1000

# deterministic token stream
vocab = llm.n_vocab()
base_tokens = [(i * 131 + 7) % (vocab - 100) + 10 for i in range(max(LENGTHS))]

def prefill(n):
    llm.reset()
    llm.eval(base_tokens[:n])          # submit n tokens
    _ = llm.scores[n - 1]              # touch logits to force sync
    return

rows = []
for n in LENGTHS:
    # warm
    prefill(n)
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        prefill(n)
        ts.append((time.perf_counter() - t) * 1000)
    m = statistics.median(ts)
    rows.append({"tokens": n, "prefill_ms_median": round(m, 2),
                 "ms_per_token": round(m / n, 3), "reps_ms": [round(x, 1) for x in ts]})
    print(rows[-1], flush=True)

# derive per-token slope + fixed overhead from two endpoints
a, b = rows[0], rows[-1]
per_tok = (b["prefill_ms_median"] - a["prefill_ms_median"]) / (b["tokens"] - a["tokens"])
fixed = a["prefill_ms_median"] - per_tok * a["tokens"]

out = {"model": MODEL, "n_gpu_layers": N_GPU_LAYERS, "load_ms": round(load_ms, 1),
       "marginal_ms_per_token": round(per_tok, 4), "fixed_overhead_ms": round(fixed, 2),
       "rows": rows}
print(json.dumps(out, indent=2))
with open(os.environ.get("OUT", "/tmp/spike_b_prefill.json"), "w") as f:
    json.dump(out, f, indent=2)
