"""GPU smoke test for a CUDA (sm_86 / RTX 3060) llama.cpp build.

Loads Ornith-1.0-9B-Q4 fully offloaded (n_gpu_layers=-1) through the installed
high-level llama_cpp, pointed at the custom CUDA lib set via LLAMA_CPP_LIB_PATH.

Verifies: (a) GPU is actually used (llama logs show CUDA devices + offloaded
layers), (b) coherent output, (c) generation tok/s vs the ~2.7 tok/s CPU
baseline (Gate 3). Also measures prefill speed on a ~512-token prompt and peak
VRAM via nvidia-smi.

Run:
  LLAMA_CPP_LIB_PATH=<out>/lib \
  LD_LIBRARY_PATH="$(cat .cuda_runtime_ld):$LD_LIBRARY_PATH" \
  .venv-spike/bin/python gpu_smoke.py
"""
import os, sys, time, subprocess

MODEL = "/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf"


def vram_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [int(x) for x in out.split()]
    except Exception as e:
        return f"nvidia-smi failed: {e}"


from llama_cpp import Llama

print(f"[env] LLAMA_CPP_LIB_PATH={os.environ.get('LLAMA_CPP_LIB_PATH')}")
print(f"[vram] before load (MiB/gpu): {vram_used_mib()}")

t0 = time.time()
llm = Llama(
    model_path=MODEL,
    n_ctx=1024,
    n_batch=512,
    n_gpu_layers=-1,   # offload ALL layers to GPU
    logits_all=False,
    verbose=True,
)
print(f"[loaded] in {time.time()-t0:.1f}s  n_ctx={llm.n_ctx()}  n_vocab={llm.n_vocab()}")
print(f"[vram] after load (MiB/gpu): {vram_used_mib()}")

# --- coherence + gen tok/s ---
prompt = "Q: What is the capital of France? A:"
t1 = time.time()
out = llm(prompt, max_tokens=64, temperature=0.0, seed=1234)
dt = time.time() - t1
txt = out["choices"][0]["text"]
gen_toks = out["usage"]["completion_tokens"]
print(f"[gen] {gen_toks} tok in {dt:.2f}s = {gen_toks/dt:.2f} tok/s")
print(f"[output] {txt!r}")
print(f"[coherence] mentions Paris: {'paris' in txt.lower()}")
print(f"[vram] after gen (MiB/gpu): {vram_used_mib()}")

# --- prefill speed on ~512-token prompt ---
big_prompt = ("The history of computing spans many decades. " * 60).strip()
llm.reset()
t2 = time.time()
res = llm.create_completion(big_prompt, max_tokens=1, temperature=0.0)
dt2 = time.time() - t2
n_prompt = res["usage"]["prompt_tokens"]
print(f"[prefill] {n_prompt} prompt tok in {dt2:.2f}s = {n_prompt/dt2:.1f} tok/s prefill")
