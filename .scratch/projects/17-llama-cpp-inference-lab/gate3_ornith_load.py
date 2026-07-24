"""Gate 3, step 1: does Ornith-1.0-9B (hybrid/GatedDeltaNet) load and generate
COHERENTLY in this llama-cpp-python on CPU? Prior sglang work never got coherent
output, so this is the prerequisite risk before any KV save/restore test.
"""
import sys, time
from llama_cpp import Llama

MODEL = "/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf"

t0 = time.time()
try:
    llm = Llama(
        model_path=MODEL,
        n_ctx=1024,
        n_batch=256,
        n_threads=8,
        logits_all=False,
        verbose=True,
    )
except Exception as e:
    print(f"[LOAD-FAIL] {type(e).__name__}: {e}")
    sys.exit(2)
print(f"[loaded] in {time.time()-t0:.1f}s  n_ctx={llm.n_ctx()}  n_vocab={llm.n_vocab()}")

prompt = "Q: What is the capital of France? A:"
t1 = time.time()
out = llm(prompt, max_tokens=24, temperature=0.0, seed=1234)
txt = out["choices"][0]["text"]
print(f"[gen] in {time.time()-t1:.1f}s")
print(f"[prompt] {prompt!r}")
print(f"[output] {txt!r}")
# crude coherence signal: does it contain 'Paris'?
print(f"[coherence] mentions Paris: {'paris' in txt.lower()}")
