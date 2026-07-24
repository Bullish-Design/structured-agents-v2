"""Probe llama-cpp-python API surface for the 3 load-bearing questions.
No model load required — pure introspection of the binding.
"""
import inspect
import llama_cpp
from llama_cpp import Llama

print(f"llama_cpp {llama_cpp.__version__}\n")

def has(obj, *names):
    for n in names:
        mark = "yes" if hasattr(obj, n) else "NO "
        print(f"  [{mark}] {n}")

# --- Q1: logit access + owned sampler ---
print("Q1  Logit access / owned sampler")
sig = inspect.signature(Llama.__init__)
for p in ("logits_all", "n_batch", "n_ubatch", "embedding"):
    print(f"  [{'yes' if p in sig.parameters else 'NO '}] Llama.__init__({p}=...)")
has(Llama, "eval_logits", "scores", "logits_to_logprobs", "sample", "_sample")
# low-level logits getter
has(llama_cpp, "llama_get_logits", "llama_get_logits_ith", "llama_decode", "llama_batch_init")

# --- Q2: multi-LoRA serving ---
print("\nQ2  LoRA adapters (multi + per-sequence)")
for p in ("lora_path", "lora_base", "lora_scale"):
    print(f"  [{'yes' if p in sig.parameters else 'NO '}] Llama.__init__({p}=...)")
has(Llama, "set_lora_adapter", "add_lora_adapter", "lora_adapters")
# low-level adapter API — the real question for mixed-batch
has(llama_cpp, "llama_adapter_lora_init", "llama_set_adapter_lora",
    "llama_rm_adapter_lora", "llama_clear_adapter_lora",
    "llama_lora_adapter_init", "llama_lora_adapter_set")
# per-sequence?  inspect llama_batch fields + set_adapter signature
for fn in ("llama_set_adapter_lora",):
    f = getattr(llama_cpp, fn, None)
    if f is not None:
        print(f"  {fn} argtypes: {getattr(f, 'argtypes', '?')}")

# --- Q3: KV cache save/restore/reuse ---
print("\nQ3  KV state save / restore / reuse")
has(Llama, "save_state", "load_state", "reset", "n_tokens")
has(llama_cpp, "llama_get_state_size", "llama_copy_state_data",
    "llama_state_seq_get_size", "llama_state_seq_get_data",
    "llama_state_seq_set_data", "llama_kv_cache_seq_cp",
    "llama_kv_self_seq_cp", "llama_memory_seq_cp")

# --- bonus: grammar / sampler chain ---
print("\nBonus  Grammar + sampler chain (xgrammar integration surface)")
has(llama_cpp, "LlamaGrammar")
has(llama_cpp, "llama_sampler_init_grammar", "llama_sampler_chain_init",
    "llama_sampler_init_logit_bias", "llama_sampler_apply")
