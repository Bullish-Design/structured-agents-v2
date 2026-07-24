"""Gate 2: does llama.cpp (GGUF) tokenization agree, token-ID for token-ID, with
the HF reference tokenizer (deepreinforce-ai/Ornith-1.0-9B)?

If they disagree, xgrammar's token masks (built over one tokenizer) would mask the
wrong IDs for the other → silent grammar corruption. This is the correctness
prerequisite for the whole grammar pillar.

Compares on: the xgrammar doc's boundary probes, Ornith-specific special/tool
markers, and a deterministic fuzz corpus (unicode / json / whitespace).
"""
import random, sys
from llama_cpp import Llama
from transformers import AutoTokenizer

MODEL = "/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf"
HF_REPO = "deepreinforce-ai/Ornith-1.0-9B"

llm = Llama(model_path=MODEL, n_ctx=64, vocab_only=True, verbose=False)
hf = AutoTokenizer.from_pretrained(HF_REPO)
print(f"[vocab] llama n_vocab={llm.n_vocab()}  hf vocab_size={hf.vocab_size} len={len(hf)}")

def llama_ids(s: str):
    return llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)

def hf_ids(s: str):
    return hf.encode(s, add_special_tokens=False)

PROBES = [
    "", "a", " a", "  a", "\n", "\n\n", "\t",
    '{"key":"value"}', '{ "key": "value" }',
    "café", "naïve", "日本語", "🙂", "\\u0000",
    "true false null -1 1.5e-4",
    # Ornith / Qwen3.5 special + tool markers (from the chat template)
    "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>",
    "<tool_call>", "</tool_call>", "<function=", "<parameter=", "</think>",
    "<|im_start|>assistant\n", "system\nYou are helpful<|im_end|>\n",
]

def fuzz_corpus(n=600, seed=17):
    rng = random.Random(seed)
    alphabets = [
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        " \t\n{}[]\":,._-/\\", "áéíóúñüçß日本語中文한국어",
        "😀🙂👍🔥✨", '{"a":1,"b":[true,false,null],"c":"x"}', "   \n\n\t  ",
    ]
    out = []
    for _ in range(n):
        parts = [rng.choice(alphabets) for _ in range(rng.randint(1, 4))]
        s = "".join(rng.choice(p) for p in parts for _ in range(rng.randint(1, 12)))
        out.append(s)
    return out

def run(label, cases):
    mism = []
    for s in cases:
        try:
            a, b = llama_ids(s), hf_ids(s)
        except Exception as e:
            mism.append((s, f"ERROR {type(e).__name__}: {e}", None)); continue
        if a != b:
            mism.append((s, a, b))
    print(f"[{label}] {len(cases)-len(mism)}/{len(cases)} match", end="")
    print("  ALL MATCH" if not mism else f"  {len(mism)} MISMATCH")
    for s, a, b in mism[:20]:
        print(f"   {s!r}\n     llama={a}\n     hf   ={b}")
    return len(mism)

m1 = run("probes", PROBES)
m2 = run("fuzz", fuzz_corpus())
total = m1 + m2
print(f"\n[GATE 2] {'PASS' if total == 0 else f'MISMATCH ({total})'}")
sys.exit(0 if total == 0 else 1)
