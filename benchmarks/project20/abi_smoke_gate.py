"""Project 20 — ABI smoke gate for the P2 fork lib (Workstream A4).

A fork build is only trustworthy after it passes a behavioral gate, not just a
symbol probe (06-LLAMACPP-BUILD-WORKFLOW.md §6). This checks, against the lib on
LLAMA_CPP_LIB_PATH:

  1. surface probe   — both fork symbols resolve (llama_set_seq_adapters/adapter)
  2. Ornith gen      — load the GGUF, greedily generate ~32 tokens (no crash)
  3. tokenizer round — encode/decode a probe string losslessly enough

Exit 0 iff all pass. GPU env per llama-cpp-gpu-driver-stub-fix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--max-tokens", default=32, type=int)
    args = p.parse_args()

    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    import llama_cpp
    from inferference.seq_routing import SEQ_ROUTING_SYMBOLS, library_supports_seq_routing
    from llama_cpp import Llama

    lib = llama_cpp.llama_cpp._lib

    # 1. surface probe
    present = {s: hasattr(lib, s) for s in SEQ_ROUTING_SYMBOLS}
    print(f"[gate:1] surface probe: {present}", flush=True)
    if not library_supports_seq_routing(lib):
        print("[gate:1] FAIL — fork routing symbols absent; this is not a P2 fork lib", flush=True)
        return 1

    # 2. Ornith generation
    llm = Llama(model_path=str(args.model), n_ctx=2048, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False)
    try:
        import itertools

        toks = llm.tokenize(b"The capital of France is", add_bos=True, special=True)
        # generate() yields until EOS/context-full; take only max_tokens.
        out = list(itertools.islice(llm.generate(list(toks), top_k=1, temp=0.0), args.max_tokens))
        text = llm.detokenize(out).decode("utf-8", errors="replace")
        print(f"[gate:2] Ornith gen: {len(out)} tokens -> {text[:60]!r}", flush=True)
        if len(out) == 0:
            print("[gate:2] FAIL — no tokens generated", flush=True)
            return 1

        # 3. tokenizer round-trip
        probe = "def add(a, b):\n    return a + b"
        rt = llm.detokenize(llm.tokenize(probe.encode("utf-8"), add_bos=False, special=True)).decode(
            "utf-8", errors="replace"
        )
        ok = probe.strip() in rt or rt.strip() in probe or rt.strip() == probe.strip()
        print(f"[gate:3] tokenizer round-trip ok={ok}: {rt[:60]!r}", flush=True)
        if not ok:
            print("[gate:3] FAIL — tokenizer round-trip drift", flush=True)
            return 1
    finally:
        llm.close()

    print("[gate] PASS — fork lib is surface-correct and behaviorally sound", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
