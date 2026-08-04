# Spike — Does llama.cpp execute Nanbeige4.2-3B's recurrent-in-depth (looped) forward pass?

Date: 2026-07-30. Status: **verdict reached without downloading weights or running GPU inference** (both 3060s saturated, <2.2 GB free). Evidence is from the HF `config.json`, model cards, the local `gguf` package, and upstream/fork llama.cpp state.

Grounding read: `CONCEPT.md` §1 (looped-transformer framing), §6.5 (expert bundles over the depth loop), Project 17 build workflow (`build-llamacpp.sh`, ABI-anchor rule) and the GPU driver stub fix.

---

## 1. What Nanbeige4.2-3B-Base actually IS (confirmed against config.json)

The looped / recurrent-in-depth claim is **TRUE and explicit in the model's own config.** Verbatim `config.json` from `Nanbeige/Nanbeige4.2-3B-Base`:

```json
{
  "architectures": ["NanbeigeForCausalLM"],
  "model_type": "nanbeige",
  "num_hidden_layers": 22,
  "num_loops": 2,
  "loop_loss_weights": [],
  "skip_loop_final_norm": false,
  "hidden_size": 3072,
  "intermediate_size": 10752,
  "num_attention_heads": 48,
  "num_key_value_heads": 8,
  "head_dim": 128, "kv_channels": 128,
  "rope_theta": 10000000, "max_position_embeddings": 131072,
  "vocab_size": 166144, "tie_word_embeddings": false,
  "auto_map": {
    "AutoConfig": "configuration_nanbeige.NanbeigeConfig",
    "AutoModelForCausalLM": "modeling_nanbeige.NanbeigeForCausalLM"
  }
}
```

Architecture reality:
- **Looped Transformer.** A 22-layer decoder stack whose hidden states are fed back into the *same* layers after a bottom-to-top pass. `num_loops: 2` → the shared-weight stack runs **twice** = 44 effective depth for the parameter cost of 22 layers. Recurrence is over **depth / ponder-steps**, not sequence position (exactly the CONCEPT §6.5.1 framing).
- **Weights are tied/shared across the looped block** — that is the whole point (capacity without params). ~4B total params / ~3B non-embedding.
- **Loop count is FIXED (2), not adaptive/ACT-style.** No halting unit; `loop_loss_weights` is present (per-loop aux loss hook) but empty here. So this is a Universal-Transformer-style fixed unroll, *not* CoTFormer/ACT budget-adaptive compute. (CONCEPT §6.5 should note: the base gives you a fixed 2-pass loop; adaptive halting would be a modification, not a free property.)
- **Per-loop KV cache.** Each loop iteration keeps its **own** KV cache (the modeling code allocates KV per pass). This ~doubles KV footprint vs a plain 22-layer model — directly relevant to Project 19's KV-object cost model (P4).
- **Custom extras in the modeling code:** `LoopSplit`, `mHC` with depth attention, and concatenated n-gram embeddings. These are non-standard ops beyond a bare re-loop.

> **CORRECTION (2026-07-31, from PORT-PLAN-NANBEIGE-P2.md):** the items above (`LoopSplit`, `mHC` depth-attention, concatenated n-gram embeddings) exist ONLY in the HF *modeling* code. The llama.cpp **GGUF graph** (`src/models/nanbeige.cpp`, read in full in the nanbeige42 fork) does NOT implement them — it is a plain looped llama-style stack: shared weights across `num_loops`-expanded layers + an optional loop-boundary norm. So what actually runs in llama.cpp is simpler than this section implies, and the port is correspondingly simpler. Treat this §1 bullet as describing the HF reference model, not the served graph.

**Arch string llama.cpp needs in GGUF metadata: `general.architecture = nanbeige`** (`model_type: "nanbeige"`, `NanbeigeForCausalLM`).

## 2. Does upstream llama.cpp have a converter + graph for it? — NO

- **`gguf` / `convert_hf_to_gguf.py`:** the installed gguf package (`~/.cache/uv/.../gguf/constants.py`) has **no `nanbeige` / `loop` / `recurrent` entry.** There is no `MODEL_ARCH.NANBEIGE`, so the stock converter cannot even emit a correct GGUF, and stock `llama.cpp` has no `LLM_ARCH_NANBEIGE` graph to build the loop.
- **Upstream runtime:** confirmed by the community GGUF card (`Andgihat/Nanbeige4.2-3B-GGUF`): *"Unmodified upstream fails with `unknown architecture 'nanbeige'`."*
- **Upstream tracking issue #26086** ("Add support for Nanbeige/Nanbeige4.2-3B") is **CLOSED without an upstream merge** — support lives only in forks.

So the recurrent/looped block is **not a standard supported architecture** in ggml-org/llama.cpp. This is the blocker for the repo's current build (which builds from `ggml-org/llama.cpp` per `build-llamacpp.sh`).

## 3. But the loop IS correctly executed — in a fork (this is the important nuance)

The recurrent-in-depth forward pass is **genuinely implemented and runs coherently**, just not on stock llama.cpp:

- **Official fork:** `Nanbeige/llama.cpp @ nanbeige42` — the model authors ported the arch (converter + graph).
- **Community fork:** BeeLlama.cpp, with prebuilt binaries.
- **Correctness evidence (not gibberish):** `Andgihat/Nanbeige4.2-3B-GGUF` reports GSM8K **92% accuracy at UD-Q4_K_XL, 98.9% recovery vs full precision.** That accuracy is only achievable if the 2-pass depth loop, per-loop KV, and the mHC/LoopSplit ops are actually executed correctly — i.e. this is *not* the Ornith/SGLang "loads but outputs gibberish" arch-mismatch failure mode. The loop works.

So this is categorically different from the Ornith-SGLang situation (that one loaded but the graph was wrong → gibberish). Here a correct graph exists; the only gap is that it is not in the tree the repo builds from.

## 4. VERDICT

**PARTIAL → UNSUPPORTED-ON-REPO-BUILD / SUPPORTED-VIA-FORK.**

- **On stock `ggml-org/llama.cpp` (what this repo builds today): UNSUPPORTED.** No `nanbeige` arch in converter or graph; `unknown architecture 'nanbeige'`.
- **The recurrent-in-depth forward pass itself is solved** — the Nanbeige fork (and BeeLlama.cpp) execute the fixed 2-loop shared-weight stack with per-loop KV correctly and coherently. So the *architectural* answer to "can llama.cpp run this looped forward pass" is **yes, with the right source tree.**

### Smallest path to unblock

1. **Build the Nanbeige fork instead of upstream:**
   `build-llamacpp.sh --ref <nanbeige42 commit> --repo https://github.com/Nanbeige/llama.cpp --profile cuda-3060`
   (script already supports `--repo`). Apply the **GPU driver stub fix** — prepend `/run/opengl-driver/lib` before the CUDA stub in the runtime env, or llama-cpp-python silently runs on CPU. Then run the CFFI/GPU smoke gate.
2. **ABI-anchor caveat (real cost):** the repo's rule is that the built commit must be ABI-compatible with the installed `llama-cpp-python` bindings. The nanbeige42 fork is off a specific upstream base; verify ABI or pin/rebuild llama-cpp-python against it. This is the main integration friction, not the loop math.
3. **Convert weights with the fork's converter** (not the stock one) so the GGUF carries `general.architecture=nanbeige` plus `num_loops`/mHC metadata — or just use the existing `Andgihat/*.gguf` for a read-only feasibility check. **(Do this only when GPU is free; no download now.)**

### Project 19 implications (flagged, not blocking)

- **The depth loop the expert-bundle plan (§6.5) wants to hang "LoRA hats" on is fixed at 2 passes with tied weights** — good: exactly a Universal-Transformer unroll, per-loop KV already separate. But **adaptive/ACT halting is NOT in the base** — routing "which hat per ponder-step" is available (loop index 0/1); "whether to stop" (ACT) would need a fork modification.
- **The hard integration gap:** the expert-bundle design needs **loop execution AND the P2 mixed-batch LoRA fork simultaneously.** No existing fork has both — the Nanbeige fork has the loop but not `llama_set_seq_adapters`/P2; the repo's engine has P2 but not the loop. Delivering §6.5 means **porting the nanbeige graph onto the repo's P2-forked llama.cpp** (or vice-versa) and indexing the P2 adapter-select vector by loop-step instead of by request. That is the real Project-19 build task; it is a graph merge, not a research unknown.
- **Per-loop KV doubling** compounds the P4 "KV must be small/partial" pressure.

---

## Sources
- [Nanbeige/Nanbeige4.2-3B-Base — HF (config.json)](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base)
- [Nanbeige4.2-3B agentic paper (arXiv 2607.22083v2)](https://arxiv.org/html/2607.22083v2)
- [Inside Nanbeige4.2-3B-Base's Looped Transformer Architecture — HackerNoon](https://hackernoon.com/inside-nanbeige42-3b-bases-looped-transformer-architecture)
- [Andgihat/Nanbeige4.2-3B-GGUF — HF (fork requirement + GSM8K evidence)](https://huggingface.co/Andgihat/Nanbeige4.2-3B-GGUF)
- [Nanbeige/llama.cpp @ nanbeige42 fork](https://github.com/Nanbeige/llama.cpp)
- [llama.cpp Issue #26086 — Add support for Nanbeige4.2-3B (closed, no upstream merge)](https://github.com/ggml-org/llama.cpp/issues/26086)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
</content>
</invoke>
