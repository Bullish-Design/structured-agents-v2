# Small-Model Survey for the tower vLLM Backend

**Date:** 2026-07-02 · **Author:** research pass for structured-agents-v2
**Question:** Which high-quality small LLMs — with good quantization (from the
provider or Unsloth) — actually work with *this* system?

> This is scoped to the real deployment, not a generic "best small model" list.
> "Works with this system" = loads on **vLLM 0.11.0**, fits the **8 GB Quadro
> RTX 4000 (Turing sm_75)**, keeps **XGrammar** structured output + **per-agent
> LoRA**, and has a **Turing-compatible quantization** (AWQ / GPTQ-INT4 / BnB —
> not FP8, not GGUF). See [[tower-vllm-live]] for the live baseline.

---

## 1. The hard filter — what "compatible" means here

Three gates, in order. A model must pass all three.

### 1a. vLLM 0.11.0 architecture support
Checked directly against the pinned image's model registry. Supported small
families (non-exhaustive, relevant ones):

- **Qwen**: `Qwen2ForCausalLM` (2/2.5), `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`
- **Llama/Mistral**: `LlamaForCausalLM` (Llama 3.x, Mistral, Ministral, InternLM3 all map here), `Llama4ForCausalLM`
- **Gemma**: `Gemma/2/3ForCausalLM`, `Gemma3nForCausalLM` — **NOT `Gemma4` / `gemma4`** (verified absent)
- **Phi**: `PhiForCausalLM`, `Phi3ForCausalLM` (Phi-3.5, **Phi-4-mini**), `PhiMoEForCausalLM`, `Phi4MM/Phi4Multimodal`
- **Others**: `GraniteForCausalLM`(+MoE), `SmolLM3ForCausalLM` (via transformers backend), `Olmo2/3`, `GlmForCausalLM`/`Glm4`, `MiniCPM/3`, `Nemotron/H`, `DeciLM`, `Exaone4`, `Cohere/Cohere2`, `Falcon/FalconH1`, `StableLm`, `InternLM2`

> ⚠️ **Newer models are blocked by the arch gate:** `gemma-4-*`, and any model
> whose arch postdates vLLM 0.11.0 (likely Qwen3.5/3.6). Running those needs a
> newer `VLLM_TAG` — and a newer tag risks **dropping sm_75/Turing support**, so
> verify before bumping. This is a strong argument for a GPU upgrade (§5).

### 1b. Turing (sm_75) hardware limits — narrows the *quantization*
The GPU is Turing. This is the decisive constraint on *format*:

| Feature | Needs | On this card | Consequence |
|---|---|---|---|
| FlashAttention-2 | sm_80+ | ❌ | falls back to FlexAttention (slower) |
| **FP8** compute | sm_89+ | ❌ | **all `-FP8` repos are unusable** |
| AWQ/GPTQ **Marlin** kernels | sm_80+ | ❌ | plain `awq`/`gptq` kernels only (slower INT4) |
| AWQ / GPTQ-INT4 (plain) | sm_75 | ✅ | **the format to use** |
| bitsandbytes 4-bit (incl. Unsloth dynamic) | sm_75 | ✅ | works, but LoRA pairing is rough (below) |
| FlashInfer sampler | — | ❌ (crash) | requires `VLLM_USE_FLASHINFER_SAMPLER=0` |

**Net:** on Turing, prefer a **pre-quantized AWQ** (or GPTQ-INT4) repo. Avoid FP8
entirely. GGUF is excluded regardless (breaks LoRA — see 1c).

### 1c. LoRA + XGrammar — the project's non-negotiables
- **Per-agent LoRA** is the core thesis. In vLLM: **AWQ + LoRA** and **GPTQ + LoRA**
  work; **BnB/Unsloth-dynamic + LoRA** has had rough edges (noted in `.env`);
  **GGUF supports no LoRA at all** → GGUF is out.
- **XGrammar** (json_schema / regex / choice) is logit-masking and works across
  these text architectures — already verified live on Qwen3-4B-AWQ.

### 1d. VRAM budget on 8 GB
With `GPU_MEMORY_UTILIZATION=0.85` (~6.8 GB) minus ~0.7 GB for the Windows
desktop compositor, **~6 GB is usable** for weights + KV cache + LoRA:

| Weights | Approx VRAM | Fit on 8 GB | Headroom for KV + LoRA |
|---|---|---|---|
| ~1.7B fp16 | ~3.4 GB | ✅ | large |
| ~3–4B AWQ-INT4 | ~2.2–3.0 GB | ✅✅ | **good — the sweet spot** |
| ~8B AWQ-INT4 | ~5.0–5.5 GB | ⚠️ tight | little (drop context / few LoRAs) |
| ~14B AWQ-INT4 | ~8–9 GB | ❌ | does not fit |

---

## 2. Recommended shortlist

All repos below were verified to exist on HF (2026-07-02). Baseline for
comparison = the incumbent **Qwen3-4B-AWQ**: ~9.6 tok/s single-stream, ~12.5×
batched at 32-way, ~2 s/command, 100% schema-valid.

### Tier 1 — deploy on the current 8 GB box today

| # | Model | Params | Quant repo (Turing-OK) | Fits | LoRA | Why |
|--:|---|---|---|:--:|:--:|---|
| 1 | **Qwen/Qwen3-4B-AWQ** | 4B | official AWQ | ✅✅ | ✅ | Incumbent, verified live. Best all-rounder: tool-calling + structured output, 128K ctx, matches Qwen2.5-7B. |
| 2 | **Qwen3-4B-Instruct-2507** | 4B | community AWQ (`cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`) or self-AWQ | ✅✅ | ✅ | The **non-thinking** 2507 refresh — better direct-instruct behavior for constrained JSON than base Qwen3 (no `<think>` budget wasted). Official repo has FP8 only (**unusable on Turing**), so use community/self AWQ. |
| 3 | **Qwen/Qwen3-8B-AWQ** | 8B | official AWQ | ⚠️ | ✅ | Highest quality that still fits (~5.5 GB). Matches previous-gen 14B. Trade: little KV/LoRA headroom, lower max concurrency. |
| 4 | **microsoft/Phi-4-mini-instruct** | 3.8B | `unsloth/...-bnb-4bit` or GPTQ-INT4 (self/community) | ✅ | ⚠️ (BnB) | Strongest **reasoning** in class (~68% MMLU, ~70% HumanEval, rivals 7–9B). Best if the agents do multi-step logic. LoRA on BnB is the caveat — GPTQ path preferred if you need adapters. |
| 5 | **HuggingFaceTB/SmolLM3-3B** | 3B | fp16 (fits) or self-AWQ | ✅ | ⚠️ | Beats Llama-3.2-3B & Qwen2.5-3B; dual-mode reasoning, 128K ctx. Small enough to run fp16 → skip quant. Caveat: runs via vLLM's **transformers backend**, so confirm LoRA support. |

### Tier 2 — solid alternatives / special cases

| Model | Params | Quant | Note |
|---|---|---|---|
| unsloth/Llama-3.2-3B-Instruct | 3B | AWQ/GPTQ (community) or fp16 | Most mature tool-calling + biggest community; ungated Unsloth mirror avoids the Meta gate. |
| mistralai/Ministral-8B-Instruct-2410 | 8B | AWQ (self/community) | Strong agentic/native-JSON niche, but 8B is tight on 8 GB. |
| google/gemma-3-4b-it | 4B | bnb-4bit / self-AWQ / QAT (`-qat-q4_0-unquantized`) | Good general model; **tool-calling weaker** than Qwen/Llama — less ideal for routing. |
| ibm-granite/granite-3.3-2b-instruct | 2B | fp16 / self-quant | Enterprise, function-calling focus, very small → high concurrency headroom. |
| allenai/OLMo-2-1124-7B-Instruct | 7B | self-quant | Fully open (data + weights) if provenance matters; 7B tight. |

### What to avoid on this card
- Anything **`-FP8`** (Qwen3-*-FP8, etc.) — no FP8 on Turing.
- **GGUF** builds (Unsloth/others) — no LoRA in vLLM.
- **`unsloth/gemma-4-*`**, and probable Qwen3.5/3.6 — arch not in vLLM 0.11.0.
- **14B+** even quantized — doesn't fit 8 GB.

---

## 3. Quantization guidance (provider vs Unsloth)

- **Best for this stack: official AWQ** from the provider (Qwen ships AWQ for
  Qwen3-4B/8B/14B). Plain `--quantization awq` (never `awq_marlin` on Turing).
  Cleanest AWQ+LoRA story.
- **GPTQ-INT4** is an equally good Turing target and often has better tooling
  (GPTQModel/AutoRound) for models lacking an official AWQ. `--quantization gptq`.
- **Unsloth**: two products, know the difference —
  - *Unsloth dynamic BnB 4-bit* (`*-unsloth-bnb-4bit`) **does load in vLLM now**
    (the danielhanchen PR merged), via `--quantization bitsandbytes`. Good
    accuracy/VRAM, but **BnB+LoRA is the shaky combo** and it's slower than AWQ.
  - *Unsloth GGUF* — **not for vLLM** (llama.cpp only; no LoRA).
- **Self-quantize when no AWQ exists** (e.g. Qwen3-4B-Instruct-2507): run AutoAWQ
  once on the fp16 repo → get a Turing-friendly INT4 that also supports LoRA.

Swapping a model in the live deploy = edit `deploy/vllm/.env` (`MODEL`,
`EXTRA_ARGS` quant flag) and `docker compose ... up -d --force-recreate`.

---

## 4. GPU upgrade path

The single biggest limiter is the **8 GB Turing** card: it caps model size (~4B
AWQ), forces slow kernels (no FA2/Marlin/FP8), needs the FlashInfer workaround,
and risks being dropped by newer vLLM. Live single-stream is ~9.6 tok/s. Decode
is **memory-bandwidth-bound**, so bandwidth ≈ single-stream speed.

### NVIDIA
| Card | Arch | VRAM | BW (GB/s) | vs current | Unlocks FA2/Marlin/FP8 | Verdict |
|---|---|---|---|---|---|---|
| Quadro RTX 4000 (current) | Turing sm_75 | 8 | ~416 | — | ❌ | baseline |
| RTX 2080 / Super | Turing sm_75 | 8 | ~448 / ~496 | +8 / +19% | ❌ | **skip — lateral move** |
| RTX 2080 Ti | Turing sm_75 | 11 | ~616 | +48% | ❌ | only for the +3 GB, still Turing-capped |
| **RTX 3060 12 GB** | Ampere sm_86 | 12 | ~360 | −13% raw | ✅ | cheap unlock: FA2+Marlin+12 GB beat raw BW for INT4 |
| **RTX 3090** | Ampere sm_86 | 24 | ~936 | +125% | ✅ (no FP8) | **best low-risk pick** — 24 GB, CUDA, nothing changes |
| RTX 4090 | Ada sm_89 | 24 | ~1008 | +142% | ✅ (+FP8) | fastest, pricier |

**A 2080 of any kind is the one option not worth doing** — same Turing ceiling as
today. The meaningful jump is **Ampere+ (sm_80+)**: FlashAttention-2, Marlin INT4
kernels, drops the FlashInfer workaround, futureproofs vLLM. A used **RTX 3060
12 GB** is the cheap unlock; an **RTX 3090 24 GB** is the sweet spot (run 8–14B, or
4B + many LoRA adapters + high concurrency).

### AMD — now genuinely viable (ROCm 7.2, 2026)
The old blocker (ROCm being Linux-only + fragile on consumer Radeon) is largely
resolved: **ROCm 7.2 made Windows first-class**, ships a blessed vLLM wheel
(`vllm==0.14.0+rocm700`), and reaches CUDA-comparable behavior on Radeon without
hand-patching. An **RX 7900 XTX (24 GB)** runs Llama-3.1-8B at **~96 tok/s (~75%
of a 4090)** — ~10× this box's single-stream.

| Card | VRAM | BW (GB/s) | Arch | ROCm 7.2 | ~$ |
|---|---|---|---|---|---|
| **RX 7900 XTX** | 24 | ~960 | RDNA3 gfx1100 | best consumer support | ~$800–900 |
| RX 7900 XT | 20 | ~800 | RDNA3 | good | ~$650 |
| RX 7900 GRE | 16 | ~576 | RDNA3 | good | ~$500 |
| ~~MI50/MI60 32 GB~~ | 32 | ~1000 | gfx906 | **deprecated — avoid** | ~$150 used |

**Stack-specific caveats (verify-before-buy):** XGrammar, per-agent LoRA, and AWQ
INT4 kernels are less battle-tested on ROCm than CUDA. Mitigation: with **24 GB
you can run a 4–8B model in bf16 and skip quantization entirely**, sidestepping
the AWQ-on-ROCm question; XGrammar (logit-masking) should port; LoRA is the one to
confirm. Deploy would move to the `rocm/vllm` image (a ROCm variant of
`deploy/vllm`).

### GPU bottom line
| Want | Buy | Why |
|---|---|---|
| Lowest risk, zero rework | **RTX 3090 (24 GB)** | CUDA — our whole stack works verbatim; 24 GB + ~936 GB/s |
| Best value, willing to verify | **RX 7900 XTX (24 GB)** | ROCm 7.2 competitive; run bf16 to dodge quant; rebuild on `rocm/vllm` |
| Cheap unlock | RTX 3060 12 GB | FA2 + Marlin + 12 GB for minimal spend |
| — | ~~any RTX 2080~~ | not worth it |

3090 and 7900 XTX are neck-and-neck on hardware (both 24 GB, ~950 GB/s); the
3090 wins on "definitely works," the 7900 XTX on value.

---

## 5. Recommendations

**Stay on the 8 GB Turing box (no hardware change):**
1. Keep **Qwen3-4B-AWQ** as the default — proven, balanced, LoRA-ready.
2. For a quality bump at the same footprint, self-AWQ (or use `cpatonn`'s AWQ of)
   **Qwen3-4B-Instruct-2507** (non-thinking) — likely the single best pick for
   constrained-JSON agents.
3. For reasoning-heavy agents, try **Phi-4-mini-instruct** (use GPTQ-INT4 if you
   need LoRA; BnB otherwise).
4. To maximize concurrency (many tiny agents), drop to **SmolLM3-3B** or
   **Qwen3-1.7B** — more KV headroom, more simultaneous sequences.
5. **Qwen3-8B-AWQ** only if you want max quality and can accept low concurrency +
   short context.

**If upgrading the GPU (recommended for a real agent fleet):** an **RTX 3090** (or
**RX 7900 XTX** if you'll verify ROCm) unlocks **8–14B unquantized + many LoRA
adapters + FA2/Marlin speed + FP8-free 24 GB**, and removes the Turing workarounds
— at which point the model shortlist widens to the 7–14B tier (Qwen3-8B/14B,
Ministral-8B, Phi-4 full, Gemma-3-12B).

---

## 6. Sources

- [The Best Open-Source SLMs in 2026 — BentoML](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [Best Open-Source LLMs under 7B (2026) — MLJourney](https://mljourney.com/best-open-source-llms-under-7b-parameters-run-locally-in-2026/)
- [Qwen3 lineup guide 2026](https://baeseokjae.github.io/posts/qwen-3-full-lineup-guide-2026/) · [Qwen3 GitHub](https://github.com/QwenLM/Qwen3) · [Qwen AWQ docs](https://qwen.readthedocs.io/en/latest/quantization/awq.html)
- [Unsloth Dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit) · [vLLM PR #12974 — Unsloth BnB in vLLM](https://github.com/vllm-project/vllm/pull/12974) · [vLLM BitsAndBytes docs](https://docs.vllm.ai/en/latest/features/quantization/bnb/)
- [Phi-4 quantization — Microsoft](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/phi-4-quantization-and-inference-speedup/4360047)
- [vLLM on ROCm attention backend — vLLM blog](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html) · [AMD ROCm local LLM: 96 tok/s on 7900 XTX](https://localaimaster.com/blog/amd-rocm-local-llm-setup) · [vLLM on Windows 2026](https://fazm.ai/t/vllm-windows-support-2026) · [7900 XTX + WSL2 + ROCm + vLLM](https://zenn.dev/troutceremony/articles/f1bf689b878a06?locale=en) · [AMD ROCm vLLM wheel — Phoronix](https://www.phoronix.com/news/AMD-ROCm-vLLM-Wheel)
- Primary checks: vLLM 0.11.0 model registry (on tower); HF model-API existence checks (2026-07-02).
