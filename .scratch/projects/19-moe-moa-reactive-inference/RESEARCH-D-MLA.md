# Research D — MLA deep dive: mechanism, models, conversion, and the KV-DB

Date: 2026-07-31. Topic: Multi-head Latent Attention (MLA) — what it is, which
models actually use it (verified from HF config.json), the post-training
conversion family, and why MLA latents are the ideal KV-DB object.
Companion: RESEARCH-C-KV-QUANTIZATION.md, NANBEIGE-45-PREVIEW.md.

## 1. The mechanism (ELI5)

Normal attention (MHA): every token writes full verbatim notes (K and V) in every
layer's notebook — the KV cache, the inference memory hog. MLA: every token
writes only a tiny **index card** — a compressed latent — plus a shared
dictionary sits on the shelf; the model expands the card back into K/V on demand.

Three steps:
1. **Compress:** squash K and V into one small latent via a down-projection
   (`W_DKV`). Only the latent is cached.
2. **Cache:** the latent sits in the KV cache (~93% smaller per DeepSeek-V2).
3. **Reconstruct:** at attention time, expand with up-projections (`W_UK`,
   `W_UV`) and attend over the reconstructed K/V.

The sneaky part: **RoPE (seat number) is not baked into the card.** Position is
applied only on the query side plus a tiny per-head key component, computed
fresh. The cached latent is **position-free** — valid at any position. This is
the "store unrotated keys" idea (§6.7.1) taken further: not just unrotated, but
*compressed* and expanded at install. Note: MLA also compresses the **query**
first (q_lora_rank), expanding it per head.

## 2. Which models actually use MLA (verified from HF config.json, 2026-07-31)

| Model | Attention (verified) | MLA? |
|---|---|---|
| DeepSeek-V2/V3/R1 | MLA (original) | ✅ |
| Kimi K2 / K2.5 / K2.6 | MLA (DeepSeek-style: kv_lora_rank=512, q_lora_rank=1536, 64 heads, qk_nope=128, qk_rope=64, v_head=128) | ✅ |
| TransMLA-LLaMA-3-8B (fxmeng) | MLA (converted; config model_type=deepseek_v3, same ranks) | ✅ |
| X-EcoMLA-1B/3B (AMD) | MLA (converted; MLA structure in sidecar MLA_config.json + custom code) | ✅ |
| GLM-4.5-Air | GQA (96/8 heads, head_dim 128) | ❌ |
| Hunyuan-A13B | GQA (32/8) | ❌ |
| MiniMax-M1 / M2 | GQA (64/8, 48/8) + linear attention | ❌ |
| Qwen3-Next-80B-A3B | GQA (16/2, head_dim 256) | ❌ |
| DeepSeek-R1-Distill-1.5B…70B | GQA/MHA (Qwen/LLaMA bases — no MLA despite the name) | ❌ |

Detection caveat: X-EcoMLA's HF `config.json` still says "llama" with GQA fields
— the MLA lives in **sidecar files** (`MLA_config.json`, `mla_layer_config.json`)
plus custom modeling code. You cannot trust `config.json` alone to detect MLA
(and conversely this is how the non-MLA models above were caught).

## 3. The conversion family (how small MLA models exist)

MLA was thought to require training from scratch. Three works disprove that:

### TransMLA [V] — arXiv 2502.07864
"Seamlessly converts any GQA-based pre-trained model into MLA." GQA's K/V
projection is a low-rank map (8 KV heads = one rank-limited matrix); MLA is a
low-rank factorization of the same projection with a latent bottleneck — so the
conversion is **algebraically exact at initialization**, then ~6B tokens of
fine-tuning adapts it. Released: LLaMA-2-7B, LLaMA-3-8B (8K/32K). 93% KV
compression, 10.6× speedup at 8K, quality recovered. Runs on DeepSeek's
vLLM/SGLang MLA kernels.

### X-EcoMLA [V] — arXiv 2503.11132 (AMD)
Same idea, **SVD-based initialization** of the down/up projections + distillation
(SFT + DPO) with only **3.6–7B tokens**. Released on HF: **1B and 3B models**
from Llama-3.2 (`amd/X-EcoMLA-1B1B-fixed-kv512`, `-dynamic-0.95`,
`X-EcoMLA-3B3B-*`). 6.4–10.6× KV compression at ~100% zero-shot performance.
**The answer to "small MLA models exist" — you can download and run a ~1B MLA
model today.**

### MHA2MLA [V] — arXiv 2502.14837 (base, LLMs) and 2601.11464 (VLM)
Converts **full MHA** models (not just GQA). Two tricks:
- **partial-RoPE:** MHA applied RoPE to *all* key dims; MLA only rotates a few.
  MHA2MLA measures which Q/K dimensions contribute least to attention scores and
  **masks RoPE off them** (keeps page number, drops line number).
- **joint SVD** of the pre-trained K and V matrices (shared gist, exact at start,
  trimmed after).
- Key insight: minimize **output activation error**, not weight distance — what
  the next layer sees matters, not how close the compressed matrices are.
- Data requirement: **0.3–0.6%** of the training corpus. Llama-2-7B: 92.19% KV
  reduction, 0.5% LongBench drop.
- VLM version: **modality-adaptive partial-RoPE** (image vs text tokens occupy
  different RoPE frequency bands) + **modality-decoupled SVD** (separate
  compressions for visual and textual KV). Integrates with KV quantization and
  cache pruning.

### Whisper-MLA [V] — arXiv 2603.00563
Same recipe applied to Whisper ASR (audio). Evidence the conversion generalizes
across modalities/domains.

## 4. Why MLA latents are the ideal KV-DB object ("cards")

| Property | K/V blob (§6.7) | MLA latent card |
|---|---|---|
| Size per token | f16 K/V, large | ~15–60× smaller (DeepSeek-V2 claims 93.3% reduction) |
| Position | unrotated + rotated at install | **position-free by construction** — nothing to rotate |
| Quantization | per-channel K, per-token V, LUTs | single dense vector; plain per-channel scales; no RoPE mess, no V-outliers |
| Install | dequant → rotate → write f16 | dequant → memcpy |

**The unification:** with an MLA model, llama.cpp's live cache *already holds
latents*. So the DB card = the live cache cell, literally the same unit. Install
= place a foreign card into a sequence's latent cache at a position; the
engine's existing MLA expansion handles the rest. **The DB becomes a spillable
extension of the live cache** — R_kv/tiering at its cleanest.

Caveats:
- **Adapter-tagging becomes more critical** (§6.6 rule (a)): the up-projections
  `W_UK`/`W_UV` are part of the attention weights, and per-hat LoRAs perturb
  them. A card captured under hat *i* expands correctly only under hat *i*'s
  dictionaries. The **dictionary weights join the P7 fingerprint** — a hat/base
  weight change invalidates all cards under it.
- The fork work does not fully disappear: installing foreign cards into a live
  latent cache at arbitrary positions is still an engine-internal-buffer
  operation — but the math is trivial (memcpy, no rotation).
- MLA is an architecture, not a format: either train from scratch (DeepSeek,
  Kimi K2) or **convert an existing GQA/MHA base** (TransMLA/X-EcoMLA/MHA2MLA).

## 5. Path for Project 19

- **Check Nanbeige's attention type first.** GQA → TransMLA recipe; MHA →
  MHA2MLA recipe. Both need only a few billion tokens of fine-tuning — well
  within the S0/S1 training infra.
- If a looped-MLA base exists (converted or 4.5-era), the §6.7 KV-DB becomes
  cards-in/cards-out: no rotation, no per-channel K complexity, simpler
  quantization (RESEARCH-C), ~14× smaller objects.
- Serving is ready: vLLM/SGLang/llama.cpp all run MLA models today (DeepSeek R1,
  Kimi K2).
- Cross-check with NANBEIGE-45-PREVIEW.md: Nanbeige 4.5 (in training) previews
  native depth-attention (mHC) which is the *model-native* answer to the
  cross-context KV validity problem — orthogonal but related to this DB story.

## Sources

- DeepSeek-V2 MLA (93.3% reduction claim) — https://arxiv.org/abs/2405.04434 [K]
- TransMLA — https://arxiv.org/abs/2502.07864 ; models: fxmeng/TransMLA-llama3-8b-8k, -32k [V]
- X-EcoMLA — https://arxiv.org/abs/2503.11132 ; models: amd/X-EcoMLA-1B1B-*, 3B3B-* [V]
- MHA2MLA (LLM) — https://arxiv.org/abs/2502.14837 [V]
- MHA2MLA-VLM — https://arxiv.org/abs/2601.11464 [V]
- Whisper-MLA — https://arxiv.org/abs/2603.00563 [V]
- Model configs verified via https://huggingface.co/{tencent/Hunyuan-A13B-Instruct, moonshotai/Kimi-K2* , zai-org/GLM-4.5-Air, MiniMaxAI/MiniMax-M1-40k, MiniMax-M2.5, Qwen/Qwen3-Next-80B-A3B-Instruct, fxmeng/TransMLA-llama3-8b-8k, amd/X-EcoMLA-1B1B-fixed-kv512-DPO}
