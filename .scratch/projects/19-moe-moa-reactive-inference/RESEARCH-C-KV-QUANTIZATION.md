# Research C — KV-cache-specific quantization formats

Date: 2026-07-31. Topic: formats/methods that beat generic block quantization
(q8_0/q4_0) for the KV cache specifically, and which fit the §6.7 KV-DB design.
Companion: RESEARCH-B-PRIOR-ART.md (Spike B prior art), RESEARCH-D-MLA.md (MLA).

## Verified baseline: what llama.cpp actually supports

The repo's fork (`.llamacpp-builds/src/src/llama-kv-cache.h`) stores cache types
as raw `ggml_type` (`type_k` / `type_v`): F32, F16, Q8_0, Q4_0, Q4_1, IQ4_NL,
Q5_0, Q6_0, … **Block formats only — no per-channel, no per-token, no
non-uniform.** Verified directly in the fork source.

## The structural insight that matters

The KV-DB blob at rest is **engine-agnostic**: the install path is
*dequant → rotate → write f16 into the live cache* (§6.7). So the blob's quant
format needs **no llama.cpp engine support** — only a dequant kernel we control
(table lookups, cheap). Research formats are therefore *available to us as
storage formats* even though llama.cpp cannot attend over them directly.

## The research landscape (by mechanism)

All numbers/titles flagged: [V] = verified live against arXiv this session;
[K] = from knowledge, verify before depending.

### 1. RoPE-aware per-channel K quantization — KVQuant [V]
arXiv 2401.18079 ("KVQuant: Towards 10 Million Context Length LLM Inference
with KV Cache Quantization"). Keys are rotated by RoPE, which **mixes channels**,
breaking naive per-channel scales. Fix: quantize K **pre-RoPE** with per-channel
scales. **This is exactly our unrotated-K storage (§6.7.1) — we get per-channel
K precision structurally, for free.**

### 2. Per-token V + non-uniform levels — KVQuant [V]
- V has **outlier tokens** → quantize V per-token, not per-block.
- KV values are roughly Gaussian per channel → **non-uniform lookup-table (LUT)
  levels** (NormalFloat-style, from QLoRA) fit better than uniform grids.
- Dequant = table lookup, trivial on install.

### 3. Mixed-precision windows — KIVI [V]
arXiv 2402.02750 ("KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV
Cache"). Keep **recent + sink tokens high precision**, old tokens at 2-bit
(attention concentrates on recent + sink positions). **This validates the §6.7
f16 sink header** — and suggests extending it: keep first-k *and* last-k tokens
of each chunk high precision (last-k are what the next chunk's link tokens
attend to, §6.7.3).

### 4. Error compensation — GEAR / WKVQuant [K]
Quantize, then measure the error and repair it with a **low-rank matrix + sparse
residual**. Makes 2–3 bits near-lossless. Costs a small side structure per blob;
dequant path stays simple. Verify numbers on our model family before depending.

### 5. Outlier elimination by rotation — QuaRot [V] / RotateKV [K]
arXiv 2404.00456 ("QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs").
Rotate weights+activations so outlier channels vanish *before* quantizing →
4-bit KV near-lossless. For keys the rotation must be applied **pre-RoPE** (same
constraint as KVQuant). More invasive (touches model weights).

### 6. Architectural compression — MLA / cross-layer KV sharing [V]
DeepSeek MLA caches one compressed latent per token (~93% KV reduction per
DeepSeek-V2); cross-layer attention (CLA) shares KV across layers. **Model
architecture changes, not formats** — see RESEARCH-D-MLA.md for the full deep
dive and the conversion path.

## Recommendation ladder for the KV-DB (§6.7)

| Step | Format | Effort | Quality |
|---|---|---|---|
| Now (decided) | q8_0 default, q4_0 cold | zero new code | safe |
| Next — best fit | KVQuant-style: **per-channel K** (we can — keys are unrotated!) + per-token V + NF4 LUT | one dequant kernel + capture-time quant | ~3–4 bits at q8_0 quality |
| Aggressive | + KIVI mixed window (extend header to first-k *and* last-k) or GEAR residual | more capture complexity | 2–3 bits near-lossless |
| Ceiling | MLA-style latent (see RESEARCH-D) | model change | ~14× KV reduction |

## Rules that carry over (unchanged from §6.7)

- Rotation only ever in float (dequant first) — issue #5652 constraint.
- f16 sink header stays — now doubly justified (KIVI mixed-precision principle).
- Heal/link-token check layers compare in float (quantization noise pollutes
  top-k deviation selection otherwise).
- **P7 fingerprint must include quant format + hyperparameters** (per-channel
  scales, LUT table, window sizes) — the format has more degrees of freedom than
  `q8_0` vs `q4_0`.
- Acceptance gate is the same as always: **continuation-equivalence of a
  quantized-blob install vs the f16 baseline**, on the actual model (Project 18
  discipline). MHA2MLA (RESEARCH-D) independently confirms this principle:
  minimize *output activation error*, not weight distance.
- Caveat [K]: GEAR/WKVQuant/QoQ/RotateKV results are mostly Llama-family; the
  Nanbeige looped base is not a standard target — re-verify.

## Sources

- KVQuant — https://arxiv.org/abs/2401.18079 [V]
- KIVI — https://arxiv.org/abs/2402.02750 [V]
- QuaRot — https://arxiv.org/abs/2404.00456 [V]
- GEAR, WKVQuant, QoQ, RotateKV — search arXiv; IDs not re-verified this session [K]
- llama.cpp cache types — `.llamacpp-builds/src/src/llama-kv-cache.h` (ggml_type)
