# DECISIONS — Ornith GGUF loader transforms (Path 1)

## D1 — Fix in the loader, keep the Q4 GGUF (not offline conversion, not bf16)
**Decision:** port the 3 transforms into SGLang's load path rather than converting the
GGUF offline to HF/NVFP4 or serving 19 GB bf16.
**Why:** the whole point of the spike is fitting a 3060's 12 GB; offline `gguf-to-nvfp4`
outputs NVFP4 (needs Blackwell — 3060 is Ampere) and its step1 needs a bf16 GGUF input,
and bf16 safetensors don't fit alongside KV cache. Loader-side transforms preserve the Q4
footprint. (See ASSUMPTIONS.md constraints.)

## D2 — Transforms in a pure, unit-tested helper module
**Decision:** `ornith_gdn_transforms.py` holds only tensor math (no SGLang imports).
**Why:** silent numerical bug — must be TDD-tested in isolation against the reference
converter's tensor-op form before wiring into the server (REPO_RULES: TDD).

## D3 — out_proj via unquantized build first (Option A)
**Decision:** start with `install_gdn_out_proj_unquantized()` + plain-tensor column
permute; defer packed block reorder (Option B).
**Why:** column permute on packed data is the one unsafe case; ~0.77 GB extra fits 12 GB
and gets correctness first. Optimize only if VRAM-bound.
**SUPERSEDED by D6** — Option A does not work here (see D6); Option B used.

## D6 — out_proj uses Option B (packed column reorder), Option A is unusable (RESOLVED ✓)
**Finding:** SGLang's `gguf_quant_weights_iterator` (weight_utils.py) **never
dequantizes** — for any non-F32 GGUF tensor it unconditionally yields
`name.replace("weight","qweight")` with the raw packed bytes. `ssm_out` is Q8_0
(quantized) in this GGUF, so it always arrives as `…out_proj.qweight`. Building
`out_proj` unquantized (Option A) would create a plain `.weight` param that the
`.qweight` tensor can never bind to (and nothing dequantizes it) → out_proj stays
at random init. So Option A is unusable as written.
**Decision:** keep `out_proj` quantized and permute its input columns **on the
packed bytes** (Option B). Probed: `ssm_out.data` is `[4096, 4352]`; 4352 bytes/row
= 32 v-heads × 136 bytes, and 136 = 4 whole Q8_0 blocks (34 bytes each). A 128-input-
feature head chunk is exactly 4 whole blocks, so reordering the 32 per-head 136-byte
segments by `perm_head` never splits a quant block. Implemented in
`ornith_gdn_transforms._perm_out_proj_cols_packed`, dispatched from
`transform_quant_rows`. **Zero extra VRAM; no `ornith_gguf_compat`/`sitecustomize`
change needed.** Falls back to no-op if a future quant type's packed width isn't a
multiple of 32.

## D4 — RMSNorm: subtract 1 (RESOLVED ✓ 2026-07-23)
**Finding:** SGLang's `Qwen3_5` uses `GemmaRMSNorm` (zero-centered `weight`, init zeros;
`gemma_rmsnorm` kernel applies `x*(1+weight)`) for `input_layernorm`,
`post_attention_layernorm`, `q_norm`, `k_norm`, and final `model.norm`. So it expects the
HF zero-centered `w` and adds 1 internally → GGUF's `1+w` must have **1 subtracted**.
`ssm_norm` uses `RMSNormGated` (init ones, no offset) → **no** subtraction.
**Source:** `qwen3_5.py:630-631,835-841,1198,265`; `layernorm.py:648-691`;
`fla/layernorm_gated.py:418`. Custom `_weight_loader` means apply `−1` to `loaded_weight`
before the loader runs.

## D5 — Packed row-permute is valid for qkv/z; out_proj needs Option A (RESOLVED ✓)
**Finding:** `gguf_quant_weights_iterator` yields quantized tensors as `…qweight` with
`tensor.data` shaped `[out_features, packed_bytes]`. Probed shapes: `attn_qkv`→(8192,3360),
`attn_gate`→(4096,2816) — dim 0 = output rows → **`qweight[row_perm]` is safe** (whole-row
reorder never splits an input-dim quant block). `ssm_out`→(4096,4352) has dim 0 = hidden
(output); its V-head axis is the **input** dim packed in dim 1 → **cannot row-index** →
build `out_proj` unquantized (Option A) and permute columns as a plain tensor.
**Source:** `weight_utils.py:1190-1285` + direct `gguf.GGUFReader` probe.

## D7 — mrope/partial-rotary RULED OUT as the residual bug (RESOLVED ✓ 2026-07-23)
**Context:** after landing the 3 transforms, output stayed gibberish. The guide
§7 named mrope/partial-rotary on the full-attn layers as the prime residual
suspect. **It is not a misconfiguration.**
**Finding:** the resolved config's `rope_parameters` **already contains**
`mrope_section=[11,11,10]`, `mrope_interleaved=True`, `partial_rotary_factor=0.25`,
`rope_theta=1e7`. SGLang's `get_rope_config` reads the v5 `rope_parameters` dict
(so rope_theta=1e7 is correct), and the rope factory's `default`+`"mrope_section"
in rope_scaling` branch builds the **interleaved `MRotaryEmbedding`**, whose
`forward_cuda` handles the text 1-D positions path. So the full-attn rope
convention is honored. (An earlier manual `get_rope` probe that omitted
`mrope_section` mis-built a plain neox `RotaryEmbedding` — a false alarm from the
incomplete test dict, not the real config.)
**Evidence the residual is elsewhere:** (a) all-transforms-ON and all-OFF are
*both* gibberish → dominant bug independent of the 3 transforms; (b) per-layer
hidden-state stats are healthy across all 32 layers (no NaN, std 0.28→1.7) → no
broken layer, a *semantic* not numeric error; (c) GGML dequant healthy (all
quantized layers fine). **Conclusion:** residual is an SGLang-`Qwen3_5` execution
detail for this checkpoint (prime suspect: fused GDN `in_proj_qkvz` split /
delta-rule head ordering), requiring the guide §6.4 bf16 per-layer bisection
(19 GB model, beyond this 12 GB spike) to pinpoint. Not fixed this session.
