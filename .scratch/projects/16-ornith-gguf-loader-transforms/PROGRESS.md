# PROGRESS — Ornith GGUF loader transforms (Path 1)

Status legend: [ ] pending · [~] in-progress · [x] done

## Research (done — prior sessions)
- [x] Establish ground-truth GGUF metadata/tensor layout (9B: r=2, g=16, d=128).
- [x] Identify the 3 missing transforms via reference converter (`gguf-to-nvfp4`).
- [x] Confirm GGUF-in-SGLang and GGUF-in-vLLM are both unsupported for `qwen35`.
- [x] Write `RESEARCH-FINDINGS.md` §4b + this implementation guide.

## Implementation (pending — next session)
- [x] 1. Verify quantized `loaded_weight` layout (guide §4.1) → DECISIONS.md **D5**:
      qweight is `[out, packed]`; row-permute safe for qkv/z; out_proj needs Option A.
- [x] 2. Verify RMSNorm direction (guide §5) → DECISIONS.md **D4**: GemmaRMSNorm →
      subtract 1 (all norms except ssm_norm/RMSNormGated).
- [x] 3. Build llama.cpp oracle (`llama-cpp-python` CPU wheel in the devenv) →
      greedy "The capital of France is" = **"Paris."**. GGUF valid, post-Feb-4
      (W4 resolved).
- [x] 4. Write `ornith_gdn_transforms.py` + unit tests (TDD, guide §6.2) — 16
      tests pass (`test_ornith_gdn_transforms.py`, runs under devenv uv).
- [x] 5. Wire transforms into `load_weights` (guide §3) — dispatch after conv1d
      unsqueeze, gated by `ORNITH_FIX_NORM/ALOG/PERM` env toggles (default on).
- [x] 6. out_proj — used **Option B** (packed Q8_0 column reorder), NOT Option A.
      D6: the GGUF iterator never dequantizes (always yields `.qweight`), so an
      unquantized out_proj would never bind the quantized `ssm_out`. ssm_out is
      Q8_0, 4352 bytes/row = 32 heads × 136 = 4 whole blocks/head → block-aligned
      segment reorder is safe, zero extra VRAM. No compat/sitecustomize change.
- [x] 7. Bisect-validate vs oracle (guide §6.3) — all-ON and all-OFF both
      gibberish → dominant residual bug is **independent of the 3 transforms**;
      no toggle combo reaches coherence. Transforms verified applied (exact
      counts) and per-layer hidden states healthy.
- [~] 8. Localization done (see DECISIONS D7): mrope/partial-rotary **RULED OUT**
      (config has `mrope_section` → interleaved MRotaryEmbedding built);
      GGML dequant healthy. Residual = semantic corruption in the layer stack
      (prime suspect: fused GDN `in_proj_qkvz` split / delta-rule head order).
      **Full §6.4 bf16 per-layer bisection NOT done** — needs the 19 GB bf16
      model, beyond this 12 GB spike. Diagnostic hooks left gated
      (`ORNITH_DEBUG_LAYERS=1`).
- [x] 9. Updated spike README "Remaining gap" + RESEARCH-FINDINGS §5;
      DECISIONS D6/D7; this PROGRESS. Server verified serving; GPU 1 freed.

## Outcome
The 3 transforms are **implemented, unit-tested (16 green), and verified applied
live**, and the oracle is established. Definition-of-done "coherent output" is
**NOT met**: a dominant residual SGLang-`Qwen3_5` execution bug survives the 3
transforms and is localized (not fixed). Next session: guide §6.4 bf16 bisection.

## Notes
- Code written: `ornith_gdn_transforms.py` (pure, 16 unit tests green),
  `ornith_text_model.py` load_weights dispatch. No `ornith_gguf_compat.py`/
  `sitecustomize.py` changes needed (Option B keeps out_proj quantized).
- Launch params: MODEL=`~/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf`;
  TOKENIZER=`~/.cache/structured-agents/sglang-ornith-tokenizer`;
  CONFIG=`~/.cache/structured-agents/sglang-ornith-resolved-config/config.json`;
  libcuda dir `/nix/store/l5smwbs4q9rni6b0pw3fr8qyl4zja14f-graphics-drivers/lib`.
  GPU1 + port 8003 both free.
