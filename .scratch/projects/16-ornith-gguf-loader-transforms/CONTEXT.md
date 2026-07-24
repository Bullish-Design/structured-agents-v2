# CONTEXT — Ornith GGUF loader transforms (Path 1)

## What this project is
Fix the gibberish output of the `deploy/sglang/native-ornith/` spike (Ornith-1.0-9B
`UD-Q4_K_XL` GGUF on SGLang 0.5.14, GPU 1, port 8003) by applying the 3 GGUF→HF weight
transforms the current patches omit.

## Where things stand (2026-07-23 implementation session)
**The 3 transforms are DONE (implemented, unit-tested, verified applied live). The
model still outputs gibberish because of a SEPARATE dominant bug that is now
localized but NOT fixed.** Definition-of-done "coherent output" is NOT met.

### Code shipped (all surgical, fully typed)
- `ornith_gdn_transforms.py` (NEW, pure, no SGLang imports): RMSNorm −1,
  `A_log=log(−x)`, value-head `(r=2,g=16)→(g,r)` unperm; `transform_plain` +
  `transform_quant_rows` (incl. packed-Q8_0 out_proj column reorder).
- `test_ornith_gdn_transforms.py` (NEW): 16 tests, all green under
  `NIXPKGS_ALLOW_UNFREE=1 devenv shell --impure -- uv run --locked --no-sync python
  test_ornith_gdn_transforms.py`.
- `ornith_text_model.py::load_weights`: transform dispatch after the conv1d
  unsqueeze, gated by `ORNITH_FIX_NORM/ALOG/PERM` (default on). Plus gated
  diagnostics (`ORNITH_DEBUG_LAYERS=1`): per-layer hidden-state stats hook +
  transform-apply counts. **out_proj = Option B** (packed reorder), so NO
  `ornith_gguf_compat.py` / `sitecustomize.py` change was needed.

### Key results
- Oracle (llama.cpp via `llama-cpp-python` CPU wheel, same GGUF): greedy "The
  capital of France is" → **"Paris."** (GGUF valid; W4 resolved).
- SGLang spike, transforms ON *and* OFF: both gibberish (different) → dominant bug
  independent of the 3 transforms.
- Per-layer hidden states healthy (no NaN, std 0.28→1.7) → semantic, not numeric.
- mrope/partial-rotary RULED OUT (config has `mrope_section` → interleaved
  MRotaryEmbedding built). GGML dequant healthy.

## Next action (the real frontier — guide §6.4)
Per-layer diff of SGLang hidden states vs the **19 GB bf16 `unsloth/Ornith-1.0-9B`**
reference (needs a bigger box / CPU offload — does not fit the 12 GB spike). First
divergent layer pinpoints the bug. Prime suspect: SGLang `Qwen3_5` fused GDN
`in_proj_qkvz` split / delta-rule head ordering. See ISSUES **W5**.

## How to run
- Launch: `.scratch/projects/16-ornith-gguf-loader-transforms/launch-spike.sh`
  (GPU 1, port 8003; sets MODEL/TOKENIZER/CONFIG/LIBRARY_PATH). Add
  `ORNITH_DEBUG_LAYERS=1` for diagnostics, `ORNITH_FIX_*=0` to bisect transforms.
- Oracle: `pip`-installed `llama-cpp-python` (CPU) is in the spike `.venv`;
  `/tmp/oracle.py` runs it.
- Unit tests: see command above.

## If you lost all memory
Read `.scratch/CRITICAL_RULES.md`, `00-IMPLEMENTATION-GUIDE.md`, then DECISIONS.md
(D6 out_proj Option B, D7 rope ruled out) and ISSUES.md (W5). NO SUBAGENTS.
