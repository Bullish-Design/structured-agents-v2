# Kickoff prompt — paste into a clean session

---

Implement the fix for the Ornith-1.0-9B GGUF gibberish in the SGLang spike. This is an
**implementation** session — you are writing and testing code, then validating real output.

## Read first (do not skip; do not trust memory over these)
1. `.scratch/CRITICAL_RULES.md` — repo workflow rules. **NO SUBAGENTS / Task tool, ever.**
   Do all work directly. Keep `PROGRESS.md`/`CONTEXT.md` current as you go.
2. `.scratch/REPO_RULES.md` — coding standards (full typing, surgical edits, TDD).
3. `.scratch/projects/16-ornith-gguf-loader-transforms/00-IMPLEMENTATION-GUIDE.md` — the
   full spec. Follow it. Its §2 table, §3 hooks, and the `ornith_gdn_transforms.py`
   skeleton are what you implement.
4. That project's `PLAN.md`, `DECISIONS.md`, `ASSUMPTIONS.md`, `PROGRESS.md`, `ISSUES.md`.
5. Root-cause context: `deploy/sglang/native-ornith/RESEARCH-FINDINGS.md` §4b.

## What is already settled (do NOT re-litigate — build on it)
- **Root cause:** the GGUF (`qwen35`) needs 3 numeric transforms the current 9 patches
  never apply. The model *loads and serves* but outputs gibberish purely from wrong weight
  values/order.
- **The 3 transforms** (9B constants: repeat r=2, groups g=16, v_heads=32, head_dim d=128,
  QK=4096, V=4096):
  1. RMSNorm **−1** on `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`,
     `model.norm` — **NOT** `ssm_norm`. *(Verified: SGLang uses GemmaRMSNorm, which adds 1
     internally — DECISIONS D4.)*
  2. **A_log = log(−ssm_a)** domain conversion.
  3. Value-head **(r,g)→(g,r)** unpermutation on A_log, dt_bias, in_proj_a, in_proj_b, the
     V-section of in_proj_qkv, in_proj_z, out_proj (columns), and the V-section of conv1d.
- **Loader layout (verified — DECISIONS D5):** quantized tensors arrive as `…qweight`,
  shape `[out_features, packed_bytes]`. `qweight[row_perm]` is **safe** for `in_proj_qkv`
  (permute rows 4096:8192) and `in_proj_z` (all rows). `out_proj`'s V-axis is the packed
  *input* dim → **cannot** row-index → build it **unquantized** (guide §4.3 Option A) and
  permute columns as a plain tensor. `A_log`/`dt_bias` arrive as **bare** names; never
  transform `.qweight_type`.
- **Hook point:** `deploy/sglang/native-ornith/ornith_text_model.py::load_weights`
  (lines ~172-262), transforming `loaded_weight` before each `weight_loader` call. Put the
  math in a new pure module `ornith_gdn_transforms.py` (no SGLang imports).

## Do this, in order (PLAN.md steps 3-9)
1. **Oracle first.** Stand up llama.cpp on the *same* GGUF as ground truth
   (`llama-server -hf deepreinforce-ai/Ornith-1.0-9B-GGUF -c 4096`, or `-m` the local
   file). Prompt `"The capital of France is"` at `temperature=0`, capture the greedy
   continuation (+first-token logprobs if possible). If llama.cpp *itself* is garbage, the
   GGUF predates the Feb-4 `key_gdiff` fix — re-download before trusting it (ISSUES W4).
2. **TDD the transforms.** Write `ornith_gdn_transforms.py` per guide §2/§3, then unit
   tests (guide §6.2): `perm_head == [0,16,1,17,…,15,31]` and is a bijection of range(32);
   the index-based `_perm_rows` path equals the reference `reshape(2,16,…).permute` op on
   random tensors (exact equality); `log(-x)` inverts `-exp(y)`. Tests pass before wiring.
3. **Wire into `load_weights`** (guide §3): `.qweight` → `transform_quant_rows`; F32/bare
   (and unquantized out_proj) → `transform_plain`; skip `.qweight_type`.
4. **out_proj Option A:** add `install_gdn_out_proj_unquantized()` in
   `ornith_gguf_compat.py` (mirror `install_gdn_ba_unquantized`, lines 370-396; find the
   `out_proj`/`o_proj` constructor on `Qwen3_5GatedDeltaNet`), wire it in
   `sitecustomize.py` next to the others.
5. **Validate incrementally** (guide §6.3), gated by env toggles so each transform is
   isolated: A_log domain → +head-permutation → +RMSNorm−1. Restart and diff vs the oracle
   after each. Success = coherent English that tracks llama.cpp's greedy output.
6. **If still wrong** after all three: per-layer bisection (guide §6.4). The prime residual
   suspect is mrope / partial-rotary on the full-attention layers (head_dim 256,
   partial_rotary 0.25, mrope_section [11,11,10]) — **out of scope** for these 3 transforms;
   log it as a new finding, don't scope-creep silently.
7. **Finalize:** update the spike `README.md` "Remaining gap" section with the resolution;
   record in `DECISIONS.md` which out_proj option you used and any surprises; check off
   `PROGRESS.md`.

## Environment / etiquette
- Spike is pinned to **GPU 1, port 8003**. Do **not** disturb the systemd inference runners
  on GPU 0 / ports 8000-8001.
- The venv only imports numpy/gguf inside the devenv shell. Run python via:
  `cd deploy/sglang/native-ornith && NIXPKGS_ALLOW_UNFREE=1 devenv shell --impure -- uv run --locked --no-sync python <script>`
  (the `NIXPKGS_ALLOW_UNFREE=1` is required or the shell eval fails on CUDA EULA).
- Launch needs `LIBRARY_PATH` → a dir with a real `libcuda.so` (graphics-drivers store
  path) for FlashInfer's `-lcuda` link (see spike README "Toolchain note").
- Privileged/systemd actions: hand me the command, don't run it.
- Model file: `/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf`.
- Reference converter to diff against (already cloned):
  `.scratch/vendor/gguf-to-nvfp4/scripts/step1_convert.py` → `gguf_tensor_to_torch()`.

## Definition of done
Greedy `temperature=0` output for "The capital of France is" from `/v1/completions` is
coherent and tracks the llama.cpp oracle; `ornith_gdn_transforms.py` unit tests pass; edits
are surgical and fully typed; spike README + project DECISIONS/PROGRESS updated.

**Reminder: NO SUBAGENTS. Work directly, keep CONTEXT.md/PROGRESS.md current, and don't
stop until output is coherent or you've localized the residual bug with evidence.**
