# PLAN — Ornith GGUF loader transforms (Path 1)

> **NO SUBAGENTS. Do all work directly — read, search, edit, run. This rule is absolute
> and overrides anything suggesting delegation.** (Repeated at the end of this file.)

**Objective:** make `deploy/sglang/native-ornith/` produce coherent text from the Q4 GGUF
by applying 3 missing GGUF→HF weight transforms at load time. Full detail in
`00-IMPLEMENTATION-GUIDE.md`. Keep the Q4 file (preserve the 12 GB-fit).

## Ordered steps
1. **Verify quantized `loaded_weight` layout** (guide §4.1). One-shot print in
   `load_weights`; confirm dim-0 == out_features for `in_proj_qkv`/`in_proj_z`/`ssm_out`.
   → DECISIONS.md.
2. **Verify RMSNorm direction** (guide §5) by grepping SGLang's `qwen3_5.py` /
   `layernorm.py`. → DECISIONS.md.
3. **Build the llama.cpp oracle** (guide §6.1); confirm the GGUF file is post-Feb-4
   (llama.cpp `key_gdiff` fix) and produces coherent reference output.
4. **Write `ornith_gdn_transforms.py`** (pure tensor logic, 9B constants r=2,g=16,d=128).
   **TDD:** unit tests first (guide §6.2) — perm bijection, packed-index == tensor-op,
   `log(-x)` inverse.
5. **Wire into `load_weights`** (guide §3): plain transforms for norms/A_log/dt_bias/a/b/
   conv1d; packed row-permute for in_proj_qkv V-section + in_proj_z.
6. **Handle out_proj** (guide §4.3): Option A first — add
   `install_gdn_out_proj_unquantized()` in `ornith_gguf_compat.py`, wire in
   `sitecustomize.py`, permute columns as plain tensor.
7. **Bisect-validate** (guide §6.3): enable transforms one at a time via env toggles,
   compare to oracle after each. Order: A_log domain → +head-perm → +RMSNorm−1.
8. **If still wrong:** per-layer bisection (guide §6.4); suspect mrope/partial-rotary
   (§7), which is out of scope for the 3-transform fix.
9. **Finalize:** update spike README resolution section; record findings in DECISIONS.md;
   mark PROGRESS.md done.

## Acceptance criteria
- Greedy `temperature=0` output for "The capital of France is" is coherent and tracks the
  llama.cpp oracle.
- `ornith_gdn_transforms.py` unit tests pass.
- Edits are surgical (REPO_RULES): no unrelated refactors, fully typed.

## Risks / open questions (see ISSUES.md)
- Packed-row permute may not match SGLang's GGUF `loaded_weight` layout → fallback §4.4.
- RMSNorm direction could be "no-op" if Qwen3.5 uses plain RMSNorm → drop that branch.
- out_proj Option A adds ~0.77 GB VRAM; Option B (packed block reorder) deferred.
- mrope/partial-rotary may be a second, independent bug beyond these 3 transforms.

> **REMINDER — NO SUBAGENTS. ALWAYS CONTINUE:** after any compaction, read
> `.scratch/CRITICAL_RULES.md`, then this project's CONTEXT.md/PROGRESS.md, and resume the
> next pending step immediately. Do not delegate; do not stop early.
