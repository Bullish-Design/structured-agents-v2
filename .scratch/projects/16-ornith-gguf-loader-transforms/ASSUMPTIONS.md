# ASSUMPTIONS — Ornith GGUF loader transforms (Path 1)

**Audience / scenario:** a later implementation session (possibly post-compaction)
executing the fix in `00-IMPLEMENTATION-GUIDE.md` against the existing
`deploy/sglang/native-ornith/` spike.

## Environment / constraints (observed, safe to rely on)
- File: `/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf`
  (427 tensors; mixed F32/Q4_K/Q5_K/Q6_K/Q8_0; `general.architecture = qwen35`).
- Stack: SGLang 0.5.14, `gguf` PyPI pkg, Transformers 5.8.1, one RTX 3060 (12 GB), GPU 1,
  port 8003. Do **not** disturb systemd runners on GPU 0 / ports 8000–8001.
- The venv only imports numpy/gguf inside the devenv shell:
  `cd deploy/sglang/native-ornith && NIXPKGS_ALLOW_UNFREE=1 devenv shell --impure -- uv run --locked --no-sync python …`
- Privileged/systemd actions: hand the user the command, don't run it.
- Goal is to KEEP the Q4 GGUF (12 GB fit). Falling back to 19 GB bf16 defeats the purpose.

## Verified architecture facts (this session, from the GGUF itself)
- GDN linear attn: k-heads 16, v-heads 32, head_dim 128 ⇒ repeat r=2, groups g=16.
- Q=2048, K=2048, V=4096 (qkv=8192); Z/gate=4096; conv over 8192; kernel 4.
- 32 layers, full-attention every 4th (first index 3); full-attn head_dim 256,
  partial_rotary 0.25, mrope_section [11,11,10], rope_theta 1e7.

## Assumptions to VERIFY before/while coding (not yet confirmed)
1. SGLang's GGUF loader hands quantized `loaded_weight` with dim 0 = out_features so that
   row-indexing a packed `qweight` is valid (guide §4.1).
2. SGLang's Qwen3.5 RMSNorm expects the HF (zero-centered) weight, so the GGUF `1+w` needs
   `−1` (guide §5). Strongly implied by native-safetensors serving working, but confirm.
3. The reference converter's `(repeat,groups)→(groups,repeat)` permutation and A_log
   `log(-x)` / RMSNorm `-1` transforms — derived for the 27B — apply to the 9B with only
   the constants changed (r=2 not 3, v_heads=32 not 48). High confidence; the GGUF shapes
   corroborate.
4. The llama.cpp GGUF build of Ornith is post-Feb-4 (`key_gdiff` GDN fix) and is a valid
   oracle. Re-download if llama.cpp itself outputs garbage.

## Invariants
- Edits surgical, fully typed, no unrelated refactors (REPO_RULES).
- Transforms live in a pure, unit-tested helper — no SGLang imports in the math.
- Validate against a reference oracle, one transform at a time — never trust "looks
  plausible" for a silent numerical bug.
