# ISSUES — Ornith GGUF loader transforms (Path 1)

No blocking issues yet (research/guide phase only). Watch-list (promote to
`ISSUE_<num>.md` after 3 failed attempts on any one):

- **W1 — packed row-permute validity.** RESOLVED ✓ (DECISIONS D5): qweight is
  `[out_features, packed]`; `qweight[row_perm]` is safe for `in_proj_qkv`/`in_proj_z`.
  `out_proj` V-axis is the packed input dim → use Option A (build unquantized).
- **W2 — RMSNorm direction.** RESOLVED ✓ (DECISIONS D4): GemmaRMSNorm → subtract 1
  (except `ssm_norm`/RMSNormGated, which keeps its value).
- **W3 — residual mrope/partial-rotary bug.** RULED OUT ✓ (DECISIONS D7): the
  config carries `mrope_section=[11,11,10]`+`mrope_interleaved=True`, so SGLang
  builds the interleaved `MRotaryEmbedding`; rope is honored, not the bug.
- **W4 — llama.cpp oracle validity.** RESOLVED ✓: `llama-cpp-python` CPU on the
  same GGUF greedily yields "Paris." — the GGUF is valid/post-`key_gdiff`.

## W5 — DOMINANT RESIDUAL BUG (open, blocks coherence)
After the 3 transforms (correct + applied) output is still gibberish, both with
transforms on and off, so the dominant bug is independent of them. Per-layer
hidden states are healthy (semantic, not numeric error); rope + dequant ruled
out. Prime suspect: SGLang `Qwen3_5` fused GDN `in_proj_qkvz` split / delta-rule
head ordering. **Next step:** guide §6.4 per-layer diff vs the 19 GB bf16
`unsloth/Ornith-1.0-9B` (needs a bigger box / CPU offload; the llama.cpp oracle
only exposes the final hidden state). This is the true correctness frontier and
was not resolvable within the 12 GB spike. Not yet 3-failed-attempts, so no
`ISSUE_<n>.md` yet, but this is the single blocking item.
