# Research B — Prior art for Spike B (cross-context KV splice / RoPE re-anchor + CacheBlend heal)

Date: 2026-07-31. Source: deep-research workflow `wf_dc72d908-cc4` (Claude Code,
~/.claude), fan-out + adversarial verification completed; final synthesis relay
was lost to a spend limit — this file is the reconstructed synthesis. Raw
evidence: `research-b-prior-art/journal.jsonl` (raw), `journal-extracted.md`
(readable: all fan-out results, claims, verification evidence, source index).

## TL;DR

**Yes — both halves of Spike B have been implemented in open source. The
RoPE-shift half already exists natively inside llama.cpp itself; the selective-
heal half is fully open-sourced in LMCache (vLLM-side).** The CONCEPT's premise
that a `kv_rope_shift` kernel is fork-blocked novelty is wrong; the kernel is
upstream. What is genuinely missing in llama.cpp is (a) position-selective
*partial* splice and (b) the CacheBlend-style heal, plus (c) adapter-tagged
admission, which upstream demonstrably does not do (llama.cpp issue #26207).

All ~90 verification agents returned `refuted: false` (high confidence); no
claim was contradicted.

## 1. The RoPE re-anchor half — already in llama.cpp

- `llama_kv_cache_seq_shift` (renamed `llama_kv_cache_seq_add` /
  `llama_kv_self_seq_add`) at `src/llama-kv-cache.cpp`. The `update()` /
  `build_graph_shift` path re-applies RoPE frequencies to **cached K tensors**
  to reflect new positions *without re-processing tokens*. This is exactly the
  "native ggml RoPE-shift op behind a shim" that `node_blend_live.py` assumes
  the stable API lacks.
- Shipped in production features: context shifting (sliding window),
  Self-Extend (PR #5104, merged by ggerganov 2024-01-27; `seq_div`+`seq_add`
  position remapping, `--grp-attn-n/-w`), StreamingLLM rolling buffers
  (discussion #3581).
- `n_cache_reuse` in llama-server (discussion #13606): finds a matching chunk
  >=N tokens **even when it is not a prefix** and RoPE-shifts that cached slice
  into its new position — the closest in-tree cross-context splice.
- Consistent with SPIKE-B-RESULTS.md: `llama_memory_seq_add` +
  `llama_memory_can_shift` already exist in `llama.h` (~lines 749/782) —
  whole-sequence re-anchor works today; partial-position splice does not.
- Recent core rework to watch: PR #13194 (SWA support) touches the same
  seq_add/shift/defrag machinery — a fork's hook point and a rebase risk.

### Constraints (all verified)
- **Issue #5652:** RoPE re-rotation of cached K crashes on quantized K cache
  (q4_0) — `GGML_ASSERT: ggml.c:12646` at context-swap/shift. Re-anchor needs
  float-domain keys. Workaround: `--no-context-shift`. See ELI5 below.
- **Discussion #24944:** KV shifting is *hard-disabled* for M-RoPE architectures
  (Qwen 3.x, Gemma 4), interleaved layers, and reasoning tokens.
  **Verify Nanbeige's rope type before betting on seq_add.**
- **Issues #4097 / #3825:** correctness pitfalls in kv shifting (the
  relative-position invariant can break in practice).
- **Issue #26207 (2026-07-28):** llama-server reuses prompt-cache KV across
  requests with *different LoRA adapters* when prefixes match → silently
  contaminated output. Upstream does **not** key cache on adapter config. This
  is empirical validation of the §6.6 adapter-tag hard rule, and shows it is
  genuinely unimplemented anywhere.
- **Discussion #22354:** when shared content is not a contiguous shifted block
  (e.g. changed system prompt mid-context), llama.cpp falls back to full
  reprocessing — no heal path in-tree. The heal half is not in llama.cpp.

## 2. The heal half — CacheBlend, fully open-source (LMCache)

- **CacheBlend**: "Fast LLM Serving for RAG with Cached Knowledge Fusion"
  (arXiv 2405.16444), EuroSys'25 Best Paper, ACM TOCS 10.1145/3790254.
- Implementation: `github.com/LMCache/LMCache`, integrated with vLLM
  (V1 + multimodal). Docs: `docs.lmcache.ai/kv_cache_optimizations/blending.html`.
- **Mechanism:** reuse precomputed KV regardless of prefix position; fully
  recompute layer 1, partially recompute layer 2 to get accurate QKV, compare V
  against precomputed → pick top-k **High-KV-Deviation (HKVD)** tokens
  (~10–15%) → recompute only those, in place.
- Config: `LMCACHE_BLEND_RECOMPUTE_RATIOS` (0.15), `LMCACHE_BLEND_CHECK_LAYERS`
  (layer 1), `LMCACHE_BLEND_SPECIAL_STR` (chunk separators),
  `LMCACHE_USE_LAYERWISE`. Chunks must be tokenized with separator strings to
  avoid tokenization drift.
- Results: TTFT 2.2–3.3×, throughput 2.8–5× vs full recompute, near-100% KV hit
  rate on 2WikiMQA.
- Note: CacheBlend handles positions by *recompute*, not by RoPE-shift. The two
  halves are complementary.

## 3. Position-independent caching (PIC) family — "both halves" designs

- **EPIC** (arXiv 2410.15332, ICML 2025): the key RoPE math — a precomputed key
  realigns to a new position with a **single RoPE rotation** since RoPE composes:
  R(δ)·R(p₀) = R(p₁). Core algorithm **LegoLink** heals the spurious "attention
  sink" at each chunk start by recomputing only the first k tokens (k = 2–32,
  O(kN) vs O(N²)) of each reused chunk. Plus AttnLink (static attention
  sparsity) and KVSplit (semantic chunking). Up to 8× TTFT, 7× throughput, <7%
  accuracy drop.
- **KVLink** (arXiv 2502.16002, NeurIPS 2025): store **unrotated** key states
  (W_k·x, no RoPE); at inference apply global RoPE at each token's correct
  full-sequence position. Plus K=5 trainable **link tokens** per segment (KV
  computed at inference, custom attention mask) to heal cross-segment attention.
  Up to 96% TTFT reduction on 5k contexts. The cleaner KV-DB design, at the
  cost of pre-rotation storage + RoPE-at-attention-time.
- **CacheClip** (arXiv 2510.10129): CacheBlend-lineage, heal-token selection via
  a **small auxiliary LLM** whose last-layer attention distribution approximates
  the primary model; + shared prefixes to eliminate redundant attention sinks,
  sliding-window grouping, CPU-GPU hybrid. At 20% recompute: 3.33× prefill
  speedup, 85.2% (NIAH) / 91.1% (LongBench) quality, beats CacheBlend/APE by
  ~16/13 pts.
- **MiniPIC** (IBM/vllm fork): stores unrotated K, defers RoPE to attention
  time; three primitives (block-aligned padding, SSep, PDep) in <100 LOC of
  core-engine change; realizes Block-Attention, EPIC, PromptCache in one vLLM
  instance. The reference "mirror in ggml" implementation.

## 4. Implications for Project 19

1. **The kernel is not the novel cost.** llama.cpp's shift is a cheap K-tensor
   rotation graph; the heal is LMCache's HKVD selection (~15% recompute ≈ the
   ~2–2.8 ms/token prefill on the 3060s). For large objects (the reframe
   SPIKE-B-RESULTS.md already endorsed): M = 1024 tokens, h = 0.15 →
   save ≈ (M − hM)·2.2 ms ≈ **1.9 s minus kernel cost** — the case where it
   beats re-prefill. Small-splice P4 economics are unchanged.
2. **Genuinely novel work:** (a) position-selective *partial* splice (llama.cpp
   shifts whole sequences, not arbitrary cell installs into a live seq);
   (b) adapter-tagged admission — upstream does the opposite (#26207);
   (c) heal selection integrated into the llama.cpp graph (LMCache's lives
   vLLM-side, engine-specific).
3. **KVLink's unrotated-K design is a serious alternative** to post-hoc
   re-anchor for the KV-DB: store W_k·x without RoPE, apply RoPE at install
   position. Cleaner math (no lossy rotation), but requires RoPE-at-attention
   engine support rather than a shift op.
4. **Verify before betting:** (a) Nanbeige rope type (M-RoPE disables shift
   upstream — discussion #24944); (b) runtime KV cache format (q4_0 K breaks
   shift — issue #5652).

## Source index (26 URLs)

- arXiv 2405.16444 (CacheBlend paper), arXiv 2410.15332 (EPIC), arXiv
  2502.16002 v1/v2 (KVLink), arXiv 2510.10129 (CacheClip)
- OpenReview qjd3ZUiHRT (EPIC, ICML 2025)
- docs.lmcache.ai/kv_cache_optimizations/blending.html, github.com/LMCache/
  LMCache, github.com/IBM/vllm (MiniPIC), blog.lmcache.ai CacheBlend EuroSys'25
- github.com/ggml-org/llama.cpp: src/llama-kv-cache.cpp, issues #5652 #4097
  #26207, discussions #13606 #22354 #24944 #3581, PRs #5104 #4815 #13194
- deepwiki.com/qualcomm/llama.cpp + spiritbuun/llama-cpp-turboquant-cuda
  (KV-cache memory management docs)
- github.com/ForceInjection/AI-fundermentals cache_blend.md

Full list with titles/snippets: `research-b-prior-art/journal-extracted.md`
(Part 4).
