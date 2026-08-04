# Deep-research workflow export — Spike B prior art
Source: ~/.claude/.../e6faa7ae.../subagents/workflows/wf_dc72d908-cc4/journal.jsonl
Run date: 2026-07-31 (workflow wf_dc72d908-cc4, 101 subagents). Final synthesis never written (spend limit); fan-out + verification completed.

## Part 1 — Fan-out search results (URLs, titles, snippets)

### agent aabb52c3c1012edd4
- **Blending — LMCache official docs (CacheBlend production API)**
  - URL: https://docs.lmcache.ai/kv_cache_optimizations/blending.html
  - Official docs for the shipped CacheBlend implementation in LMCache. Documents the actual 'blending' API/config used to reuse non-prefix KV chunks and selectively recompute HKVD tokens — the concrete integration surface to study before building a native kernel.
- **LMCache/LMCache — GitHub (CacheBlend source, maintained)**
  - URL: https://github.com/LMCache/LMCache
  - The maintained OSS repo where CacheBlend is actually implemented and integrated with vLLM. Contains the real code for non-prefix KV reuse and selective token recomputation (HKVD selection). Primary implementation reference; merged and actively maintained.
- **CacheBlend (Best Paper @ EuroSys'25): 100% KV Cache Hit Rate in RAG — LMCache blog**
  - URL: https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/
  - Authors' own writeup of the mechanism: reuse cached KV at any position, selectively recompute a small token subset to heal cross-attention. Explains the healing rationale and reports TTFT 2.2-3.3x, throughput 2.8-5x vs full recompute.
- **CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion (paper PDF)**
  - URL: https://arxiv.org/pdf/2405.16444
  - The source paper detailing HKVD token selection: fully recompute layer 1, partially recompute layer 2, compare V matrix against precomputed to pick top-k highest-deviation tokens to recompute. Defines the exact selective-recompute scoping needed to replicate the method.
- **CacheBlend deep-dive notes — ForceInjection/AI-fundermentals (GitHub)**
  - URL: https://github.com/ForceInjection/AI-fundermentals/blob/main/09_inference_system/kv_cache/02_systems/lmcache/cache_blend.md
  - Third-party engineering walkthrough of the LMCache CacheBlend implementation, tracing how partial recompute and HKVD selection map onto code. Useful cross-check on how the healing is scoped in practice.
- **CacheClip: Accelerating RAG with Effective KV Cache Reuse (paper PDF)**
  - URL: https://arxiv.org/pdf/2510.10129
  - Directly relevant to focus (c): a newer position-independent/non-prefix KV reuse method building on the CacheBlend line, describing an alternative token-selection/recompute mechanism for cross-context chunk reuse.

### agent aec07475cf9d4cd12
- **MiniPIC: Flexible Position-Independent Caching in <100 LOC (IBM/vllm fork)**
  - URL: https://github.com/IBM/vllm
  - Directly answers focus (c): a real open-source implementation. MiniPIC stores UNROTATED K vectors in the KV cache and exposes three primitives (block-aligned padding, SSep, PDep) in <100 LOC of core-engine change plus a custom attention backend, realizing Block-Attention, EPIC and PromptCache inside one running vLLM instance. Shows the exact mechanism (defer RoPE, apply at attention time for arbitrary position) I'd want to mirror in ggml. Code from commit 6631ff3 onward.
- **LMCache — CacheBlend selective-recompute KV blending (official impl)**
  - URL: https://github.com/lmcache/lmcache
  - Answers focus (b): CacheBlend is implemented in LMCache. Mechanism: reuse precomputed KV, fully recompute layer 1, partially recompute layer 2 to get accurate QKV, compare V against precomputed to pick top-k High-KV-Deviation (HKVD) tokens (~10-15%), recompute only those to heal cross-attention. Reports 2.2-3.3x TTFT reduction, 2.8-5x throughput. This is the selective-heal blueprint.
- **KVLink: Accelerating LLMs via Efficient KV Cache Reuse**
  - URL: https://arxiv.org/html/2502.16002v2
  - Focus (c). KVLink precomputes each chunk's KV independently at position 0 and makes it valid at arbitrary positions via position re-encoding plus trained link tokens and selective recomputation to restore cross-chunk attention. Directly relevant to RoPE re-anchoring + healing; has code.
- **EPIC: Efficient Position-Independent Caching for Serving LLMs (ICML 2025)**
  - URL: https://openreview.net/forum?id=qjd3ZUiHRT
  - Focus (c). EPIC's LegoLink algorithm mitigates the 'attention sink' at each document start when independently-encoded KV chunks are concatenated at new positions, using O(kN) recompute instead of O(N^2). Core position-independent-reuse method; MiniPIC (IBM/vllm) provides a runnable EPIC implementation.
- **CacheClip: Accelerating RAG with Effective KV Cache Reuse**
  - URL: https://arxiv.org/pdf/2510.10129
  - Focus (c). Newer selective-recompute variant explicitly named in the question; refines which tokens to heal for cross-chunk attention in RAG reuse. Useful comparison point to CacheBlend/KVLink for choosing the heal set.
- **llama.cpp: KV-cache shifting (seq_add/seq_shift, RoPE re-anchor) and its arch limits**
  - URL: https://github.com/ggml-org/llama.cpp/discussions/24944
  - Focus (a)/(d). Confirms llama.cpp's actual API: llama_kv_cache_seq_rm/seq_cp/seq_add perform in-cache RoPE-frequency updates ('context shifting', self-extend/StreamingLLM) by rewriting positions of cached K tensors. Discussion + issues (#3825 delta bug, #5652 q4_0 incompatibility) document where shifting is hard-disabled (M-RoPE, interleaved layers) — key constraints for a native ggml RoPE-shift kernel in a fork.

### agent a2a6b425472a52455
- **Tutorial: KV cache reuse with llama-server (Discussion #13606)**
  - URL: https://github.com/ggml-org/llama.cpp/discussions/13606
  - Directly documents llama.cpp's n_cache_reuse mechanism — the closest thing in-tree to cross-context splicing. It detects a matching chunk of >=N tokens shared with the cached sequence even when it is NOT a pure leading prefix, and RoPE-shifts that cached KV slice forward/backward into its new position via KV-shifting instead of recomputing. Explains exactly the RoPE re-anchoring path relevant to the splice goal, plus practical chunk-size tuning (256).
- **llama_kv_cache_seq_shift does not work with cache type q4_0 (Issue #5652)**
  - URL: https://github.com/ggml-org/llama.cpp/issues/5652
  - Concrete detail on the actual position-shift API (llama_kv_cache_seq_shift / seq_add) that applies RoPE frequency updates to cached K tensors. The bug — shift fails on quantized (q4_0) KV — reveals the mechanism's constraint: RoPE re-rotation of already-quantized keys is lossy/unsupported, a key correctness caveat for a native ggml RoPE-shift kernel.
- **llama.cpp/src/llama-kv-cache.cpp (master source)**
  - URL: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp
  - The actual implementation. Contains the K-shift graph construction: applies RoPE adjustments to the K cache tensors based on accumulated per-cell position deltas (cell.delta), i.e. the raw tensor manipulation you'd fork. Shows which tensors are touched and how the shift graph is built — the reference for building a native RoPE-shift + selective-heal kernel.
- **How do I reuse cache properly? for OpenCode agent switching (Discussion #22354)**
  - URL: https://github.com/ggml-org/llama.cpp/discussions/22354
  - Real-world report of the limits of --cache-reuse: when shared content is not a contiguous shifted block (e.g. a changed system prompt mid-context), llama.cpp still falls back to full reprocessing. Illustrates why pure RoPE-shift is insufficient and motivates CacheBlend-style selective recompute/healing of the boundary tokens.
- **server: prompt cache reused across requests with different per-request lora (Issue #26207)**
  - URL: https://github.com/ggml-org/llama.cpp/issues/26207
  - Documents a correctness hazard when KV computed under one condition (adapter A) is spliced/reused for a request under condition B via prefix/chunk matching — output silently contaminated. Directly relevant to correctness scoping when reusing precomputed KV chunks across contexts; a cautionary case for the healing strategy.
- **Rolling buffer cache (Discussion #3581)**
  - URL: https://github.com/ggml-org/llama.cpp/discussions/3581
  - Design discussion of StreamingLLM/rolling-buffer behavior in llama.cpp using seq_add with a negative shift to slide the window and update RoPE frequencies of existing cached tokens. Useful background on the position-slide primitive that underlies context shift and self-extend.

### agent a527803dc6e35df85
- **llama.cpp/src/llama-kv-cache.cpp (master source)**
  - URL: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp
  - The actual implementation of the KV-cache seq operations. This is where llama_kv_cache_seq_add/seq_div/seq_rm and the RoPE-shift 'update()' path live — the update() function re-applies RoPE frequencies to cached K tensors to reflect new relative positions without re-processing tokens. Direct source for how the raw K tensors are manipulated for position shifting, exactly the primitive to fork for a native RoPE-shift + selective-heal kernel.
- **llama_kv_cache_seq_shift does not work with cache type q4_0 · Issue #5652**
  - URL: https://github.com/ggml-org/llama.cpp/issues/5652
  - Documents that the seq_shift / seq_rm position-shift ops are broken with quantized (q4_0) K cache — a concrete correctness constraint. RoPE re-anchoring of cached keys requires float-domain rotation, so any splice/shift kernel must account for quantized-K limitations. Directly relevant to (a) and (d).
- **Port of self extension (Self-Extend) to server · PR #5104 (and PR #4815, Issue #4886)**
  - URL: https://github.com/ggml-org/llama.cpp/pull/5104
  - llama.cpp's Self-Extend implementation (grouped attention). Uses llama_kv_cache_seq_div and seq_add to remap cached token positions into a compressed group space at decode time — a working example of re-anchoring cached KV to different effective positions via RoPE. Shows the exact API sequence (seq_div/seq_add) to re-position an existing cache; the closest in-tree analog to cross-context splicing.
- **LMCache — Fastest KV Cache Layer (CacheBlend implementation)**
  - URL: https://github.com/LMCache/LMCache
  - The maintained open-source home of CacheBlend (EuroSys'25 best paper). Implements selective-recompute KV blending: reuses non-prefix precomputed KV chunks and recomputes a small top-k subset of tokens to heal cross-attention. Token selection heals highest-deviation tokens (V-value discrepancy). Targets vLLM; production-stack integrated. The reference implementation to learn healing/recompute scoping from for (b).
- **Blending | LMCache docs**
  - URL: https://docs.lmcache.ai/kv_cache_optimizations/blending.html
  - Developer-facing docs for CacheBlend's 'blending' feature: how LMCache concatenates independently-cached KV chunks at arbitrary positions and selectively recomputes tokens to restore cross-attention. Explains configuration and the mechanism (first-layer full recompute to pick recompute tokens, then partial healing). Concrete how-to for the selective-recompute path in (b)/(c).
- **kv-cache : add SWA support by ggerganov · PR #13194**
  - URL: https://github.com/ggml-org/llama.cpp/pull/13194
  - Recent core rework of llama.cpp's KV cache to support sliding-window attention (StreamingLLM-style). Touches the same seq_add/shift/defrag machinery and shows current maintainer direction for cache-cell position management — useful context for where a fork's RoPE-shift kernel would hook into the modern kv-cache abstraction (a)/(d).

### agent a9df09c2207c4e83e
- **llama.cpp KV cache seq_add / seq_shift RoPE-shift mechanism (DeepWiki)**
  - URL: https://deepwiki.com/spiritbuun/llama-cpp-turboquant-cuda/2.2-context-kv-cache-and-batching
  - Directly answers focus (a): documents llama.cpp's actual API — seq_add (shift positions), seq_div, seq_rm, seq_cp. The llama_kv_cache::update 'shift' operation updates the RoPE frequencies of existing cached tokens to reflect new relative positions WITHOUT re-processing them; seq_add with a negative delta re-rotates cached keys to slide the window. This is exactly the native ggml RoPE-shift-on-cached-K mechanism the fork wants to learn from.
- **llama_kv_cache_seq_shift does not work with cache type q4_0 (llama.cpp Issue #5652)**
  - URL: https://github.com/ggml-org/llama.cpp/issues/5652
  - Concrete llama.cpp-specific evidence (focus a/d): confirms the seq_shift/RoPE-shift op operates by re-rotating the raw cached K tensors, and breaks on quantized (q4_0) KV cache because you cannot losslessly re-apply RoPE rotation to quantized keys. Critical correctness caveat for anyone building a native ggml RoPE-shift kernel on a quantized cache.
- **CacheBlend (EuroSys'25 Best Paper): 100% KV Cache Hit Rate in RAG — LMCache blog**
  - URL: https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/
  - Directly answers focus (b): authors'/LMCache implementation of CacheBlend. Reuses precomputed KV for non-prefix chunks and selectively recomputes the KV of a small subset of highly-deviated tokens to heal cross-attention. Reports 2.2–3.3x TTFT reduction, 2.8–5x throughput. Open-source code at github.com/LMCache/LMCache; docs at docs.lmcache.ai/kv_cache_optimizations/blending.html. Describes how healing tokens are selected (highest deviation) and pipelined with retrieval.
- **EPIC: Efficient Position-Independent Context Caching for Serving LLMs (ICML 2025)**
  - URL: https://arxiv.org/abs/2410.15332
  - Answers focus (c) and the core RoPE re-anchoring math: modular position-independent KV reuse. States a precomputed Key state can be realigned to a new position via a single RoPE rotation (R(δ)R(p0)=R(p1) since RoPE composes). Two mechanisms — AttnLink (exploits static attention sparsity to minimize recompute for accuracy recovery) and KVSplit (semantic-coherent chunking). Public ICML poster/proceedings; the CacheBlend-style selective-recompute alternative with a re-rotation angle.
- **KVLink: Accelerating LLMs via Efficient KV Cache Reuse**
  - URL: https://arxiv.org/html/2502.16002v1
  - Focus (c): position-independent cross-context KV reuse implementation. Precomputes KV of each chunk independently then re-rotates/re-anchors keys to concatenated positions plus lightweight trained link tokens to heal cross-segment attention. Useful contrast to CacheBlend's training-free selective recompute — KVLink instead fine-tunes to tolerate spliced KV.
- **Possible issue with kv shifting (llama.cpp Issue #4097)**
  - URL: https://github.com/ggml-org/llama.cpp/issues/4097
  - llama.cpp-specific (focus a/d): discussion of correctness pitfalls when shifting KV positions via RoPE re-rotation — which tensors are touched and how the relative-position invariant can break in practice. Relevant real-world debugging context before building a native ggml RoPE-shift + selective-heal kernel.

## Part 2 — Claim-generation (fan-out synthesis of claims from sources)

### agent ae070553d5fc8576e (sourceQuality=primary, publishDate=)
- [central] LMCache implements CacheBlend to extend KV reuse beyond prefix caching by reusing cached KV blocks at ANY position in the prompt (position-independent reuse), using selective token recomputation for quality recovery.
  - quote: extending "KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt. This leverages CacheBlend to selectively recompute tokens for quality recovery."
- [central] LMCache is the maintained open-source home of CacheBlend, targeting mainstream open-source serving engines, specifically vLLM (including vLLM V1 and multimodal models).
  - quote: extends its turbo-boost to multimodal models in vLLM V1 ... integrated with mainstream open-source serving engines
- [supporting] CacheBlend's stated purpose is fast LLM serving for RAG via cached knowledge fusion, i.e. combining multiple independently-cached KV chunks.
  - quote: Cacheblend: Fast large language model serving for rag with cached knowledge fusion

### agent a182135dd1c600bae (sourceQuality=primary, publishDate=)
- [central] CacheBlend, as productized in LMCache, enables KV cache reuse at non-prefix positions by selectively recomputing a subset of tokens to heal cross-attention.
  - quote: CacheBlend enables KV cache reuse for non-prefix positions by recomputing a subset of tokens at non-prefix positions.
- [central] The fraction of tokens recomputed (healed) is a configurable ratio, exposed via LMCACHE_BLEND_RECOMPUTE_RATIOS with an example value of 0.15 (15%).
  - quote: LMCACHE_BLEND_RECOMPUTE_RATIOS: Sets the proportion of tokens to recalculate (example: 0.15 or 15%)
- [central] Which tokens get recomputed is decided using a specific check layer (e.g. layer 1), configured via LMCACHE_BLEND_CHECK_LAYERS.
  - quote: LMCACHE_BLEND_CHECK_LAYERS: Specifies which layers determine token recomputation (example: layer 1)
- [supporting] LMCache's CacheBlend implementation targets the vLLM engine and requires layerwise execution (LMCACHE_USE_LAYERWISE) plus chunk separator strings (LMCACHE_BLEND_SPECIAL_STR).
  - quote: CacheBlend targets vLLM and is documented as part of LMCache's legacy in-process mode
- [supporting] The blending path combines multiple precomputed KV caches for concatenated text chunks (RAG), and requires tokenizing chunks before concatenation to avoid tokenization drift.
  - quote: combining multiple pre-computed KV caches when their corresponding texts are concatenated in the LLM input, making it particularly useful for RAG scenarios

### agent aac897004a4f4e815 (sourceQuality=blog, publishDate=2025-03-31)
- [central] CacheBlend reuses non-prefix KV cache chunks across different contexts by selectively recomputing only a small fraction of tokens' cross-attention to heal quality, rather than recomputing whole sequences.
  - quote: selectively recomputes cross-attention layers to maintain generation quality while preventing redundant computations
- [central] CacheBlend updates positional encoding when a cached chunk is placed at a new sequence position, preserving attention integrity.
  - quote: CacheBlend efficiently updates the positional encoding during reuse, preserving the integrity
- [central] CacheBlend is open source, integrated into the LMCache library and vLLM Production Stack.
  - quote: Deployed through LMCache library and vLLM Production Stack for Kubernetes-native deployment.
- [supporting] CacheBlend achieves roughly 3x reduction in TTFT and 3x throughput increase with near-100% KV cache hit rate on 2WikiMQA with Llama 70B on dual A40 GPUs.
  - quote: 3x reduction in Time to First Token (TTFT)

### agent a01ee734ea7565e67 (sourceQuality=primary, publishDate=)
- [supporting] IBM maintains a public fork of vLLM to develop support for 'span semantics', a concept plausibly related to position-independent KV cache reuse across contexts.
  - quote: a fork of vLLM which we are using to develop support for _span semantics_
- [tangential] The IBM/vllm fork inherits vLLM's prefix caching support, but the fetched repository page contains no implementation details of MiniPIC, RoPE re-anchoring, position-shift, or selective token recomputation.
  - quote: Prefix caching support

### agent a864bcfb20b26c63b (sourceQuality=secondary, publishDate=)
- [central] CacheBlend selects which tokens to recompute by measuring the L2 distance between newly computed and cached keys at an early check layer (typically Layer 1) and picking the top tokens with the largest deviation per a configured recomp_ratio (~15%).
  - quote: 根据配置的 `recomp_ratios` 选取误差最大的 Token ... Computing 新旧 Key 的 L2 距离 (L2 distance between new and old keys) at a check layer (typically Layer 1)
- [central] CacheBlend recomputes KV only for the selected critical tokens rather than the whole context, trading minimal recompute for accurate reuse of non-prefix KV cache.
  - quote: 仅对筛选出的关键 Token 进行 KV 重算 — recomputes KV only for selected critical tokens
- [central] CacheBlend requires applying RoPE to Q/K before computing the key differences because the cached keys are already RoPE-processed, handling position encoding for relocated chunks.
  - quote: 在计算差异之前，必须先对 Q/K 进行旋转位置编码 (RoPE) because cached keys are already RoPE-processed
- [supporting] CacheBlend is implemented in LMCache via an LMCBlender component that coordinates GPU computation (vllm_model) and storage I/O (cache_engine) using a layerwise pipeline and in-place KV updates.
  - quote: LMCache integrates CacheBlend through `LMCBlender`, which 协调 GPU 计算（通过 `vllm_model`）和存储 I/O（通过 `cache_engine`）
- [supporting] CacheBlend reports TTFT reduction of 2.2x-3.3x and throughput increase of 2.8x-5x.
  - quote: 首字延迟降低 2.2x - 3.3x ... 推理吞吐量提升 2.8x - 5x

### agent a4d1b0258c0829295 (sourceQuality=primary, publishDate=2024-05-26)
- [central] CacheBlend enables reuse of precomputed KV caches at non-prefix positions by selectively recomputing the KV values of a small subset of tokens to partially update each reused cache, healing cross-attention.
  - quote: selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache
- [central] CacheBlend reuses precomputed KV caches regardless of whether the chunk is at a prefix position, i.e. supports position-independent reuse.
  - quote: reuses the precomputed KV caches, regardless prefix or not
- [supporting] CacheBlend reduces time-to-first-token by 2.2-3.3x and increases inference throughput by 2.8-5x versus full KV recompute, without compromising generation quality.
  - quote: CacheBlend reduces time-to-first-token (TTFT) by 2.2-3.3x and increases the inference throughput by 2.8-5x from full KV recompute without compromising generation quality
- [supporting] The extra recompute delay can be pipelined with KV cache retrieval, allowing caches to be stored on slower/larger devices without added inference delay.
  - quote: the small extra delay for recomputing some tokens can be pipelined with the retrieval of KV caches within the same job, allowing CacheBlend to store KV caches in slower devices with more storage capacity while retrieving them without increasing the inference delay
- [central] CacheBlend has an open-source implementation available in the LMCache project.
  - quote: The code is available at: https://github.com/LMCache/LMCache

### agent a164b9008284c762e (sourceQuality=primary, publishDate=2025-10-11)
- [central] CacheClip is a CacheBlend-style selective KV-cache recomputation method: it precomputes per-chunk KV independently and heals cross-chunk attention by recomputing a selected subset of tokens, rather than using RoPE position-shifting/re-anchoring.
  - quote: CacheClip integrates four techniques: (1) auxiliary-model-guided token selection for selective KV cache recomputation, (2) shared prefixes to eliminate redundant attention sinks, (3) a sliding-window grouping strategy to maintain local coherence during partial KV cache updates
- [central] CacheClip selects which tokens to recompute/heal by using a small auxiliary LLM whose last-layer attention distribution approximates the primary model's, identifying tokens critical for restoring inter-chunk attention.
  - quote: small auxiliary LLMs exhibit similar last-layer attention distributions to primary LLMs (the target model for generation), enabling efficient identification of tokens critical for restoring inter-chunk attention
- [supporting] CacheClip's recomputation ratio is user-adjustable, and at 20% recomputation it achieves up to 3.33x prefill speedup while retaining 85.2% (NIAH) and 91.1% (LongBench) of full-attention quality, outperforming CacheBlend and APE.
  - quote: CacheClip retains up to 85.2% and 91.1% of full-attention performance on NIAH and LongBench, outperforming CacheBlend and APE by 16.1 and 12.8 points on NIAH... Meanwhile, CacheClip accelerates LLM inference by up to 3.33× in prefill time (with recomp% = 20%)
- [supporting] CacheClip explicitly positions itself against APE and CacheBlend as prior selective-recompute KV reuse methods, and eliminates repeated attention sinks via shared prefixes, addressing missing inter-chunk attention in direct precomputation.
  - quote: Recent methods like APE and CacheBlend partially address these issues but remain inadequate for robust RAG applications

### agent a2a603d5917db666f (sourceQuality=primary, publishDate=2025-02-22)
- [central] KVLink decouples RoPE from stored KV states by caching pre-rotation key states (W_k*x) and reapplying global rotary embeddings at each token's correct full-sequence position at inference time, so a chunk's KV becomes valid at a new position.
  - quote: we decouple the key-value states from the positional embeddings when storing them ... apply the global rotary embedding for the KV states of each token appropriate to its location in the full sequence.
- [central] Instead of recomputing a subset of document tokens, KVLink inserts a small set of trainable 'link tokens' between independently encoded segments whose KV is computed at inference to heal cross-segment attention.
  - quote: we append a small set of trainable tokens between each segment's precomputed KV states before concatenation. The KV representations for these tokens are computed during inference.
- [supporting] Link tokens use a customized attention mask that attends to all preceding tokens while document tokens keep local causal attention, restoring cross-attention lost by independent encoding.
  - quote: all tokens (including the link tokens) in the preceding documents
- [supporting] KVLink reduces time-to-first-token by up to 96% on 5,000-token contexts and improves QA accuracy up to 6.6% on Natural Questions and 7.3% on HotpotQA versus baselines.
  - quote: reducing time-to-first-token by up to 96% compared to standard LLM inference
- [tangential] KVLink requires model fine-tuning (trainable link tokens) rather than being a training-free splice; code is provided only via arXiv supplementary material with public-release status unclear.
  - quote: Code is available in the supplementary material.

### agent a2c9e99fcc43b3699 (sourceQuality=primary, publishDate=2024-10-20 (v3: 2025-05-27))
- [central] EPIC enables position-independent KV cache reuse (PIC), allowing KV vectors of a text chunk to be reused regardless of the preceding prefix/position, rather than recomputing prefill for every new context.
  - quote: The paper introduces Position-Independent Caching (PIC) to enable KV vector reuse regardless of prefix variations.
- [central] EPIC's core algorithm LegoLink heals cross-context attention by targeting the 'attention sink' effect at document/chunk boundaries, recomputing only a minimal set of tokens to restore accuracy.
  - quote: The core innovation is the LegoLink algorithm, which addresses the 'attention sink' effect that occurs at document boundaries to maintain accuracy with minimal computation overhead.
- [supporting] EPIC reports up to 8x TTFT improvement and 7x throughput gains over existing systems with negligible or no accuracy loss.
  - quote: up to 8x improvements in Time-To-First-Token (TTFT) and 7x throughput gains over existing systems, with negligible or no accuracy loss
- [tangential] The EPIC paper (arXiv:2410.15332) was submitted October 2024 with a v3 revision in May 2025, corresponding to ICML 2025.
  - quote: Initial submission: October 20, 2024 - Latest version (v3): May 27, 2025

### agent aa98a1b98e7966a85 (sourceQuality=primary, publishDate=2024-02-21)
- [central] llama.cpp exposes a KV-cache sequence-shift API (llama_kv_cache_seq_shift) that performs RoPE-based position shifting of cached keys during context swapping, invoked when the context window fills.
  - quote: User @ngxson reported that `llama_kv_cache_seq_shift` and `llama_kv_cache_seq_rm` functions fail when the K cache uses q4_0 quantization format. The crash occurs during context swapping—the mechanism that removes old tokens to make room for new ones when the context window fills.
- [central] The position-shift/RoPE operation manipulates the raw K (key) tensor in the cache, and fails when that K tensor is stored quantized as q4_0 (V remains f16), crashing with assertions in ggml.c.
  - quote: The llama_kv_cache_seq_shift or llama_kv_cache_seq_rm (or all two of them) is broken with cache type q4_0 for K.
- [supporting] llama.cpp's context-shift mechanism can be disabled with --no-context-shift, confirming context shifting is a real built-in feature of the engine.
  - quote: The temporary fix provided was: "disable context shift via --no-context-shift"
- [supporting] The shift is triggered automatically at context capacity, dropping old tokens and repositioning remaining ones (n_past=10, n_left=9 for n_ctx=10).
  - quote: The log shows: "context full, swapping: n_past = 10, n_left = 9, n_ctx = 10" followed by multiple assertion failures.

### agent a8ba63e8c0a13b09a (sourceQuality=forum, publishDate=2025-05-17)
- [supporting] llama-server reuses KV cache via slot-based prompt-prefix matching controlled by the -sps parameter (default 0.5), not via explicit RoPE re-anchoring or position-shifting of cached keys.
  - quote: By default, `llama-server` attempts to assign slot to a new request based on prompt similarity. The `-sps` parameter controls this behavior. A value of 0.5 (the default) means a slot is considered a match if at least 50% of the prompt context matches.
- [supporting] llama-server exposes slot save/restore endpoints (POST /slots/{id}?action=save and restore) to persist and reload KV cache state to/from disk.
  - quote: `POST /slots/{id}?action=save` - saves KV cache to disk
- `POST /slots/{id}?action=restore` - restores cached state
- [tangential] Prefix reuse yields large reductions in prompt-processing work, e.g. only 1 new token evaluated at n_past=43 (~97% reduction).
  - quote: prompt processing progress, n_past = 43, n_tokens = 1, progress = 0.023256
- [tangential] The tutorial's cache reuse keeps slots isolated per conversation and does not address cross-context KV splicing/reuse at a new position.
  - quote: Cross-context sharing isn't detailed; slots appear isolated per conversation.

### agent a8429b5cd682b5f92 (sourceQuality=forum, publishDate=)
- [central] llama.cpp hard-disables KV-cache shifting (context shift) for certain architectures once the context limit is reached, rather than rotating the ring buffer.
  - quote: whenever a user hits the context limit (e.g., 32,768 tokens) on these architectures, llama.cpp hard-disables KV-cache shifting
- [central] The architectural reasons llama.cpp cites for disabling KV-cache/RoPE shift are multi-dimensional RoPE (M-RoPE), interleaved layers, and reasoning tokens.
  - quote: Multi-dimensional RoPE (M-RoPE), interleaved layers, or reasoning tokens
- [supporting] When context is exhausted and shifting is disabled, llama.cpp either stops generation or falls back to full history recomputation (prefill), rather than shifting cached KV positions.
  - quote: The engine either stops generation entirely or forces a full, agonizingly slow history re-computation (prefill overhead)
- [supporting] llama.cpp blocks the ring-buffer rotation at the kernel level even when the user forces full global attention, preventing manual position shifting of the cache.
  - quote: Even when we are fully willing to sacrifice VRAM and explicitly force full global attention...the engine still proactively blocks the ring-buffer rotation on the kernel level

### agent a15eadb399b0384dd (sourceQuality=primary, publishDate=)
- [central] llama.cpp's KV cache implementation exposes a seq_add() operation that shifts cached token positions forward by a delta (RoPE re-anchoring), enabling reuse of cached K/V at a new position offset.
  - quote: **`seq_add()`** - Shifts positions forward: "if (cells.pos_add(i, shift)) { if (new_head == cells.size())"
- [central] llama.cpp provides a seq_div() operation that divides/compresses cached token positions by a divisor, the mechanism used by Self-Extend for position compression of the KV cache.
  - quote: **`seq_div()`** - Divides positions by a divisor: "cells.pos_div(i, d);" for position compression
- [central] When positions of cached K change (context shift), llama.cpp builds a dedicated compute graph (build_graph_shift) that re-applies rotary position embeddings to the cached K tensors, but only when the model uses RoPE.
  - quote: if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
    auto * gf = build_graph_shift(res, lctx);
- [supporting] llama.cpp supports removing arbitrary position ranges from a sequence's KV cache via seq_rm(), operating per-cell over a [p0,p1) range.
  - quote: **`seq_rm()`** - Removes position ranges: iterates through cells checking "if (!cells.pos_in(i, p0, p1))"
- [supporting] The KV cache supports cross-stream (multi-sequence) copies of cached K/V tensors between sequence streams via backend tensor copy operations.
  - quote: Cross-stream sequence copies handle multi-stream scenarios: "ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst])"

### agent ac5c2f6139dbb6ffc (sourceQuality=forum, publishDate=2026-04-25)
- [supporting] llama.cpp's --cache-reuse flag keeps the tail end of a prefilled sequence after deleting earlier parts while making the model behave as if the earlier tokens were still present, i.e. it is a position-manipulation trick rather than true cross-context splicing.
  - quote: It's a processing trick to allow you to keep the tail end of prefilled sequence after deleting earlier parts, but crucially while pretending that the earlier parts are still there.
- [tangential] The --cache-reuse mechanism does not function with models that use recurrent state (e.g. Qwen3.6-35B-A3B), limiting KV-reuse tricks to standard attention KV caches.
  - quote: The linked comment mentions that `--cache-reuse` doesn't work with models that use recurrent state. `Qwen3.6-35B-A3B` does use recurrent state
- [supporting] Changing a system prompt invalidates the cache because reuse is gated on Longest Common Prefix matching, so there is no partial-splice healing path in llama.cpp; the entire history must be reprocessed.
  - quote: if you want to replace your original system prompt and have it take effect, then unfortunately the only way is to reprocess the entire history

### agent aadcbe2e8c0b13de8 (sourceQuality=primary, publishDate=2026-07-28)
- [tangential] llama.cpp's server reuses the cached KV computed under one LoRA adapter for a later request selecting a different adapter whenever the prompt prefix matches, contaminating output.
  - quote: KV computed while adapter A was active is reused for a request selecting adapter B whenever the prompt prefix matches.
- [supporting] llama.cpp server keys its prompt cache on prompt-prefix match alone, not on the adapter/LoRA configuration, so cache identity ignores the compute context.
  - quote: invalidate the slot's cached prefix (or key the cache on the lora config)
- [tangential] The only current mitigation for cross-context KV misuse in llama.cpp server is disabling caching entirely via cache_prompt:false, forcing full recompute.
  - quote: Using `cache_prompt: false` on every request disables caching but incurs full re-processing costs.

### agent a5f1ae964aa774bad (sourceQuality=secondary, publishDate=)
- [central] llama.cpp exposes a KV-cache position-shift operation (seq_add / K-Shift) that updates the RoPE frequencies of existing cached tokens rather than physically moving data, enabling 'Context Shifting'.
  - quote: by using `seq_add` with a negative shift, the engine updates the RoPE frequencies of existing tokens in the cache, effectively 'sliding' the window back to make room for new tokens
- [supporting] The position-shift operates through the llama_memory_i interface abstraction and modifies positional encodings (RoPE) of cached tokens in place.
  - quote: The position-shift mechanism operates through the `llama_memory_i` interface abstraction, which manages the underlying KV cache implementation
- [supporting] llama.cpp provides seq_rm to remove tokens for a sequence within a position range (p0,p1), plus seq_cp and seq_keep for KV state copy/retention without recomputation.
  - quote: Removes tokens belonging to a specific sequence within a designated position range (p0, p1)
- [tangential] This documentation source does not describe StreamingLLM or self-extend, and does not use the legacy names llama_kv_cache_seq_shift or llama_kv_self_seq_add/div.
  - quote: The documentation provided does not explicitly mention `llama_kv_cache_seq_shift`, `llama_kv_self_seq_add/div`, StreamingLLM, or self-extend functionality.

### agent a43ac1200d06f0ba1 (sourceQuality=primary, publishDate=2024-01-27)
- [central] llama.cpp merged a port of Self-Extend (grouped-attention context extension) into the HTTP server in PR #5104, making runtime position/RoPE manipulation of the KV cache a maintained upstream feature.
  - quote: Pull Request #5104 was **merged on January 27, 2024** by ggerganov into the master branch of ggml-org/llama.cpp.
- [central] Self-Extend in llama.cpp is driven by two runtime parameters, --grp-attn-n (group factor G) and --grp-attn-w (window W), that reshape token positions to fit within the model's trained context, demonstrating an existing position-remapping mechanism over cached tokens.
  - quote: Both parameters must satisfy: G ≥ (desired_context / training_context) and G must be a multiple of W
- [supporting] The feature was empirically validated on solar-10.7b-instruct with a 16384 context using --grp-attn-n 4 --grp-attn-w 2048, showing functional information-retrieval parity across sequential server requests.
  - quote: The implementation was validated with solar-10.7b-instruct-v1.0.Q5_K_M.gguf using `-c 16384 --grp-attn-n 4 --grp-attn-w 2048`
- [supporting] Self-Extend coexists with prompt caching in the server, with a fix ensuring cache behavior is unchanged when self-extend is off, indicating the position-shift logic interacts directly with KV cache management.
  - quote: A fix was applied to ensure "prompt caching should work as before without self extend," addressing initial cache management issues.

### agent a571f554d0a4899d3 (sourceQuality=primary, publishDate=2024-10-20 (v1); last revised 2025-05-27 (v3))
- [central] EPIC implements Position-Independent Caching (PIC), enabling reuse of a chunk's KV vectors regardless of what prefix precedes it, rather than requiring exact prefix matches.
  - quote: Position-Independent Caching (PIC) addresses this issue by enabling modular reuse of the KV vectors regardless of prefixes.
- [central] EPIC's core mechanism is the LegoLink algorithm, which heals cross-attention by mitigating the spurious 'attention sink' effect that appears at every reused document's beginning, using minimal recomputation.
  - quote: introducing EPIC, a serving system incorporating our new LegoLink algorithm, which mitigates the inappropriate "attention sink" effect at every document beginning, to maintain accuracy with minimal computation
- [supporting] EPIC reports up to 8x TTFT improvement and 7x throughput gains over existing systems with negligible or no accuracy loss.
  - quote: Experiments show that EPIC achieves up to 8x improvements in Time-To-First-Token (TTFT) and 7x throughput gains over existing systems, with negligible or no accuracy loss.
- [supporting] EPIC targets RAG and few-shot serving scenarios where immutable documents appear under varying prefixes, which is exactly the cross-context KV reuse case.
  - quote: limiting reuse cases in settings such as few-shot learning and retrieval-augmented generation, where immutable content (e.g., documents) remains unchanged across requests but is preceded by varying prefixes

## Part 3 — Adversarial verification (all refuted=false, high confidence)

### agent aa465862d027941b0 — refuted=False, confidence=high
PR #5104 "Port of self extension to server" was merged by ggerganov on Jan 27, 2024. Contributor Maximilian-Winter ported self-extend (grouped attention) into the llama.cpp HTTP server, adding --grp-attn-n / --grp-attn-w flags, mirroring the main-binary capability. Self-Extend is a grouped-attention context-extension method that manipulates token positions (runtime RoPE/position handling of the KV cache), so the claim's characterization is correct. It is in upstream master. Minor caveat: Self-Extend does position grouping at attention time rather than being a pure standalone KV-shift op like llama_kv_cache_seq_add, but the claim only asserts "runtime position/RoPE manipulation... upstream feature," which holds.

### agent a64e2028042f57390 — refuted=False, confidence=high
GitHub PR ggml-org/llama.cpp#5104 "server : ported self-extend/ngram-cache" ports the Self-Extend feature to the HTTP server and was merged Jan 27, 2024 by ggerganov into master. Self-Extend is grouped-attention context extension exposed at runtime via --grp-attn-n / --grp-attn-w (tester used -c 16384 --grp-attn-n 4 --grp-attn-w 2048 on a 4096-trained model). It works by runtime position remapping of the KV cache (grouped/divided positions via llama_kv_cache_seq_div/seq_add-style ops), i.e. RoPE/position manipulation of cached keys. The claim's specifics (PR number, merge date, merger, server target, Self-Extend/grouped-attention, runtime position manipulation) all match the primary source. Minor caveat: the same discussion notes a related KV-cache context-shift bug later fixed in PR #5195, but that does not contradict the claim.

### agent a89842c3a2a011127 — refuted=False, confidence=high
Verified against llama.cpp PR #5104 ("Port of self extension to server") and server docs. Self-Extend in llama.cpp is genuinely controlled by two runtime params: --grp-attn-n (group attention factor G, default 1=disabled) and --grp-attn-w (group attention width W, default 512), exposing the SelfExtend paper's group-size and neighbor-window hyperparameters. These do reshape/remap token positions (grouped floor-division of positions) to keep them within the model's trained context, per LongLM Self-Extend (arXiv 2401.01325), extended length = (L - w)*G + w. llama.cpp enforces a divisibility constraint between grp-attn-n and grp-attn-w (n must be a multiple of w), matching the supporting quote. It is a real, existing position-remapping mechanism. Minor caveat: Self-Extend remaps positions of all tokens in the window during attention (a position-mapping trick), not literally a splice of a foreign precomputed KV chunk — but the claim only asserts a position-remapping mechanism exists, which is true. Primary source (merged PR + paper) is sufficient quality.

### agent a10c89a98db8be9d3 — refuted=False, confidence=high
PR #5104 (ggml-org/llama.cpp) ports Self-Extend to the server and confirms both parameters: --grp-attn-n (G, group factor, default 1=disabled) and --grp-attn-w (W, group attention width, default 512). The PR discussion and llama.cpp server README both confirm they expose the SelfExtend paper's group-size and neighbor-window hyperparameters, applying grouped attention (floor operation) to distant tokens to fit positions within trained context. The constraint quote "G must be a multiple of W" matches the PR's stated "G has to be multiple of W", and G ≥ desired_context/training_context is the documented sizing rule (contributor tested -c 16384 --grp-attn-n 4 --grp-attn-w 2048 on a 4096 model). This is a primary source (merged upstream feature) and the claim accurately describes it as a position-remapping mechanism. Minor caveat: Self-Extend remaps positions at attention/decode time via floor-division grouping, not by rotating raw cached K tensors in place like the RoPE-shift ops the research question targets — but the claim only asserts a position-remapping mechanism exists, which is correct.
counterSource: https://github.com/ggml-org/llama.cpp/pull/5104

### agent ae37575e2ed6fac92 — refuted=False, confidence=high
Primary source confirms the claim. GitHub PR ggml-org/llama.cpp#5104 "Port of self extension to server" by Maximilian-Winter was merged into master on 2024-01-27 by ggerganov (18 commits). It ports Self-Extend / grouped-attention context extension into the HTTP server, exposing --grp-attn-n and --grp-attn-w flags; the feature already existed in the main/CLI path (issue #4886 requested the server port). Self-Extend is a runtime position/RoPE manipulation technique (grouped attention divides token positions, floor(pos/G), re-mapping RoPE positions over the KV cache), and it remains a documented feature in the current tools/server/README.md, i.e. maintained upstream. The quote about the merge date/author is accurate and the claim does not overreach. Minor nuance: grouped-attention position remapping is related to but distinct from the explicit llama_kv_cache_seq_shift/seq_add ops, yet both are runtime KV position-manipulation features and the characterization holds.

### agent a160f4b8e35c863ad — refuted=False, confidence=medium
PR #5104 (primary) and PR #4815 confirm the central claim: llama.cpp's Self-Extend is driven by --grp-attn-n (group factor G) and --grp-attn-w (neighbor window W), and it remaps/reshapes token positions via grouped attention to fit the model's trained context — a genuine position-remapping mechanism. The server README documents both flags (grp-attn-n default 1=disabled; grp-attn-w default 512), and the tested example --grp-attn-n 4 --grp-attn-w 2048 extends 4096-trained to 16384. The G >= desired_context/training_context relation is correct. CAVEAT: the supporting quote's 'G must be a multiple of W' is backwards — the real constraint is W must be a multiple of G (ga_w % ga_n == 0), consistent with the PR's own example (W=2048, G=4). This is a minor factual slip in one sub-detail, not a refutation of the core claim, which is well-supported by the primary source. Note also this is decode-time grouped positioning, not cross-context KV splicing/RoPE re-anchoring of foreign cached chunks — but the claim only asserts a position-remapping mechanism exists, which is true.
counterSource: https://github.com/ggml-org/llama.cpp/pull/4815 and llama.cpp server/completion README (grp-attn constraint ga_w % ga_n == 0)

### agent ad20bf38a83ab5983 — refuted=False, confidence=high
The primary source (docs.lmcache.ai blending page) directly confirms the claim. It documents the env var LMCACHE_BLEND_RECOMPUTE_RATIOS as "Ratio of tokens to recompute" with the example `os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"`, i.e. 15% of tokens recomputed. This matches the claim's wording (configurable ratio, exposed via that env var, example 0.15/15%) exactly. Source is the official LMCache docs — appropriate primary source for a config-detail claim. Not extraordinary, not outdated, not marketing. The only minor nuance is the var name is plural ("RATIOS"), consistent with LMCache/CacheBlend supporting per-layer or per-source lists, but the scalar 0.15 example is genuinely shown, so the claim is not an overreach.

### agent ab60e60dd9298926f — refuted=False, confidence=high
Verified against the primary source (LMCache official docs, docs.lmcache.ai/kv_cache_optimizations/blending.html). The page contains a code example: `# Ratio of tokens to recompute` / `os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"`. This directly confirms all three parts of the claim: (1) the env var LMCACHE_BLEND_RECOMPUTE_RATIOS exists, (2) it is a configurable recompute ratio, and (3) the documented example value is 0.15 (15%). LMCache is the maintained open-source engine that ships CacheBlend as its "blending" feature, so the source quality matches the claim strength. No overreach: the quote and claim align exactly. Nothing contradicts it. Claim is current (docs are live/maintained).

### agent adb53e5780b8694ed — refuted=False, confidence=high
Primary source (docs.lmcache.ai/kv_cache_optimizations/blending.html) states verbatim: "CacheBlend enables KV cache reuse for non-prefix positions by recomputing a subset of tokens at non-prefix positions." The quote directly supports the claim word-for-word. Corroborated by the CacheBlend paper (arXiv:2405.16444, EuroSys'25 Best Paper) and LMCache's own blog (blog.lmcache.ai, 2025-03-31), which describe the mechanism: fully recompute layer 1, partially recompute layer 2, compare recomputed V matrix against precomputed to pick top-k tokens with largest V-value discrepancy for selective recompute — i.e., healing cross-attention. CacheBlend is genuinely productized in LMCache (integrated as a documented feature, one-line-change RAG acceleration). Source quality is high: primary vendor docs + peer-reviewed best paper. Not outdated (2025), not marketing-only (backed by academic paper with public code at github.com/LMCache). No contradicting source found.

### agent a586e5a4127539154 — refuted=False, confidence=high
Claim is accurately supported by the quote and corroborated by multiple sources. Primary: LMCache official docs (docs.lmcache.ai/kv_cache_optimizations/blending.html) state verbatim "CacheBlend enables KV cache reuse for non-prefix positions by recomputing a subset of tokens at non-prefix positions." Corroborated by the peer-reviewed EuroSys'25 Best Paper "CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion" (arXiv 2405.16444; ACM TOCS 10.1145/3790254), which specifies the concrete healing mechanism: fully recompute layer 1, partially recompute layer 2, then select top-k tokens with largest V-matrix discrepancy for recomputation. CacheBlend is a real, integrated component of LMCache (open-source), not vaporware. Reported ~3x TTFT reduction / 100% RAG hit rate. Current (2025), primary-sourced, mechanism matches the claim exactly.

### agent a4b958367c1303925 — refuted=False, confidence=high
Primary source (LMCache docs) states verbatim: "CacheBlend enables KV cache reuse for non-prefix positions by recomputing a subset of tokens at non-prefix positions." The underlying CacheBlend paper (arXiv:2405.16444, EuroSys'25 best paper, also in ACM TOCS 2025) confirms the "heal cross-attention" framing: precomputed non-prefix KV caches "ignore the text's cross-attention with the preceding texts," and CacheBlend "selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache" — selecting recompute tokens via top-k V-value discrepancy at the first/second layer. LMCache's blending module implements this ("leverages CacheBlend to selectively recompute tokens for quality recovery"). Reported TTFT reduction 2.2–3.3x. Source quality matches claim strength: peer-reviewed paper + official product docs + open-source code (github.com/LMCache/LMCache), current (2025). The only minor gap is the docs page not literally using the words "cross-attention," but the mechanism (recompute to correct cross-attention) is exactly the paper's stated purpose, so the claim is not an overreach.

### agent af4c4f70258230b7d — refuted=False, confidence=high
The primary LMCache docs (docs.lmcache.ai/kv_cache_optimizations/blending.html) confirm LMCACHE_BLEND_RECOMPUTE_RATIOS exists as a configuration option, described as the "ratio of tokens to recompute," with an example value of "0.15" (15% of tokens recomputed during CacheBlend blending). This matches the claim verbatim. Source is the authoritative first-party documentation for the LMCache project (which houses the official CacheBlend integration), so source quality matches the claim's strength. The claim is a modest, factual statement about a config knob — not extraordinary, not marketing, and current. No contradicting evidence found; the doc directly supports it.

### agent ab782dbeebf116aae — refuted=False, confidence=high
Primary source (LMCache docs, blending.html) directly supports the claim: LMCACHE_BLEND_CHECK_LAYERS "specifies which layers determine token recomputation" with the config example setting it to layer 1. This is consistent with CacheBlend's published method, where token selection for selective recompute is driven by attention deviation measured at a specific check layer. The claim ("which tokens get recomputed is decided using a specific check layer, e.g. layer 1, configured via LMCACHE_BLEND_CHECK_LAYERS") is an accurate restatement, not an overreach. Source is the official LMCache documentation — appropriate quality for this implementation-detail claim, and current (LMCache is actively maintained).

### agent a6e071d6748645662 — refuted=False, confidence=high
Verified via primary source https://docs.lmcache.ai/kv_cache_optimizations/blending.html. The doc confirms LMCACHE_BLEND_CHECK_LAYERS specifies which layer(s) determine token recomputation, with the config example value being layer 1, and prose stating "Determining which tokens to recompute at layer 1". This is a config-variable-level factual claim well-supported by the official LMCache docs, and it is consistent with CacheBlend's published mechanism of selecting high-deviation (HKVD) tokens using a single check layer's attention deviation. Source quality (project's own documentation) matches the modest strength of the claim; no contradicting evidence found.

### agent a63630d083c261be9 — refuted=False, confidence=high
Primary source (docs.lmcache.ai/kv_cache_optimizations/blending.html) confirms LMCACHE_BLEND_CHECK_LAYERS is documented as "Determining which tokens to recompute at layer 1" — i.e. it specifies which layer(s) decide token recomputation, with layer 1 as the example. This is corroborated by the CacheBlend paper (arXiv:2405.16444), whose mechanism fully recomputes the first layer and uses its QKV/value discrepancies to select the top-k tokens to heal. The claim accurately restates a documented, current config parameter matching the underlying algorithm; no contradicting source found. Companion param LMCACHE_BLEND_RECOMPUTE_RATIOS (~0.15) sets how many tokens. Claim is not an overreach.

### agent a9430a39ab731dcb3 — refuted=False, confidence=high
LMCache's official README and docs (docs.lmcache.ai/kv_cache_optimizations/blending.html) directly state: "Non-prefix KV reuse: Extend KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt. This leverages CacheBlend to selectively recompute tokens for quality recovery." Docs further describe position-independent KV cache reuse requiring recomputation of a subset of tokens. CacheBlend is the EuroSys 2025 method; LMCache is its maintained integration (also into vLLM). The blog reports up to 4.5x TTFT speedup with ~15% recomputation on non-prefix caches. The claim matches the primary source verbatim in mechanism and scope — no overreach, current, high-quality primary source.

### agent a945bf774d8115a00 — refuted=False, confidence=high
Claim is directly supported by the primary source (official LMCache docs, docs.lmcache.ai/kv_cache_optimizations/blending.html): "CacheBlend enables KV cache reuse for non-prefix positions by recomputing a subset of tokens at non-prefix positions." The LMCache repo and blog corroborate: LMCache "extends KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt... leveraging CacheBlend to selectively recompute tokens for quality recovery." CacheBlend is the ACM EuroSys'25 best paper, and LMCache is the authors' maintained OSS implementation (active repo). CacheBlend's mechanism — recompute only the small subset of tokens whose KV values diverge most between naive reuse and full recompute — matches the claim's "selective token recomputation." Current (docs and 2025-2026 blog posts, active project). No contradicting source found; the only nuance is that "position-independent" is achieved via selective recompute healing, not truly position-free KV, which the claim states accurately.

### agent a111ba60ba7f1e532 — refuted=False, confidence=high
LMCache's official docs host a dedicated "Blending" page (docs.lmcache.ai/kv_cache_optimizations/blending.html) documenting CacheBlend, which reuses KV even when the prompt is not a prefix (RAG doc-swap) and blends multiple KV caches to preserve cross-attention. Search results confirm CacheBlend is now compatible with vLLM V1 HMA with dual-RoPE for SWA models and extends turbo-boost to multimodal models in vLLM V1. The repo has active releases and example files (kv_cache_sharing_lmcache_v1.py) targeting vLLM V1. CacheBlend's authors (U. Chicago, Kuntai Du et al.) are the LMCache team, making LMCache its maintained home. Claim targets vLLM specifically, which matches LMCache's stated integration focus.

### agent af7b95f91e44f589c — refuted=False, confidence=high
Claim is well-supported by the primary source. LMCache official docs (docs.lmcache.ai/kv_cache_optimizations/blending.html) and blog state it "extends KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt, leveraging CacheBlend to selectively recompute tokens for quality recovery." CacheBlend is the ACM EuroSys'25 best paper; its mechanism = recompute only ~10-15% High-KV-Deviation (HKVD) tokens whose KV differs most between naive reuse and full recompute, achieving near full-recompute accuracy with ~3x TTFT reduction. LMCache is maintained by the CacheBlend authors and integrates with vLLM, so source quality matches the claim strength. Quote matches claim verbatim; no contradicting or qualifying evidence found. Current (2025-2026 blog posts, actively maintained repo).

### agent a69e6ec564184f47d — refuted=False, confidence=high
Confirmed by LMCache's own docs and GitHub. LMCache docs have a dedicated "Blending" page (docs.lmcache.ai/kv_cache_optimizations/blending.html) describing CacheBlend: "blends KV caches with minimum computation to preserve cross-attention... reuses cached blocks at any position in the prompt, not just shared prefixes." DeepWiki documents CacheBlend as a feature of the LMCache/lmcache-vllm integration. LMCache explicitly targets vLLM (integration to latest vLLM) and is integrated into mainstream serving stacks (NVIDIA Dynamo, IBM's LLM serving stack, CoreWeave, official vLLM production-stack), indicating active maintenance. The claim that LMCache is the maintained OSS home of CacheBlend targeting vLLM (incl. V1 and multimodal) is well-supported by primary sources. The only minor nuance: some CacheBlend/blending code historically lived in the separate lmcache-vllm repo, but it is part of the LMCache project. No contradicting evidence found.

### agent a31c0ebc3a266f339 — refuted=False, confidence=high
Claim is verbatim-supported by the primary source. CacheBlend abstract (arxiv 2405.16444, also ACM TOCS 10.1145/3790254): "reuses the precomputed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache." Purpose is exactly to handle non-prefix reuse where chunks ignore cross-attention with preceding text (RAG). Mechanism confirmed: full first-layer recompute, second-layer V-value comparison to select top-k discrepant tokens for healing. Peer-reviewed (EuroSys 2025 / ACM TOCS), primary and current. No contradicting source found; supporting works (CacheClip 2510.10129) cite it as the reference method.

### agent a55b921abf1f99f63 — refuted=False, confidence=high
Corroborated by LMCache's own blog and docs: the CacheBlend EuroSys'25 post (blog.lmcache.ai) states "CacheBlend is deployed with vLLM through two open source projects: LMCache and vLLM Production Stack." LMCache docs (docs.lmcache.ai) describe it as an open-source KV cache layer for vLLM (and SGLang), maintained/active (2026 MLSys talk, Oct 2025 arXiv 2510.09665). The multimodal vLLM V1 turbo-boost claim matches LMCache's own release messaging. Minor nuance: LMCache is *a* maintained home (alongside vLLM Production Stack), and it targets SGLang too, not only vLLM — but "specifically vLLM incl. V1 and multimodal" is accurate and current.
counterSource: https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/

### agent a9e0691a38867b5b5 — refuted=False, confidence=high
CacheBlend (arXiv 2405.16444, EuroSys 2025, Microsoft Research + U. Chicago) abstract states verbatim: it "reuses the precomputed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache." The stated motivation is that reused chunks are not always the input prefix, so precomputed KV ignores cross-attention with preceding text (RAG multi-chunk case) — exactly the "non-prefix positions / healing cross-attention" framing in the claim. The supporting quote is a near-exact match to the paper's own wording. No contradicting source found; the work is peer-reviewed (EuroSys 2025 / ACM TOCS 2025) with reported 2.2-3.3x TTFT reduction and 2.8-5x throughput gains, and has public code (LMCache integration). The claim is well-supported, current (2024-2025, actively maintained), and backed by a primary peer-reviewed source matching the claim's strength.

### agent a1ac71ba0411ca695 — refuted=False, confidence=high
CacheBlend (arxiv 2405.16444, EuroSys 2025) explicitly claims and implements position-independent KV reuse. The paper's own framing: reused chunks "are not always the input prefix," making precomputed KV caches "not directly usable since they ignore cross-attention with the preceding texts," and CacheBlend "fuses multiple pre-computed KV caches, regardless of prefix or not, by selectively recomputing the KV cache of a small fraction of tokens." This exactly matches the claim. The quote is not an overreach — non-prefix reuse is the paper's entire raison d'être (vs. prefix-only reuse like PagedAttention/prefix caching). Source is the primary paper (arxiv abs/pdf, v3), and it is backed by a public implementation in LMCache (github, e.g. ForceInjection/AI-fundermentals cache_blend.md documents the LMCache integration). No credible source disputes this; it is a well-known, current (2025) result. Note: reuse is enabled by selective recompute to heal cross-attention, not by making the raw cached KV magically valid at arbitrary positions with zero cost — but the claim as worded ("supports position-independent reuse") is correct.

### agent ac722bdce11ee66a5 — refuted=False, confidence=high
The claim precisely matches the CacheBlend paper (arXiv 2405.16444) abstract and mechanism. The supporting quote is verbatim from the paper: CacheBlend "reuses pre-computed KV caches, regardless of prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache." This is corroborated by independent, credible sources: the LMCache implementation blog (won Best Paper at ACM EuroSys 2025), the ACM Transactions on Computer Systems published version (10.1145/3790254), and LMCache's official Blending docs. The mechanism described (selective per-layer recompute of <15% of tokens, layer-by-layer prefill, healing cross-attention because non-prefix chunks lack cross-attention with preceding text) is exactly as the claim states. No contradicting source found; the method is real, implemented (LMCache/vLLM), current (2025), and peer-reviewed. Not marketing fluff.

### agent a82ac772d2d23d328 — refuted=False, confidence=high
The claim is verbatim-supported by the CacheBlend primary source (arXiv:2405.16444, EuroSys 2025, Microsoft Research). The paper's abstract states it "reuses precomputed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache" to supplement cross-attention. This is precisely CacheBlend's core contribution — non-prefix / position-independent reuse — not an overreach. Source is the primary paper (also ACM/dl.acm.org peer-reviewed publication), current (2024/2025), no contradicting source found. Note: CacheBlend achieves this via selective recompute (healing) rather than pure positional invariance, but the claim as worded ("supports position-independent reuse") is accurate.

### agent ac64fc5fbf3f3d2da — refuted=False, confidence=high
LMCache's official blog (blog.lmcache.ai, 2025-03-31) states CacheBlend "is now released as part of the LMCache and vLLM Production Stack packages," confirming an open-source implementation in the LMCache repo (github.com/LMCache/LMCache). This matches the paper's supporting quote pointing to the same repo. CacheBlend won ACM EuroSys 2025 Best Paper; the LMCache implementation selectively recomputes a small subset of tokens to heal cross-attention when reusing KV at arbitrary positions (TTFT 2.2-3.3x, throughput 2.8-5x). Corroborated by LMCache docs, ACM DL, and 2026 third-party reviews. Claim is current and primary-sourced.

### agent aeb819f0bce30d918 — refuted=False, confidence=high
The claim is verbatim-supported by the primary source, CacheBlend (arXiv 2405.16444, "You Only Prefill Once"/EuroSys 2025, now also ACM TOCS 2026 dl.acm.org/doi/10.1145/3790254). The paper's own abstract states CacheBlend "reuses the pre-computed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache." This is CacheBlend's central design contribution, not an overreach: unlike prefix-caching (which requires exact prefix match), CacheBlend reuses any chunk's precomputed KV at a non-prefix position and heals cross-attention by recomputing only ~10-15% High-KV-Deviation (HKVD) tokens, selected by comparing recomputed vs. precomputed V-values. Multiple independent secondary sources corroborate (Microsoft Research publication page, LMCache blog 2026-04 which ships a production CacheBlend implementation, and follow-on papers CacheClip/RedKnot/ProphetKV that cite CacheBlend precisely as the position-independent selective-recompute baseline). Source quality (primary peer-reviewed EuroSys/TOCS) matches the claim strength; not marketing, not outdated, not disputed. One caveat that does NOT refute: CacheBlend's mechanism is selective recomputation to fix attention, not RoPE position re-anchoring; it addresses cross-attention healing but positions are handled by recompute rather than a RoPE-shift op. The claim as stated (position-independent reuse) is nonetheless correct.

### agent a0edb422f2ccb12e7 — refuted=False, confidence=high
Claim is well-supported and current. (1) The primary arXiv source (2405.16444, CacheBlend by the LMCache/vLLM authors) states "The code is available at: https://github.com/LMCache/LMCache" — the quote directly supports the claim. (2) The official LMCache blog (blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25) independently confirms "CacheBlend is released as part of the LMCache and vLLM Production Stack packages" and describes it as integrated into open-source LMCache. (3) CacheBlend won Best Paper at ACM EuroSys 2025, and a 2026 LMCache follow-up blog ("Accelerating OpenClaw Agents with CacheBlend") shows the implementation is actively maintained. The mechanism (reuse non-prefix KV caches + selectively recompute a small subset of high-attention-deviation tokens to heal cross-attention) matches the research question's description of CacheBlend. No contradicting source found; the LMCache authors ARE the CacheBlend authors, so this is a primary/authoritative source, not third-party marketing.

### agent ab21f22b94805be7c — refuted=False, confidence=high
Primary source arxiv 2510.10129 (CacheClip: Accelerating RAG with Effective KV Cache Reuse) directly confirms the claim. It "integrates four techniques: (1) auxiliary-model-guided token selection for selective KV cache recomputation, (2) shared prefixes to eliminate redundant attention sinks, (3) a sliding-window grouping strategy... (4) CPU-GPU hybrid design." The paper's stated insight is that small auxiliary LLMs share last-layer attention distributions with the primary model, "enabling efficient identification of tokens critical for restoring inter-chunk attention" — i.e. it precomputes per-chunk KV independently and heals cross-chunk attention by recomputing a selected subset of tokens. This is explicitly a CacheBlend-lineage selective-recompute approach (adjustable recomputation ratio), NOT RoPE position-shifting/re-anchoring — no rotation math is used. The characterization ("CacheBlend-style selective recompute, not RoPE re-anchoring") is accurate. Source is the primary arXiv paper, recent (Oct 2025), matching claim strength.

### agent ad0961747bfc237f8 — refuted=False, confidence=high
The claim is well-supported and current. CacheBlend (Best Paper @ ACM EuroSys'25, arXiv 2405.16444) was authored by the same group behind LMCache, and its selective-recompute KV blending is released as part of the LMCache open-source library (github.com/LMCache/LMCache). LMCache's own blog (blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25...) states CacheBlend "is now released as part of the LMCache and vLLM Production Stack packages," and a 2026 follow-up post ("Accelerating OpenClaw Agents with CacheBlend") shows continued maintenance. The mechanism (reuse KV blocks at any position, selectively recompute a small subset of tokens to heal cross-attention) matches the research question's description. The claim understates rather than overstates: CacheBlend is not merely "available" but actively integrated and maintained. Multiple independent corroborating sources (paper's own code link, LMCache docs, TOCS/EuroSys ACM publications). No contradicting evidence found.

### agent a865f7f6d1acf39b6 — refuted=False, confidence=high
Primary source (arXiv 2510.10129, CacheClip) confirms every element of the claim. The paper states CacheClip uses "auxiliary-model-guided token selection for selective KV cache recomputation" to "restore inter-chunk attention capabilities that would otherwise be lost" — i.e., it precomputes per-chunk KV independently (per-chunk precompute) and heals cross-chunk attention by recomputing a selected token subset. It explicitly positions itself against CacheBlend, reporting it "outperform[s] CacheBlend and APE by 16.1 and 12.8 points on NIAH... with recomp%=20%," establishing it as a CacheBlend-style selective-recompute method operating at low recompute fractions. The paper does NOT use RoPE position-shifting/re-anchoring (no mention of RoPE), consistent with the claim's contrast. The additional techniques (shared prefixes, sliding-window grouping) are complementary to, not contradictory of, the core selective-recompute characterization. Source is the primary paper itself and current (Oct 2025). No overreach detected between quote and claim.

### agent a4656c82de798fad9 — refuted=False, confidence=high
Primary source (arXiv 2510.10129, CacheClip) directly supports the claim. Its abstract/paper states CacheClip uses "auxiliary-model-guided token selection for selective KV cache recomputation," with the stated goal of "identifying tokens critical for restoring inter-chunk attention" — i.e. it precomputes per-chunk KV independently, then heals cross-chunk (inter-chunk) attention by recomputing a selected subset of tokens. This is explicitly the CacheBlend paradigm (selective partial recompute), and there is no use of RoPE position-shifting/re-anchoring anywhere in its four techniques (token selection, shared prefixes, sliding-window grouping, CPU-GPU hybrid offload). The claim matches the quote and the full source; nothing contradicts it. Only nuance: 'CacheBlend-style' is the reviewer's characterization, but it is accurate — CacheClip is positioned as an improvement over CacheBlend's selective-recompute approach, using a learned auxiliary model to pick tokens rather than CacheBlend's attention-deviation heuristic.

### agent a27b225c2d17a802e — refuted=False, confidence=high
The claim faithfully paraphrases CacheClip's own abstract/method. The paper (arXiv 2510.10129, "CacheClip: Accelerating RAG with Effective KV Cache Reuse") states: "small auxiliary LLMs exhibit similar last-layer attention distributions to primary LLMs (the target model for generation), enabling efficient identification of tokens critical for restoring inter-chunk attention." Independent sources (arXiv HTML/abstract, Semantic Scholar, ResearchGate) confirm component (1) is "auxiliary-model-guided token selection for selective KV cache recomputation," where the auxiliary model uses query attention scores at the last layer as the token-importance metric and is finetuned to improve selection accuracy. The claim's assertions — (a) small auxiliary LLM, (b) last-layer attention distribution approximating the primary model, (c) used to pick tokens critical for restoring inter-chunk attention — all match. No contradicting source found; source is the primary paper, adequate for the claim's strength. Recent (Oct 2025), not marketing.

### agent a797b4336e01cb2ec — refuted=False, confidence=high
Primary source arxiv 2502.16002v2 directly confirms the claim. It states KVLink "decouple[s] the key-value states from the positional embeddings when storing them, i.e., W{k,v}·x_i" (pre-rotation states, excluding the RoPE rotation matrix R_i), and at inference "apply[s] the global rotary embedding for the KV states of each token appropriate to its location in the full sequence." This is exactly the mechanism claimed: cache pre-RoPE key states, reapply global rotary at correct full-sequence positions so a chunk's KV is valid at a new position. Minor imprecision: RoPE rotates only keys, not values, so "reapplying rotary to KV" is loosely worded — but this mirrors the paper's own wording and does not falsify the claim. Note KVLink's full method also adds trained 'link' tokens to heal cross-attention; the RoPE-decoupling sub-claim is accurate on its own. Source is the primary paper (v2), current (2025), matching claim strength.

### agent a710fabb4e26b5306 — refuted=False, confidence=high
The claim faithfully paraphrases the primary source. CacheClip's abstract (arxiv.org/abs/2510.10129) states its core insight verbatim: "small auxiliary LLMs exhibit similar last-layer attention distributions to primary LLMs (the target model for generation), enabling efficient identification of tokens critical for restoring inter-chunk attention." The paper lists as technique #1 "auxiliary-model-guided token selection for selective KV cache recomputation, where the auxiliary model is finetuned to improve selection accuracy." This matches the claim exactly: a small auxiliary LLM whose last-layer attention approximates the primary model selects which tokens to recompute/heal. No contradicting source found; the claim is not an overreach. One minor nuance: the auxiliary model is finetuned to improve selection accuracy (claim omits this but does not contradict it). Source is the primary paper itself (Oct 2025), current for a fast-moving field, quality appropriate to the descriptive claim.

### agent ad6a7ad2fd431aaa5 — refuted=False, confidence=high
The primary source (arXiv 2510.10129, "CacheClip: Accelerating RAG with Effective KV Cache Reuse") directly supports the claim. The paper's abstract/HTML (https://arxiv.org/html/2510.10129v1) states the key insight verbatim: "small auxiliary LLMs exhibit similar last-layer attention distributions to primary LLMs (the target model for generation), enabling efficient identification of tokens critical for restoring inter-chunk attention." It further lists technique (1) as "auxiliary-model-guided token selection for selective KV cache recomputation." This is exactly what the claim states — an auxiliary LLM's last-layer attention approximates the primary model's and is used to pick which tokens to recompute/heal for inter-chunk attention. The claim is a faithful paraphrase, not an overreach. Source is the primary paper (Oct 2025, current for a fast-moving field), matching the claim's strength. No contradicting evidence found.

### agent afda925ff65809127 — refuted=False, confidence=high
Primary source arxiv.org/html/2502.16002v2 (KVLink) directly confirms the claim. Storage phase: the paper stores KV "without positional embeddings, i.e., W_{k,v}x_i" — exactly the pre-rotation key/value projection W_k*x. Inference phase: "At inference time, the key-value states of all documents are concatenated, and we apply the global rotary embedding for the KV states of each token appropriate to its location in the full sequence." This is verbatim the RoPE-decoupling / re-anchoring mechanism the claim describes. The claim does not overreach: it accurately restates the storage-without-RoPE + reapply-global-RoPE-at-correct-position mechanism. Source is the authors' primary paper (Feb 2025, v2), matching the claim's strength; field-current. Note: KVLink additionally trains link tokens and does light fine-tuning to heal cross-attention, but that does not contradict the narrower positional-decoupling claim under review.

### agent a212c4ad1d6e37166 — refuted=False, confidence=high
Primary source arxiv.org/html/2502.16002v2 (KVLink paper) directly supports the claim verbatim: "the KV cache can be stored without positional embeddings, i.e., W_{k,v}x_i. At inference time, the key-value states of all documents are concatenated, and we apply the global rotary embedding for the KV states of each token appropriate to its location in the full sequence." This exactly matches the claim: cache pre-rotation key states (W_k·x), reapply global RoPE at each token's correct full-sequence position, making a chunk's KV valid at a new position. Source is the authoritative primary paper (not marketing/forum). Minor nuance: KVLink additionally uses trained link tokens to heal cross-attention, but that does not contradict the RoPE-decoupling claim as stated.

### agent a2e030ea9b803dd5a — refuted=False, confidence=high
Primary source arxiv.org/html/2502.16002v2 (KVLink) directly supports the claim. Verbatim paper text: "KVLink appends a small set of trainable tokens between each segment's precomputed KV states before concatenation. The KV representations for these tokens are computed during inference." This matches the claim exactly: trainable link tokens between independently-encoded segments, KV computed at inference, purpose = restore/heal cross-segment attention ("different documents are implicitly connected through these link tokens during inference"). Source is the primary paper itself; claim strength matches. Current (2025 paper). No contradicting evidence found.

### agent a215f45ddee537d5a — refuted=False, confidence=high
Primary source arxiv.org/html/2502.16002v2 (KVLink paper) confirms the mechanism verbatim: "For every document c with length L, we append K (e.g., K=5) link tokens" and "A customized attention map ensures that the link tokens of each document can attend to (i) all tokens (including the link tokens) in the preceding documents and (ii) tokens within the current document." The link tokens are trainable, inserted between/appended to independently-encoded segments, and their KV is computed at inference time (not precomputed) — restoring lost cross-document attention. This exactly matches the claim. Source is the primary paper; claim strength matches. Minor nuance: link tokens are appended per-document rather than strictly "between" segments, but functionally they sit between segments in the concatenated sequence, so the claim is accurate.

### agent af3483b08a826b962 — refuted=False, confidence=high
EPIC (arXiv:2410.15332, ICML 2025 poster, OpenReview qjd3ZUiHRT) explicitly introduces Position-Independent Caching (PIC), enabling KV cache reuse of a text chunk regardless of preceding prefix/position, avoiding full prefill recomputation per new context. The abstract states existing caching requires exact prefix matches (a limitation in RAG/few-shot where immutable docs sit behind varying prefixes), and EPIC's PIC enables modular reuse via AttnLink/LegoLink (selective recompute to recover accuracy) + KVSplit chunking. Reported up to 8x TTFT and 7x throughput gains with negligible accuracy loss. The claim faithfully paraphrases the paper — no overreach. Source is the primary peer-reviewed paper (ICML 2025), current (2024-2025), not marketing. Claim strength matches source strength.

### agent ade8155362feeaa0f — refuted=False, confidence=high
Claim is directly supported by the arXiv 2502.16002v2 quote and corroborated by search summaries. KVLink "appends a small set of trainable tokens between each segment's precomputed KV states before concatenation," and "the KV representations for these tokens are computed during inference." These cross-segment link tokens attend to preceding tokens to restore self/cross-attention across independently encoded segments. The claim accurately paraphrases this without overreach. Minor caveat: KVLink also fine-tunes the model and re-adjusts positional embeddings to global positions — the link tokens are one of several components, not the sole healing mechanism — but the specific claim about link tokens is faithful. Source is the primary paper (NeurIPS 2025 poster), current (2025), not marketing.

### agent a5b991be631cb6d88 — refuted=False, confidence=high
EPIC (arXiv:2410.15332, "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," Hu et al., ICML 2025 / PMLR v267, OpenReview qjd3ZUiHRT) directly supports the claim. The abstract states existing prefix-based caching "requires exact token prefix matches, limiting cache reuse in few-shot learning, multi-document QA, or RAG, where prefixes may vary," and EPIC introduces position-independent caching (PIC) enabling "modular KV cache reuse regardless of token chunk position." Its two mechanisms — AttnLink (attention-sparsity-based selective recompute for accuracy recovery) and KVSplit (semantic chunking) — are precisely a position-independent KV reuse method that avoids re-prefilling per context. The primary source is a peer-reviewed ICML 2025 paper, current (2024-2025), with follow-up work (MEPIC, arXiv:2512.16822) confirming and extending it. Source quality and recency match the claim. The claim is a faithful summary, not an overreach.

### agent a21d86ccd70dcc511 — refuted=False, confidence=high
EPIC (arXiv 2410.15332, "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," Hu et al., published at ICML 2025) explicitly introduces Position-Independent Caching (PIC) to "enable modular reuse of the KV vectors regardless of prefixes." The paper's stated problem is exactly that existing context caching requires exact prefix matches, and PIC enables reuse of immutable chunk KV (e.g., RAG documents) across varying preceding prefixes/positions without full re-prefill. This precisely matches the claim. The claim is well-supported, the source is the primary peer-reviewed paper (not marketing), and it is current (2024-2025, with follow-up MEPIC in 2025). The quote in the review understates but does not misrepresent. No contradicting source found. Note: the mechanism (LegoLink algorithm mitigating spurious attention-sink at each chunk start) is somewhat different from RoPE re-anchoring, but the CLAIM itself only asserts position-independent reuse, which is accurate.

### agent a9fcc1992fef30a82 — refuted=False, confidence=high
Multiple independent sources corroborate the claim. EPIC (arXiv 2410.15332, ICML 2025 poster 43926, OpenReview qjd3ZUiHRT) introduces the LegoLink algorithm for position-independent caching (PIC). Sources confirm the exact mechanism: because a reused KV chunk placed at a non-zero position would otherwise start at position 0, its first few tokens develop a spurious "attention sink" (initial tokens absorbing excessive attention). LegoLink recomputes only the first k tokens of each chunk (except the first chunk), typically k=2 to 32, so those tokens "lose their sink behaviour" and later tokens attend correctly. Recompute cost is O(kN). Reported results: up to 8x TTFT and 7x throughput improvement with <7% accuracy drop. This matches the claim's assertion that LegoLink heals cross-context attention by targeting the attention-sink effect at chunk/document boundaries via minimal-token recomputation. The claim is well-supported by the primary source and secondary reviews; source quality (ICML 2025 peer-reviewed paper) matches claim strength; and it is current (2025). One minor nuance: the sink appears at chunk boundaries generally, not merely "document boundaries," but this is a wording refinement, not a refutation.

### agent ab601a9e83a99d44b — refuted=False, confidence=high
Multiple sources (arXiv 2410.15332, OpenReview qjd3ZUiHRT, ICML 2025 poster/slides) confirm EPIC's core algorithm is LegoLink, which targets the "attention sink" effect. Because each cached chunk's position IDs start at 0, its first few tokens acquire disproportionate sink attention; when spliced at a non-zero position this is wrong. LegoLink recomputes only the first k tokens of each chunk (except the genuinely-first chunk) so those tokens "learn they are not the start" and lose sink behavior, restoring accuracy with O(kN) work instead of O(N^2) full recompute. This matches the claim's three assertions: (1) heals cross-context attention, (2) targets attention sink at chunk/document boundaries, (3) recomputes a minimal token set. Source quality is strong: authors' own paper, ICML 2025 acceptance, independent literature reviews.

### agent a19a4dee067fa2714 — refuted=False, confidence=high
Claim matches primary source exactly. arxiv 2410.15332 (EPIC, ICML 2025 poster #43926) states PIC "enables modular reuse of the KV vectors regardless of prefixes," and abstract frames the problem as existing caching "requires exact prefix matches." Corroborated by independent summaries (emergentmind, themoonlight). EPIC implements PIC via AttnLink/LegoLink (static-sparsity selective recompute) and KVSplit chunking. Reported 8x TTFT / 7x throughput, accuracy drop <7%. Current (Oct 2024, ICML 2025), primary-sourced, no contradicting evidence found.

### agent a313dc9493638a6c5 — refuted=False, confidence=high
The claim accurately describes EPIC's LegoLink. Multiple sources corroborate: LegoLink selects k tokens at each chunk/document boundary for recomputation to correct the "attention sink" phenomenon (initial tokens absorbing excessive attention), enabling adjacent tokens to attend across chunks. It does this with O(kN) work instead of O(N^2), minimizing recompute. Reported results: up to 8x TTFT and 7x throughput improvement with accuracy drops below 7%. Sources: primary paper OpenReview qjd3ZUiHRT / arXiv:2410.15332 (Hu et al., ICML 2025 poster 43926), plus independent literature reviews (themoonlight.io, emergentmind PIC topic). The supporting quote directly matches the mechanism. No contradicting source found; the claim matches source strength (peer-reviewed ICML 2025 paper).

### agent ae8d3f27108637b90 — refuted=False, confidence=high
The claim is directly supported by the primary source (arXiv:2410.15332, "EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models," Hu et al., also published at PMLR proceedings.mlr.press/v267/hu25j.html). The paper explicitly introduces Position-Independent Caching (PIC) to enable "modular reuse of the KV vectors regardless of prefixes," solving the exact-prefix-match limitation of standard prefix caching (relevant for RAG/few-shot where immutable chunks appear after varying prefixes). EPIC's mechanism (LegoLink/AttnLink using static attention sparsity for minimal recompute at chunk boundaries, plus KVSplit chunking) is a concrete system implementation, not just a naming claim. The wording in the claim ("regardless of what prefix precedes it") accurately paraphrases the quote without overreach. Source quality (primary arXiv + peer-reviewed PMLR) matches the claim's modest strength. Note: the claim only asserts EPIC "implements PIC" as a system — it does not assert public code availability, so no stronger burden applies. No contradicting source found.

### agent af70cfa2f0d336c62 — refuted=False, confidence=high
The EPIC paper (arXiv:2410.15332) abstract explicitly states: "Position-Independent Caching (PIC) addresses this issue by enabling modular reuse of the KV vectors regardless of prefixes." The claim paraphrases this faithfully with no overreach — EPIC does implement PIC to reuse a chunk's KV regardless of preceding prefix, avoiding exact-prefix-match requirements. The paper further describes EPIC as the system embodying PIC via LegoLink (static sparsification recompute) and AttnLink mechanisms. Source is the primary paper; claim strength matches. No contradicting source found; this is a well-known 2024/2025 work in the KV-reuse literature (cited alongside CacheBlend). Claim is current and appropriately sourced.

### agent a51c29488420a4caa — refuted=False, confidence=high
Primary source (arXiv 2410.15332) abstract states EPIC "incorporat[es] our new LegoLink algorithm, which mitigates the inappropriate 'attention sink' effect at every document beginning, to maintain accuracy with minimal computation" — a direct match to the claim. Corroborated by OpenReview/ICML 2025 poster and literature reviews: LegoLink dynamically selects a small fixed number k (typically 2-32) of initial tokens per reused chunk to recompute, addressing the attention sink caused by position IDs restarting at zero, at O(kN) cost instead of O(N^2). Only nuance: EPIC's umbrella contribution is 'position-independent caching' (PIC); LegoLink is the specific healing algorithm — but characterizing it as the 'core mechanism' is accurate since it is the novel accuracy-preserving technique. No contradicting source found.

### agent acb0bc742949d5260 — refuted=False, confidence=high
The primary source abstract (arXiv:2410.15332) states EPIC "incorporat[es] our new LegoLink algorithm, which mitigates the inappropriate 'attention sink' effect at every document beginning, to maintain accuracy with minimal computation" — directly matching the claim. Independent summaries (Moonlight review, EmergentMind, OpenReview/ICML 2025 poster) confirm the mechanism: LegoLink dynamically recomputes a small number k of initial tokens per chunk (O(kN), k<<N) to neutralize the attention-sink effect where initial chunk tokens absorb excessive attention. The claim accurately characterizes LegoLink as the healing/recompute mechanism with minimal recomputation. Minor framing nit: EPIC's overarching contribution is Position-Independent Caching (PIC), with LegoLink as its central algorithm — but calling LegoLink the 'core mechanism' for healing cross-attention is fair and consistent with the sources. Source is the peer-reviewed primary paper (ICML 2025), quality sufficient; claim is current (2024-2025).

### agent af6ae83506704d97d — refuted=False, confidence=high
The claim is a faithful paraphrase of the primary source. EPIC's abstract (arxiv 2410.15332) states its "new LegoLink algorithm... mitigates the inappropriate 'attention sink' effect at every document beginning, to maintain accuracy with minimal computation." The paper body confirms the mechanism: "The initial tokens of each immutable chunk disproportionately absorb attention... a phenomenon known as attention sink. LegoLink recomputes initial tokens of each chunk (except the first chunk), allowing these tokens to recognize their non-initial positions and crippling their attention-sink ability." This matches the claim's assertions that (1) LegoLink is EPIC's core mechanism, (2) it heals cross-attention by mitigating the attention-sink at every reused document's beginning, and (3) it uses minimal recomputation. WebSearch found no contradicting or qualifying source. Primary peer-reviewed-track source matches claim strength; not marketing, not outdated.

### agent a9dbf51ecbc18e118 — refuted=False, confidence=high
Verified against llama.cpp source and docs. llama.cpp's KV-cache update/shift mechanism (originally llama_kv_cache_seq_shift, later renamed llama_kv_cache_seq_add) builds a computation graph that applies RoPE adjustments to the cached K tensors to reflect new relative positions without re-processing tokens — exactly the "RoPE-based position shifting of cached keys" the claim describes. It is invoked during "context shift" when the window fills: seq_add with a negative offset slides the window back to make room. Issue #5652 confirms the function's existence and its role in context swapping (the q4_0 crash occurred precisely during this shift). DeepWiki (ggml-org/llama.cpp, qualcomm/llama.cpp) corroborates: "seq_add (shift positions)... updates the RoPE frequencies of existing tokens in the cache." The only minor imprecision is the API name (renamed seq_shift→seq_add), which does not affect the substantive claim. Current: Discussion #24944 (2026) shows the mechanism is still active/maintained, with shifting hard-disabled only on M-RoPE architectures (Qwen3.5/3.6, Gemma4).
counterSource: Issue #5652; DeepWiki ggml-org/llama.cpp Memory Management; Discussion #24944 (renamed to seq_add is the only caveat)

### agent a193477351e46668b — refuted=False, confidence=high
Confirmed accurate. Issue #5652 (ggml-org/llama.cpp) exists exactly as cited: "llama_kv_cache_seq_shift does not work with cache type q4_0," documenting a crash during context swapping. Independent DeepWiki documentation of llama.cpp confirms the mechanism: llama_kv_cache_seq_shift (later renamed seq_add) is used for "context swapping" — removing old tokens to make room for new ones when the context fills — and that llama_kv_cache::update performs "shifting" which "updates the RoPE frequencies of existing tokens in the cache to reflect their new relative positions without re-processing them." Using seq_add with a negative shift slides the window back by rotating cached K RoPE frequencies. This matches every element of the claim: (1) the API name, (2) RoPE-based position shifting of cached keys, (3) invoked during context-window-full swapping. Minor caveat (does not refute): the function was renamed to llama_kv_cache_seq_add/llama_kv_self_seq_add in later versions, and shifting is now hard-disabled for some architectures (Qwen 3.x, Gemma) due to M-RoPE — but the historical/original claim is factually correct and primary-sourced.
counterSource: https://github.com/ggml-org/llama.cpp/issues/5652 ; https://deepwiki.com/qualcomm/llama.cpp/2.2-llama_context-and-memory-management

### agent a5fd0bdec7fa642af — refuted=False, confidence=high
Claim confirmed by primary and secondary sources. llama.cpp's llama_kv_cache_seq_shift (later renamed llama_kv_cache_seq_add / llama_kv_self_seq_add) applies a position delta to cached tokens and, in llama_kv_cache::update, builds a compute graph that re-applies RoPE to the K-cache tensors to reflect their new relative positions without recomputation — exactly the "RoPE-based position shifting during context swapping" the claim states. Issue #5652 (ggml-org/llama.cpp) documents it failing with q4_0 K-cache during context swap, and issue #3825/#4097 discuss its RoPE delta behavior. DeepWiki docs for ggml-org/llama.cpp describe the shift op updating RoPE frequencies of existing cache tokens to slide the window. The supporting quote (crash during context swapping with q4_0) is accurate and the broader RoPE mechanism is independently corroborated. Minor caveat: the API has been renamed in current llama.cpp (seq_shift -> seq_add), and for some architectures (M-RoPE: Qwen3.5/3.6, Gemma 4) KV-shifting is now hard-disabled (Discussion #24944) — but this does not refute the claim that the API exists and does RoPE-based shifting.
counterSource: https://github.com/ggml-org/llama.cpp/discussions/24944 (shifting now disabled for M-RoPE archs) and API rename to seq_add — qualifiers, not refutations

### agent af8e35b0b4fda9fb1 — refuted=False, confidence=high
Primary source llama.cpp issue #5652 states verbatim: "The llama_kv_cache_seq_shift or llama_kv_cache_seq_rm (or all two of them) is broken with cache type q4_0 for K." The reporter runs with --cache-type-k q4_0 and on context-swap/shift the program crashes with repeated "GGML_ASSERT: ggml.c:12646: false" assertion failures. Workaround given is --no-context-shift. This matches the claim on all points: the shift operation touches the raw K tensor (RoPE-based position shift), fails specifically when K is quantized q4_0 (V is f16 by default), and crashes with ggml.c assertions. The claim is not an overreach — it faithfully restates the issue. Source is the authoritative upstream repo (github.com/ggml-org/llama.cpp), matching the claim's strength.

### agent af16ae4090710898b — refuted=False, confidence=high
llama.cpp issue #5652 confirms the claim. The reporter states "llama_kv_cache_seq_shift or llama_kv_cache_seq_rm ... is broken with cache type q4_0 for K." The context-shift operation applies RoPE to the cached K tensor to re-position tokens; with K quantized as q4_0 (V stays f16), the ggml RoPE path hits "GGML_ASSERT: ggml.c:12646: false" repeatedly and aborts with "IOT instruction (core dumped)". Workaround is --no-context-shift. This matches the claim's three elements: (1) op manipulates raw K tensor, (2) fails when K is q4_0 with V f16, (3) crashes via ggml.c assertion. Primary source is the actual GitHub issue cited.

### agent a0eb7ab621616c81a — refuted=False, confidence=high
llama.cpp issue #5652 confirms the claim precisely. Running `./main ... --cache-type-k q4_0 -c 10` (K set to q4_0, V left at its default f16) crashes during context shifting with repeated `GGML_ASSERT: ggml.c:12646: false` followed by a core dump. This occurs exactly at the context-swap/shift step ("context full, swapping..."), i.e. when the RoPE position-shift is applied to the cached K tensor. The workaround is `--no-context-shift`. This matches the claim's three parts: (1) the shift/RoPE op touches the raw K tensor, (2) it fails when K is quantized q4_0 while V stays f16, and (3) it asserts in ggml.c. The supporting quote is faithful to the issue. This is a primary source (the official ggml-org/llama.cpp tracker) for a narrow, technical, verifiable claim, so source quality matches claim strength.

## Part 4 — All source URLs
- https://arxiv.org/abs/2410.15332
- https://arxiv.org/html/2502.16002v1
- https://arxiv.org/html/2502.16002v2
- https://arxiv.org/html/2510.10129v1
- https://arxiv.org/pdf/2405.16444
- https://arxiv.org/pdf/2510.10129
- https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/
- https://deepwiki.com/qualcomm/llama.cpp/2.2-llama_context-and-memory-management
- https://deepwiki.com/spiritbuun/llama-cpp-turboquant-cuda/2.2-context-kv-cache-and-batching
- https://docs.lmcache.ai/kv_cache_optimizations/blending.html
- https://github.com/ForceInjection/AI-fundermentals/blob/main/09_inference_system/kv_cache/02_systems/lmcache/cache_blend.md
- https://github.com/IBM/vllm
- https://github.com/LMCache/LMCache
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp
- https://github.com/ggml-org/llama.cpp/discussions/13606
- https://github.com/ggml-org/llama.cpp/discussions/22354
- https://github.com/ggml-org/llama.cpp/discussions/24944
- https://github.com/ggml-org/llama.cpp/discussions/3581
- https://github.com/ggml-org/llama.cpp/issues/26207
- https://github.com/ggml-org/llama.cpp/issues/4097
- https://github.com/ggml-org/llama.cpp/issues/5652
- https://github.com/ggml-org/llama.cpp/pull/13194
- https://github.com/ggml-org/llama.cpp/pull/4815
- https://github.com/ggml-org/llama.cpp/pull/5104
- https://github.com/lmcache/lmcache
- https://openreview.net/forum?id=qjd3ZUiHRT
