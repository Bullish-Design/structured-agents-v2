# Nanbeige 4.5 architecture preview — LoopSplit, mHC depth attention, n-gram embeddings

Date: 2026-07-31. Source: Nanbeige/Nanbeige4.2-3B HF page + `modeling_nanbeige.py`
(fetched 2026-07-31, 2671 lines) + released `config.json`. Technical report:
https://arxiv.org/abs/2607.22083

## Headline

The 4.2-3B HF page states the modeling code "also includes our latest
architectural improvements, including **LoopSplit**, **mHC with depth
attention**, and **concatenated n-gram embeddings**. These features have been
incorporated into Nanbeige4.5, whose training is underway for release later in
2026."

**Verified: all three are disabled in the released 4.2-3B.** Its `config.json`
has only `num_loops: 2` and empty `loop_loss_weights`; the features exist in the
modeling code (gated behind config flags) but are staged for 4.5. The served
4.2 GGUF graph is therefore exactly what the project memory already established:
a plain looped llama-style stack (+ optional loop-boundary norm). LoopSplit /
mHC / n-gram are **HF-modeling-only today** — none of them are in the llama.cpp
graph.

## 1. LoopSplit — loop the middle, not the stack

`_get_double_loop_split_layer_order` (modeling_nanbeige.py): for N layers and a
middle block of M layers:
- `(N−M)/2` **prefix** layers run once,
- M **middle** layers repeat `(N+M)/M` times (each pass carries a `repeat_idx`),
- `(N−M)/2` **suffix** layers run once.

Hourglass: encode → think in a loop → decode. Each middle pass knows which
depth-pass it is on (`repeat_idx`). Concentrates depth where it matters and
gives an explicit, stable per-pass index.

## 2. mHC with depth attention — parallel residual streams + cross-depth KV

`NanbeigeHyperConnectionModule`: **Hyper-Connection** maintains
`num_residual_streams` parallel residual streams; each layer mixes them with a
learned mapping (`mapping_proj` producing an n×n mixing matrix) plus `alpha`
gates, **initialized near-identity** (bias init ±20 so each stream feeds itself
first). mHC = the "multi-head" full-mixing variant.

**Depth attention:** `_apply_loop_shared_kv` + `loop_share_kv_cache` +
`loop_share_kv_repeat_idx` — on loop pass 0 a layer saves its KV; later passes'
attention can attend to that saved KV again. The model can **read its own output
from earlier depth passes** — attention *across depth*, not just across
sequence.

## 3. Concatenated n-gram embeddings — a parametric phrasebook

- Hashed n-gram embedding tables of growing sizes (`_ngram_embedding_vocab_sizes`:
  m, m+3, m+5, …; prime-hashed via `_next_prime_after` to spread collisions).
- "Concatenated": embeddings across the n-gram orders are stitched together.
- `NanbeigeNgramLayerFusion` fuses them into the hidden state **at every layer**
  via a learned gate (normalized dot-product similarity → sigmoid → gated add).
- Fixed-size parametric memory that does not grow with context — a third
  knowledge axis alongside distillation-in-weights and KV prefixes.

## Implications for Project 19

1. **Depth attention is P1 arriving as a trained-in feature.** Project 19's
   hardest wall is cross-context KV validity (CacheBlend problem, §6.7). mHC
   depth attention is the *cross-depth* version — KV computed in pass 0 read in
   pass 1 — trained into the model, not spliced at runtime. If 4.5 works, it
   changes the calculus: the model reads its own prior passes natively instead
   of the harness staging KV objects. **Watch 4.5.**
2. **LoopSplit gives per-depth structure for the hats for free.** §6.5.4 hats
   are per-depth-pass LoRAs indexed by loop-step. LoopSplit makes this cheaper:
   hats only need to cover the **middle block**, and `repeat_idx` is an explicit
   per-pass index — a natural P2 pool index. Hourglass shape maps onto
   frame→critique→commit-style modes.
3. **n-gram embeddings are a third knowledge axis.** Could absorb the
   "common phrases / verbatim boilerplate" cases where distillation degrades
   (§6.5.3).
4. **The llama.cpp port cost rises for 4.5.** A 4.5 base would need real graph
   work: loop-split layer ordering, multi-stream residual mixing, shared-KV
   depth attention, n-gram tables. The 4.2 port plan (plain loop) stays cheap.
5. Timeline per HF page: 4.5 training underway, release later 2026. The
   looped-MLA question (RESEARCH-D) and this preview are independent bets on
   the same future: a base that handles cross-depth context natively.

## Sources

- HF page: https://huggingface.co/Nanbeige/Nanbeige4.2-3B (README quote)
- modeling_nanbeige.py: fetched from that repo (functions cited above)
- config.json: fetched from that repo (feature flags absent/disabled)
- Technical report: https://arxiv.org/abs/2607.22083
