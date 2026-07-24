# Ornith-1.0-9B GGUF-in-SGLang — Research Findings (2026-07-23)

Web-research + primary-source session. **No code changed, no server launched.** This
report re-derives ground truth and interrogates the prior session's README claims.
Every architecture claim below is checked against either the GGUF file's own metadata
(read this session with `gguf.GGUFReader`) or an authoritative upstream source.

---

## 1. What the model ACTUALLY is

### 1a. Ground truth from the GGUF file itself
Read directly from `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (427 tensors, 46 KV pairs):

| Field | Value |
|---|---|
| `general.architecture` | **`qwen35`** |
| `general.base_model.0.repo_url` | `deepreinforce-ai/Ornith-1.0-9B` |
| `qwen35.block_count` | 32 |
| `qwen35.embedding_length` | 4096 |
| `qwen35.feed_forward_length` | 12288 |
| `qwen35.attention.head_count / head_count_kv` | 16 / 4 |
| `qwen35.attention.key_length / value_length` | **256 / 256** |
| `qwen35.rope.dimension_count` | **64** |
| `qwen35.rope.dimension_sections` | **`[11, 11, 10, 0]`** (mrope) |
| `qwen35.rope.freq_base` | 1.0e7 |
| `qwen35.full_attention_interval` | **4** |
| `qwen35.ssm.conv_kernel / state_size` | 4 / 128 |
| `qwen35.ssm.group_count / time_step_rank / inner_size` | 16 / 32 / 4096 |
| tokenizer | `gpt2` BPE, pre=`qwen35`, eos=248046, vocab ≈ 248320 |
| quant | file_type 15 (Q4_K_M base), imatrix (unsloth), quantization_version 2 |

Tensor-type histogram: **225× F32, 101× Q4_K, 43× Q5_K, 34× Q6_K, 24× Q8_0.**
More than half the tensors are full-precision F32 — this is a genuine unsloth
"dynamic" (UD) mixed-precision quant, exactly as the name implies.

### 1b. Verdict on the prior session's architecture claims
- **CONFIRMED** — `head_dim = 256` (`key_length`/`value_length` = 256).
- **CONFIRMED** — `partial_rotary_factor = 0.25` (rope `dimension_count` 64 ÷ head_dim
  256 = 0.25; only 64 of 256 dims are rotated).
- **CONFIRMED** — `mrope_section = [11,11,10]` (`rope.dimension_sections`
  = `[11,11,10,0]`; 11+11+10 = 32, ×2 = 64 = rope dims).
- **CONFIRMED** — hybrid Gated-DeltaNet + full-attention, `full_attention_interval = 4`
  (every 4th layer full attention; ~75% linear / ~25% full). Independently confirmed
  by upstream: *"Qwen3.5 (model_type qwen3_5) inherits from Qwen3-Next; hybrid that
  interleaves full-attention (GQA + RoPE) and Gated-DeltaNet linear-attention layers;
  by default every 4th layer is full attention."* (SGLang Qwen3.5 docs / cookbook.)
- **CONTRADICTED (naming, not substance)** — the README repeatedly calls the model
  "brand-new … not yet in wide production." It is a **derivative of Qwen3.5 /
  Qwen3-Next**, an architecture with an established (if young) upstream and its own
  known bug history (§3). It is not sui generis.
- **OPEN / CONTRADICTED (multimodality)** — README says Ornith is a multimodal VLM
  with a separate vision `mmproj`. The **official `unsloth/Ornith-1.0-9B` card says it
  is text-only** ("strictly a text-based reasoning and coding model without vision").
  MarkTechPost's launch write-up calls the family "vision-language." The `architectures`
  string `Qwen3_5ForConditionalGeneration` is a multimodal-capable *class name*, and the
  GGUF's chat template carries image/video macros — but the GGUF ships **zero** vision
  tensors. Net: the *checkpoint under test is text-only*; whether the base model can do
  vision is disputed across sources and **irrelevant to the gibberish**.

### 1c. The tensor layout — the important part the README gets subtly wrong
GGUF (`qwen35`, llama.cpp naming) stores a **linear-attention** layer (`blk.0`) as:

```
attn_qkv.weight     [4096, 8192]  Q6_K   <- Q+K+V  ONLY  (2048+2048+4096)
attn_gate.weight    [4096, 4096]  Q5_K   <- the Z output gate, SEPARATE
ssm_conv1d.weight   [4, 8192]     F32    <- depthwise causal conv over the 8192 qkv
ssm_beta.weight     [4096, 32]    F32    <- b projection (per-head)
ssm_alpha.weight    [4096, 32]    F32    <- a projection (per-head)
ssm_a               [32]          F32    <- A_log (per-head decay)
ssm_dt.bias         [32]          F32    <- dt_bias (per-head)
ssm_norm.weight     [128]         F32    <- gated RMSNorm over head_dim
ssm_out.weight      [4096, 4096]  Q8_0   <- out_proj
```

Full-attention layer (`blk.3`) is standard: `attn_q [4096,8192]`, `attn_k/v [4096,1024]`,
`attn_q_norm`/`attn_k_norm [256]`, `attn_output [4096,4096]`. (Note `attn_q` width 8192 >
16 heads × 256 = 4096 — consistent with a **query/output gate** on full attention too,
i.e. the README's `attn_output_gate` is plausibly real; flagged, not central.)

Contrast with the **HF/SGLang fused** layout (from the community
`qwen3.5-gated-deltanet-analysis` gist and Qwen3-Next reference):
- `in_proj_qkvz`: 4096 → **12288** = Q(2048)+K(2048)+V(4096)+**Z(4096)** — *Z is fused in*.
- `in_proj_ba`: 4096 → **64** = **β(32) then α(32)** — *b and a are fused, β first*.

**So the GGUF and SGLang disagree on how these projections are packed:**
- GGUF keeps **Z separate** (`attn_gate`, 4096) from QKV (`attn_qkv`, 8192);
  SGLang wants them fused into `in_proj_qkvz` (12288).
- GGUF keeps **b and a separate** (`ssm_beta`, `ssm_alpha`, 32 each); SGLang wants
  them fused into `in_proj_ba` (64), **β before α**.

Bridging this mismatch is precisely what the prior session's patches #7/#8 do by
re-fusing/splitting **block-quantized** tensors at load time. This is the single most
suspicious surface for the gibberish (§4).

---

## 2. Is GGUF serving of this checkpoint in SGLang a supported path?

**No — it is explicitly off the blessed path, on every axis.**

- **Official Ornith card:** the documented serving paths are **vLLM ≥ 0.19.1** and
  **SGLang ≥ 0.5.9** *on the native safetensors* (`--model-path
  deepreinforce-ai/Ornith-1.0-9B`), with Transformers ≥ 5.8.1. GGUF is offered **only**
  for **llama.cpp / Ollama** (`llama-server -hf …-GGUF`). There is **no** documented
  GGUF-in-SGLang recipe for Ornith.
- **Qwen3.5 upstream (SGLang):** Qwen3.5 support requires **SGLang from `main`**, again
  serving safetensors. This spike is pinned to a **release, 0.5.14**, not main.
- **SGLang GGUF loader scope:** SGLang's GGUF path is per-architecture and narrow.
  There are open issues even for *plain* dense Qwen (`#6281` "GGUF model with
  architecture qwen3 is not supported yet"; tokenizer issue `#3427`; `#7404` "Cannot
  use gguf"). The `qwen35`/`qwen3_next` **hybrid GDN** architecture is **not** in
  SGLang 0.5.14's GGUF allowlist — which is exactly why the prior session had to
  monkey-patch the arch redirect, the tensor-name map, the packed loader, and the
  hybrid-config detection to get it to load at all.
- **SGLang docs, quantization page:** warns directly that *"due to vLLM's layer fusion
  (e.g. QKV fusion), applying different bit-widths to components within the same fused
  layer can lead to compatibility issues."* That is this checkpoint exactly: `attn_qkv`
  Q6_K fused with a separate F32 `ssm_beta`/`ssm_alpha` and Q5_K `attn_gate`, all being
  re-merged into SGLang's single fused params.

**Conclusion:** GGUF-in-SGLang-0.5.14 for this architecture is an unsupported,
hand-patched path. The prior README's framing that the nine patches merely "wire up
gaps" that are "safe and reusable once upstream fixes land" **overstates the case** —
upstream's own answer is "serve the safetensors, or use llama.cpp for GGUF."

### Is llama.cpp's own qwen35/qwen3-next GGUF support known-buggy?
**Yes, historically — and this matters for interpreting "gibberish."**
- Unsloth's Qwen3-Coder-Next docs: *"Feb 4: llama.cpp fixed a bug correcting the
  calculation for vectorized `key_gdiff`. This fixes previous looping and output
  issues,"* with instructions to **re-download the GGUFs and update llama.cpp**. That is
  a Gated-DeltaNet numerical bug in the reference implementation itself.
- Additional recent qwen3.5 GGUF issues: gibberish over RPC+Vulkan (`#22235`, closed
  not-planned, distributed-only), hard crashes (`#19906`, `#19860`), server crash on
  Strix Halo (`#19355`).

**Important caveat for our case:** the `key_gdiff` fix is a fix to *llama.cpp's own GDN
kernel*. **SGLang does not use llama.cpp's kernels** — it only reads the GGUF's weights
and runs its *own* GDN implementation. So that specific fix does **not** clear SGLang's
output. It does, however, establish that (a) GDN math is fiddly and error-prone across
implementations, and (b) if this **GGUF file itself was produced before the fix or with
a mismatched converter**, the stored weights (e.g. sign/scale of `ssm_a`/`dt_bias`)
could already encode a convention SGLang doesn't expect.

---

## 3. Known issues matching the symptom (loads clean → gibberish)

The symptom class — **weights all bind, forward runs, output is garbled multilingual
tokens at temp 0** — is a textbook "silently-wrong-weights/wrong-math" signature, not a
loader failure. Matching precedents found:
- **SGLang docs' own fused-mixed-bitwidth warning** (above) — the closest documented
  match: different bit-widths inside one fused layer → "compatibility issues."
- **SGLang GGUF tokenizer bug `#3427`** — GGUF load produced *different token IDs* than
  llama.cpp for the same file. Gibberish at the *decode/detokenize* layer can look
  identical to numerical gibberish. Our file uses a `gpt2`-class BPE with a `qwen35`
  pre-tokenizer and a 248k vocab; worth ruling out that SGLang's GGUF tokenizer
  reconstruction matches llama.cpp's.
- **Generic GGUF community reports** ("Gibberish output from GGUF" HF threads) — the
  consistent community read is *"looks like a broken quant / wrong dequant,"* usually
  traced to a fused-projection or a bit-width/converter mismatch, not the sampler.
- **llama.cpp qwen3.5 GDN `key_gdiff` bug** — same *architecture family*, same *looping/
  garbled* symptom, root-caused to a GDN vector-math error.

---

## 4. Ranked root-cause hypotheses (evidence-backed)

**H1 — Re-fusion of the split, block-quantized GDN projections is numerically wrong
(HIGHEST).** GGUF splits what SGLang fuses (Z out of `in_proj_qkvz`; b/a out of
`in_proj_ba`). Patches #7/#8 re-assemble these. Two concrete failure modes:
  - **b/a ordering:** SGLang's `in_proj_ba` expects **β then α**; the GGUF names are
    `ssm_beta` and `ssm_alpha`. If the patch concatenated them α-then-β (or mapped
    `ssm_a`↔`ssm_alpha` by name similarity), the decay gate
    `g = exp(-exp(A_log)·softplus(a + dt_bias))` is fed the wrong tensor → total garbage
    with clean load. Highest-value single thing to verify next session.
  - **Z placement / QKV interleave:** re-inserting `attn_gate` as the Z slice of
    `in_proj_qkvz`, and the exact Q/K/V (2048/2048/4096) slice boundaries, must match
    SGLang's reshape into (16 k-heads, 32 v-heads, head_dim 128). Off-by-a-block on a
    Q6_K super-block boundary corrupts silently.
  Evidence: SGLang's own mixed-bitwidth-fusion warning; the confirmed layout mismatch in
  §1c; this is the least-tested, most-custom code in the spike.

**H2 — mrope / partial-rotary mismatch on full-attention layers (HIGH).** `head_dim=256`
but only 64 dims rotated (`partial_rotary_factor 0.25`), `mrope_section [11,11,10]`,
`freq_base 1e7`. If SGLang applies rope to the wrong 64 sub-dims, or treats it as full
rotary, or mis-sections mrope, full-attention layers (every 4th) inject wrong positions
→ progressively garbled decode. Config-property grafting (patch #9) is where these
values are hand-plumbed; a wrong default there wouldn't crash, just corrupt.

**H3 — quantized-embedding / lm_head dequant (MEDIUM).** `token_embd` Q4_K and `output`
Q6_K are large fused vocab tensors the Qwen3.5 decoder doesn't natively build a quant
param for (hence the embed-rebuild hack). A wrong dequant here garbles both input
representation and final logits. Plausible but less likely to produce *fluent-looking*
multilingual gibberish than H1/H2.

**H4 — GGUF↔SGLang convention/sign mismatch on GDN scalars (`ssm_a`/`dt_bias`)
(MEDIUM-LOW).** Given llama.cpp's own `key_gdiff` history, the stored convention for the
decay scalars may differ from what SGLang's GDN kernel assumes (log-space, sign of A).

**H5 — SGLang GGUF tokenizer reconstruction (LOW-MEDIUM, easy to exclude).** Per `#3427`,
detokenization from a GGUF-embedded 248k BPE could mis-map IDs. Cheap to rule out by
comparing token IDs / logprobs against llama.cpp on the identical prompt.

**H6 — the whole path is unsupported and not worth hardening (META).** Upstream's
blessed answer is safetensors-in-SGLang or GGUF-in-llama.cpp. Even if H1–H5 are fixed,
this remains an unmaintained fork of the loader.

---

## 4b. DECISIVE EVIDENCE — a reference converter names the exact missing transforms

Cloned two community repos to `.scratch/vendor/`:
- **`li-yifei/gguf-to-nvfp4`** — a working GGUF→HF→NVFP4 pipeline for Qwen3.5 hybrid
  GDN (targets `huihui-ai/Huihui-Qwen3.5-27B`). Its `scripts/step1_convert.py` +
  `README.md` §"Qwen3.5 GGUF → HF Conversion Pitfalls" are a **reference reshape/remap
  oracle**. Its README states verbatim: *"Getting any of these wrong produces a model
  that loads cleanly but generates garbage"* — our exact symptom.
- **`Thireus/GGUF-Tool-Suite`** — NOT relevant: a per-tensor quant-mixing/PPL-recipe
  tool for GGUF (`quant_assign.py`, etc.). Does no tensor remapping/reshaping and has no
  qwen35/GDN support. Useful only if we later want to re-quant a GGUF blend; ignore for
  the gibberish.

The converter reveals **THREE numerical transforms the GGUF `qwen35` weights require to
match the HF/SGLang Qwen3.5 model — none of which the prior session's nine patches
perform** (its patches only *bind* tensors + fix config detection; they never touch
tensor *values* or *head order*). This reframes the root cause:

1. **RMSNorm +1.0 offset (`step1_convert.py:120-125`).** GGUF stores RMSNorm weights as
   `1 + learned_param`; HF stores `learned_param`. Fix: `w = w - 1.0` on
   `attn_norm`, `post_attention_norm`, `output_norm`, `attn_q_norm`, `attn_k_norm`
   — **but NOT `ssm_norm`** (that's a GroupNorm). README: without this *"the first few
   tokens may look correct but output rapidly becomes incoherent."* (Ornith is derived
   from Gemma 4 + Qwen 3.5, and Gemma-lineage norms bake the +1 — consistent.)
   *Caveat:* whether this fix is needed in SGLang depends on whether SGLang's own
   `Qwen3_5` RMSNorm already adds 1 in its forward (Gemma-style). Direction must be
   checked against SGLang's norm impl — but SGLang's generic GGUF loader almost
   certainly does **not** special-case it.
2. **A_log domain conversion (`step1_convert.py:127-129`).** GGUF stores the SSM decay
   as the *materialized* value `A = -exp(A_log)`; HF expects `A_log` (log-space). Fix:
   `A_log = (-ssm_a).log()`. If SGLang binds the raw GGUF `ssm_a` into a param its GDN
   kernel treats as log-space, **every linear-attention layer's decay gate is wrong** →
   garbage. (This is my prior H4 — now CONFIRMED as a mandatory transform, not a maybe.)
3. **Value-head (repeat, groups) → (groups, repeat) unpermutation
   (`step1_convert.py:131-158`).** llama.cpp stores the GQA-repeated V-heads grouped as
   `(repeat, n_kv_groups)`; HF expects `(n_kv_groups, repeat)`. For the **27B** that's
   `(3,16)→(16,3)` over 48 V-heads; **for our 9B it is `(2,16)→(16,2)` over 32 V-heads**
   (GGUF `head_count_kv`-derived repeat = 32/16 = 2). Applied to *A_log, dt_bias,
   in_proj_a, in_proj_b, the V-section of in_proj_qkv, in_proj_z, out_proj (columns), and
   the V-section of conv1d*. This is the classic "llama.cpp permutes heads" bug; skip it
   and the linear-attention heads are scrambled → garbage. README calls it *"the most
   subtle bug."*

Also note the converter's **actual HF param names keep the projections SPLIT**
(`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` — separate), matching the GGUF's
own split. This partially **CONTRADICTS the README's premise** that SGLang needs them
re-fused into `in_proj_qkvz`(12288)/`in_proj_ba`(64): the *checkpoint* layout is split;
any fusion is an internal SGLang detail. So H1's "wrong β/α fusion order" is *less*
likely than first thought, and **H2b (these three missing value/order transforms)
supersedes it as the leading cause.** The `.scratch/vendor/gguf-to-nvfp4/scripts/
step1_convert.py` `gguf_tensor_to_torch()` + `build_llm_mapping()` are the ground-truth
mapping to diff the SGLang loader against.

### Revised hypothesis ranking
1. **Missing A_log domain conversion** (transform #2) — mandatory, unconditional, absent.
2. **Missing value-head (2,16)→(16,2) unpermutation** (transform #3) — mandatory, absent.
3. **Missing / wrong-direction RMSNorm +1 offset** (transform #1) — mandatory unless
   SGLang's norm forward already compensates; must verify direction.
4. mrope/partial-rotary on full-attn layers (prior H2) — still possible, now secondary.
5. β/α fusion order (prior H1) — demoted; the reference keeps them split.

## 5. How the community validates such a setup (for the next session)

- **llama.cpp as ground-truth reference.** Run the *same* GGUF via
  `llama-server -hf deepreinforce-ai/Ornith-1.0-9B-GGUF` (the officially-supported GGUF
  runtime) on the identical prompt at temp 0, and diff tokens/logprobs against SGLang.
  This isolates *loader/dequant/GDN-math* bugs (SGLang-only) from *file* bugs (both
  wrong ⇒ re-download a post-Feb-4 GGUF and update llama.cpp first).
- **bf16 safetensors as the numerical gold standard.** Serve `unsloth/Ornith-1.0-9B`
  (~19 GB) natively in SGLang-main/vLLM and compare per-layer hidden states; this is the
  documented, working path and sidesteps all GGUF split/mixed-precision dequant.
- **Layer-by-layer bisection** against either reference, starting at the *first*
  linear-attention layer's `in_proj_ba`/`in_proj_qkvz` outputs (H1), then the first
  full-attention layer's rope (H2), then embeddings/lm_head (H3).

---

## 6. Scorecard on the prior session's assumptions

| Prior-session claim | Verdict |
|---|---|
| model_type `qwen3_5`, hybrid GDN + full-attn, interval 4 | **CONFIRMED** (GGUF meta + upstream) |
| head_dim 256, partial_rotary 0.25, mrope [11,11,10] | **CONFIRMED** (GGUF meta) |
| Ornith is "brand-new, not in production anywhere" | **CONTRADICTED** — it's Qwen3.5/Qwen3-Next-derived, with an upstream + bug history |
| Ornith checkpoint is a multimodal VLM w/ vision tower | **OPEN/CONTRADICTED** — official card says text-only; GGUF has no vision tensors |
| `Qwen3_5ForCausalLM` in SGLang is "not a servable class" → needed a custom wrapper | **PLAUSIBLE but OPEN** — not independently verified; may be an artifact of the unsupported GGUF path rather than a real SGLang gap. Re-check against SGLang-main. |
| The 9 patches are "narrow, correct, semantics-preserving" | **CONTRADICTED as stated** — at least the fused-projection re-assembly (#7/#8) is a prime numerical-corruption suspect; "correct" is unproven and the gibberish is evidence against it |
| GGUF-in-SGLang is a "sensible, reusable" path pending upstream fixes | **CONTRADICTED** — upstream blesses safetensors-in-SGLang / GGUF-in-llama.cpp; no GGUF-in-SGLang recipe exists for this arch |
| Gibberish is "numerical, not templating" | **CONFIRMED-ish** — temp-0 garble from raw `/v1/completions` rules out chat templating; but SGLang GGUF *tokenizer* reconstruction (H5) is still numerically-adjacent and unexcluded |
| bf16 safetensors is the lower-risk route | **CONFIRMED** — this is literally upstream's documented path |

---

## Sources
- GGUF metadata/tensors: read this session from
  `Ornith-1.0-9B-UD-Q4_K_XL.gguf` via `gguf.GGUFReader`.
- Official model card: https://huggingface.co/unsloth/Ornith-1.0-9B ,
  https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B (base),
  https://huggingface.co/OsaurusAI/Ornith-1.0-9B-MXFP8
- Launch/architecture write-ups: https://www.marktechpost.com/2026/06/25/deepreinforce-releases-ornith-1-0-an-open-source-coding-model-family-that-learns-its-own-rl-scaffolds/ ,
  https://deep-reinforce.com/ornith_1_0.html
- Qwen3.5 / GDN architecture & SGLang support: https://docs.sglang.io/basic_usage/qwen3_5.html ,
  https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.5 ,
  https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15
- llama.cpp GDN `key_gdiff` fix + re-download guidance: https://unsloth.ai/docs/models/qwen3-coder-next
- llama.cpp qwen3.5 GGUF bugs: https://github.com/ggml-org/llama.cpp/issues/22235 ,
  https://github.com/ggml-org/llama.cpp/issues/19906 ,
  https://github.com/ggml-org/llama.cpp/issues/19860 ,
  https://github.com/ggml-org/llama.cpp/issues/19355
- SGLang GGUF support/limitations: https://docs.sglang.io/docs/advanced_features/quantization ,
  https://github.com/sgl-project/sglang/issues/6281 ,
  https://github.com/sgl-project/sglang/issues/3427 ,
  https://github.com/sgl-project/sglang/issues/7404

---

## 5. IMPLEMENTATION SESSION (2026-07-23) — 3 transforms landed; residual bug localized

The three GGUF→HF weight transforms from §4b were implemented, unit-tested, and
run live. **They are correct and necessary but not sufficient**: a dominant
residual bug survives them. Full detail in
`.scratch/projects/16-ornith-gguf-loader-transforms/`.

### 5.1 What was built
- `ornith_gdn_transforms.py` — pure tensor math (no SGLang imports): RMSNorm −1,
  A_log `log(−x)` domain, value-head `(r=2,g=16)→(g,r)` unpermutation. 16 unit
  tests (`test_ornith_gdn_transforms.py`) assert the perm is a bijection, the
  index path equals the reference converter's `reshape/permute`, `log(−x)`
  inverts `−exp(y)`, and every dispatch branch. All green under the devenv uv.
- `ornith_text_model.py::load_weights` — dispatch applies `transform_quant_rows`
  to `.qweight` tensors and `transform_plain` to F32/bare tensors, before each
  `weight_loader`. Gated by `ORNITH_FIX_NORM/ALOG/PERM` (default on).
- **out_proj used Option B, not Option A.** SGLang's
  `gguf_quant_weights_iterator` **never dequantizes** (always yields `.qweight`),
  so an unquantized `out_proj` (Option A) would never bind the quantized
  `ssm_out`. Instead the packed Q8_0 bytes are column-permuted at head
  granularity: `ssm_out` is `[4096, 4352]` = 32 heads × 136 bytes = 4 whole Q8_0
  blocks/head, so reordering per-head 136-byte segments never splits a block.
  Zero extra VRAM; no `ornith_gguf_compat`/`sitecustomize` change.

### 5.2 Live evidence
- **Oracle (llama.cpp, `llama-cpp-python` CPU on the *same* GGUF):** greedy
  `temperature=0` "The capital of France is" → **`" Paris.\nThe capital of France
  is Paris."`** The GGUF is valid and post-`key_gdiff` (ISSUES W4 **resolved**).
- **SGLang spike, all 3 transforms ON:** still gibberish
  (`'阿克老城 CheneyD…'`); first-token top-5 logprobs ~**−5.2** (near-uniform).
- **SGLang spike, all 3 transforms OFF (baseline):** also gibberish, *different*
  (`'Wyerceрок…'`), first-token top1 ~−1.86. → Both toggle states are garbage, so
  **the dominant bug is independent of these 3 transforms**; no toggle
  combination can reach coherence.
- **Transforms verified applied** (per-load counter): input_layernorm ×32,
  post_attention_layernorm ×32, model.norm ×1, q_norm/k_norm ×8 each (the 8
  full-attn layers), A_log/dt_bias/in_proj_a/in_proj_b/conv1d/in_proj_qkv/
  in_proj_z/out_proj ×24 each (the 24 linear-attn layers). Exactly right.
- **Per-decoder-layer hidden-state stats (all 32 layers):** healthy — no NaN, no
  explosion, std grows smoothly 0.28 → 1.7, absmax ≤ ~102 at layer 31. **No single
  catastrophically-broken layer.** The error is *semantic* (healthy-magnitude but
  wrong-content vectors), not numeric.

### 5.3 Suspects ruled out this session
- **mrope / partial-rotary (the §7 prime suspect) — RULED OUT as misconfig.** The
  resolved config's `rope_parameters` **does** contain `mrope_section=[11,11,10]`
  + `mrope_interleaved=True` + `partial_rotary_factor=0.25`, so SGLang's rope
  factory (`layers/rotary_embedding/factory.py`, `default`+`mrope_section` branch)
  builds the **interleaved `MRotaryEmbedding`**, whose `forward_cuda` handles the
  text 1-D positions path. rope_theta 1e7 is read correctly via `get_rope_config`
  (v5 `rope_parameters` path). So the full-attn rope convention is honored.
- **GGML dequant — likely fine.** All quantized layers show healthy stats; a
  broken dequant kernel would corrupt every layer's magnitude, which is not seen.

### 5.4 Remaining frontier (needs the §6.4 bf16 reference)
The residual is a subtle, pervasive semantic error in the layer stack that keeps
activations healthy but scrambles meaning — consistent with an SGLang
`Qwen3_5` execution detail for this checkpoint (e.g. the fused `in_proj_qkvz`
split/head-interleave inside the GDN Triton kernel, or the q/k/v/z ordering the
delta-rule expects) rather than any of the 3 weight transforms. Pinpointing it
requires the guide §6.4 **per-layer diff against the bf16 `unsloth/Ornith-1.0-9B`
reference** (the standalone `llama.cpp` oracle only exposes the final hidden
state, not per-layer). That reference run needs the 19 GB bf16 model on a bigger
box / CPU-offload and is beyond this 12 GB spike. The diagnostic hooks
(`ORNITH_DEBUG_LAYERS=1` → per-layer stats + transform-apply counts) are left in
place, gated, to seed that bisection.
