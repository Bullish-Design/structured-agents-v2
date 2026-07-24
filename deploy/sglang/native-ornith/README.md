# Isolated SGLang Ornith-1.0-9B GGUF spike

This is an intentionally disabled-by-default compatibility test for the
`unsloth/Ornith-1.0-9B-GGUF` `Ornith-1.0-9B-UD-Q4_K_XL.gguf` quantization.
Ornith is `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) — a
brand-new, multimodal, hybrid Gated-DeltaNet linear-attention + periodic
full-attention architecture, not yet in wide production use anywhere. When
enabled, it binds only `127.0.0.1:8003`, uses GPU 1, has one request slot,
uses a dedicated cache, and forces Hugging Face and Transformers offline.
This spike is text-only: the GGUF's vision tower ships as a separate
`mmproj` file which `serve.sh` explicitly refuses to load.

`devenv.nix` and `pyproject.toml` pin the same standalone Python 3.12 /
SGLang 0.5.14 environment style as the sibling `deploy/sglang/native/`
(Gemma 4) spike. `run.sh` enters that environment. `serve.sh` requires
pre-existing local target and tokenizer paths, uses the SGLang
`--load-format gguf` and `--tokenizer-path` interface, and logs the fully
resolved invocation.

## Config resolution

Unlike Gemma 4, there is **no live-GGUF-derive path at all** for Ornith:
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` raises
`ValueError: GGUF model with architecture qwen35 is not supported yet.`
before SGLang can build any config — its GGUF metadata parser's
architecture allowlist has no `qwen35` entry. `resolve_ornith_gguf_config.py`
instead publishes a static `config.json` built from the real,
already-published `config.json` in the non-GGUF `unsloth/Ornith-1.0-9B`
repo:

```
python resolve_ornith_gguf_config.py /path/to/real/config/dir /path/to/resolved-config-dir
```

then set `SGLANG_GGUF_CONFIG_PATH` to the `config.json` it writes before
running `serve.sh`/`run.sh`. `serve.sh` refuses to start without it.

`sitecustomize.py` / `ornith_gguf_compat.py` / `ornith_text_model.py`
install a set of narrow, well-understood compatibility patches, none of
which change model semantics beyond what a genuinely-supported checkpoint
would already have. The first five get SGLang far enough to *build* the
config; the rest were needed to make the weights actually bind and the
hybrid Gated-DeltaNet runtime actually execute (see "Runtime evidence"):

1. **`install_static_config_redirect`** — redirects every
   `AutoConfig.from_pretrained(<gguf path>, ...)` call (there are two
   independent call sites in SGLang 0.5.14: the main engine config and
   `multimodal_processor.get_processor`) to the static resolved config
   instead of the broken GGUF-metadata parser.
2. **`install_causal_lm_registry_entry`** — `sglang.srt.utils.hf_transformers
   .config.get_config` unconditionally rewrites a GGUF checkpoint's
   `architectures` to Transformers' `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
   [model_type]` entry (`Qwen3_5ForCausalLM` for `qwen3_5`/`qwen3_5_text`).
   But **despite its name, `sglang.srt.models.qwen3_5.Qwen3_5ForCausalLM`
   is not a servable causal LM** — it is the flat decoder stack
   (`embed_tokens` / `layers` / `norm`); its `params_dict` has no `model.`
   prefix and no `lm_head`, and its `forward` returns raw hidden states, not
   logits. The only complete top-level Qwen3.5 classes are the multimodal
   `…ForConditionalGeneration` wrappers, which build a vision tower this
   text-only spike excludes. So this patch registers, under the name GGUF
   rewrites to, the local **`OrnithTextForCausalLM`** (`ornith_text_model.py`)
   — a minimal text-only wrapper that owns the flat decoder as `self.model`
   plus a `ParallelLMHead` and a `LogitsProcessor`, giving a `model.`-prefixed
   param tree that matches the GGUF-derived tensor names and an actual logits
   path, with no vision stack. **This corrects an earlier revision of this
   spike, which registered the flat decoder directly and consequently failed
   to bind every one of the 629 GGUF tensors.**
3. **`install_text_only_mm_processor_skip`** — the post-GGUF-rewrite
   `Qwen3_5ForCausalLM` architecture has no matching entry in SGLang's
   multimodal processor mapping (only the `ConditionalGeneration` wrapper
   does), and there is no CLI flag to force multimodal off. Since this
   spike's GGUF never has vision weights to load in the first place, the
   processor lookup for this one known architecture is a deliberate no-op.
4. **`install_dense_expert_location_skip`** — `Qwen3_5ForCausalLM
   .get_model_config_for_expert_location` is shared by the dense and MoE
   subclasses and always returns a `ModelConfigForExpertLocation` even for
   a dense model with `num_experts=0`, which crashes
   `eplb.expert_location._pad_nested_array` (`max()` over an empty
   sequence). Returning `None` for `num_experts=0` is what a model class
   without the method would already produce. (`OrnithTextForCausalLM`
   independently returns `None` from this classmethod for the same reason.)
5. **`install_gguf_arch_name_translation`** — `GGUFModelLoader
   ._get_gguf_weights_map` looks up the tensor-name map by searching the
   standalone `gguf` PyPI package's `MODEL_ARCH_NAMES` for a value equal to
   `config.model_type`, with existing hardcoded translations for other
   naming mismatches (`cohere` -> `command-r`, `qwen3_moe` -> `qwen3moe`).
   The `gguf` package already has a full `qwen35` tensor-name map
   (confirmed to include the Gated-DeltaNet tensors); it was simply never
   wired up because SGLang's model_type string (`qwen3_5`/`qwen3_5_text`)
   doesn't match the GGUF library's arch key (`qwen35`).
6. **`install_gdn_ssm_tensor_map_fix`** — `_get_gguf_weights_map` builds its
   name map by splitting the final dotted component off each HF parameter as
   a `.weight`/`.bias` suffix. That breaks for the two bare `nn.Parameter`
   GDN tensors: `linear_attn.A_log` (whose gguf name the map *does* know, but
   the `rsplit` mangles the lookup key) and `linear_attn.dt_bias` (absent
   from the `gguf` package's `qwen35` map entirely, though the file ships it
   as `blk.N.ssm_dt.bias`). Without this, 48 of 427 tensors (24 layers ×
   {A_log, dt_bias}) are silently dropped.
7. **`install_gdn_ba_unquantized`** — this `UD-Q4_K_XL` GGUF keeps the tiny
   `in_proj_b`/`in_proj_a` GDN projections in **F32** (unsloth's dynamic
   quant leaves small, precision-sensitive tensors full width), so they
   arrive as plain `.weight`. SGLang builds the merged `in_proj_ba` with the
   layer quant config anyway, so its param is `qweight` and the F32 sources
   miss. Building `in_proj_ba` unquantized matches the checkpoint.
8. **`install_gdn_packed_gguf_loader_binding`** — the GDN merges the split
   `in_proj_qkv`/`in_proj_z` into one `MergedColumnParallelLinear` and loads
   `in_proj_qkv` with a *tuple* shard id `(0,1,2)`. The base loader rejects
   tuple shard ids; the model works around this by rebinding a packed loader
   — but only onto `.weight`/`*_scale` params, not the GGUF `qweight`/
   `qweight_type` this checkpoint actually uses. This extends the rebinding
   to the GGUF params (splitting the fused projection along its output rows,
   which is safe for llama.cpp's input-dim block quant), and fans the single
   quant-type scalar out to each shard.
9. **`install_qwen3_5_text_config_hybrid_properties`** /
   **`install_hybrid_gdn_config_recognition`** — SGLang builds the hybrid
   linear-attention backend and SSM state cache only when it recognizes the
   config as a GDN model, keyed on `isinstance(hf_config.get_text_config(),
   Qwen3NextConfig | Qwen3_5Config | …)` and on a `config.mamba2_cache_params`
   property. Because Transformers 5.8.1 natively owns `model_type=
   "qwen3_5_text"`, `AutoConfig` resolves the resolved config to
   *Transformers'* `Qwen3_5TextConfig`, which is neither in that isinstance
   set nor carries the SSM-cache properties. The first patch grafts SGLang's
   own `Qwen3NextConfig` hybrid properties (`mamba2_cache_params`,
   `linear_layer_ids`, `full_attention_layer_ids` — each reads only fields
   the resolved config already has) onto the Transformers config class; the
   second teaches `ModelRunner.hybrid_gdn_config` to accept it. Without both,
   the GDN layers hit the full-attention backend at runtime and raise
   `AttentionBackend.forward() missing … 'q', 'k', 'v'`.

`ornith_text_model.py`'s `OrnithTextForCausalLM.load_weights` also reshapes
the GDN `conv1d` weight from the GGUF's 2-D `[conv_dim, kernel]` to SGLang's
3-D `[conv_dim, 1, kernel]`, and rebuilds `embed_tokens` with the quant
config (the Qwen3.5 decoder, unlike `qwen2.py`, omits it, so the quantized
`token_embd` would otherwise have no `qweight` param to bind).

`resolve_ornith_gguf_config.py` additionally publishes the dense
`Qwen3_5TextConfig` (not the full multimodal wrapper) with two synthetic
fields SGLang's `qwen3_5.py` reads unconditionally but this Transformers
version's config class never defines or defaults:

- `layers_block_type` — a straight rename/value-translation of the
  Transformers field `layer_types` (`"full_attention"` -> `"attention"`,
  `"linear_attention"` unchanged).
- `output_gate_type` — genuinely absent from both the installed
  Transformers `Qwen3_5TextConfig` class and the source repo's own
  `config.json`; set to `None`, matching that same code's own
  `if self.output_gate_type is not None` handling for "no special
  output-gate activation." This is the one value in this list that is a
  reasoned default rather than a confirmed fact — flagged as such below.

## Runtime evidence (2026-07-23)

### Earlier run — wrong model class, weights unbound

The first real GPU-1 run (patches 1–5 only, registering the flat
`Qwen3_5ForCausalLM` decoder directly) **started the HTTP server** but bound
almost no weights: `Load weight end` reported `mem usage=1.96 GB` (vs the
~6 GB GGUF) with 629 `Parameter … not found in params_dict` warnings across
every layer. Root cause, traced by dumping the model's actual `params_dict`:
the class SGLang's `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` rewrite selects —
`sglang.srt.models.qwen3_5.Qwen3_5ForCausalLM` — is the *flat decoder*, whose
params have no `model.` prefix and no `lm_head`, while every GGUF-derived name
is `model.`-prefixed and includes `lm_head`. That class also cannot emit
logits. Fixed by patch #2's `OrnithTextForCausalLM` wrapper.

### Current run — full weight load, hybrid GDN executes, server serves

With patches 1–9 and the resolved config in place, a real GPU-1 run against
the Unsloth Ornith-1.0-9B `UD-Q4_K_XL` GGUF:

- **Loads every tensor.** `Load weight end` reports `mem usage=5.83 GB` with
  **zero** `not found in params_dict` warnings — all 427 GGUF tensors bind,
  including the split→fused GDN projections and the mixed F32/Q4 precision.
  (Reaching zero required, in order: the tensor-map fixes #6, the F32
  `in_proj_ba` #7, the packed GGUF shard loader #8, the `conv1d` reshape and
  the quantized `embed_tokens` rebuild.)
- **Builds the hybrid GDN runtime.** After patch #9, the tree cache logs
  `hybrid_ssm=True` with a `MambaRadixCache`, the SSM state cache allocates,
  and the FlashInfer full-attention kernel (`head_dim_qk=256`) compiles once
  `LIBRARY_PATH` points at a real `libcuda.so` (see below).
- **Serves.** `The server is fired up and ready to roll!`, `GET /health`
  returns `200`, `GET /model_info` `200`, and both `/v1/chat/completions` and
  `/v1/completions` return completions — the full forward/decode path
  (embedding → 24 GDN linear-attention layers + 8 full-attention layers →
  lm_head → sampling) executes end to end without error.

**Remaining gap — output is not yet coherent (residual bug localized).** The
three GGUF→HF weight transforms the arch requires (RMSNorm −1, `A_log = log(−x)`,
value-head `(r=2,g=16)→(g,r)` unpermutation) are now implemented and applied at
load time (`ornith_gdn_transforms.py`, wired into `load_weights`; 16 unit tests;
`out_proj` handled as a packed-Q8_0 column reorder — see RESEARCH-FINDINGS §5).
Generations are **still gibberish** at `temperature=0`, but the failure is now
*localized*, not open-ended:

- **Oracle established:** `llama.cpp` (`llama-cpp-python`, CPU) on the *same*
  GGUF greedily continues "The capital of France is" → **"Paris."** The GGUF is
  valid; the bug is entirely on the SGLang side.
- **The 3 transforms are correct but not sufficient:** output is gibberish both
  with all transforms on *and* with them all off (a different gibberish), so the
  dominant bug is **independent** of these transforms — no toggle combination
  reaches coherence.
- **Per-layer hidden states are healthy** across all 32 layers (no NaN, smooth
  std 0.28→1.7): no single broken layer — a *semantic* corruption, not numeric.
- **mrope/partial-rotary is RULED OUT** as a misconfiguration: the resolved
  config's `rope_parameters` carries `mrope_section=[11,11,10]` +
  `mrope_interleaved=True`, so SGLang builds the correct interleaved
  `MRotaryEmbedding`. GGML dequant also looks healthy.

The residual is a subtle SGLang-`Qwen3_5` *execution* detail for this checkpoint
(prime suspect: the fused GDN `in_proj_qkvz` split / delta-rule head ordering).
Pinning it down needs the guide §6.4 **per-layer diff vs the 19 GB bf16
`unsloth/Ornith-1.0-9B`** (the llama.cpp oracle exposes only the final hidden
state), which does not fit this 12 GB spike. Diagnostic hooks are left gated
behind `ORNITH_DEBUG_LAYERS=1`.

### Toolchain note: FlashInfer `-lcuda`

The FlashInfer JIT build of the `head_dim=256` prefill kernel fails in this
devenv with `ld.bfd: cannot find -lcuda` — it searches a CUDA stubs dir with
no `libcuda.so`. Exporting `LIBRARY_PATH` to a directory containing a real
`libcuda.so` (the graphics-drivers store path) before launching lets the link
succeed. This is a devenv CUDA-packaging gap, orthogonal to the GGUF/model
work; it is applied as an env var at launch, not baked into `serve.sh`.

## Conclusion

This spike now produces a **fully-loading, fully-serving** SGLang endpoint for
the Ornith-1.0-9B `UD-Q4_K_XL` GGUF — every tensor binds, the hybrid
Gated-DeltaNet + full-attention runtime executes, and the OpenAI endpoints
respond. The nine patches above are real, narrow SGLang 0.5.14 / devenv gaps
(a mis-registered non-servable model class, an incomplete GGUF tensor-name
map, no packed-loader support for GGUF-quantized fused projections, mixed
F32/Q4 tensors, and hybrid-GDN config detection defeated by a Transformers vs
SGLang config-class name clash), each safe and reusable once upstream fixes
land. The one unresolved item is **numerical correctness of the generated
text**, which requires reference-based layer bisection.

If a working endpoint is needed before that debugging is done, serving the
bf16 safetensors checkpoint (`unsloth/Ornith-1.0-9B`, ~19 GB) natively —
SGLang already has complete native `Qwen3_5` model code — remains the
lower-risk route, since it sidesteps the GGUF split/mixed-precision dequant
entirely (though ~19 GB does not fit alongside the KV cache on a single 12 GB
3060 the way the ~6 GB GGUF does).

## Source evidence

- Ornith-1.0-9B model card (`unsloth/Ornith-1.0-9B`): confirms
  `--tool-call-parser qwen3_coder --reasoning-parser qwen3` for SGLang
  ≥0.5.9, `--context-length 262144`, `--mem-fraction-static 0.85`; this
  spike deliberately caps context at 16384, same conservative posture as
  the gemma4 spike.
- `unsloth/Ornith-1.0-9B/config.json`: `architectures:
  ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`, hybrid
  `layer_types` (`linear_attention` / `full_attention`, `full_attention_interval: 4`).
