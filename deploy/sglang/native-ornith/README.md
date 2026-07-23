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

`sitecustomize.py` / `ornith_gguf_compat.py` install five narrow, well
-understood compatibility patches, none of which change model semantics
beyond what a genuinely-supported checkpoint would already have:

1. **`install_static_config_redirect`** — redirects every
   `AutoConfig.from_pretrained(<gguf path>, ...)` call (there are two
   independent call sites in SGLang 0.5.14: the main engine config and
   `multimodal_processor.get_processor`) to the static resolved config
   instead of the broken GGUF-metadata parser.
2. **`install_causal_lm_registry_entry`** — `sglang.srt.utils.hf_transformers
   .config.get_config` unconditionally rewrites a GGUF checkpoint's
   `architectures` to Transformers' `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
   [model_type]` entry (`Qwen3_5ForCausalLM` for `qwen3_5`/`qwen3_5_text`).
   But `sglang.srt.models.qwen3_5`'s `EntryClass` list only registers
   `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`
   — the plain `Qwen3_5ForCausalLM` class is defined in the same file but
   never registered. This is a genuine gap in SGLang's model registry, not
   a missing architecture; the patch adds the one missing entry.
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
   without the method would already produce.
5. **`install_gguf_arch_name_translation`** — `GGUFModelLoader
   ._get_gguf_weights_map` looks up the tensor-name map by searching the
   standalone `gguf` PyPI package's `MODEL_ARCH_NAMES` for a value equal to
   `config.model_type`, with existing hardcoded translations for other
   naming mismatches (`cohere` -> `command-r`, `qwen3_moe` -> `qwen3moe`).
   The `gguf` package already has a full `qwen35` tensor-name map
   (confirmed to include the Gated-DeltaNet tensors); it was simply never
   wired up because SGLang's model_type string (`qwen3_5`/`qwen3_5_text`)
   doesn't match the GGUF library's arch key (`qwen35`).

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

A real GPU-1 run of `run.sh` against the Unsloth Ornith-1.0-9B `UD-Q4_K_XL`
GGUF, with all five compat patches and the resolved config in place,
**started the HTTP server successfully** (`Uvicorn running on
http://127.0.0.1:8003`, `GET /model_info` returned `200 OK`). However:

- **Weight loading is incomplete.** `Load weight end` reported
  `mem usage=1.96 GB` — far short of the ~6 GB Q4_K_XL GGUF file size — and
  the load logged 581 `Parameter ... not found in params_dict` warnings
  spanning every one of the 32 layers (`linear_attn.in_proj_qkvz`,
  `linear_attn.out_proj`, `mlp.*`, `o_proj`, `qkv_proj`, `q_norm`,
  `input_layernorm`, and more). The tensor-name map SGLang derives via the
  `gguf` package for `qwen35` does not fully match the parameter names
  `Qwen3_5ForCausalLM.load_weights` expects, so a large fraction of the
  model's weights were **not** loaded from the GGUF and were left at their
  randomly-initialized values. **This is not a working model** — it would
  produce incoherent output, not a correctness bug you'd notice only under
  load.
- `GET /health` returned `503`, and the scheduler then crashed on the
  warmup forward pass with `RuntimeError: Ninja build failed` compiling a
  FlashInfer CUDA kernel (`batch_prefill`, `head_dim_qk=256`) — a separate
  CUDA/build-toolchain issue in this devenv, not a GGUF/architecture
  problem, and never reached because the weight-loading gap above would
  have produced garbage output regardless.

**Conclusion: this spike does not produce a usable Ornith-1.0-9B GGUF
endpoint.** The five registry/config/patch-level fixes above are real,
narrow SGLang 0.5.14 gaps and are safe/reusable once upstream fixes land.
The remaining gap — the GGUF tensor-name map not fully covering
`Qwen3_5ForCausalLM`'s actual parameter names, particularly the
Gated-DeltaNet linear-attention tensors — is a deeper mismatch between the
`gguf` library's `qwen35` tensor map and this specific SGLang model
implementation's expected names, and was not resolved. It would need
either an upstream SGLang/`gguf`-library fix or a hand-written tensor-name
remapping layer, which is a substantially larger undertaking than the
patches above and was not attempted here.

Given GGUF's fundamental unsuitability for this checkpoint right now, the
straightforward alternative — serving the bf16 safetensors checkpoint
(`unsloth/Ornith-1.0-9B`, ~19 GB) natively with SGLang, which already has
complete native `Qwen3_5` model code — was not attempted in this spike but
is far more likely to actually work, since it sidesteps the entire
GGUF-loading problem.

## Source evidence

- Ornith-1.0-9B model card (`unsloth/Ornith-1.0-9B`): confirms
  `--tool-call-parser qwen3_coder --reasoning-parser qwen3` for SGLang
  ≥0.5.9, `--context-length 262144`, `--mem-fraction-static 0.85`; this
  spike deliberately caps context at 16384, same conservative posture as
  the gemma4 spike.
- `unsloth/Ornith-1.0-9B/config.json`: `architectures:
  ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`, hybrid
  `layer_types` (`linear_attention` / `full_attention`, `full_attention_interval: 4`).
