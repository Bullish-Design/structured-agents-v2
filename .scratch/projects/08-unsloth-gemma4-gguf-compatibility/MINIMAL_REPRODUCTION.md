# Upstream-ready reproduction: Gemma 4 GGUF config conversion fails

## Expected behavior

An SGLang GGUF startup for a valid text-only Gemma 4 GGUF should progress past
Hugging Face configuration construction so that the tensor-name/quantization
loader can be tested.

## Actual behavior

The current Transformers GGUF converter accepts `general.architecture=gemma4`,
but copies the 48-element Gemma 4 mixed-attention
`gemma4.attention.head_count_kv` array directly into the scalar
`Gemma4Config.num_key_value_heads`. Strict config validation then aborts before
weights load:

```text
huggingface_hub.errors.StrictDataclassFieldValidationError:
Validation error for field 'num_key_value_heads':
  TypeError: Field 'num_key_value_heads' expected int, got list
```

## Environment

- SGLang 0.5.14
- Python 3.12.13
- Torch 2.11.0+cu130
- Transformers 5.14.0.dev0, commit
  `ab1771c9e42891d893189978a8009426d70b4688`
- GPU: RTX 3060 12 GiB; this error occurs before GPU model allocation.

## Command

```bash
env -u SGLANG_GGUF_CONFIG_PATH \
  MODEL_PATH=/path/to/gemma4-12b-qat.gguf \
  TOKENIZER_PATH=/path/to/cached/gemma4-config \
  MODEL_LOAD_FORMAT=gguf CUDA_VISIBLE_DEVICES=0 PORT=8002 \
  CONTEXT_LENGTH=16384 MAX_RUNNING_REQUESTS=1 CPU_OFFLOAD_GB=0 \
  ENABLE_MTP=0 SGLANG_CACHE_DIR=/tmp/sglang-gguf-cache \
  bash deploy/sglang/native/run.sh
```

## Non-sensitive metadata excerpt

```text
general.architecture = gemma4
gemma4.block_count = 48
gemma4.attention.head_count = 16
gemma4.attention.head_count_kv = [8, 8, 8, 8, 8, 1, ...]  # 48 values; 5:1 repeat
gemma4.attention.key_length = 512
gemma4.attention.value_length = 512
gemma4.attention.sliding_window = 1024
```

The corresponding native Gemma 4 config has scalar
`num_key_value_heads=8`, scalar `num_global_key_value_heads=1`, and explicit
`layer_types` identifying the sixth layer of each group as full attention.

## Suspected component and smallest credible fix

The blocker is Transformers' `load_gguf_checkpoint` Gemma 4 config conversion,
not SGLang's CUDA implementation. `GGUF_CONFIG_MAPPING["gemma4"]` currently
maps `attention.head_count_kv` directly to `num_key_value_heads`, while the
Gemma-4-specific post-processing only changes `model_type` and EOS tokens.

Add a Gemma-4-specific conversion that reconstructs scalar local/global KV-head
counts and per-layer attention types from the metadata (or introduce an
explicitly supported per-layer representation). Do not coerce with `max()`:
that loses the full-attention layer semantics and can silently generate
incorrect output. Once fixed, retry to determine tensor-name and Q4_0 loader
compatibility independently.

## Evidence

Complete traceback and resolved command:
`artifacts/sglang-gemma4-spike/20260714T234306Z/clean-stock-gguf-launch.txt`.
Read-only metadata and mapping report: [METADATA_REPORT.md](METADATA_REPORT.md).

## Follow-on reproduction: constructor-time SDPA rejection

After the local metadata conversion supplies a valid `gemma4_text` config,
SGLang 0.5.14 reaches `Load weight begin` and selects its Triton Gemma 4
attention backend. It then fails during construction of SGLang's own
`Gemma4ForCausalLM`, before target tensors are loaded:

```text
ValueError: Gemma4ForCausalLM does not support an attention implementation
through torch.nn.functional.scaled_dot_product_attention yet.
```

The relevant path is `sglang.srt.model_loader.loader._initialize_model` to
`model_class(config=model_config.hf_config, ...)`, then Transformers'
`PreTrainedModel.__init__`. The resolved config has an explicit SDPA request,
so Transformers treats it as requested rather than applying its normal
unspecified-attention fallback.

The local, reversible test patch changes only
`Gemma4TextConfig._attn_implementation` to `"eager"` immediately before that
constructor call. It intentionally leaves SGLang's server-selected Triton
backend unchanged; SGLang's Gemma 4 module implements runtime attention with
`RadixAttention`, not Transformers SDPA/eager attention. Reproduce with the
same command above plus `SGLANG_GGUF_CONFIG_PATH` set to the cached config
file, after applying the local metadata compatibility layer.

## Follow-on reproduction: expert implementation validation

With the eager attention construction workaround installed, the exact GGUF
reaches `Load weight begin` and fails before tensors are read with:

```text
ValueError: Gemma4ForCausalLM does not support setting experts implementation.
```

The exception is raised by Transformers'
`PreTrainedModel._grouped_mm_can_dispatch` while SGLang constructs its own
Gemma 4 module. Transformers defaults an unspecified expert implementation to
`grouped_mm`; the raised path means the resolved config is requesting that
implementation explicitly. This must be investigated separately from the
attention workaround: determine where the request is set and compare SGLang
main's constructor/config handling before changing it. The error precedes
GGUF tensor loading, so it establishes no tensor, quantization, API, MTP, or
context-capacity result.
