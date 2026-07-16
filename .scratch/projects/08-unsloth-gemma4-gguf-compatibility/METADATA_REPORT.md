# Exact Unsloth Gemma 4 GGUF metadata and current-stack mapping

**Read-only capture:** 2026-07-14T23:46Z  
**Target:** immutable blob `cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165`

The installed isolated environment is SGLang 0.5.14, Python 3.12.13, Torch
2.11.0+cu130, Transformers `5.14.0.dev0` (the checkout locked in the project),
and the installed `gguf` package. `SGLANG_GGUF_CONFIG_PATH` was unset for this
inspection. Raw captures are in
`artifacts/sglang-gemma4-spike/20260714T234306Z/`.

## GGUF contents

| Item | Value |
| --- | --- |
| `general.architecture` | `gemma4` |
| Name | `Gemma-4 12B IT (smart Q4_0, QAT-lossless)` |
| File type / quantization version | `2` / `2` |
| Context / blocks | 262144 / 48 |
| Hidden / FFN width | 3840 / 15360 |
| Attention heads | 16 |
| Key/value length | 512 / 512 |
| Sliding window | 1024 |
| RoPE base | 1,000,000 |
| Tensors | 667: 338 F32 and 329 Q4_0 |
| Tensor namespaces | `blk` (664), `output_norm`, `rope_freqs`, `token_embd` |
| Vision/projector tensors | none found; this GGUF is the text backbone |

`gemma4.attention.head_count_kv` is a 48-element per-layer list: five
successive `8` values followed by `1`, repeated eight times. The shapes in the
raw dump corroborate it: for example, full-attention layer 47 has a 512-wide
K projection and no V projection, while a sliding layer has 2048-wide K and V
projections. This is semantically mixed attention, not malformed metadata.

## Current Transformers mapping result

The current checkout now contains `gemma4` in `GGUF_CONFIG_MAPPING` and
`GGUF_SUPPORTED_ARCHITECTURES`; it converts the architecture to
`gemma4_text`, and `get_gguf_hf_weights_map` maps `gemma4_text` back to the
GGUF `gemma4` tensor-name map.

However, its Gemma 4 config mapping is:

```text
gemma4.attention.head_count_kv -> num_key_value_heads
```

with no Gemma-4-specific conversion of the list. Read-only conversion therefore
emits `num_key_value_heads: [8, 8, ..., 1]`, whereas the cached native text
configuration represents this as scalar `num_key_value_heads: 8`, scalar
`num_global_key_value_heads: 1`, and an explicit six-layer repeating
`layer_types` sequence. Instantiating that converted config is expected to
fail strict validation because `num_key_value_heads` is an `int` field. The
existing adapter trace demonstrated exactly that validation error; the current
source confirms why it remains applicable after the architecture-support
change.

The smallest credible upstream fix is not `max(list)`: the converter must
recognize Gemma 4 mixed attention and reconstruct the native Gemma 4 fields
(`num_key_value_heads`, `num_global_key_value_heads`, and layer types) from
the per-layer GGUF values, or the configuration/model implementation must
accept the equivalent per-layer representation. Only after that conversion is
correct can the SGLang GGUF tensor loader and CUDA path be assessed.

## Evidence files

- `gguf-metadata.json` — concise reader-derived metadata, tensor types, and names.
- `gguf-dump-raw.txt` — complete local `gguf-dump` output.
- `transformers-current-config-from-gguf.json` — current conversion output.
- `transformers-gguf-mapping.txt` — installed mapping and weight-map source.
- `environment-current.txt` — package/runtime versions and adapter state.
