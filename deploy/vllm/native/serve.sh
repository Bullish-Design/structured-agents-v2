#!/usr/bin/env bash
# Start the native vLLM process with the Gemma 4 production profile.
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${MODEL_REVISION:?set MODEL_REVISION to an immutable Hugging Face commit}"
: "${TOKENIZER:?set TOKENIZER to the Hugging Face tokenizer repository}"
: "${TOKENIZER_REVISION:?set TOKENIZER_REVISION to an immutable Hugging Face commit}"
: "${HF_CONFIG_PATH:?set HF_CONFIG_PATH to the Hugging Face configuration repository}"
: "${HF_CONFIG_REVISION:?set HF_CONFIG_REVISION to an immutable Hugging Face commit}"
: "${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
: "${MAX_MODEL_LEN:?set MAX_MODEL_LEN}"
: "${GPU_MEMORY_UTILIZATION:?set GPU_MEMORY_UTILIZATION}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the dedicated GPU}"

[[ "$MAX_MODEL_LEN" == "16384" ]] || { echo "Gemma 4 profile requires MAX_MODEL_LEN=16384" >&2; exit 1; }
[[ "$GPU_MEMORY_UTILIZATION" == "0.82" ]] || { echo "Gemma 4 profile requires GPU_MEMORY_UTILIZATION=0.82" >&2; exit 1; }
[[ "$CUDA_VISIBLE_DEVICES" == "1" ]] || { echo "Gemma 4 profile requires CUDA_VISIBLE_DEVICES=1" >&2; exit 1; }
[[ -z "${LORA_MODULES:-}" ]] || { echo "Gemma 4 profile prohibits LORA_MODULES" >&2; exit 1; }

# vLLM applies --revision to --hf-config-path too. The GGUF weights and the
# upstream Gemma config live in different repositories, so materialize the
# latter locally at its own immutable revision before passing it to vLLM.
hf_config_dir="${HF_HOME:-$PWD/.hf}/gemma4-config-$HF_CONFIG_REVISION"
if [[ ! -f "$hf_config_dir/processor_config.json" ]]; then
  # Keep the multimodal processor assets alongside config.json. Do not fetch
  # the upstream safetensors weights: vLLM loads the pinned GGUF separately.
  hf download "$HF_CONFIG_PATH" \
    config.json generation_config.json processor_config.json \
    tokenizer.json tokenizer_config.json chat_template.jinja \
    --revision "$HF_CONFIG_REVISION" --local-dir "$hf_config_dir"
fi

args=(
  serve "$MODEL"
  --revision "$MODEL_REVISION"
  --tokenizer "$TOKENIZER"
  --tokenizer-revision "$TOKENIZER_REVISION"
  --hf-config-path "$hf_config_dir"
  --host 127.0.0.1 --port 8000
  --served-model-name "$SERVED_MODEL_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --quantization gguf --dtype bfloat16
  # Gemma 4's compiled CUDA-graph warm-up exceeds the remaining headroom on
  # the dedicated 12 GiB GPU after this exact 12B GGUF is loaded. Eager mode
  # keeps the requested 16k/0.82 capacity profile without that transient peak.
  --enforce-eager
  # Gemma 4 is a prefix-LM multimodal model. Without an explicit cap, vLLM
  # raises the startup profiling batch to 2,496 tokens for a video item; on
  # this 12 GiB GPU that transient BF16 activation peak OOMs after the GGUF
  # weights are resident. This caps prefill/concurrency, not request context:
  # --max-model-len remains 16,384. Gemma 4 needs 750 tokens for one image
  # because bidirectional multimodal input cannot be chunked, so use the
  # smallest aligned value above that budget.
  --max-num-batched-tokens 768
  # The Gemma processor advertises video, but this deployment does not serve
  # video requests. Reject them at the OpenAI-compatible API boundary.
  --limit-mm-per-prompt '{"video": 0}'
  # Do not allow model-weight fallback capacity in host RAM. vLLM 0.25 has no
  # CPU swap cache; its KV offload remains disabled unless explicitly sized.
  --cpu-offload-gb 0
  --structured-outputs-config.backend xgrammar
)

if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
fi

# vLLM 0.25's FlashInfer sampler may JIT-compile a CUDA extension. The locked
# Python environment ships its matching nvcc under nvidia/cu*/bin; expose that
# toolkit explicitly so NixOS does not need a global /usr/local/cuda install.
if [[ -z "${CUDA_HOME:-}" ]]; then
  venv_root="${VIRTUAL_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv}"
  shopt -s nullglob
  nvcc_candidates=("$venv_root"/lib/python*/site-packages/nvidia/cu*/bin/nvcc)
  shopt -u nullglob
  if [[ "${#nvcc_candidates[@]}" -eq 1 ]]; then
    export CUDA_HOME="${nvcc_candidates[0]%/bin/nvcc}"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

exec "${VLLM_BIN:-vllm}" "${args[@]}"
