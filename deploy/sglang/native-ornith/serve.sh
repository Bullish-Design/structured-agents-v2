#!/usr/bin/env bash
# Start the isolated SGLang GGUF spike for unsloth/Ornith-1.0-9B-GGUF
# (Ornith-1.0-9B-UD-Q4_K_XL.gguf). Ornith is `Qwen3_5ForConditionalGeneration`
# (model_type "qwen3_5"): a hybrid Gated-DeltaNet linear-attention + periodic
# full-attention, multimodal architecture. SGLang 0.5.14 (pinned here) has
# native Qwen3_5 model code, but GGUF weight loading for this brand-new
# architecture is unproven — this script does not assume it works.
# Deliberately not a production launcher: no downloads, CPU weight offload,
# or speculative drafter are allowed without a separately captured runtime
# result.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the exact local target GGUF}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH to the existing local Ornith tokenizer directory}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the isolated GPU}"
: "${SGLANG_GGUF_CONFIG_PATH:?set SGLANG_GGUF_CONFIG_PATH to a config.json produced by resolve_ornith_gguf_config.py}"
[[ -f "$SGLANG_GGUF_CONFIG_PATH" ]] || { echo "SGLANG_GGUF_CONFIG_PATH does not exist: $SGLANG_GGUF_CONFIG_PATH" >&2; exit 1; }

if [[ "${MODEL_LOAD_FORMAT:-gguf}" == "gguf" ]]; then
  [[ -f "$MODEL_PATH" ]] || { echo "target GGUF does not exist: $MODEL_PATH" >&2; exit 1; }
else
  [[ -d "$MODEL_PATH" ]] || { echo "native model directory does not exist: $MODEL_PATH" >&2; exit 1; }
fi
[[ -f "$TOKENIZER_PATH/tokenizer.json" ]] || { echo "tokenizer.json does not exist under: $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH/tokenizer_config.json" ]] || { echo "tokenizer_config.json does not exist under: $TOKENIZER_PATH" >&2; exit 1; }

[[ "$CUDA_VISIBLE_DEVICES" == "1" ]] || { echo "Ornith spike is pinned to GPU 1; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 1; }
[[ "${PORT:-8003}" == "8003" ]] || { echo "Ornith spike port is reserved as 8003; got PORT=${PORT:-}" >&2; exit 1; }
[[ "${CONTEXT_LENGTH:-16384}" == "16384" ]] || { echo "spike requires CONTEXT_LENGTH=16384" >&2; exit 1; }
[[ "${MAX_RUNNING_REQUESTS:-1}" == "1" ]] || { echo "spike starts with exactly one slot" >&2; exit 1; }
[[ "${CPU_OFFLOAD_GB:-0}" == "0" ]] || { echo "CPU weight offload is prohibited for this spike" >&2; exit 1; }

# Ornith's GGUF ships a separate mmproj (vision) file. This spike is text-only
# until a runtime result proves the multimodal path; refuse rather than
# silently dropping image support.
[[ -z "${MMPROJ_PATH:-}" ]] || { echo "multimodal (MMPROJ_PATH) is out of scope for this text-only spike" >&2; exit 1; }

export HF_HOME="${SGLANG_CACHE_DIR:?set SGLANG_CACHE_DIR to the dedicated persistent cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME"

args=(
  --model-path "$MODEL_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --load-format "${MODEL_LOAD_FORMAT:-gguf}"
  --host 127.0.0.1 --port "${PORT:-8003}"
  --served-model-name "${SERVED_MODEL_NAME:-base}"
  --context-length "${CONTEXT_LENGTH:-16384}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-1}"
  --cpu-offload-gb 0
  --grammar-backend "${GRAMMAR_BACKEND:-xgrammar}"
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --disable-cuda-graph
)

printf '+ %q ' "${SGLANG_PYTHON:-python3}" -m sglang.launch_server "${args[@]}"
printf '\n'
exec "${SGLANG_PYTHON:-python3}" -m sglang.launch_server "${args[@]}"
