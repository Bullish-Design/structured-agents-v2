#!/usr/bin/env bash
# Start the isolated SGLang GGUF spike. This is deliberately not a production
# launcher: no downloads, CPU weight offload, or speculative drafter are allowed
# without a separately captured runtime result.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the exact local target GGUF}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH to the existing local Gemma tokenizer directory}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the isolated GPU}"
: "${SGLANG_GGUF_CONFIG_PATH:?set SGLANG_GGUF_CONFIG_PATH to a config.json produced by resolve_gemma4_gguf_config.py}"
[[ -f "$SGLANG_GGUF_CONFIG_PATH" ]] || { echo "SGLANG_GGUF_CONFIG_PATH does not exist: $SGLANG_GGUF_CONFIG_PATH" >&2; exit 1; }

if [[ "${MODEL_LOAD_FORMAT:-gguf}" == "gguf" ]]; then
  [[ -f "$MODEL_PATH" ]] || { echo "target GGUF does not exist: $MODEL_PATH" >&2; exit 1; }
else
  [[ -d "$MODEL_PATH" ]] || { echo "native model directory does not exist: $MODEL_PATH" >&2; exit 1; }
fi
[[ -f "$TOKENIZER_PATH/tokenizer.json" ]] || { echo "tokenizer.json does not exist under: $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH/config.json" ]] || { echo "config.json does not exist under: $TOKENIZER_PATH" >&2; exit 1; }
# Config resolution comes from a static config.json produced once, offline,
# by `resolve_gemma4_gguf_config.py` -- not from a live GGUF parse inside the
# server process. See sitecustomize.py for why the live-derive path was
# dropped (an import-order bug silently discarded the sliding/full attention
# head_dim split on every server start).
[[ "$CUDA_VISIBLE_DEVICES" == "0" ]] || { echo "SGLang spike is pinned to GPU 0; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 1; }
[[ "${PORT:-8002}" == "8002" ]] || { echo "SGLang spike port is reserved as 8002; got PORT=${PORT:-}" >&2; exit 1; }
[[ "${CONTEXT_LENGTH:-16384}" == "16384" ]] || { echo "spike requires CONTEXT_LENGTH=16384" >&2; exit 1; }
[[ "${MAX_RUNNING_REQUESTS:-1}" == "1" ]] || { echo "spike starts with exactly one slot" >&2; exit 1; }
[[ "${CPU_OFFLOAD_GB:-0}" == "0" ]] || { echo "CPU weight offload is prohibited for this spike" >&2; exit 1; }

# There is no proven SGLang command line for the target GGUF + Q8 GGUF MTP
# assistant. Refuse rather than silently trying a mismatched HF drafter.
[[ "${ENABLE_MTP:-0}" == "0" ]] || {
  echo "MTP is deliberately disabled pending a version-specific runtime-proven configuration" >&2
  exit 1
}

# Keep this cache entirely separate from vLLM's cache. Offline settings turn an
# absent local tokenizer/config into a clear error instead of a hidden download.
export HF_HOME="${SGLANG_CACHE_DIR:?set SGLANG_CACHE_DIR to the dedicated persistent cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME"

args=(
  --model-path "$MODEL_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --load-format "${MODEL_LOAD_FORMAT:-gguf}"
  --host 127.0.0.1 --port "${PORT:-8002}"
  --served-model-name "${SERVED_MODEL_NAME:-base}"
  --context-length "${CONTEXT_LENGTH:-16384}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-1}"
  --cpu-offload-gb 0
  --grammar-backend "${GRAMMAR_BACKEND:-xgrammar}"
  --disable-cuda-graph
)

printf '+ %q ' "${SGLANG_PYTHON:-python3}" -m sglang.launch_server "${args[@]}"
printf '\n'
exec "${SGLANG_PYTHON:-python3}" -m sglang.launch_server "${args[@]}"
