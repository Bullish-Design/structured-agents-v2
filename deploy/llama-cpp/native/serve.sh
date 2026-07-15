#!/usr/bin/env bash
# Start the independent llama.cpp Gemma 4 endpoint.  This service intentionally
# shares no process, port, GPU, or cache writes with the vLLM endpoint.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the local Gemma GGUF file}"
: "${DRAFT_MODEL_PATH:?set DRAFT_MODEL_PATH to the local Gemma MTP drafter GGUF file}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the dedicated GPU}"

[[ -f "$MODEL_PATH" ]] || { echo "GGUF model does not exist: $MODEL_PATH" >&2; exit 1; }
[[ -f "$DRAFT_MODEL_PATH" ]] || { echo "MTP drafter GGUF does not exist: $DRAFT_MODEL_PATH" >&2; exit 1; }
[[ "$CUDA_VISIBLE_DEVICES" == "0" ]] || {
  echo "llama.cpp comparison profile requires CUDA_VISIBLE_DEVICES=0" >&2
  exit 1
}

args=(
  --model "$MODEL_PATH"
  --host 127.0.0.1 --port "${PORT:-8001}"
  --alias "${SERVED_MODEL_NAME:-base}"
  --ctx-size "${CTX_SIZE:-16384}"
  # Keep all weights and KV cache on GPU 0.  The q8 KV cache is the practical
  # 16k-context profile for this 12 GiB card; it avoids host-memory fallback.
  --n-gpu-layers 999
  --cache-type-k q8_0 --cache-type-v q8_0
  --flash-attn on
  # The target verifies every MTP proposal, so this improves throughput without
  # changing the generated output.
  --spec-type draft-mtp
  --spec-draft-model "$DRAFT_MODEL_PATH"
  --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-4}"
  --n-gpu-layers-draft 999
  --parallel "${PARALLEL_SLOTS:-1}"
  --threads "${CPU_THREADS:-8}"
  --threads-batch "${CPU_BATCH_THREADS:-8}"
)

if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "$API_KEY")
fi

echo "+ llama-server ${args[*]}"
exec "${LLAMA_SERVER_BIN:-llama-server}" "${args[@]}"
