#!/usr/bin/env bash
# Build a `vllm serve` invocation from env vars. Keeps the container declarative:
# every knob the xgrammar concept needs (XGrammar backend, per-agent LoRA, API key,
# batching, context length) is set here from the environment.
set -euo pipefail

: "${MODEL:?set MODEL (HF repo id, or local path under /models/base)}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_MODEL_NAME:-$MODEL}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SO_BACKEND="${STRUCTURED_OUTPUTS_BACKEND:-xgrammar}"

args=(
  "$MODEL"
  --host "$HOST" --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
)

# API key — vLLM enforces it on every request when set (do this even on Tailscale).
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

# Structured-outputs backend (XGrammar). The CLI flag for this MOVED across vLLM
# versions, so it's overridable:
#   older:  --guided-decoding-backend xgrammar
#   newer:  --structured-outputs-config.backend xgrammar   (0.10+)
SO_FLAG="${STRUCTURED_OUTPUTS_FLAG:---guided-decoding-backend}"
args+=("$SO_FLAG" "$SO_BACKEND")

# Per-agent LoRA. Set LORA_MODULES to a space-separated list of name=path, e.g.
#   LORA_MODULES="file-edit=/models/lora/file-edit git-ops=/models/lora/git-ops"
# Each name becomes addressable via the OpenAI `model` field (how a PydanticAI
# agent selects its adapter — verified in the request-path spike).
if [[ -n "${LORA_MODULES:-}" ]]; then
  # shellcheck disable=SC2206  # intentional word-splitting of the module list
  modules=(${LORA_MODULES})
  args+=(--enable-lora
         --max-loras "${MAX_LORAS:-4}"
         --max-lora-rank "${MAX_LORA_RANK:-32}"
         --lora-modules "${modules[@]}")
fi

# Anything else passed through verbatim, e.g. EXTRA_ARGS="--quantization awq --dtype half"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${EXTRA_ARGS})
  args+=("${extra[@]}")
fi

echo "+ vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
