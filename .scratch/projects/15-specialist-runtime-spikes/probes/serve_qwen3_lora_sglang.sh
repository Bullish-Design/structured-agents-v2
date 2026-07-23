#!/usr/bin/env bash
# Native Qwen3-4B-AWQ + dual-LoRA SGLang qualification runner.
#
# This is the parity counterpart to the vLLM serve_qwen3_lora_spike profile. It
# deliberately does NOT reuse deploy/sglang/native/serve.sh: that launcher is
# hardened for the (failed) Gemma-4 GGUF path and is pinned to GPU 0. Here we
# qualify the native AWQ + multi-adapter + xgrammar path on GPU 1 / port 8002.
set -euo pipefail

AWQ_SNAPSHOT=/var/lib/structured-agents-vllm/hf/hub/models--Qwen--Qwen3-4B-AWQ/snapshots/136f16ffdca9c9e49527391169e042634b9ad0d6
# SGLang's fused-QKV LoRA backend needs q/k/v projections present in each adapter,
# so this runner uses the q/k/v control set (the vLLM run used the q-only set).
LORA_DIR=/home/andrew/Documents/Projects/structured-agents-v2/.scratch/projects/15-specialist-runtime-spikes/runtime/qwen3-loras-qkv
SGLANG_PY=/home/andrew/Documents/Projects/structured-agents-v2/deploy/sglang/native/.venv/bin/python

[[ -f "$AWQ_SNAPSHOT/config.json" ]] || { echo "AWQ snapshot missing: $AWQ_SNAPSHOT" >&2; exit 1; }
[[ -f "$LORA_DIR/control-a/adapter_config.json" ]] || { echo "control-a adapter missing" >&2; exit 1; }
[[ -f "$LORA_DIR/control-b/adapter_config.json" ]] || { echo "control-b adapter missing" >&2; exit 1; }
[[ -x "$SGLANG_PY" ]] || { echo "sglang venv python missing: $SGLANG_PY" >&2; exit 1; }

# Isolate GPU + caches from the vLLM and llama.cpp runners.
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# awq_marlin repack JIT-compiles a kernel with ninja, which lives in the venv's
# bin dir; that dir is not on PATH when python is invoked by absolute path.
export PATH="$(dirname "$SGLANG_PY"):$PATH"

args=(
  --model-path "$AWQ_SNAPSHOT"
  --host 127.0.0.1 --port 8002
  --served-model-name base
  --context-length 4096
  --mem-fraction-static 0.70
  --max-running-requests 8
  --grammar-backend xgrammar
  --disable-cuda-graph
  --disable-flashinfer-autotune
  # flashinfer's awq_marlin+LoRA decode kernel deadlocks the forward pass on this
  # RTX 3060 (sm_86); triton attention avoids that path and also skips the slow
  # flashinfer JIT warmup.
  --attention-backend triton
  --sampling-backend pytorch
  --lora-paths control-a="$LORA_DIR/control-a" control-b="$LORA_DIR/control-b"
  --max-loras-per-batch 2
  --disable-radix-cache
)

printf '+ %q ' "$SGLANG_PY" -m sglang.launch_server "${args[@]}"; printf '\n'
exec "$SGLANG_PY" -m sglang.launch_server "${args[@]}"
