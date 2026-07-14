#!/usr/bin/env bash
# Focused contract test for serve.sh; does not require CUDA, vLLM, or a model download.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
native_dir="$repo_root/deploy/vllm/native"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/vllm" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$VLLM_ARGS_FILE"
EOF
chmod +x "$tmpdir/vllm"

cat >"$tmpdir/hf" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--local-dir" ]]; then
    mkdir -p "$2"
    printf '{"model_type":"gemma4_unified"}\n' >"$2/config.json"
    exit 0
  fi
  shift
done
exit 1
EOF
chmod +x "$tmpdir/hf"

MODEL=unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
MODEL_REVISION=f18012b8f690e563b7f872cb764b4cb3de90b14a \
TOKENIZER=google/gemma-4-12B-it \
TOKENIZER_REVISION=0e2b1058541244490925fbacf8972041435691ac \
HF_CONFIG_PATH=google/gemma-4-12B-it \
HF_CONFIG_REVISION=0e2b1058541244490925fbacf8972041435691ac \
SERVED_MODEL_NAME=base \
MAX_MODEL_LEN=16384 \
GPU_MEMORY_UTILIZATION=0.82 \
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/tmp/hf \
PATH="$tmpdir:$PATH" VLLM_BIN="$tmpdir/vllm" VLLM_ARGS_FILE="$tmpdir/args" \
  bash "$native_dir/serve.sh"

expected=(
  serve unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL
  --revision f18012b8f690e563b7f872cb764b4cb3de90b14a
  --tokenizer google/gemma-4-12B-it
  --tokenizer-revision 0e2b1058541244490925fbacf8972041435691ac
  --hf-config-path /tmp/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac
  --host 127.0.0.1 --port 8000
  --served-model-name base
  --max-model-len 16384
  --gpu-memory-utilization 0.82
  --quantization gguf --dtype bfloat16
  --enforce-eager
  --max-num-batched-tokens 768
  --limit-mm-per-prompt '{"video": 0}'
  --cpu-offload-gb 0
  --structured-outputs-config.backend xgrammar
)
mapfile -t actual <"$tmpdir/args"
[[ "${actual[*]}" == "${expected[*]}" ]]
