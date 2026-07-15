#!/usr/bin/env bash
# Contract test for the llama.cpp launch profile; it needs neither CUDA nor weights.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
touch "$tmpdir/model.gguf" "$tmpdir/draft.gguf"
cat >"$tmpdir/llama-server" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$LLAMA_ARGS_FILE"
EOF
chmod +x "$tmpdir/llama-server"

MODEL_PATH="$tmpdir/model.gguf" DRAFT_MODEL_PATH="$tmpdir/draft.gguf" CUDA_VISIBLE_DEVICES=0 PORT=8001 \
SERVED_MODEL_NAME=base CTX_SIZE=16384 PARALLEL_SLOTS=1 CPU_THREADS=8 \
LLAMA_SERVER_BIN="$tmpdir/llama-server" LLAMA_ARGS_FILE="$tmpdir/args" \
  bash "$here/serve.sh"

expected=(
  --model "$tmpdir/model.gguf" --host 127.0.0.1 --port 8001 --alias base
  --ctx-size 16384 --n-gpu-layers 999 --cache-type-k q8_0 --cache-type-v q8_0
  --flash-attn on --spec-type draft-mtp --spec-draft-model "$tmpdir/draft.gguf"
  --spec-draft-n-max 4 --n-gpu-layers-draft 999 --parallel 1 --threads 8 --threads-batch 8
)
mapfile -t actual <"$tmpdir/args"
[[ "${actual[*]}" == "${expected[*]}" ]]
