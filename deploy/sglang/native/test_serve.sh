#!/usr/bin/env bash
# Static shell contract test. Does not import SGLang, access CUDA, or load a model.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
touch "$tmpdir/model.gguf"
printf '{}\n' >"$tmpdir/tokenizer.json"
printf '{}\n' >"$tmpdir/config.json"
cat >"$tmpdir/python" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$SGLANG_ARGS_FILE"
EOF
chmod +x "$tmpdir/python"

MODEL_PATH="$tmpdir/model.gguf" TOKENIZER_PATH="$tmpdir" CUDA_VISIBLE_DEVICES=0 \
SGLANG_GGUF_CONFIG_PATH="$tmpdir/config.json" SGLANG_CACHE_DIR="$tmpdir/cache" SGLANG_PYTHON="$tmpdir/python" SGLANG_ARGS_FILE="$tmpdir/args" \
  bash "$here/serve.sh"

expected=(
  -m sglang.launch_server --model-path "$tmpdir/model.gguf" --tokenizer-path "$tmpdir"
  --load-format gguf --host 127.0.0.1 --port 8002 --served-model-name base
  --context-length 16384 --mem-fraction-static 0.80 --max-running-requests 1
  --cpu-offload-gb 0 --grammar-backend xgrammar --disable-cuda-graph
)
mapfile -t actual <"$tmpdir/args"
[[ "${actual[*]}" == "${expected[*]}" ]]

if MODEL_PATH="$tmpdir/model.gguf" TOKENIZER_PATH="$tmpdir" CUDA_VISIBLE_DEVICES=0 \
  SGLANG_GGUF_CONFIG_PATH="$tmpdir/config.json" SGLANG_CACHE_DIR="$tmpdir/cache" SGLANG_PYTHON="$tmpdir/python" ENABLE_MTP=1 \
  bash "$here/serve.sh" >/dev/null 2>&1; then
  echo "MTP safety gate unexpectedly passed" >&2
  exit 1
fi
