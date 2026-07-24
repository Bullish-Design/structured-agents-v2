#!/usr/bin/env bash
# Launch the patched Ornith GGUF SGLang spike on GPU 1 / port 8003.
# Env toggles ORNITH_FIX_NORM/ALOG/PERM (default on) select which of the 3 GDN
# transforms are applied — pass e.g. ORNITH_FIX_PERM=0 to bisect.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
spike="/home/andrew/Documents/Projects/structured-agents-v2/deploy/sglang/native-ornith"

export MODEL_PATH="/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf"
export TOKENIZER_PATH="/home/andrew/.cache/structured-agents/sglang-ornith-tokenizer"
export SGLANG_GGUF_CONFIG_PATH="/home/andrew/.cache/structured-agents/sglang-ornith-resolved-config/config.json"
export SGLANG_CACHE_DIR="/home/andrew/.cache/structured-agents/sglang-ornith-cache"
export CUDA_VISIBLE_DEVICES=1
export LIBRARY_PATH="/nix/store/l5smwbs4q9rni6b0pw3fr8qyl4zja14f-graphics-drivers/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

cd "$spike"
export NIXPKGS_ALLOW_UNFREE=1
exec devenv shell --impure -- uv run --locked --no-sync bash "$spike/serve.sh"
