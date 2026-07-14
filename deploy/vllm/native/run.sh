#!/usr/bin/env bash
# Execute the locked native vLLM environment. Used by the systemd unit.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# The vendored GGUF extension is compiled with the matching Nix CUDA toolkit.
# CUDA packages are unfree, so devenv must be allowed to evaluate that toolkit
# when systemd starts the already-approved NixOS service.
export NIXPKGS_ALLOW_UNFREE=1
exec devenv shell --impure -- uv run --locked --no-sync bash "$here/serve.sh"
