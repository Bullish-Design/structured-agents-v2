#!/usr/bin/env bash
# Enter only the dedicated SGLang devenv, then run its locked uv environment.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
export NIXPKGS_ALLOW_UNFREE=1
exec devenv shell --impure -- uv run --locked --no-sync bash "$here/serve.sh"
