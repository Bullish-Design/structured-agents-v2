#!/usr/bin/env bash
# Enter only the dedicated SGLang devenv, then run its locked uv environment.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
export NIXPKGS_ALLOW_UNFREE=1
# Makes the duplicate AutoConfig-registration compatibility shim available.
# The GGUF config adapter remains disabled unless its explicit environment
# variable is supplied. Keep any caller-supplied PYTHONPATH after this directory.
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec devenv shell --impure -- uv run --locked --no-sync bash "$here/serve.sh"
