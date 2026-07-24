#!/usr/bin/env bash
# Build a chosen llama.cpp (release tag, fork, or local) into a shared-lib set
# that llama-cpp-python can load via LLAMA_CPP_LIB_PATH (Mode B) or that a
# source rebuild of llama-cpp-python can bundle (Mode A).
#
# Run inside `devenv shell` or the project CUDA nix-shell so gcc, cmake, and
# the CUDA toolkit are present. See 06-LLAMACPP-BUILD-WORKFLOW.md.
#
# Usage:
#   build-llamacpp.sh --ref <git-ref|local:PATH> --profile <cpu-light|cuda-3060> [--out DIR]
#
# The ABI-anchor rule (workflow doc §1): the built commit MUST be ABI-compatible
# with the installed llama-cpp-python's bindings. Run the smoke gate after.
set -euo pipefail

REF=""
PROFILE="cuda-3060"
OUT=""
REPO="https://github.com/ggml-org/llama.cpp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)     REF="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --repo)    REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$REF" ]] || { echo "--ref required (git ref, or local:/abs/path)" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${here}/.llamacpp-builds"
mkdir -p "$work"

# --- resolve source tree -----------------------------------------------------
if [[ "$REF" == local:* ]]; then
  src="${REF#local:}"
  built_ref="$(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo local-worktree)"
else
  src="${work}/src"
  if [[ ! -d "$src/.git" ]]; then
    git clone "$REPO" "$src"
  fi
  git -C "$src" fetch --tags --force origin
  git -C "$src" checkout --detach "$REF"
  built_ref="$(git -C "$src" rev-parse --short HEAD)"
fi

# --- flag profiles (workflow doc §3) -----------------------------------------
common=(
    -DCMAKE_BUILD_TYPE=Release
    -G Ninja
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
    -DGGML_NATIVE=ON
    -DGGML_CCACHE=ON
)
case "$PROFILE" in
  cpu-light)
    flags=( "${common[@]}" -DGGML_CUDA=OFF )
    ;;
  cuda-3060)
    # 3060 = compute capability 8.6 / sm_86
    flags=( "${common[@]}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 )
    ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

: "${OUT:=${work}/out-${PROFILE}-${built_ref}}"
build="${work}/build-${PROFILE}-${built_ref}"
rm -rf "$build"; mkdir -p "$build" "${OUT}/lib"

echo ">> building llama.cpp ref=${built_ref} profile=${PROFILE}"
echo ">> flags: ${flags[*]}"

cmake -S "$src" -B "$build" "${flags[@]}"
# Build the shared llama library target only. The default `all` target also
# builds the `app/llama` executable; with LLAMA_BUILD_SERVER=OFF that executable
# can still reference the disabled llama-server-impl/llama-cli-impl targets and
# fail after all required libraries have already built.
cmake --build "$build" --target llama --config Release -j"$(nproc)"

# --- collect the shared-lib set ----------------------------------------------
# llama-cpp-python loads libllama + libggml{,-base,-cpu} (+ libmtmd if present).
find "$build" -name 'libllama.so*' -o -name 'libggml*.so*' -o -name 'libmtmd.so*' \
  | while read -r f; do cp -av "$f" "${OUT}/lib/"; done

# Keep the matching public headers beside the library set so cffi API-mode
# bindgen can compile against exactly the source that produced these .so files.
mkdir -p "${OUT}/include"
cp -a "${src}/include/." "${OUT}/include/"
cp -a "${src}/ggml/include/." "${OUT}/include/"

# --- build manifest ----------------------------------------------------------
ggml_ver="$(basename "$(ls "${OUT}"/lib/libggml.so.* 2>/dev/null | head -1 || echo unknown)")"
cat > "${OUT}/build-manifest.json" <<JSON
{
  "repo": "${REPO}",
  "ref_requested": "${REF}",
  "ref_built": "${built_ref}",
  "profile": "${PROFILE}",
  "cmake_flags": "${flags[*]}",
  "ggml_lib": "${ggml_ver}"
}
JSON

echo ">> done. lib set in ${OUT}/lib"
echo ">> Mode B:  export LLAMA_CPP_LIB_PATH=${OUT}/lib"
echo ">> Next:    run the ABI smoke gate (workflow doc §5) before trusting this build."
