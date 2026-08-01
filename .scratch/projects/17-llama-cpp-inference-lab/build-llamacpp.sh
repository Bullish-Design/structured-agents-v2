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
PATCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)     REF="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --repo)    REPO="$2"; shift 2 ;;
    --patch)   PATCH="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${here}/.llamacpp-builds"
mkdir -p "$work"

# --- p2fork profile: pinned base + private mixed-batch multi-LoRA patch -------
# (Project 20, Workstream A) The fork is a private build of the pinned anchor
# commit with patches/p2-mixed-batch-lora.patch applied. No llama_batch ABI
# change, so it stays ABI-compatible with the installed llama-cpp-python 0.3.34
# bindings. This profile bakes in the ref + patch so one command reproduces the
# fork lib with a manifest recording base commit + patch sha256.
patch_sha=""
if [[ "$PROFILE" == p2fork ]]; then
  : "${REF:=c588c4f47}"                       # pinned anchor for llama-cpp-python 0.3.34
  : "${PATCH:=${here}/patches/p2-mixed-batch-lora.patch}"
  [[ -f "$PATCH" ]] || { echo "p2fork: patch not found: $PATCH" >&2; exit 2; }
  patch_sha="$(sha256sum "$PATCH" | cut -d' ' -f1)"
fi
[[ -n "$REF" ]] || { echo "--ref required (git ref, or local:/abs/path)" >&2; exit 2; }

# --- resolve source tree -----------------------------------------------------
if [[ "$REF" == local:* ]]; then
  src="${REF#local:}"
  built_ref="$(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo local-worktree)"
else
  # p2fork uses a dedicated src checkout so the patch never collides with an
  # unpatched cuda-3060 build sharing ${work}/src.
  if [[ "$PROFILE" == p2fork ]]; then
    src="${work}/src-p2fork"
  else
    src="${work}/src"
  fi
  if [[ ! -d "$src/.git" ]]; then
    git clone "$REPO" "$src"
  fi
  git -C "$src" fetch --tags --force origin
  git -C "$src" reset --hard >/dev/null 2>&1 || true
  git -C "$src" checkout --detach "$REF"
  git -C "$src" reset --hard "$REF" >/dev/null 2>&1 || true
  git -C "$src" clean -fd >/dev/null 2>&1 || true
  built_ref="$(git -C "$src" rev-parse --short HEAD)"
  # apply the private fork patch onto the clean pinned tree
  if [[ -n "$PATCH" ]]; then
    echo ">> verifying patch applies against pinned tree ${built_ref}: $PATCH"
    if ! git -C "$src" apply --check "$PATCH"; then
      echo "!! patch does NOT apply cleanly against ${built_ref}." >&2
      echo "!! The llama-cpp-python anchor may have moved; re-anchor the patch." >&2
      exit 3
    fi
    git -C "$src" apply "$PATCH"
    echo ">> patch applied (sha256=${patch_sha})"
  fi
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
  cuda-3060 | p2fork)
    # 3060 = compute capability 8.6 / sm_86. p2fork uses the identical CUDA flags
    # (ABI-compatible with the stock cuda-3060 lib); only the source tree differs.
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
  "ggml_lib": "${ggml_ver}",
  "ggml_version": "${ggml_ver}",
  "llama_cpp_commit": "${built_ref}",
  "build_id": "${PROFILE}-${built_ref}",
  "patch": "${PATCH}",
  "patch_sha256": "${patch_sha}",
  "seq_adapter_routing": $([[ -n "$patch_sha" ]] && echo true || echo false)
}
JSON

echo ">> done. lib set in ${OUT}/lib"
echo ">> Mode B:  export LLAMA_CPP_LIB_PATH=${OUT}/lib"
echo ">> Next:    run the ABI smoke gate (workflow doc §5) before trusting this build."
