# Build investigation report — 2026-07-24

## Failure boundary

The first CUDA build compiled the llama.cpp libraries and CUDA kernels, then
failed while linking the separate `app/llama` executable. The failure was:

```text
cannot find -lllama-server-impl
cannot find -llama-cli-impl
```

The configured flags intentionally set `LLAMA_BUILD_SERVER=OFF` and
`LLAMA_BUILD_EXAMPLES=OFF`. The required runtime libraries were already
present, and the subsequent GPU smoke test loaded Ornith and generated
coherently.

## Local source evidence

At the pinned source commit, `src/CMakeLists.txt` declares
`add_library(llama ...)`. The failing target was `app/llama`, not that library.
Therefore the minimal fix is to invoke:

```sh
cmake --build "$build" --target llama
```

instead of building the default aggregate target.

## Chosen fix

The build script now:

1. selects the `llama` shared-library target explicitly;
2. uses the Ninja generator;
3. enables llama.cpp's ccache integration; and
4. exposes a persistent ccache directory from the CUDA shell.

The target change is required for correctness. Ninja and ccache are isolated
build-performance changes and do not alter model or kernel semantics.

## Post-fix verification

A fresh rebuild with the fixed script completed all 344 Ninja steps and exited
0. It produced `libllama.so`, the CUDA/CPU/base ggml libraries, and copied the
matching public headers into the output artifact. The cffi API-mode bindgen then
compiled and imported successfully against that artifact, reporting
`supports_gpu_offload=True` and CUDA architecture 860.

The same rebuilt library set passed the 1024-context Ornith GPU smoke test:
44.67 generated tokens/s, 922.7 prompt tokens/s, coherent output, and the same
~2.6/3.6 GiB per-GPU model footprint. The first rerun without the host driver
directory produced the expected NixOS `CUDA driver is a stub library` failure;
rerunning with the recorded `.cuda_runtime_ld` path passed.

The second clean rebuild achieved 334 direct ccache hits out of 668 cacheable
calls (50% overall), confirming that the persistent cache is active. The first
build was cold and had 334 misses.

## Primary upstream references

- [llama.cpp b10103 top-level CMake configuration](https://github.com/ggml-org/llama.cpp/blob/b10103/CMakeLists.txt) — project build options and subdirectories.
- [llama.cpp b10103 library target](https://github.com/ggml-org/llama.cpp/blob/b10103/src/CMakeLists.txt) — defines the `llama` shared-library target.

## Removal condition

The explicit target workaround can be removed if a future llama.cpp release
makes the aggregate target valid with server and examples disabled, and a
clean build verifies that behavior. Ninja/ccache should remain unless measured
regressions show they are harmful for this machine.

## Evidence limits

The original failure log and post-fix artifacts are preserved under
`artifacts/20260724-postfix/`. The initial smoke measured 46.95 tok/s and
976.8 tok/s prefill; the post-fix rerun measured 44.67 tok/s and 922.7 tok/s.
These are valid only for their recorded smoke configurations and are not a
controlled performance comparison.
