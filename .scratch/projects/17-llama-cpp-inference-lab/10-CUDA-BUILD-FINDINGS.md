# CUDA build and 1024-context smoke findings — 2026-07-24

## Scope

This records the completed Phase-0 CUDA spike for the `cuda-3060` profile and
the 1024-context Ornith run. The fixed inputs were:

- llama.cpp ref `b10103`, resolved to commit `c588c4f47683e73ad2d69f50480bec6cc85fd0f7`
- CUDA 12.9, `CMAKE_CUDA_ARCHITECTURES=86`
- two NVIDIA RTX 3060 GPUs, with the 9B Q4 GGUF split across both devices
- `n_ctx=1024`, model `/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf`

## Result

The CUDA backend built successfully for the required shared-library set. The
original default `all` build exited at the final `app/llama` executable link:

```text
cannot find -lllama-server-impl
cannot find -llama-cli-impl
```

This was not a CUDA or model failure. The required `libllama.so` and
`libggml*.so` libraries had already built. The build script now requests the
library target `llama` explicitly, avoiding the unrelated CLI target.

After the target fix, a fresh Ninja rebuild completed all 344 steps with exit
code 0. The rebuilt artifact contains the shared libraries and matching public
headers for cffi bindgen. The post-fix GPU smoke test then passed:

| Measurement | Result |
|---|---:|
| CUDA devices | 2 × RTX 3060, compute capability 8.6 |
| Model load | 6.3 s |
| Context | 1024 tokens |
| VRAM after load | 2595 MiB / 3585 MiB |
| Generation | 64 tokens in 1.43 s = 44.67 tok/s |
| Prefill | 480 tokens in 0.52 s = 922.7 tok/s |
| Coherence | Passed; output included the expected `Paris` answer |

CUDA graph warmup and reuse also completed. The cffi bindgen smoke compiled
against the rebuilt artifact's header/library pair and passed
`llama_backend_init/free`, reporting GPU offload support.

## Build-speed changes

The build workflow now uses Ninja and persistent ccache when run through
`cuda-shell.nix`. The cache defaults to `$HOME/.cache/llamacpp-ccache` and can
be overridden with `CCACHE_DIR`. A second clean build recorded 334 direct hits
out of 668 cacheable calls (50% overall; the first build was cold).

## Evidence

- Raw build log: `build-b10103.log`
- Raw GPU log: `gpu_smoke.log`
- Fresh post-fix artifacts: `artifacts/20260724-postfix/`
- Collected library manifest: `.llamacpp-builds/out-cuda-3060-c588c4f47/build-manifest.json`
- Smoke harness: `gpu_smoke.py`
- Upstream library target definition: [llama.cpp b10103 `src/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/b10103/src/CMakeLists.txt)
- Upstream build entry point: [llama.cpp b10103 `CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/b10103/CMakeLists.txt)

## What this does not prove

This proves that the selected build can load and generate with Ornith on this
two-GPU system, that the fixed library-only build path exits cleanly, that the
artifact can compile the cffi binding, and that the binding can initialize the
CUDA backend. It does not yet prove multi-LoRA scheduling, xgrammar
correctness, KV-cache break-even, or performance under batching.
