# Build-speed optimization (llama.cpp CUDA builds)

Measured env (2026-07-24): **8 cores, 62 GB RAM (48 free), NVMe btrfs /home
(297 GB free), /tmp on the same NVMe btrfs (not tmpfs).** ccache and Ninja are
now provided by the project CUDA shell; the build uses Ninja with `-j8`.

The long pole is **nvcc compiling the CUDA kernels** (`fattn.cu`, `ggml-cuda.cu`,
mmq, template explosion). Levers, most impactful first.

## 1. ccache — THE big win
nvcc object compiles are cacheable. llama.cpp's CMake has `GGML_CCACHE=ON` by
default and auto-wires ccache as the compiler launcher (C/C++/CUDA) when it finds
ccache on PATH. `build-llamacpp.sh` does `rm -rf` the build dir each run, so a
**persistent CCACHE_DIR** is what makes it pay off — cached objects survive the
wipe and serve near-instantly across ref/flag-sweep rebuilds.

Done already: created `/home/andrew/.cache/llamacpp-ccache`.

Activated in `cuda-shell.nix`; the build shell exports a persistent default. To
override it:
  ```
  export CCACHE_DIR=/home/andrew/.cache/llamacpp-ccache
  export CCACHE_MAXSIZE=25G
  ```
- Nothing else needed — GGML_CCACHE default ON picks ccache up. (Belt-and-braces:
  pass `-DGGML_CCACHE=ON`.)
The first post-change build was cold; the second clean build recorded 334 direct
hits out of 668 cacheable calls (50% overall). Verify future runs with
`ccache -s`.

## 2. Ninja generator
Ninja is now enabled in `build-llamacpp.sh` and provided by `cuda-shell.nix`.
Faster dependency scheduling + much faster null/incremental builds. Modest but
free.

## 3. Keep the arch/kernel trims we already have
- `-DCMAKE_CUDA_ARCHITECTURES=86` (single arch, not a fat binary) — already set,
  already a large saving vs the default multi-arch.
- `-DGGML_CUDA_FA_ALL_QUANTS=OFF` (default) — leave OFF; ON compiles every
  flash-attn quant combo and is a major time sink. Only turn on if a benchmark
  needs an otherwise-missing quant path.
- tests/examples/server already OFF in the profile.

## 4. -j is already optimal
8 cores → `-j8`. Don't oversubscribe (nvcc is CPU-bound; more jobs thrash). RAM
is not the constraint here.

## 5. Optional / marginal
- **tmpfs build dir:** 48 GB free RAM could host the build tree in RAM to cut
  small-file + link I/O. Compute-bound nvcc means the gain is small; skip unless
  ccache+ninja aren't enough. (nvcc already stages intermediates in /tmp.)
- **nvcc `--threads`** (`-t0`): parallelizes per-arch codegen; with a single arch
  the benefit is limited.
- **Incremental (drop `rm -rf`):** faster than clean builds, but risks stale
  state across refs. ccache gives ~the same speed with clean-build correctness —
  prefer ccache over incremental.
- **Prebuilt CUDA wheel:** when we DON'T need a custom llama.cpp, skip building
  entirely and use the upstream cu wheel; reserve source builds for custom/fork/
  tailored work.

## Net recommendation
Add **ccache (persistent dir)** + **Ninja** to the CUDA build shell. Those two are
the whole game for our rebuild-often workflow; everything else is already right.

## Verification note
The rebuild and GPU smoke evidence are recorded in `10-CUDA-BUILD-FINDINGS.md`
and the dated local artifacts under `artifacts/20260724-postfix/`.
