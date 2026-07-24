# Build-speed optimization (llama.cpp CUDA builds)

Measured env (2026-07-24): **8 cores, 62 GB RAM (48 free), NVMe btrfs /home
(297 GB free), /tmp on the same NVMe btrfs (not tmpfs).** ccache NOT installed;
ninja present in nix store; build uses Make + `-j$(nproc)`=`-j8`.

The long pole is **nvcc compiling the CUDA kernels** (`fattn.cu`, `ggml-cuda.cu`,
mmq, template explosion). Levers, most impactful first.

## 1. ccache — THE big win (missing today)
nvcc object compiles are cacheable. llama.cpp's CMake has `GGML_CCACHE=ON` by
default and auto-wires ccache as the compiler launcher (C/C++/CUDA) when it finds
ccache on PATH. `build-llamacpp.sh` does `rm -rf` the build dir each run, so a
**persistent CCACHE_DIR** is what makes it pay off — cached objects survive the
wipe and serve near-instantly across ref/flag-sweep rebuilds.

Done already: created `/home/andrew/.cache/llamacpp-ccache`.

To activate (apply after the in-flight build/agent finishes — see "coordination"):
- Add `ccache` to the CUDA nix-shell (`cuda-shell.nix` buildInputs).
- Export in the build shell:
  ```
  export CCACHE_DIR=/home/andrew/.cache/llamacpp-ccache
  export CCACHE_MAXSIZE=25G
  ```
- Nothing else needed — GGML_CCACHE default ON picks ccache up. (Belt-and-braces:
  pass `-DGGML_CCACHE=ON`.)
Expected: first build cold (no change); subsequent builds of the same/nearby ref
or a different flag profile drop from ~tens-of-minutes to a few minutes for the
unchanged CUDA TUs. Verify with `ccache -s` (hit rate).

## 2. Ninja generator (present in nix store)
Make is a weaker scheduler than Ninja for large parallel graphs. Switch:
- `cmake -S "$src" -B "$build" -G Ninja "${flags[@]}"`
- Add `ninja` to the nix-shell buildInputs.
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

## Coordination note
Do NOT edit `cuda-shell.nix` / `build-llamacpp.sh` while the CUDA agent is live —
it re-enters `cuda-shell.nix` for the GPU smoke test. Apply §1–2 after the agent
completes and its work is committed, then trigger one rebuild to warm the cache
and confirm `ccache -s` hits on a second build.
