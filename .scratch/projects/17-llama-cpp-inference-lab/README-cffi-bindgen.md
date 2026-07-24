# cffi bindgen — drift-proof llama.cpp bindings

`llama_cffi_build.py` builds a **cffi API-mode** Python↔C binding against a chosen
llama.cpp. Unlike llama-cpp-python's hand-written ctypes (ABI mode), the C
compiler binds the real `llama.h` against the real `libllama.so`, so the binding
**cannot silently drift** — a signature/struct change fails the compile instead
of segfaulting at runtime. The binding `_llama_cffi*.so` and the lib it was
compiled against are one paired, versioned unit. See the header comment in
`llama_cffi_build.py` for the full API-mode-vs-ABI-mode explanation, and
`06-LLAMACPP-BUILD-WORKFLOW.md §1` for the ABI-anchor rule.

## What was validated (2026-07, this host)

- `cffi_smoke_trivial.py` — trivial API-mode compile over libc + a 3-line inline
  C snippet. **Compiled and ran** (proves cffi + gcc toolchain works end-to-end,
  no llama.h needed).
- `llama_cffi_build.py --from-wheel --check` — the REAL minimal llama.h cdef
  **compiled against the wheel's bundled `llama.h` and linked against the wheel's
  `libllama.so`**, then imported and called `llama_backend_init` /
  `llama_print_system_info` / `llama_supports_gpu_offload` successfully. This was
  possible because the installed `llama-cpp-python 0.3.34` CPU wheel DOES ship the
  headers under `site-packages/include/` (see caveat below).

Toolchain used: spike venv python + `cffi 2.1.0` + `setuptools` (had to be
installed) + a nix gcc-wrapper on PATH, with `$(cat .stdcxx_dir)` and the wheel's
`llama_cpp/lib` on `LD_LIBRARY_PATH`.

## Run it locally

Against a `build-llamacpp.sh` output (the pinned path):

```bash
cd .scratch/projects/17-llama-cpp-inference-lab
./build-llamacpp.sh --ref <lcpp-ref> --profile cpu-light   # emits .llamacpp-builds/out-...
export PATH="/nix/store/<gcc-wrapper-15.2.0>/bin:$PATH"
LD_LIBRARY_PATH="$(cat .stdcxx_dir):$LD_LIBRARY_PATH" \
  .venv-spike/bin/python llama_cffi_build.py \
    --build-dir .llamacpp-builds/out-cpu-light-<sha> --profile cpu-light --check
```

Against the installed wheel's bundled header+lib (convenience/self-test):

```bash
export PATH="/nix/store/<gcc-wrapper-15.2.0>/bin:$PATH"
LD_LIBRARY_PATH="$(cat .stdcxx_dir):.venv-spike/lib/python3.13/site-packages/llama_cpp/lib:$LD_LIBRARY_PATH" \
  .venv-spike/bin/python llama_cffi_build.py --from-wheel --check
```

Explicit dirs / env vars also work: `--include-dir X --lib-dir Y`, or
`LLAMA_CFFI_INCLUDE` / `LLAMA_CFFI_LIB`. `--profile` must match how the lib was
built (it forwards preprocessor defines, e.g. `GGML_USE_CUDA` for `cuda-3060`).

## The header-not-in-wheel caveat

An API-mode compile needs `llama.h` (+ the ggml headers) at the **matching
commit**, plus `libllama.so` to link. The `0.3.34` CPU wheel here happens to ship
the headers, so `--from-wheel` works today — **but do not rely on that**. Other /
older wheels ship only the `.so` set and no header; then `--from-wheel` fails with
a clear "llama.h not found" message and the real compile is **deferred** until
`build-llamacpp.sh` has produced a source checkout+build (which always yields
`<out>/include`). The pinned path is `--build-dir`, not `--from-wheel`.

## What a BINDGEN failure means

The CI `BINDGEN` stage (`ci/llama-cpp-bindgen.yml`) running `llama_cffi_build.py`
is the **ABI-drift gate**. A non-zero exit there means our `cdef` no longer
matches this `llama.h`/`libllama.so` — a struct was reordered, a signature
changed, or a symbol was renamed/removed. That is a *good* failure: it stops a
mismatched binding before it can corrupt memory. The fix is to reconcile the cdef
against the new header (rename the symbol, adjust the prototype), together with
the llama-cpp-python re-anchor per the workflow doc. The `nightly-canary` job runs
the same BINDGEN against llama.cpp `master` (allowed to fail) purely as early
warning that upstream moved the ABI.

## Known reconciliation point (real drift found)

The shipped ctypes binding still exposes the **old** `llama_set_adapter_lora(ctx,
adapter, scale)`, but `0.3.34`'s `libllama.so` actually exports the **new** batch
API `llama_set_adapters_lora(ctx, adapters**, n_adapters, scales*)`. The cdef
binds the real (plural) symbol. This is the ABI-drift risk in miniature and the
clearest example of why API mode is safer than the hand-mirrored ctypes.
