# llama.cpp integration & build workflow

**Goal:** freely swap the llama.cpp underneath `llama-cpp-python` — a new upstream
release, a custom fork, or a personal build tailored to this rig for performance
and lightness — with a repeatable, ABI-safe process.

Verified against the installed `llama-cpp-python 0.3.34` (`.venv-spike`):
- `llama_cpp/llama_cpp.py:36` reads `LLAMA_CPP_LIB_PATH` to override where the
  shared libs load from → runtime swap is supported.
- The package ships a full lib set: `libllama.so`, `libggml.so`,
  `libggml-base.so`, `libggml-cpu.so`, `libmtmd.so` (ggml 0.16.0). A custom build
  must produce the matching set.
- Bindings are hand-maintained ctypes mirroring a specific `llama.h`. **This is
  the constraint everything else orbits.**

---

## 1. The ABI-anchor rule (read this first)

`llama-cpp-python`'s ctypes bindings are written by hand to match one `llama.h`
ABI — struct layouts, function signatures, enum values. They are NOT generated
from headers at install time. Therefore:

- Every `llama-cpp-python` release has an **anchor**: the exact llama.cpp commit
  it vendors. Its bindings are correct for that commit's ABI.
- A custom llama.cpp is ABI-safe iff its `llama.h` is unchanged relative to the
  anchor (same commit, or a nearby commit / your fork branched off the anchor
  that didn't touch the public ABI).
- To ride a llama.cpp far ahead of the anchor, you must **also** move
  `llama-cpp-python` to a release whose bindings track that ABI, then re-anchor.
- Failure mode if you ignore this: signature/struct drift → segfaults or, worse,
  silent memory corruption and wrong outputs. Hence the mandatory smoke gate (§5).

**Practical policy:** fork/branch off the anchor commit. Rebase your fork onto a
new anchor only together with a `llama-cpp-python` bump. Record the whole tuple
(§6).

---

## 2. Two integration modes

### Mode A — source rebuild (coordinated build) — the *pinned/distributable* path
Rebuild `llama-cpp-python` from source so its bundled lib set IS your build.
```
CMAKE_ARGS="<tailoring flags>" FORCE_CMAKE=1 \
  pip install --no-binary llama-cpp-python \
  "llama-cpp-python @ git+https://github.com/abetlen/llama-cpp-python@<lcp-ref>"
```
To use a custom llama.cpp, point the vendored submodule at your ref before build
(clone the lcp repo, `git -C vendor/llama.cpp checkout <your-ref>`, then
`pip install .`). One install, bindings + lib from one tree. This is what we pin
for benchmarks and releases.

### Mode B — runtime lib swap — the *fast-iteration* path
Keep the installed `llama-cpp-python`; build only the llama.cpp lib set and point
at it:
```
LLAMA_CPP_LIB_PATH=/path/to/your/build/lib python your_script.py
```
No reinstall; rebuild the C library and rerun. Same ABI-anchor rule applies — the
installed package's bindings must match your lib. Best for tailoring-flag sweeps
and fork experiments; promote a winner to Mode A.

---

## 3. Tailoring for this rig (performance + lightness)

Target: 2×3060 (CUDA compute capability **8.6**, sm_86), a text-gen teaching
workload. Candidate CMake flags (validate each with the bench harness):

Performance:
- `-DGGML_NATIVE=ON` — march=native for the host CPU (prefill/CPU offload paths).
- `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` — build only for the 3060s, not
  the full arch fatbin (smaller, faster to build).
- `-DGGML_CUDA_FA_ALL_QUANTS=OFF` and friends — trim CUDA kernel variants we
  don't use.
- LTO / `-DCMAKE_BUILD_TYPE=Release`.

Lightness (drop what we don't use):
- Disable unused backends: no Vulkan/SYCL/HIP/BLAS/OpenCL.
- `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF`.
- Skip `libmtmd` (multimodal) if the binding version allows it and we stay
  text-only — reduces the shipped lib set.

Every flag change is a teaching data point: rebuild → bench → record delta.

---

## 4. The workflow (repeatable)

```
1. PICK target llama.cpp ref
     (a) upstream release tag   (b) our fork branch   (c) local tailored build
2. RESOLVE anchor
     - determine the llama-cpp-python release ABI-compatible with that ref
     - if far ahead → bump llama-cpp-python too, re-anchor
3. BUILD  (build-llamacpp.sh)
     - checkout llama.cpp @ ref
     - cmake configure with tailoring flags (§3) inside devenv (toolchain + CUDA)
     - build the .so set → out dir
4. INTEGRATE
     - Mode B (iterate): export LLAMA_CPP_LIB_PATH=<out>/lib
     - Mode A (pin):     rebuild llama-cpp-python from source against the ref
5. VERIFY — ABI smoke gate (§5). MUST pass before use.
6. RECORD the tuple in versions.py (§6).
```

---

## 5. ABI smoke gate (mandatory after every swap)

A swap is only "done" when all pass:
1. `probe_api.py` — the low-level surface still resolves (functions + flags
   present, argtypes intact). Catches gross ABI/symbol drift.
2. Load Ornith GGUF + generate ~32 tokens greedily — catches struct-layout drift
   that `probe_api.py` can't see (wrong output / crash).
3. Tokenizer round-trip on the probe corpus — catches vocab/token ABI drift.
4. (When the cache pillar exists) a state save→restore round-trip — the most
   ABI-sensitive surface.
Green on 1–3 is the minimum bar to trust a new lib for development.

---

## 6. Version tuple (single source of truth)

Recorded by `versions.py` and stamped into every benchmark/diagnostic:
```
llama-cpp-python : <release or git sha>
llama.cpp anchor : <commit the bindings target>
llama.cpp built  : <commit actually built>   # == anchor for ABI safety
build flags      : <tailoring flag set / a named profile>
ggml version     : <from lib>                 # e.g. 0.16.0
xgrammar         : <release>
torch (if used)  : <release>
```
Do NOT trust version claims from the intern docs (e.g. a specific vendored
commit hash) — read the actual submodule/lib at build time and record what is
real.

---

## 7. NixOS specifics

- No `nvcc` on the host. The CUDA toolkit must come via nix/devenv (unfree),
  matching the existing spike pattern (`launch-spike.sh` used
  `NIXPKGS_ALLOW_UNFREE=1`, nix graphics-drivers lib path).
- Prebuilt-wheel `.so`s need nix `libstdc++` on `LD_LIBRARY_PATH` (recipe in
  `.stdcxx_dir`). A source build against the nix toolchain avoids that mismatch.
- Provide a devenv target that exposes gcc, cmake, and the CUDA toolkit so the
  build script runs hermetically.

---

## 8. Deliverables (fold into PLAN Phase 0)

- `build-llamacpp.sh` — parameterized by ref + flag profile; emits the lib set +
  a `build-manifest.json` (ref, flags, ggml version).
- Named build profiles: `cpu-light`, `cuda-3060`, `cuda-3060-fat` (debug).
- ABI smoke-gate script wrapping §5.
- `versions.py` tuple emitter (shared with the rest of Phase 0).
- Short "how to pull a new llama.cpp release" and "how to run my fork" runbook.
```
