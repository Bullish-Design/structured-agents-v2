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
  silent memory corruption and wrong outputs. Hence the mandatory smoke gate (§6).

**Practical policy:** fork/branch off the anchor commit. Rebase your fork onto a
new anchor only together with a `llama-cpp-python` bump. Record the whole tuple
(§7).

---

## 2. Binding generation strategy (own the ABI, don't re-type it)

The ABI-anchor rule (§1) is a workaround for a deeper fragility: the low-level
bindings hand-**re-assert** the C ABI in Python. `llama_cpp/llama_cpp.py` is a
hand-maintained ctypes transcription of one `llama.h` — struct field orders,
enum values, `argtypes`/`restype` — mirrored by eye. When the built header and
the Python transcription disagree, most disagreements are **silent**: a wrong
struct offset reads the wrong bytes → memory corruption or plausible-but-wrong
outputs, with no exception. Only a symbol *rename or removal* fails loudly (an
unresolved lookup). Everything subtler slips through to runtime.

Generating the binding **from the exact header you build against** makes ABI
mismatch structurally impossible rather than a discipline you enforce by hand.
Three approaches:

| Approach | Build step | ABI truth source | Drift is silent? | You find out about drift... |
| --- | --- | --- | --- | --- |
| Hand-written ctypes (status quo, what llama-cpp-python ships) | none | a human reading `llama.h` | **yes** (except symbol rename/removal) | in production / demo — wrong output or corruption |
| Generated ctypes (`clang2py` / `ctypesgen`) | codegen only (no compile) | libclang parse of the real `llama.h` at generation time | yes, if the runtime lib differs from the header it was generated from | in production / demo — regeneration is automated, but it's still a runtime-libffi snapshot per header |
| **cffi API mode (RECOMMENDED — pinned path)** | **C compile** | the **C compiler** resolving `#include "llama.h"` against that build | **no — cannot silently drift** | **CI build time — a compile error** |

The recommended path is **cffi API mode**: `ffi.set_source(...)` emits C that
`#include`s the real `llama.h`, and `ffi.compile()` invokes the actual C
compiler to build a CPython extension module. The compiler — not a human, not a
one-time parse — derives every struct offset, enum value, and call signature
from the same header the library was built from. You declare *intent* with
cffi's `...` ("dotdotdot": incomplete structs, unknown enum values, unspecified
array sizes) and the compiler fills in the **binary truth**. A field that moved,
an enum that shifted, a signature that changed → the compile fails, loudly, at
build time. That early-failure property is the main reason to adopt it.

The generated-ctypes middle option is strictly better than hand-writing (the
transcription is automated from the real header), but it still ships a
runtime-libffi snapshot: run it against a lib other than the one it was
generated from and silent drift returns. Only the compile step closes that gap.

Costs, stated honestly:

- **Needs a compiler + header + lib at build time.** cffi API mode is not a
  pure-Python install; it's a compile.
- **Binding + library become a UNIT** — the extension is compiled against one
  header, so it is only valid for a lib of that ABI. This tightens Mode B (§3):
  the rigorous pinned model becomes "rebuild the binding **and** the lib
  together"; loose `LLAMA_CPP_LIB_PATH` swapping stays fine **only** for
  same-ABI experiments (the ABI-anchor rule still gates it).
- **It's a compiled, per-platform artifact** (here: linux / x86-64 / CUDA
  sm_86), not a universal wheel. For a self-hosted teaching rig that already
  builds llama.cpp from source this is the right tradeoff — but it is a real
  constraint, not a free lunch.

### Two adoption shapes

- **Hybrid (start here):** keep `llama-cpp-python`'s high-level `Llama` for
  model lifecycle and ergonomics; add a cffi low-level module only for the
  surfaces we own on the hot path — logits access, the sampler chain, the state
  save/restore APIs, LoRA adapter control. Smallest surface, immediate
  ABI-safety where it matters most.
- **Full:** own the entire low-level cffi layer plus a thin ergonomic wrapper.
  This is the "own the substrate" teaching thesis carried to its end, but it's a
  larger build and takes ownership of surfaces the high-level package handles
  today.

Start Hybrid; graduate surfaces to Full as the pillars justify it.

---

## 3. Two integration modes

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

## 4. Tailoring for this rig (performance + lightness)

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

## 5. The workflow (repeatable)

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

## 6. ABI smoke gate (mandatory after every swap)

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

## 7. Version tuple (single source of truth)

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

## 8. NixOS specifics

- No `nvcc` on the host. The CUDA toolkit must come via nix/devenv (unfree),
  matching the existing spike pattern (`launch-spike.sh` used
  `NIXPKGS_ALLOW_UNFREE=1`, nix graphics-drivers lib path).
- Prebuilt-wheel `.so`s need nix `libstdc++` on `LD_LIBRARY_PATH` (recipe in
  `.stdcxx_dir`). A source build against the nix toolchain avoids that mismatch.
- Provide a devenv target that exposes gcc, cmake, and the CUDA toolkit so the
  build script runs hermetically.

---

## 9. Deliverables (fold into PLAN Phase 0)

- `build-llamacpp.sh` — parameterized by ref + flag profile; emits the lib set +
  a `build-manifest.json` (ref, flags, ggml version).
- Named build profiles: `cpu-light`, `cuda-3060`, `cuda-3060-fat` (debug).
- ABI smoke-gate script wrapping §5.
- `versions.py` tuple emitter (shared with the rest of Phase 0).
- Short "how to pull a new llama.cpp release" and "how to run my fork" runbook.
```
