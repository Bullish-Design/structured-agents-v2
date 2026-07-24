"""Trivial cffi API-mode smoke test — proves the toolchain (cffi + C compiler)
works end-to-end on THIS host WITHOUT needing llama.h.

API mode = cffi runs the real C compiler at build time. It emits a CPython
C-extension (`_cffi_trivial*.so`) whose C source `#include`s the target headers
and calls the declared functions directly. This is the mode we want for the
pinned llama.cpp binding, because the compiler — not us — fixes struct layout,
enum values, and calling convention. If a signature we `cdef` disagrees with the
header, the compile FAILS loudly here rather than corrupting memory at runtime
(the failure mode of hand-written ctypes drift).

This file exercises only libc (`strlen`, `snprintf`) plus a 3-line inline C
snippet, so it needs no third-party header. Success => cffi + gcc are wired up
and `llama_cffi_build.py` can run the same machinery against the real llama.h.

Run (inside the spike venv, with a C compiler on PATH):
    LD_LIBRARY_PATH="$(cat .stdcxx_dir):$LD_LIBRARY_PATH" \
      PATH="/path/to/gcc-wrapper/bin:$PATH" \
      .venv-spike/bin/python cffi_smoke_trivial.py
"""
import os
import sys

from cffi import FFI

ffibuilder = FFI()

# --- cdef: declarations the extension will expose to Python -------------------
# Note the `...;` on snprintf is cffi's "the compiler knows the rest" marker for
# variadic functions; for the fixed-arg funcs we give the exact prototype.
ffibuilder.cdef(
    r"""
    size_t strlen(const char *s);
    int add_two(int a, int b);         /* from our inline C below */
    int fill_hi(char *buf, size_t n);  /* from our inline C below */
    """
)

# --- set_source: the C that actually gets compiled ---------------------------
# In the real binding this becomes `#include "llama.h"` + libraries=["llama"].
ffibuilder.set_source(
    "_cffi_trivial",
    r"""
    #include <string.h>
    #include <stdio.h>
    static int add_two(int a, int b) { return a + b; }
    static int fill_hi(char *buf, size_t n) { return snprintf(buf, n, "hi"); }
    """,
)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = ffibuilder.compile(tmpdir=here, verbose=True)
    print(f"[build] compiled extension -> {so_path}")

    sys.path.insert(0, here)
    import _cffi_trivial  # noqa: E402  (built just above)

    lib = _cffi_trivial.lib
    ffi = _cffi_trivial.ffi

    assert lib.strlen(b"llama") == 5, "strlen mismatch"
    assert lib.add_two(20, 22) == 42, "add_two mismatch"

    buf = ffi.new("char[]", 8)
    n = lib.fill_hi(buf, 8)
    got = ffi.string(buf).decode()
    assert (n, got) == (2, "hi"), f"fill_hi mismatch: {n!r} {got!r}"

    print("[check] strlen('llama')==5  add_two(20,22)==42  fill_hi->'hi'  OK")
    print("[result] cffi API mode compiled and ran on this host.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
