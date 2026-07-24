#!/usr/bin/env python3
"""Build drift-proof Python<->C bindings for llama.cpp with cffi API mode.

================================ WHY API MODE ================================
llama-cpp-python ships HAND-WRITTEN ctypes bindings (ABI mode): a human mirrors
each struct layout, enum value, and function prototype from one specific
`llama.h`. Nothing checks that the mirror still matches the .so you actually
load. When upstream reorders a struct field or changes a signature, ctypes keeps
calling the old layout -> segfault, or worse, silent memory corruption and wrong
tokens. That is the exact failure mode 06-LLAMACPP-BUILD-WORKFLOW.md §1 warns
about ("the ABI-anchor rule").

cffi *API mode* removes the human mirror. At build time cffi generates a small
CPython C-extension whose source literally `#include "llama.h"` and calls the
real functions. The **C compiler** — not us — fills in struct offsets, enum
values, and the calling convention. Our `cdef()` only lists *which* symbols we
want and their prototypes; wherever we write `...` cffi asks the compiler for the
truth (struct layouts, macro values). Consequences:

  * It CANNOT silently drift. If our prototype disagrees with the header, or a
    symbol vanished, the **compile fails loudly** (that is the ABI-drift signal
    the CI BINDGEN stage keys on). You never get a mismatched binding at runtime.
  * The binding and the lib are a PAIRED UNIT. `_llama_cffi*.so` is compiled
    against one `llama.h` and linked against one `libllama.so`. Ship/pin them
    together; rebuilding the lib at a new ref means rebuilding this binding.
  * Real drift discovered while writing this file: the wheel's `libllama.so`
    exports `llama_set_adapters_lora(ctx, adapters**, n, scales*)` (plural, new
    batch API), but the shipped ctypes binding still exposes the *old* singular
    `llama_set_adapter_lora(ctx, adapter, scale)` name. API mode forces us onto
    the real symbol; ctypes let the stale name linger. This is the teaching
    point in miniature.

============================ HEADER-NOT-ALWAYS-IN-WHEEL =======================
A real API-mode compile needs `llama.h` (+ ggml.h etc.) at the MATCHING commit,
plus `libllama.so` to link. Where they come from:

  * Preferred / pinned path: a llama.cpp checkout built by build-llamacpp.sh,
    which emits `<out>/include` and `<out>/lib`. Point --build-dir there.
  * Convenience: some llama-cpp-python wheels DO bundle the headers under
    `site-packages/include` (the 0.3.34 CPU wheel in .venv-spike does — see
    --from-wheel). Older/other wheels ship only the .so set and NO header; then
    the real compile is DEFERRED until build-llamacpp.sh has produced a checkout.
    Do not assume the header is present — this script checks and says so.

============================ ABI-anchor / defines ============================
The preprocessor defines passed here MUST match how the lib was built (profile).
A CUDA lib built with -DGGML_CUDA=ON exposes CUDA-guarded declarations in the
ggml headers; compiling the binding with a mismatched view can shift what the
compiler sees. We forward a small, explicit define set per profile and keep it
next to the build profiles in build-llamacpp.sh. Extend as first compiles reveal
what each header path actually gates.

Usage:
    # against a build-llamacpp.sh output:
    python llama_cffi_build.py --build-dir .llamacpp-builds/out-cpu-light-<sha>
    # against the installed wheel's bundled header+lib (convenience/self-test):
    python llama_cffi_build.py --from-wheel --check
    # explicit dirs + drift check:
    python llama_cffi_build.py --include-dir X --lib-dir Y --profile cuda-3060 --check

Run inside the spike venv with a C compiler on PATH and (on NixOS) nix
libstdc++ on LD_LIBRARY_PATH:
    LD_LIBRARY_PATH="$(cat .stdcxx_dir):$LD_LIBRARY_PATH" \
      PATH="/nix/store/<gcc-wrapper>/bin:$PATH" \
      .venv-spike/bin/python llama_cffi_build.py --from-wheel --check
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cffi import FFI

MODULE_NAME = "_llama_cffi"

# --- profile -> extra preprocessor defines -----------------------------------
# Keep in lockstep with build-llamacpp.sh's flag profiles. These are a starting
# point; the FIRST real compile against a given header will tell you if the
# header needs a backend define to expose/hide a declaration. Comment additions.
PROFILE_DEFINES: dict[str, list[tuple[str, str | None]]] = {
    # CPU-only lib: no CUDA symbols compiled into ggml; nothing extra needed.
    "cpu-light": [],
    # CUDA lib for the 3060s (sm_86). GGML_USE_CUDA gates CUDA-only bits in some
    # ggml headers; forward it so the binding's header view matches the lib.
    "cuda-3060": [("GGML_USE_CUDA", "1")],
}


# ------------------------------------------------------------------ cdef ------
# MINIMAL, REAL subset mapped to what project 17 actually needs (probe_api.py +
# PLAN Phases 1-3): backend/model/context init, decode + zero-copy logits, the
# sampler chain, vocab/tokenize/detokenize/eog, per-sequence state save/restore
# (the cache pillar), and LoRA adapters (the flagship). Signatures below were
# copied from the real llama.h in the 0.3.34 wheel; lines flagged RECONCILE are
# the ones most likely to shift across refs — the compile will catch it.
#
# `...` (dotdotdot) => "compiler, fill this in": used for the param structs
# passed by value (layout), the llama_batch struct (layout), and the STATE_SEQ
# flag macros (values). Opaque handles are declared as bodyless `struct` types;
# we only ever hold pointers to them, so cffi never needs their layout.
CDEF = r"""
    /* --- scalar typedefs (stable; int32 in every ref to date) --- */
    typedef int32_t  llama_pos;
    typedef int32_t  llama_token;
    typedef int32_t  llama_seq_id;
    typedef uint32_t llama_state_seq_flags;

    /* --- opaque handles: pointer-only, no layout needed --- *
     * The real llama.h declares these as bare `struct X` with NO bare-name
     * typedef, so we must reference them as `struct X` throughout (a plain
     * `typedef struct llama_model llama_model;` here makes cffi emit an
     * undefined bare type -> compile error; that mismatch is caught, not run). */
    struct llama_model;
    struct llama_context;
    struct llama_vocab;
    struct llama_sampler;
    struct llama_adapter_lora;

    /* --- param structs: passed BY VALUE, layout supplied by compiler --- *
     * Also bare `struct X` in the header (no bare-name typedef). */
    struct llama_model_params          { ...; };
    struct llama_context_params        { ...; };
    struct llama_sampler_chain_params  { ...; };

    /* --- llama_batch: real fields, but let compiler own the exact layout --- *
     * The named fields are the ones we touch on the hot path (decode loop).
     * `...;` lets cffi verify/pad against the true struct. Confirmed fields in
     * 0.3.34: n_tokens, token, embd, pos, n_seq_id, seq_id, logits. */
    typedef struct llama_batch {
        int32_t        n_tokens;
        llama_token   *token;
        float         *embd;
        llama_pos     *pos;
        int32_t       *n_seq_id;
        llama_seq_id **seq_id;
        int8_t        *logits;
        ...;
    } llama_batch;

    /* --- STATE_SEQ flag macros: values supplied by compiler --- */
    #define LLAMA_STATE_SEQ_FLAGS_NONE ...
    #define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY ...
    #define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY ...
    #define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE ...

    /* --- backend lifecycle --- */
    void llama_backend_init(void);
    void llama_backend_free(void);
    const char * llama_print_system_info(void);
    bool llama_supports_gpu_offload(void);

    /* --- default params (return structs by value) --- */
    struct llama_model_params         llama_model_default_params(void);
    struct llama_context_params       llama_context_default_params(void);
    struct llama_sampler_chain_params llama_sampler_chain_default_params(void);

    /* --- model / context lifecycle --- */
    /* RECONCILE: load/init were renamed from the *_with_model era; these are the
     * current (non-deprecated) names in 0.3.34. */
    struct llama_model *   llama_model_load_from_file(const char *path_model, struct llama_model_params params);
    void                   llama_model_free(struct llama_model *model);
    struct llama_context * llama_init_from_model(struct llama_model *model, struct llama_context_params params);
    void                   llama_free(struct llama_context *ctx);
    uint32_t               llama_n_ctx(const struct llama_context *ctx);

    /* --- vocab --- */
    const struct llama_vocab * llama_model_get_vocab(const struct llama_model *model);
    int32_t                    llama_vocab_n_tokens(const struct llama_vocab *vocab);
    bool                       llama_vocab_is_eog(const struct llama_vocab *vocab, llama_token token);
    llama_token                llama_vocab_bos(const struct llama_vocab *vocab);
    llama_token                llama_vocab_eos(const struct llama_vocab *vocab);

    /* --- decode + zero-copy logits (Phase 1 hot path) --- */
    llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    llama_batch llama_batch_get_one(llama_token *tokens, int32_t n_tokens);
    void        llama_batch_free(llama_batch batch);
    int32_t     llama_decode(struct llama_context *ctx, llama_batch batch);
    float *     llama_get_logits_ith(struct llama_context *ctx, int32_t i);

    /* --- tokenize / detokenize --- */
    int32_t llama_tokenize(const struct llama_vocab *vocab, const char *text, int32_t text_len,
                           llama_token *tokens, int32_t n_tokens_max,
                           bool add_special, bool parse_special);
    int32_t llama_token_to_piece(const struct llama_vocab *vocab, llama_token token,
                                 char *buf, int32_t length, int32_t lstrip, bool special);

    /* --- sampler chain (owned decode loop) --- */
    struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
    void                   llama_sampler_chain_add(struct llama_sampler *chain, struct llama_sampler *smpl);
    struct llama_sampler * llama_sampler_init_greedy(void);
    struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
    struct llama_sampler * llama_sampler_init_top_k(int32_t k);
    struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
    struct llama_sampler * llama_sampler_init_temp(float t);
    llama_token            llama_sampler_sample(struct llama_sampler *smpl, struct llama_context *ctx, int32_t idx);
    void                   llama_sampler_accept(struct llama_sampler *smpl, llama_token token);
    void                   llama_sampler_free(struct llama_sampler *smpl);

    /* --- per-sequence state save/restore (Phase 2 cache pillar) --- */
    size_t llama_state_seq_get_size(struct llama_context *ctx, llama_seq_id seq_id);
    size_t llama_state_seq_get_data(struct llama_context *ctx, uint8_t *dst, size_t size, llama_seq_id seq_id);
    size_t llama_state_seq_set_data(struct llama_context *ctx, const uint8_t *src, size_t size, llama_seq_id dest_seq_id);

    /* --- LoRA adapters (Phase 3 flagship) --- *
     * RECONCILE / DRIFT NOTE: the shipped ctypes binding exposes the OLD name
     * `llama_set_adapter_lora(ctx, adapter, scale)`, but the 0.3.34 libllama.so
     * actually exports the NEW batch API below. API mode binds the real symbol. */
    struct llama_adapter_lora * llama_adapter_lora_init(struct llama_model *model, const char *path_lora);
    void                        llama_adapter_lora_free(struct llama_adapter_lora *adapter);
    int32_t                     llama_set_adapters_lora(struct llama_context *ctx, struct llama_adapter_lora **adapters,
                                                        size_t n_adapters, float *scales);
"""


def build(include_dir: Path, lib_dir: Path, profile: str, out_dir: Path) -> Path:
    if profile not in PROFILE_DEFINES:
        raise SystemExit(f"unknown profile {profile!r}; known: {sorted(PROFILE_DEFINES)}")

    header = include_dir / "llama.h"
    if not header.is_file():
        raise SystemExit(
            f"llama.h not found under {include_dir} .\n"
            "  The header is required for an API-mode compile and is NOT shipped by\n"
            "  every llama-cpp-python wheel. Run build-llamacpp.sh to produce a\n"
            "  checkout+build (emits <out>/include and <out>/lib), then point\n"
            "  --build-dir at it. This is the deferred real-header compile."
        )

    ffibuilder = FFI()
    ffibuilder.cdef(CDEF)
    ffibuilder.set_source(
        MODULE_NAME,
        '#include "llama.h"',
        libraries=["llama"],
        include_dirs=[str(include_dir)],
        library_dirs=[str(lib_dir)],
        # embed an rpath so the built .so finds libllama + its ggml deps at import
        # without needing LD_LIBRARY_PATH for THEM (nix libstdc++ is separate).
        extra_link_args=[f"-Wl,-rpath,{lib_dir}"],
        define_macros=[(name, val) for name, val in PROFILE_DEFINES[profile]],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = ffibuilder.compile(tmpdir=str(out_dir), verbose=True)
    return Path(so_path)


def self_check(out_dir: Path) -> None:
    """Import the freshly built module and call a couple of harmless funcs."""
    sys.path.insert(0, str(out_dir))
    mod = __import__(MODULE_NAME)
    lib = mod.lib
    ffi = mod.ffi
    lib.llama_backend_init()
    info = ffi.string(lib.llama_print_system_info()).decode(errors="replace")
    gpu = bool(lib.llama_supports_gpu_offload())
    lib.llama_backend_free()
    print(f"[check] llama_backend_init/free OK; supports_gpu_offload={gpu}")
    print(f"[check] system info: {info.strip()[:120]}...")


def _wheel_dirs() -> tuple[Path, Path]:
    import llama_cpp

    root = Path(llama_cpp.__file__).resolve().parent
    site = root.parent
    include = site / "include"
    # 0.3.34 lays libs under llama_cpp/lib; fall back to site/lib64.
    for cand in (root / "lib", site / "lib64", site / "lib"):
        if (cand / "libllama.so").exists() or list(cand.glob("libllama.so*")):
            return include, cand
    raise SystemExit("could not locate libllama.so under the installed wheel")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--build-dir", type=Path,
                     help="build-llamacpp.sh output dir (expects <dir>/include and <dir>/lib)")
    src.add_argument("--from-wheel", action="store_true",
                     help="use the installed llama-cpp-python's bundled header+lib (if present)")
    ap.add_argument("--include-dir", type=Path, help="override include dir (has llama.h)")
    ap.add_argument("--lib-dir", type=Path, help="override lib dir (has libllama.so)")
    ap.add_argument("--profile", default="cpu-light", choices=sorted(PROFILE_DEFINES),
                    help="match the build-llamacpp.sh profile the lib was built with")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="where to emit _llama_cffi*.so (default: this dir)")
    ap.add_argument("--check", action="store_true", help="import the built module and self-test")
    # env-var fallbacks
    ap.set_defaults(
        include_dir=None if os.environ.get("LLAMA_CFFI_INCLUDE") is None else Path(os.environ["LLAMA_CFFI_INCLUDE"]),
    )
    args = ap.parse_args(argv)

    if args.include_dir and args.lib_dir:
        include_dir, lib_dir = args.include_dir, args.lib_dir
    elif args.from_wheel:
        include_dir, lib_dir = _wheel_dirs()
    elif args.build_dir:
        include_dir = args.build_dir / "include"
        lib_dir = args.build_dir / "lib"
    elif os.environ.get("LLAMA_CFFI_INCLUDE") and os.environ.get("LLAMA_CFFI_LIB"):
        include_dir = Path(os.environ["LLAMA_CFFI_INCLUDE"])
        lib_dir = Path(os.environ["LLAMA_CFFI_LIB"])
    else:
        ap.error("supply --build-dir, --from-wheel, or --include-dir + --lib-dir "
                 "(or LLAMA_CFFI_INCLUDE / LLAMA_CFFI_LIB env vars)")

    print(f"[build] include={include_dir}")
    print(f"[build] lib    ={lib_dir}")
    print(f"[build] profile={args.profile}  defines={PROFILE_DEFINES[args.profile]}")

    so_path = build(include_dir, lib_dir, args.profile, args.out_dir)
    print(f"[build] compiled extension -> {so_path}")

    if args.check:
        self_check(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
