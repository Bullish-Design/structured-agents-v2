# S0 — Nanbeige arch-patch extraction (read + local git analysis only)

Date: 2026-07-31. No build, no GPU, no sudo. GPU 0 / GPU 1 untouched.

Deliverable: `patches/nanbeige-arch.patch` under
`.scratch/projects/17-llama-cpp-inference-lab/patches/`, isolating the nanbeige
architecture from the 48-commit nanbeige42 window so it can be applied onto the
P2 anchor `c588c4f47` and then have `patches/p2-mixed-batch-lora.patch` applied
on top.

## 1. Merge-base confirmation

Local trees (already on disk, nothing cloned):
- `.llamacpp-builds/src-nanbeige` @ `c6640a1c0` (nanbeige42 HEAD, build b10151)
- `.llamacpp-builds/src-p2fork` / `src` @ `c588c4f47` (P2 anchor, llama-cpp-python 0.3.34)

```
git merge-base c6640a1c0 c588c4f47  = c588c4f47   (anchor IS the merge base)
git rev-list --count c588c4f47..c6640a1c0 = 48
```

Confirmed: nanbeige42 HEAD is a direct descendant of the P2 anchor. Matches the
plan headline.

## 2. File classification — arch vs noise

The 48-commit window is dominated by an upstream `master` merge
(`af4b8e17c`) that pulls in unrelated PRs. Rather than filter the entangled
`git diff c588c4f47..c6640a1c0` by hand, the nanbeige arch turned out to be
cleanly isolated in **four dedicated Nanbeige-authored commits**, none of which
touch the noisy files:

| commit | subject | files |
|---|---|---|
| `26cfdc440` | support nanbeige4.2 model | conversion/__init__.py, conversion/nanbeige.py, gguf-py constants + writer, llama-arch.{cpp,h}, llama-context.cpp, llama-model.cpp, models/models.h, models/nanbeige.cpp |
| `03327d628` | fix | conversion/__init__.py, llama-model.cpp |
| `d28da865b` | fix flake8 Lint check | conversion/nanbeige.py, models/models.h |
| `c6640a1c0` | fix loop bound check / drop redundant head_dim | conversion/nanbeige.py, models/nanbeige.cpp |

**ARCH (included in patch):**
- `src/models/nanbeige.cpp` — new file, 184 lines (the looped graph).
- `src/models/models.h` — `llama_model_nanbeige` struct (+16).
- `src/llama-arch.{cpp,h}` — `LLM_ARCH_NANBEIGE`, `LLM_KV_NUM_LOOPS`,
  `LLM_KV_SKIP_LOOP_FINAL_NORM` registration (+3 / +3).
- `src/llama-model.cpp` — arch→`llama_model_nanbeige` dispatch + `LLAMA_ROPE_TYPE_NORM` (+3).
- `src/llama-context.cpp` — `graph_max_nodes` gets the NANBEIGE branch (+1 line, minimax-m3 excluded).
- `gguf-py/gguf/constants.py` (+25), `gguf-py/gguf/gguf_writer.py`
  (`add_num_loops`, `add_skip_loop_final_norm`, +6).
- `conversion/__init__.py`, `conversion/nanbeige.py` — the converter.

**NOISE (excluded — entangled in the same window but NOT nanbeige):**
- `src/llama-kv-cache.{h,cpp}` (+311) — MiniMax-M3 MSA single-head indexer.
- `src/models/minimax-m3.cpp` (new, +562), `src/models/glm-dsa.cpp` (+397).
- `src/llama-hparams.{h,cpp}` — MSA indexer hparams (`n_embd_k_idx`,
  `is_indexer_full`, `indexer_block_size`), NOT nanbeige loop hparams.
- `src/llama-model.h` — `LLM_TYPE_428B_A23B` + `index_{q,k}_{proj,norm}` (MiniMax MSA).
- `src/llama-model-loader.{cpp,h}`, `src/llama-model-saver.cpp` — the
  `use_mmap/use_direct_io/use_mlock → llama_load_mode` refactor (PR #20834).
- `src/llama-graph.cpp` — `LLM_FFN_SWIGLU_OAI_MOE`, `kq_mask` const change.
- **`include/llama.h`** — the `enum llama_load_mode` + `llama_model_params`
  struct change (drops 3 bools, adds `load_mode`). **This is ABI-BREAKING and is
  exactly what Direction A must avoid.** Correctly excluded; the nanbeige arch
  adds ZERO public API. (The plan's §1.2 note that nanbeige touches
  `llama-hparams` / `convert_hf_to_gguf.py` was slightly off — see §6 correction.)

Noise scan of the final patch (`grep -i minimax|indexer|load_mode|use_mmap|msa|
SWIGLU_OAI`) returns nothing added; the only hit is a pre-existing
`case LLM_ARCH_GLM_DSA:` **context** line above the added nanbeige rope case.

## 3. Exact extraction command

Cherry-picked the four nanbeige commits onto a fresh anchor checkout in a
scratch cache worktree (outside the repo tree), then diffed:

```
# scratch worktree of the on-disk nanbeige clone, detached at the anchor
git -C .llamacpp-builds/src-nanbeige worktree add --detach \
    ~/.cache/structured-agents/nanbeige-port-s0 c588c4f47
cd ~/.cache/structured-agents/nanbeige-port-s0 && git checkout -b nanbeige-arch-s0
git cherry-pick -x 26cfdc440 03327d628 d28da865b c6640a1c0   # all clean, auto-merge only
git diff c588c4f47..HEAD > <repo>/.scratch/.../patches/nanbeige-arch.patch
```

All four cherry-picks applied with no conflicts (only benign auto-merges on
llama-model.cpp / models.h). Result: 10 files, +266/-2, 417 lines of patch.

## 4. git apply --check results (onto clean c588c4f47)

Fresh detached worktree at `c588c4f47`, `git clean -fdq`:

- `git apply --check patches/nanbeige-arch.patch` → **ARCH APPLIES CLEAN**
- `git apply --check --3way patches/nanbeige-arch.patch` → all 8 tracked files
  "cleanly" (2 fall-back-to-direct on new files — expected).
- Then `git apply nanbeige-arch.patch` (real apply), followed by:
  - `git apply --check patches/p2-mixed-batch-lora.patch` → **P2 APPLIES CLEAN ON TOP**
  - `git apply patches/p2-mixed-batch-lora.patch` (real apply) → **no rejects**;
    touches include/llama.h, llama-context.cpp, llama-context.h, llama-graph.cpp,
    llama-graph.h (+156).

Overlap note: both patches edit `src/llama-context.cpp`, in disjoint regions
(arch = `graph_max_nodes` ~L2338; P2 = seq-adapter setters). They coexist with
no reject. This matches the build script's mechanism, which uses **plain**
`git apply --check` (not `--3way`).

Caveat (honest): `git apply --check --3way` of the **P2** patch reported
`src/llama-context.cpp: does not match index`. This is NOT a real conflict — it
is because the arch patch was applied to the working tree but never staged, so
`--3way` has no index blob to merge against. The plain `git apply --check`
(authoritative for build-llamacpp.sh) passes, and the real `git apply` of P2
produced zero rejects. If you ever want a clean `--3way` you must `git add -A`
after applying the arch patch first.

## 5. Patch location (matches existing P2 layout)

Written to:
`/.../17-llama-cpp-inference-lab/patches/nanbeige-arch.patch`
(sha256 `862a33250b6406647715126f9631b67ae7fb94b513f3e493752eb3fef23a9f5c`)

This sits beside the existing `patches/p2-mixed-batch-lora.patch`, which
`build-llamacpp.sh` resolves as `${here}/patches/...`. No repo source files were
modified. **build-llamacpp.sh NOT touched (that is S1).**

## 6. What S1 will need (do NOT do now)

- Add a `--profile nanbeige-p2` to `build-llamacpp.sh` that:
  - pins `REF=c588c4f47`;
  - uses a dedicated src tree `${work}/src-nanbeige-p2` (mirror the `src-p2fork`
    convention so it never collides with `src` / `src-p2fork`);
  - applies **two** patches in order: `patches/nanbeige-arch.patch` **then**
    `patches/p2-mixed-batch-lora.patch` (the current code path only supports a
    single `$PATCH`; S1 must extend it to a patch list + `--check` each before
    applying);
  - records **both** patch sha256s in `build-manifest.json`.
- Build target stays `--target llama` (shared lib), CUDA flags identical to
  `cuda-3060`/`p2fork` (sm_86) — no ABI change, so still loads under
  llama-cpp-python 0.3.34.
- ABI header check (plan §4.2): diff `c588c4f47:include/llama.h` vs the merged
  `include/llama.h`; assert changes are ONLY additive `extern "C"` decls (P2's
  two symbols; nanbeige adds none). The arch patch does not touch `include/llama.h`.

## 7. Corrections / unknowns

- **Correction to plan §1.2:** the nanbeige converter lives in **`conversion/nanbeige.py`
  + `conversion/__init__.py`** (a fork-local module), NOT `convert_hf_to_gguf.py`.
  And the nanbeige loop hparams (`n_loops`, `n_layer_phys`, `skip_loop_final_norm`)
  live as members on `llama_model_nanbeige` in `src/models/models.h` / read in
  `nanbeige.cpp::load_arch_hparams` — **not** in `src/llama-hparams.{h,cpp}` (those
  diffs were entirely MiniMax MSA and are excluded).
- **UNVERIFIED (S1 build-time, not S0):** whether `nanbeige.cpp` compiles against
  the anchor. It references `hparams.n_layer_all`, `create_tensor_qkv`, `build_qkv`,
  `build_attn_inp_kv`, `build_cvec`; the plan asserts all exist at `c588c4f47`.
  Patch APPLIES clean; compile is the S1 gate.
- **UNVERIFIED (R2):** the on-disk Nanbeige GGUF was produced by the b10151
  converter; reader-side key compatibility on the b10103 base is a GPU/S2 concern.

Scratch worktrees used (`~/.cache/structured-agents/nanbeige-port-s0`,
`nanbeige-verify`) are pruned after extraction; nothing left in the repo tree.
