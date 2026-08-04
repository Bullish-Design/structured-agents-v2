# Port Plan — Nanbeige recurrent-in-depth graph onto the P2 mixed-batch LoRA fork

Goal: one `libllama.so` that has **both** (a) the Nanbeige looped (recurrent-in-depth)
architecture graph and (b) the P2 per-sequence mixed-batch LoRA routing, so that a
LoRA adapter ("hat") can be selected **per depth-loop pass** (CONCEPT §6.5 Expert
Bundles). Today the loop lives only in `Nanbeige/llama.cpp@nanbeige42` and P2 lives
only in the repo's private patch; no tree has both.

Date: 2026-07-31. Status: **read/analysis only** — no build, no GPU, no sudo. All
findings verified against the two local source trees under
`.scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/` unless flagged
UNVERIFIED.

---

## 0. Headline (TL;DR)

- **The two forks share history and are close.** The nanbeige42 branch is a **direct
  descendant of the exact commit the P2 fork is pinned to** (`c588c4f47`, = the
  `llama-cpp-python 0.3.34` ABI anchor, build `b10103`). nanbeige HEAD `c6640a1c0`
  (build `b10151`, ggml 0.17.0) is that anchor **+48 commits** (one of which is an
  upstream `master` merge; the rest are unrelated upstream PRs + the nanbeige arch).
- **The P2 patch applies cleanly onto the nanbeige HEAD tree** (`git apply --check
  --3way` reports all 5 files "cleanly"). So *mechanically*, either direction works.
- **Recommended merge direction: port the nanbeige *arch* ONTO the P2 anchor**
  (`c588c4f47`), NOT P2 onto nanbeige HEAD. Reason: it keeps the public ABI **exactly
  at the 0.3.34 anchor** (no bindings regen / no llama-cpp-python bump), and the
  nanbeige arch depends only on helpers that **already exist at the anchor**
  (`src/models/` per-model split, `create_tensor_qkv`, `build_qkv` are all present at
  `c588c4f47`). See §2.
- **Hardest sub-problem:** not the merge — it is **threading a loop-step index into
  the adapter selection** inside `build_lora_mm`, which today knows nothing about the
  layer index. See §3.
- **Honest correction to the spike docs:** the fork's *llama.cpp graph* does **NOT**
  implement LoopSplit / mHC depth-attention / concatenated n-gram embeddings. Those
  are in the HF `modeling_nanbeige.py`; the GGUF graph (`src/models/nanbeige.cpp`) is
  a **plain looped llama-style stack** — weights shared across `n_loops`-expanded
  layers, with an optional loop-boundary RMS norm. This *simplifies* the hats port
  (there is no exotic loop op to route through) and should be corrected in
  SPIKE-NANBEIGE-LOOP.md §1. See §1.3.

---

## 1. Divergence map

### 1.1 Bases and distance

| | P2 fork | Nanbeige fork |
|---|---|---|
| Repo | private patch on `ggml-org/llama.cpp` | `github.com/Nanbeige/llama.cpp` |
| Ref | `c588c4f47` + `patches/p2-mixed-batch-lora.patch` | branch `nanbeige42` @ `c6640a1c0` |
| Build id | `b10103` | `b10151` |
| ggml lib | 0.16.0 | 0.17.0 |
| libllama | (0.3.34 anchor ABI) | 0.0.10151 |
| Local tree | `.llamacpp-builds/src-p2fork` | `.llamacpp-builds/src-nanbeige` |

`git merge-base c6640a1c0 c588c4f47 = c588c4f47`. **The anchor IS the merge base** —
nanbeige42 branched *from* (or through) the exact P2 anchor. Distance:
`git rev-list --count c588c4f47..c6640a1c0 = 48` commits. Of those 48, most are
unrelated upstream PRs (UI, httplib vendor bump, MiniMax-M3, GLM-5.2, opencl/hexagon
backends) pulled in via the `af4b8e17c "Merge branch 'master'"` commit; only a small
subset is the nanbeige arch itself.

### 1.2 Files each fork touches vs the shared base `c588c4f47`

**P2 fork** (267-line patch, surgical — 5 files):
- `include/llama.h` — +2 additive API decls (`llama_set_seq_adapters`,
  `llama_set_seq_adapter`) appended inside `extern "C"`. **No struct/enum change.**
- `src/llama-context.{h,cpp}` — two setters + two members (`seq_loras`,
  `seq_adapter_map`); pass both into `llm_graph_params`.
- `src/llama-graph.{h,cpp}` — `llm_graph_input_seq_lora_mask`, `build_inp_seq_lora_mask`,
  the masked-sum branch inside `build_lora_mm`, and two `llm_graph_params` /
  `llm_graph_context` members.

**Nanbeige fork** (arch-relevant C/C++ only; UI/vendor churn ignored):
- `src/llama-arch.{h,cpp}` — `LLM_ARCH_NANBEIGE`, `LLM_KV_NUM_LOOPS`,
  `LLM_KV_SKIP_LOOP_FINAL_NORM` registration (+24 lines).
- `src/llama-hparams.{h,cpp}` — loop hparams (+32 lines).
- `src/llama-model.{h,cpp}` — `LLM_ARCH_NANBEIGE` dispatch → `llama_model_nanbeige`,
  `n_loops` / `n_layer_phys` / `skip_loop_final_norm` members (+28 lines).
- `src/models/nanbeige.cpp` — **new file, 184 lines**: the loop graph (see §1.3).
- `src/models/models.h` — `llama_model_nanbeige` decl (+43, shared w/ minimax/glm).
- `src/llama-model-loader.{h,cpp}`, `src/llama-model-saver.cpp` — minor loader plumbing.
- `convert_hf_to_gguf.py` + `gguf-py/gguf/constants.py` — converter emits
  `general.architecture=nanbeige` + `num_loops`/`skip_loop_final_norm` keys.
- **Incidental (NOT nanbeige, but in the same 48 commits):** heavy churn in
  `src/llama-kv-cache.{h,cpp}` (+301 lines) is the **MiniMax-M3 MSA single-head
  indexer** (`k_idx`, `msa_strict_slots`), and `src/models/{minimax-m3,glm-dsa}.cpp`
  are new — **unrelated to nanbeige** and must be excluded from the port.

### 1.3 What the nanbeige *graph* actually does (verified from `src/models/nanbeige.cpp`)

- `load_arch_hparams`: reads `num_loops` (default 1), `skip_loop_final_norm`; records
  `n_layer_phys` (= physical 22); **expands the logical layer count** to
  `n_layer_phys * n_loops` (44), replicating per-layer hparam arrays.
- `load_arch_tensors`: allocates weights for the 22 physical layers, then
  `layers[i + j*n_phys] = layers[i]` — **physical weights shared across loops**, each
  loop slot has a distinct logical index (hence a distinct KV cache slot).
- `graph::graph`: a single `for il in [0, n_layer)` over the **expanded** 44 layers:
  RMSNorm → `build_qkv` → RoPE(Q,K) → `build_attn` → residual → FFN(SILU/PAR) →
  residual → `build_cvec`. At each loop boundary (`(il+1) % n_phys == 0`, not the last)
  it optionally applies `output_norm` as a loop-boundary norm.
- **Per-loop KV** is therefore *free/automatic*: because layers are expanded, loop
  pass `j`'s attention writes to KV layer index `il = i + j*n_phys`, distinct from
  other passes. (This is the "per-loop KV doubling" the concept flags for P4.)
- **NOT present in the graph:** no `LoopSplit`, no `mHC` depth-attention, no n-gram
  embedding concat. The GSM8K-92% coherence result (SPIKE-NANBEIGE-BUILD) is achieved
  by this simplified looped stack. **Correct the spike docs.** (UNVERIFIED whether the
  simplification costs accuracy vs the HF model — out of scope, GPU-gated.)

**Consequence for hats:** the loop-step for any graph node is the compile-time constant
`loop_step = il / n_phys`. This is the single most important fact for §3.

### 1.4 Overlap / conflict files

Files touched by **both** P2 and the nanbeige delta: `include/llama.h`,
`src/llama-context.cpp`, `src/llama-graph.cpp`. All three are small, disjoint regions
(P2 appends API + adapter members; nanbeige adds arch registration). `git apply
--check --3way` of the P2 patch onto nanbeige HEAD reports **all five files apply
cleanly** — so even the naive direction has no real textual conflict. `llama-graph.h`
and `llama-context.h` are touched only by P2 (nanbeige delta = 0 lines there).

---

## 2. Merge direction — recommendation

**Port the nanbeige *arch* files ONTO the P2 anchor `c588c4f47`, then apply the P2
patch on top. Do NOT rebase P2 onto nanbeige HEAD `b10151`.**

Two candidate directions:

**Direction A — nanbeige-arch → P2 anchor (RECOMMENDED).**
- ABI stays **exactly** at `c588c4f47` = the `llama-cpp-python 0.3.34` anchor. The
  existing repo bindings (`llama_cpp.py` ctypes + `seq_routing.py`) are already
  correct for this ABI; the P2 fork lib already loads under 0.3.34 today. Adding the
  nanbeige arch introduces **zero new public API** (it is internal arch registration
  + GGUF metadata keys), so the merged lib's `llama.h` is ABI-identical to the current
  P2 lib. **No bindings regen, no llama-cpp-python bump.**
- Feasible because the nanbeige graph depends only on helpers **already present at the
  anchor**: `src/models/` per-model architecture, `create_tensor_qkv`, `build_qkv`,
  `LLAMA_LOAD_LOCALS`, `build_attn_inp_kv` — all verified to exist at `c588c4f47`.
  So porting `nanbeige.cpp` does **not** drag in a post-anchor refactor.
- Cost: extract the nanbeige-only hunks from shared files (`llama-arch`,
  `llama-hparams`, `llama-model`, `models.h`, loader, converter, gguf constants) and
  the standalone `models/nanbeige.cpp`. Mechanical; the surface is ~5 small shared-file
  edits + 1 new file + converter/constants. Must **exclude** the MiniMax-M3/GLM MSA
  kv-cache churn from the same window.

**Direction B — P2 → nanbeige HEAD (NOT recommended).**
- The P2 patch applies cleanly (verified), but the resulting lib is `b10151`/ggml
  0.17.0 — **ABI-drifted from 0.3.34** (ggml minor bump strongly implies struct-layout
  changes across the 48 commits; UNVERIFIED which exact structs). Loading it under the
  installed bindings would be ABI-unsafe. You would have to either regen the low-level
  bindings from the b10151 header (BINDGEN path, workflow §2) or bump
  llama-cpp-python and re-anchor the whole tuple. That is strictly more work and more
  risk than Direction A, for no benefit (we do not need any of the 48 upstream PRs).

**Decision:** Direction A. Smaller ABI surface, closer to the anchor, and the nanbeige
graph is self-contained enough to lift. Keep a dedicated `src-nanbeige-p2` checkout so
the port never collides with the stock `src` / `src-p2fork` trees (mirror the
`build-llamacpp.sh` `p2fork` convention). A new `--profile nanbeige-p2` in
`build-llamacpp.sh` should pin `REF=c588c4f47`, apply an arch patch
(`patches/nanbeige-arch.patch`) **then** `patches/p2-mixed-batch-lora.patch`, and
record both shas in the manifest.

---

## 3. The hats mechanism — adapter select indexed by loop-step

### 3.1 The gap

P2 today routes **by sequence**: `build_lora_mm`'s masked branch reads a per-token mask
built from `seq_adapter_map[ubatch->seq_id[t][0]]`. It has **no notion of the layer /
loop index** — `build_lora_mm(w, cur, w_s)` is not passed `il`. To make a hat swap
per depth-pass we must make the adapter selection a function of
`loop_step = il / n_phys`.

Crucially (§1.3) **`loop_step` is a graph-build-time constant**, not a runtime
per-token property. That makes the cheapest version of hats simpler than P2's runtime
mask.

### 3.2 Where the delta is applied

In `models/nanbeige.cpp`, every projection that a LoRA can target funnels through
`build_lora_mm` (called inside `build_qkv`, `build_attn` on `wo`, `build_ffn` on the
up/gate/down, and the final `build_lora_mm(model.output, ...)`). So a hat active for
loop pass `j` is applied by having `build_lora_mm` add that hat's low-rank delta at
every projection of every physical layer while `il/n_phys == j`. The KV written during
pass `j` is thereby computed under hat `j`'s perturbed W_k/W_v — i.e. per-loop KV is
**hat-consistent by construction** (this is exactly CONCEPT §6.5.2 rule (a): a hat's KV
is captured under that hat's weights). No extra KV machinery is needed; the arch's
existing layer-expansion already separates KV per pass.

### 3.3 Three implementation options (pick H1 for the prototype)

**H1 — build-time hat selection (RECOMMENDED first prototype).** Thread `il` into
`build_lora_mm` (add a defaulted `int il = -1` arg; the nanbeige graph passes it).
Add a context-level "hat map": `std::vector<int32_t> loop_hat_map` sized `n_loops`,
`loop_hat_map[j] = pool index for pass j` (-1 = base), set by a new
`llama_set_loop_adapters(ctx, adapters, n)` + `llama_set_loop_adapter(ctx, loop_step,
idx)` (mirror the P2 API surface, reuse the same `seq_loras` pool storage). In
`build_lora_mm`, when a loop-hat map is active and `il >= 0`, select
`seq_loras[loop_hat_map[il/n_phys]]` and add its delta to **all tokens** with a plain
`ggml_add` — **no per-token mask tensor needed**, because a whole layer shares one hat.
Cheapest, no new runtime input, matches the "fixed program of thought" case (frame →
critique/commit) which is the default in CONCEPT §6.5.1.

**H2 — runtime mask, per loop-step (for per-request hat programs).** Keep P2's
`build_inp_seq_lora_mask` but parametrize the built mask by `loop_step`, so the
selected column is `k = program[seq_id][loop_step]`. Only `n_loops` (=2) distinct masks
exist, so build one mask per loop-step and index it by `il/n_phys` in `build_lora_mm`.
This composes the two axes (per-request AND per-pass) — the full CONCEPT §6.5.4 vision
where each sequence carries its own hat program. More work; defer past prototype.

**H3 — pure per-token (status quo P2).** Unchanged; hats not possible. Listed for
contrast.

### 3.4 Per-loop KV implication

Because layers are expanded (`il` distinct per pass), enabling hats needs **no change**
to KV allocation — pass `j` already owns KV slots `[j*n_phys, (j+1)*n_phys)`. The only
correctness rule is that a KV object *staged as a read-only prefix* for hat `j` must
have been captured under hat `j` (CONCEPT §6.5.2 rule (a)); the runtime decode path
gets this for free since the hat is live while that pass writes its KV. Fingerprinting
(P7) must now key on `(engine, arch=nanbeige, num_loops, loop_step, hat_id)`.

---

## 4. The ABI gate

**Target:** load the merged `libllama.so` under the installed `llama-cpp-python 0.3.34`
(anchor `c588c4f47`, ggml 0.16.0). Direction A makes this the *same* ABI the current P2
lib already satisfies, so the gate is an extension of the existing
`benchmarks/project20/abi_smoke_gate.py`, not a new bindings effort.

What must hold / be checked:
1. **Symbol probe** — `llama_set_seq_adapters` + `llama_set_seq_adapter` resolve
   (existing gate step 1); plus the new `llama_set_loop_adapter(s)` if H1 lands.
2. **ABI invariance** — the merged `include/llama.h` must differ from the anchor
   *only* by additive `extern "C"` decls (P2's two + hats' two). **No struct field, no
   enum value, no function-signature change** to any pre-existing symbol → the 0.3.34
   ctypes bindings stay valid. Verify with a header diff `c588c4f47:include/llama.h`
   vs merged, asserting all changes are new top-level declarations. (The nanbeige arch
   adds none — its keys are GGUF metadata strings, not C ABI.)
3. **Arch load** — load a `Nanbeige4.2-3B` GGUF (`general.architecture=nanbeige`,
   `num_loops=2`) through `llama_cpp.Llama`, confirm no `unknown architecture` and that
   `n_layer` reports the expanded count.
4. **Coherence + continuation-equivalence** (project-18 discipline: "a load is not a
   pass; only a matching deterministic continuation counts") — greedy-decode a fixed
   prompt at temp=0 and assert **token-exact** against the standalone nanbeige `llama`
   binary (build `b10151`) on the same prompt/quant. This catches any graph regression
   introduced by porting onto the older base.
5. **LoRA still routes** — apply a converted GGUF LoRA (Ornith path is a confirmed GO)
   and run a seq-routed 2-adapter mixed batch; assert token-exact vs each adapter
   decoded alone (existing P2 gate). Then (H1) set a loop-hat map and assert the hatted
   run differs from base in the expected direction and is reproducible.

Model for the gate: `~/.cache/structured-agents/models/Nanbeige4.2-3B-UD-Q4_K_XL.gguf`
(already on disk). GPU env per the driver-stub memory (prepend
`/run/opengl-driver/lib`). **GPU-gated — run only when GPU 1 is free.**

Open ABI question (must verify at port time, cheap): the Nanbeige GGUF was produced by
the fork's `b10151` converter. Confirm its tensor names / metadata keys are read
correctly by the arch code as ported onto the `b10103` base (they should be — the keys
are the same `nanbeige.*` strings — but the converter side may assume post-anchor gguf
helpers; if so, re-convert with a b10103-based converter or verify key compatibility).

---

## 5. Risks, unknowns, and staged task breakdown

### 5.1 Risks & unknowns
- **R1 (low).** Extracting nanbeige-only hunks from shared files without dragging in
  MiniMax-M3/GLM MSA changes. Mitigate: port by hand-written `nanbeige-arch.patch`
  against `c588c4f47`, not by cherry-picking the entangled 48-commit range.
- **R2 (low/med).** GGUF converter/metadata compatibility across the b10151→b10103
  base gap (§4 open question). UNVERIFIED. Cheap to check: inspect key reads; worst
  case re-run the fork converter pinned differently (GPU-free, but weights download —
  defer, or use the existing on-disk GGUF and only fix reader-side).
- **R3 (med).** `build_lora_mm` is `const`; H1 adds an `il` arg and reads a context
  member — fine, but confirm the graph-result reuse cache (`can_reuse`) invalidates
  when `loop_hat_map` changes (same caveat the P2 design flagged for its mask input).
- **R4 (med).** Looped-weight stability under a per-pass perturbation (CONCEPT §6.5.4):
  a hat must not make the shared stack diverge across passes. This is a *training*
  concern, not a port concern, but the prototype should sanity-check that a
  near-identity hat leaves output ~unchanged.
- **R5 (unknown, GPU-gated).** Whether the simplified graph (no mHC/LoopSplit/n-gram,
  §1.3) matters for accuracy — orthogonal to this port; the fork already ships it.
- **R6 (low).** `LLAMA_MAX_LAYERS` bound with expanded layers × hats: unchanged
  (nanbeige.cpp already asserts `n_layer_phys*n_loops <= LLAMA_MAX_LAYERS`).

### 5.2 Staged tasks (smallest first)

1. **S0 — Arch-patch extraction (no build).** Produce `patches/nanbeige-arch.patch`:
   `git diff c588c4f47..c6640a1c0 -- <nanbeige-only files>` filtered to exclude
   minimax/glm/kv-cache-MSA hunks. `git apply --check` it onto a fresh `c588c4f47`
   checkout. Deliverable: patch that applies clean + P2 patch applies clean on top.
2. **S1 — Merged build (GPU nix-shell, no GPU inference).** Add `--profile nanbeige-p2`
   to `build-llamacpp.sh` (pin `c588c4f47`, apply arch patch then P2 patch, manifest
   records both shas). Build the `llama` shared-lib target. Gate: compiles; `llama.h`
   header-diff shows only additive decls (ABI check §4.2).
3. **S2 — ABI smoke gate (GPU-gated).** Extend `abi_smoke_gate.py` with the nanbeige
   arch-load + continuation-equivalence vs the standalone `b10151` binary (§4 steps
   3–4). Gate: token-exact greedy match. This proves Direction A didn't regress the
   graph.
4. **S3 — Existing P2 routing on nanbeige (GPU-gated).** Run the existing seq-routed
   2-adapter mixed-batch correctness check with a nanbeige base + Ornith-style GGUF
   LoRA. Gate: token-exact vs single-adapter contexts (proves P2 survives the merge).
5. **S4 — Hats API + H1 wiring (build).** Add `llama_set_loop_adapter(s)` + the
   `loop_hat_map`, thread `il` into `build_lora_mm`, build-time selection branch.
   Bindings: extend `seq_routing.py` with the two new symbols (fail-closed like the
   existing pair).
6. **S5 — Per-pass-hat prototype (GPU-gated).** Set a 2-hat program over `num_loops=2`
   (pass0=hatA, pass1=hatB), decode, and assert: (a) reproducible, (b) differs from
   base and from either single hat, (c) near-identity hats ≈ base (R4 sanity). This is
   the first working per-pass-hat prototype.

Everything through S1 is off-GPU. S2–S5 are GPU-gated and must wait for GPU 1 to free
(distillation spike running now).

---

## 6. Sources / verification notes

- Local trees: `.llamacpp-builds/src-nanbeige` (@`c6640a1c0`), `.llamacpp-builds/src-p2fork`
  (@`c588c4f47`), `.llamacpp-builds/src` (stock @`c588c4f47`).
- P2 patch: `.scratch/projects/17-.../patches/p2-mixed-batch-lora.patch` (267 lines);
  design `19-P2-FORK-DESIGN.md`; bindings `src/structured_agents/llama_core/seq_routing.py`,
  router `.../router.py` (`seq_routed`/`context_pool`/`auto` backends).
- Merge base / distance: `git merge-base` = `c588c4f47`; `git rev-list --count
  c588c4f47..c6640a1c0` = 48.
- Patch-applies-clean-onto-nanbeige: `git apply --check --3way` (all 5 files clean).
- Anchor has models/-split + `create_tensor_qkv`/`build_qkv`: `git ls-tree`/`git grep`
  at `c588c4f47` (verified).
- Graph reality: `src/models/nanbeige.cpp` (read in full) — plain looped stack, no
  mHC/LoopSplit/n-gram; corrects SPIKE-NANBEIGE-LOOP.md §1.
- ABI anchor rule: `06-LLAMACPP-BUILD-WORKFLOW.md` §1–2 (0.3.34 → ggml 0.16.0 anchor).
- UNVERIFIED (flagged): b10151→b10103 GGUF converter/metadata compat (R2); accuracy
  impact of the simplified graph (R5); exact struct drift across the 48 commits
  (only relevant if Direction B were chosen).
</content>
</invoke>
