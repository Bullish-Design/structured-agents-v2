# Per-Sequence Mixed-Batch LoRA in llama.cpp — Deep-Dive Guide

> Status: investigation/planning kickoff document (2026-07-24). Self-contained:
> everything needed to start a deeper design/planning session from scratch.
> Pinned runtime: `out-cuda-3060-postfix2` (llama.cpp build
> `build-cuda-3060-c588c4f47`), llama-cpp-python 0.3.34, CUDA on 2×RTX 3060.
> All `file:line` references are into the pinned source tree at
> `.scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/src/`.

---

## 0. How to use this guide

Read §1–§3 for context, §4–§6 for *why the library cannot do this today*, §7–§9
for *how to implement it on our own fork*, §10 for the no-fork alternative, §11
for prerequisites, and §12–§13 to drive the planning session. §14 is the
reference list (local source + upstream + papers). If you only read one thing:
**per-sequence LoRA is structurally MoE routing; the CUDA primitive already
exists in our build; the work is a C++ graph patch on our fork, not a Python
binding change and not a from-scratch kernel.**

---

## 1. What we are trying to build (project context)

Decision D2 (`02-DECISIONS.md`): the flagship demo is a **multi-LoRA agent-router
fleet** — many small fine-tuned LoRA adapters (one per agent/router), served over
**one base model** with **massively parallel batched generation**. Small
specialized routers should beat a big general model on tool/function-call routing
(latency + always-valid via xgrammar), while sharing the base weights and KV.

The single hardest capability behind that story is **mixed-batch multi-LoRA**:
running one `llama_decode` step where *different sequences in the batch use
different adapters* — the vLLM/Punica/S-LoRA "thousands of LoRAs, one batch"
trick. This guide is about whether and how we get that on llama.cpp.

Base model for all experiments: **Ornith-1.0-9B** (`Ornith-1.0-9B-UD-Q4_K_XL.gguf`,
`n_vocab=248320`), a **hybrid attention + GatedDeltaNet (linear-attention /
recurrent)** architecture (Qwen3-Next family). llama.cpp loads and serves it
correctly (`08-GATE3-ORNITH-RESTORE.md`).

---

## 2. What is already proven (reuse this; do not re-litigate)

All GPU-only, CUDA-synchronized, GPU 1 idle. See `RESEARCH_REPORT.md` dated
sections and these artifacts:

- **Per-sequence KV/state reuse works** (`artifacts/project17-seq-reuse-20260724T201749Z/`):
  a cached per-sequence blob restores into a **nonzero seq slot** of a **live
  multi-sequence context** with correct continuation *and* neighbour isolation
  (K up to 256), and survives **cross-process restart**. Constraint: a blob loads
  only into a context with the **same `n_seq_max`** (mismatch →
  `llama_state_seq_set_data` returns 0, fails safe; blob size encodes `n_seq_max`).
- **Parallel multi-sequence decode works** and **batches ~4×**
  (`artifacts/project17-seq-batch-20260724T202826Z/`): S sequences, one token
  each per `llama_decode`, scale from 48 tok/s (S=1) to 195 tok/s (S=16); decode
  is weight-memory-bandwidth bound (step 75→82 ms from S=8→16). Matched-`n_seq_max`
  portability holds at n_seq_max 8 and 16.
- **In-context per-sequence reuse breaks even at ≈192 tokens** (restore ~130 ms
  flat vs cold prefill ~0.8 ms/token).
- **Key harness lesson** (`RESEARCH_REPORT.md`): after `llama_state_seq_set_data`
  you must replay Python n_past bookkeeping (`n_tokens`, `input_ids`) or
  `Llama.eval` runs `kv_cache_seq_rm` and wipes the restore. Own-batch decode
  (build the `llama_batch` yourself with explicit `seq_id`/`pos`) avoids this;
  the high-level `Llama.eval` only ever uses seq 0. Reference own-batch decoders:
  `benchmarks/project17/run_seq_reuse.py`, `run_seq_batch_breakeven.py`.

**Implication:** the "many sequences in flight, each with its own cached prefix"
half of the router is done. The missing half is *per-sequence adapters*.

---

## 3. Glossary

- **LoRA** (Low-Rank Adaptation): a frozen base weight `W` gets an additive
  low-rank delta `ΔW = (α/r)·B·A`, with `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}`, rank
  `r ≪ in,out`. Inference: `y = Wx + scale·B(Ax)`. Cheap to store/swap.
- **Mixed-batch / multi-LoRA batching**: one forward pass where different rows
  (tokens/sequences) apply different adapters. Requires a *segmented/grouped*
  matmul (SGMV) or per-token gather (BGMV) — see Punica/S-LoRA (§14).
- **MoE routing**: each token is routed to a subset of "expert" weight matrices,
  selected by an `ids` tensor, then a routed matmul (`ggml_mul_mat_id`) applies
  the right expert per token. **Structurally identical to per-token LoRA
  selection.**
- **`n_seq_max`**: max number of independent sequences (distinct recurrent states)
  in a context. Sizes the recurrent/KV allocation; part of any per-seq cache key.
- **ubatch**: the physical micro-batch llama.cpp splits a `llama_batch` into
  before running the graph.

---

## 4. How llama.cpp applies LoRA (the execution model)

### 4.1 Adapter data structures (`src/llama-adapter.h`)

```cpp
struct llama_adapter_lora_weight {           // one (a,b) pair per targeted tensor
    ggml_tensor * a = nullptr;               // A: [in, r]
    ggml_tensor * b = nullptr;               // B: [r, out]
    float get_scale(float alpha, float adapter_scale) const;  // (alpha/r)*adapter_scale
};

struct llama_adapter_lora {
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;  // keyed by BASE tensor name
    float alpha;
    llama_adapter_lora_weight * get_weight(ggml_tensor * w);  // name lookup; null if not targeted
};
```

`get_weight` (`src/llama-adapter.cpp:138`) looks up by `w->name`, so an adapter
only affects the base tensors it was trained on; it returns `nullptr` for others.

### 4.2 How adapters attach to a context (`src/llama-context.cpp:1258`)

`llama_context::set_adapters_lora(adapters**, n, scales*)` rebuilds a
`llama_adapter_loras` map = `{adapter → scale}` on the **context**. Zero-scale
adapters are dropped. The context passes `loras.get()` into the graph params
(`llama-context.cpp:2428`). Public API: `llama_set_adapters_lora`
(header `llama.h:690`); the singular `llama_set_adapter_lora` is now a deprecated
shim in the Python binding (`llama_cpp.py:2210`).

### 4.3 Where the delta is applied (`src/llama-graph.cpp:1382`)

```cpp
ggml_tensor * llm_graph_context::build_lora_mm(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);        // base matmul, cur = WHOLE ubatch
    if (w_s) res = ggml_mul(ctx0, res, w_s);
    for (const auto & lora : *loras) {                     // every context-set adapter
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) continue;                       // adapter doesn't target this tensor
        const float scale = lw->get_scale(lora.first->alpha, lora.second);
        ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));  // B(Ax)
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);                 // y = Wx + scale·B(Ax)
    }
    return res;
}
```

`build_lora_mm` is called for every projection: `wqkv`/`wq`/`wk`/`wv`
(`llama-graph.cpp:1501–1541`) and FFN `up`/`gate`/`down` (`:1602–1619`).

**The crux:** `cur` holds hidden states for *all tokens of all sequences* in the
ubatch. The delta `scale·B(Ax)` is computed and added **uniformly across every
column of `cur`**. There is no per-column (per-token/per-sequence) selection of
which adapter to apply. Setting N adapters *sums* all their deltas onto every
token (a blend), which is not routing.

---

## 5. Why mixed-batch multi-LoRA is unsupported today

1. **No per-token adapter channel.** `llama_batch` (`llama.h:244`) =
   `{n_tokens, token, embd, pos, n_seq_id, seq_id, logits}`. No adapter field.
   The ubatch (`src/llama-batch.cpp`) carries per-token `seq_id` but nothing that
   maps a token to an adapter.
2. **Application is context-scoped.** `*loras` is a context-level set (§4.2), and
   `build_lora_mm` applies it uniformly (§4.3).
3. **Bindings cannot fix this.** Our ctypes bindings wrap the *compiled*
   `libllama.so`; the LoRA math lives in the compiled ggml graph. Owning the
   Python sampling/decode loop does not help — LoRA is applied *inside*
   `llama_decode`, upstream of the logits we hook.

So the literal capability requires a **C++/ggml change**, i.e. a fork + rebuild.
Confirmed independently and previously in `03-SPIKE-FINDINGS.md` Q2.

---

## 6. The key insight: per-sequence LoRA ≈ MoE routing

llama.cpp already selects **per-token weight matrices** for Mixture-of-Experts,
via `ggml_mul_mat_id` (`ggml/include/ggml.h:1445`):

```cpp
GGML_API struct ggml_tensor * ggml_mul_mat_id(ggml_context * ctx,
    ggml_tensor * as,   // stacked expert weights [k, n, n_expert]
    ggml_tensor * b,    // inputs
    ggml_tensor * ids); // per-token expert selection
```

- It is **CUDA-backed in our pinned build** (`ggml/src/ggml-cuda/ggml-cuda.cu`,
  `GGML_OP_MUL_MAT_ID`), so no new kernel is needed for the routed matmul.
- llama.cpp even has a **routed LoRA path already**, `build_lora_mm_id`
  (`src/llama-graph.cpp:1413`), used where the *base* matmul is itself routed
  (MoE FFN): it applies `mul_mat_id(B, mul_mat_id(A, cur, ids), ids)`. This is
  the exact structure per-sequence LoRA needs — it just needs to be driven by a
  **seq→adapter** `ids` tensor instead of the MoE gating `ids`.

Per-sequence LoRA = "route each token to its sequence's adapter's A/B." That is
`build_lora_mm_id` with `ids` derived from `seq_id`. This is what makes the
feature a *wiring* problem on top of existing ops rather than a kernel project.

---

## 7. Implementation design (own-fork C++)

### 7.1 Data layout — stack adapters as "experts"

For each targeted base tensor `w`, build once (at adapter-pool registration time)
two stacked tensors over the N pool adapters:
- `A_stack[w]`: shape `[in, r_max, N]` — adapter i's `A` in slice i, zero-padded
  to the common `r_max = max_i r_i`.
- `B_stack[w]`: shape `[r_max, out, N]` — adapter i's `B` in slice i, zero-padded
  on the rank axis so padded ranks contribute nothing.
- `scale[w]` per adapter (fold `α_i/r_i · adapter_scale_i` into a per-expert
  vector, or bake into `B_stack`).

Zero-padding to `r_max` keeps the routed matmul a single uniform op while
preserving each adapter's effective rank.

### 7.2 The routing tensor `ids`

At graph-build time, build a per-token `ids` vector of length `n_tokens(ubatch)`
where `ids[t] = adapter_index_for(ubatch.seq_id[t])`. Each token routes to
exactly one adapter (single-expert-per-token; `n_expert_used = 1`), so the reshape
to `mul_mat_id`'s expected `[.., 1, n_tokens]` layout is straightforward — mirror
`build_lora_mm_id` (`llama-graph.cpp:1413–1441`) for the exact tensor shapes.

### 7.3 The graph change — a routed `build_lora_mm`

Add `build_lora_mm_routed(w, cur, ids)` that, per targeted `w`:

```
delta = mul_mat_id(B_stack[w], mul_mat_id(A_stack[w], cur, ids), ids)  // per-token A then B
res   = ggml_mul_mat(ctx0, w, cur) + scale_route(delta, ids)
```

Call it from the same projection sites as `build_lora_mm`
(`llama-graph.cpp:1501–1619`) when a per-sequence adapter map is active; fall back
to plain `build_lora_mm` (or none) otherwise. Include a "no adapter" sentinel
slice (all-zero A/B) so a sequence with no adapter routes to a null delta.

### 7.4 Plumbing per-token adapter ids (recommended: side-channel, no batch ABI change)

Do **not** add a field to `llama_batch` (ABI churn, touches every caller). Instead
model the adapter as **sequence state**, consistent with our per-sequence KV
design:
- New context API: `llama_set_seq_adapters(ctx, adapters**, n)` registers the
  pool (builds the stacked tensors, §7.1); `llama_set_seq_adapter(ctx, seq_id,
  adapter_idx)` assigns an adapter to a sequence (a `seq_id → idx` map on the
  context).
- The graph builds `ids` from the ubatch's per-token `seq_id` via that map. The
  ubatch already carries `seq_id` per token (`src/llama-batch.cpp`), so no batch
  format change is needed. Expose both via our own ctypes bindings.

This composes cleanly with cached-prefix restore: a router assigns
`seq_id → (cached prefix, adapter)` together.

### 7.5 Build & bindings

We already compile llama.cpp from source with a patched CMake target
(`06-LLAMACPP-BUILD-WORKFLOW.md`, `RESEARCH_REPORT.md` build section). Add the new
symbols to the build and to our ctypes bindings. No new kernels; `mul_mat_id`
CUDA path is reused.

### 7.6 Correctness validation (the gate)

Ground truth = run each sequence **separately** on a single-adapter context with
only its adapter set (the proven, supported path). Then run all sequences in
**one mixed-batch decode** with per-seq adapters and require **exact greedy
token-by-token equality** for every sequence, swept over: adapter count,
different adapters per seq, ranks (incl. mixed ranks → padding), the "no adapter"
sentinel, and interaction with restored per-sequence KV prefixes. Same fail-closed
discipline as the existing runners.

### 7.7 Performance expectation

The win is replacing S sequential single-adapter decodes with one routed decode
over S sequences/adapters, amortizing base-weight memory traffic (decode is
weight-bandwidth bound — see §2). `mul_mat_id` adds gather overhead and the
padded-rank stack wastes some FLOPs; net benefit grows with S and with base-model
size relative to adapter rank. Compare aggregate tok/s vs the sequential baseline
and vs the context-pool alternative (§10). Cite Punica/S-LoRA for the expected
scaling shape (§14).

---

## 8. Complications & mitigations

| Complication | Mitigation |
| --- | --- |
| Adapters have different ranks | Pad `A`/`B` to `r_max` with zeros (§7.1). Wasted FLOPs bounded by rank spread. |
| Adapters target different tensor subsets | For a router fleet, train all adapters with the **same `target_modules`** (homogeneous). Otherwise include zero A/B for non-participating adapters per tensor. |
| Quantized base (Q4_K) + f16 adapters | `mul_mat_id` already handles mixed types on the MoE path; validate numerics against the sequential baseline. |
| Sequence with no adapter | Reserve a zero-slice "null adapter" index; assign it by default. |
| ubatch splitting reorders tokens | Build `ids` from the *ubatch's* `seq_id` at graph time, after the split — never from the original batch order. |
| Hybrid recurrent layers | Confirm which tensors adapters target on Ornith; GatedDeltaNet projections may or may not be adapter-targeted. Depends on how the adapters were trained. |
| alora ("activated LoRA") | Orthogonal — `llama_adapter_get_alora_*` (`llama-adapter.cpp:487`) triggers adapters by invocation tokens, still context-level. Ignore for this feature. |

---

## 9. Risks / unknowns to resolve in planning

- Does llama.cpp apply a LoRA to Ornith's **hybrid GatedDeltaNet** arch at all?
  (Needs a real adapter — see §11. First go/no-go.)
- Exact `mul_mat_id` tensor-shape contract for single-expert-per-token routing
  (study `build_lora_mm_id` and the CUDA op).
- Upstream drift: is there an in-flight llama.cpp PR for batched multi-LoRA we
  should track/rebase onto instead of maintaining a private patch? (§14 — check.)
- Maintenance cost of a fork vs. contributing upstream.

---

## 10. Alternative without a fork: the context-pool router (Decision D2 path a)

If the C++ fork is deferred, the router is still buildable with **zero library
changes**, using only proven primitives:

- Load base once; keep a **pool of contexts**, each pinned to one adapter via
  `llama_set_adapters_lora`.
- Scheduler **batches requests within a context** (proven ~4× multi-seq decode)
  and **multiplexes across contexts** (round-robin / grouped-by-adapter).
- Reuse per-sequence cached prefixes inside each context (proven).

New measurables for path (a): adapter-swap cost of `llama_set_adapters_lora`
(only if swapping adapters on a shared context instead of a pool), per-context
VRAM and the 3060 context-count ceiling, and whether adapter application composes
with restored per-sequence KV state. Path (a) is the pragmatic teaching MVP; the
§7 fork is the flagship stretch that actually matches vLLM-style batching.

---

## 11. Asset prerequisites (block ALL empirical LoRA work — do first)

There are currently **zero LoRA adapters on this machine** (no GGUF LoRA, no HF
`adapter_config.json`/`adapter_model`). Before any experiment:

1. Fine-tune (or obtain) **≥2 LoRA adapters on the Ornith base** (distinct
   router behaviours), HF PEFT format, ideally identical `target_modules` and
   rank.
2. Convert to GGUF with the pinned
   `.llamacpp-builds/src/convert_lora_to_gguf.py` (requirements file alongside it).
3. **Smoke test**: load base + one adapter via `llama_adapter_lora_init` +
   `llama_set_adapters_lora` (or `Llama(lora_path=…)`), verify coherent generation
   changes vs. base. This is the first go/no-go for the whole line of work and
   needs items 1–2.

---

## 12. Proposed investigation phases (for the planning session)

1. **P0 — assets**: produce 2 Ornith GGUF adapters; smoke-test single-adapter
   application on the hybrid arch. *(Gate: adapter changes output coherently.)*
2. **P1 — context-pool router (no fork)**: build path (a); measure adapter-swap
   cost, per-context VRAM, aggregate tok/s vs a single big model; integrate
   xgrammar routing + cached prefixes. *(Deliver the teaching MVP.)*
3. **P2 — fork spike**: implement `build_lora_mm_routed` + `set_seq_adapter`
   plumbing behind a build flag; validate exact-match vs sequential baseline on
   2–4 adapters. *(Gate: token-exact equivalence.)*
4. **P3 — scale + benchmark**: sweep adapter count and S; compare mixed-batch
   routed decode vs context-pool vs sequential; publish synchronized throughput.
5. **P4 — decide**: upstream contribution vs. maintained fork vs. ship path (a)
   only.

---

## 13. Open questions to answer up front

- How many router adapters, what rank, what `target_modules`? (Drives padding &
  stacking cost.)
- Latency vs throughput target for the router? (Path a may suffice for latency.)
- Is exact-match equivalence the acceptance bar, or is approximate acceptable?
- Fork maintenance appetite (rebase cadence against upstream llama.cpp)?

---

## 14. References

### Local source (pinned build `.llamacpp-builds/src/`)

- `src/llama-graph.cpp:1382` — `build_lora_mm` (context-level LoRA application).
- `src/llama-graph.cpp:1413` — `build_lora_mm_id` (routed LoRA via `mul_mat_id`;
  the template for §7.3).
- `src/llama-graph.cpp:1501–1619` — projection call sites of `build_lora_mm`.
- `src/llama-adapter.h:48–86` — `llama_adapter_lora_weight` / `llama_adapter_lora`.
- `src/llama-adapter.cpp:138` — `get_weight` (base-tensor-name lookup).
- `src/llama-adapter.cpp:487` — alora invocation tokens (orthogonal).
- `src/llama-context.cpp:1258` — `set_adapters_lora`; `:2428` passes `loras` to graph.
- `src/llama-batch.cpp` — batch→ubatch, per-token `seq_id` (source of routing ids).
- `ggml/include/ggml.h:1445` — `ggml_mul_mat_id`.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `GGML_OP_MUL_MAT_ID` CUDA support.
- Pinned header `include/llama.h`: `:244` `llama_batch`; `:657`
  `llama_adapter_lora_init`; `:690` `llama_set_adapters_lora`; `:845–915`
  `llama_state_seq_*` + flags.
- `convert_lora_to_gguf.py` (+ `requirements/requirements-convert_lora_to_gguf.txt`).

### Our evidence & prior docs

- `RESEARCH_REPORT.md` — per-sequence reuse, break-even, batched throughput,
  LoRA readiness/feasibility (dated 2026-07-24 sections).
- `03-SPIKE-FINDINGS.md` Q2 — multi-LoRA context-level confirmation.
- `02-DECISIONS.md` D2 — flagship router demo & path (a)/(b) framing.
- `08-GATE3-ORNITH-RESTORE.md` — Ornith hybrid state restore.
- `06-LLAMACPP-BUILD-WORKFLOW.md` — how we build/patch llama.cpp.
- Runners: `benchmarks/project17/run_seq_reuse.py`,
  `run_seq_batch_breakeven.py`, `run_native_state_decompose.py`.
- Artifacts: `artifacts/project17-seq-reuse-20260724T201749Z/`,
  `…-seq-batch-20260724T202826Z/`, `…-native-decompose-20260724T194604Z/`.

### Upstream (verify against the pinned commit before relying on line numbers)

- llama.cpp repo: `https://github.com/ggml-org/llama.cpp` — `src/llama-graph.cpp`,
  `src/llama-adapter.cpp`, `ggml/…/ggml-cuda`. Track the PR history around
  `llama_set_adapters_lora` (plural) and any "batched/per-sequence LoRA" work.
- llama.cpp LoRA docs / `convert_lora_to_gguf.py` usage in the repo `docs/`.
- llama-cpp-python: `https://github.com/abetlen/llama-cpp-python` (binding we mirror).

### Papers / prior art (the batched-multi-LoRA technique)

- **LoRA** — Hu et al., 2021, arXiv:2106.09685 (the `y = Wx + BA x` formulation).
- **Punica** — Chen et al., 2023, arXiv:2310.18547 (**SGMV**: segmented gather
  matmul for serving many LoRAs in one batch; the algorithm §7 mirrors).
- **S-LoRA** — Sheng et al., 2023, arXiv:2311.03285 (scalable multi-LoRA serving,
  unified paging; motivates the pooled/cached-prefix design).
- **vLLM multi-LoRA** — vLLM docs on LoRA serving (Punica-based) for the
  reference behaviour and API shape we are emulating.
- Qwen3-Next / GatedDeltaNet — background for Ornith's hybrid architecture (why
  recurrent state and per-seq `n_seq_max` matter, §2–§3).
