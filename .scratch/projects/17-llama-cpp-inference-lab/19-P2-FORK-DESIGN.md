# P2 — Mixed-Batch Multi-LoRA Fork: Design (grounded in the pinned source)

> 2026-07-25. True single-`llama_decode` multi-LoRA: different sequences in one
> ubatch use different adapters. Private fork of the pinned build only (llama.cpp
> `AGENTS.md` forbids fully-autonomous UPSTREAM contribution but exempts private
> forks; we are NOT submitting upstream). Supersedes the guide-14 §7 sketch where
> the real code differs.

## Key correction to the guide-14 §7 plan

Guide §7 says "add `build_lora_mm_routed` and call it from the 18 projection
sites (`llama-graph.cpp:1501-1619`)." The real code (verified) already funnels
every projection through **one** method — `llm_graph_context::build_lora_mm`
(`src/llama-graph.cpp:1382`) — and the adapter set enters via a single member
`loras` (`llama-graph.h:927`, set from `llm_graph_params`). So the clean change
is to make **`build_lora_mm` itself route** when a per-sequence adapter map is
active. Zero call-site edits; matches `AGENTS.md`'s "simpler change" ethos.

## Two-stage plan

### P2a — correctness-first, mask-based (implement now)

For a small router fleet (N adapters, N ~ 2-8, low rank), the LoRA matmuls are
cheap vs the base weight matmul. So the simplest CORRECT mixed-batch path is: in
`build_lora_mm`, compute EACH pool adapter's full delta over all tokens, then
mask it to only the tokens whose sequence selected that adapter, and sum:

```
res = ggml_mul_mat(w, cur)                       // base, all tokens  [out, n_tokens]
for k, adapter in enumerate(seq_loras):
    lw = adapter.get_weight(w); if !lw: continue
    delta_k = scale_k * B_k(A_k @ cur)           // [out, n_tokens]
    mask_k  = view row k of inp_seq_lora_mask     // [1, n_tokens] f32, 1.0 where seq→k
    res = res + delta_k * mask_k                  // ggml_mul broadcasts over out
```

Cost: N x the LoRA FLOPs (each adapter computed for every token), but NO stacked
tensors, NO `mul_mat_id`, NO r_max padding, NO new GPU-resident weights — reuse
each adapter's existing `a`/`b`. Ops used: `ggml_mul_mat`, `ggml_mul` (broadcast),
`ggml_add` — all trivially CUDA-backed. This is the 90%-simpler change that proves
the capability; optimize later.

### P2b — performance, stacked `mul_mat_id` (follow-on, guide §7.1-7.3)

Once P2a validates exact-match, replace the N-loop with one routed matmul:
`A_stack[in,r_max,N]`, `B_stack[r_max,out,N]`, `ids[1,n_tokens]` = seq→adapter,
`delta = mul_mat_id(B_stack, mul_mat_id(A_stack, cur, ids), ids)`. Reuses
`build_lora_mm_id`'s exact shape contract (`llama-graph.cpp:1413`). Heaviest part:
allocating the stacked GPU tensors (mirror `llama-adapter.cpp` tensor loading).
Deferred until P2a is green.

## Touch points (file:line in the pinned tree)

1. **`llama.h`** — public API:
   - `llama_set_seq_adapters(ctx, llama_adapter_lora ** adapters, size_t n)` —
     register the ordered pool (index = routing id).
   - `llama_set_seq_adapter(ctx, llama_seq_id, int32_t adapter_idx)` — assign a
     sequence to a pool slot (-1 = no adapter).
2. **`llama-context.{h,cpp}`** — store `std::vector<llama_adapter_lora*> seq_loras`
   and `std::array<int32_t, LLAMA_MAX_SEQ> seq_adapter_map` (init -1). Implement
   the two setters. In `graph_params(...)` (`llama-context.cpp:1329`) pass both
   into `llm_graph_params`.
3. **`llama-graph.h`** — add to `llm_graph_params` + `llm_graph_context`:
   `const std::vector<llama_adapter_lora*> * seq_loras`, `const int32_t *
   seq_adapter_map`. Add `llm_graph_input_seq_lora_mask` (holds `ggml_tensor *
   mask` [N, n_tokens]) with `set_input`. Add a mutable `ggml_tensor *
   seq_lora_mask` on the context + `build_inp_seq_lora_mask()`.
4. **`llama-graph.cpp`**:
   - `build_inp_seq_lora_mask()`: create f32 [N, n_tokens] input, `res->add_input`.
   - `llm_graph_input_seq_lora_mask::set_input`: for each token t, primary
     `seq_id = ubatch->seq_id[t][0]`; `k = seq_adapter_map[seq_id]`; write 1.0 at
     (k, t) for the matching row, 0 elsewhere. `ggml_backend_tensor_set`.
   - Build the mask once (lazily on first `build_lora_mm` when `seq_loras` set, or
     eagerly) and store on the mutable member so `build_lora_mm` (const) can read it.
   - `build_lora_mm`: if `seq_loras && !empty`, take the masked-sum branch above;
     else the existing uniform `*loras` loop.
5. **Bindings** — add the two symbols to our ctypes bindings (`llama_cpp.py`).

## Build & validate

- Rebuild the pinned CUDA target (`build-llamacpp.sh`; ccache keeps CUDA kernels
  cached, only the touched .cpp recompile → minutes). Point `LLAMA_CPP_LIB_PATH`
  at the new `out`/lib.
- **Gate (guide §7.6):** ground truth = each sequence decoded ALONE on a
  single-adapter context (proven path). Then one mixed-batch decode with per-seq
  adapters must be **token-exact greedy** for every sequence. Sweep: 2-4 adapters,
  different adapters per seq, the "no adapter" (-1) sentinel, mixed with base.
  Compare aggregate tok/s vs the P1 context-pool router (one decode vs N contexts).

## Risks

- `build_lora_mm` is `const`; storing the built mask needs a `mutable` member or
  building it in the graph-context setup before any projection. Confirm ordering.
- Hybrid arch: adapters must target tensors `build_lora_mm` sees (attn/ffn proj);
  linear-attention `in_proj_*` go through the same method, fine.
- `graph_result` reuse/caching (`can_reuse`): the mask input must invalidate reuse
  when the seq→adapter map changes (like other inputs key on ubatch).
