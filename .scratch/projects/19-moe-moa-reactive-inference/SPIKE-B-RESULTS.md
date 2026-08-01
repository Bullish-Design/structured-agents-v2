# Spike B — KV-splice kernel go/no-go (results)

Date: 2026-07-30. Target: does reusing a *small partial* precomputed KV segment
(spliced to a new position + a cheap CacheBlend-style heal) beat re-prefilling
that text in wall-clock, while staying coherent? This is the Project 19 gate.

## TL;DR verdict

**BLOCKED for the true kernel; lean NO-GO for the "small partial splice"
premise on the current model/GPU.**

1. The make-or-break path (`blend_by_reanchor`: splice a *foreign* KV object at
   arbitrary positions + selectively heal a fraction of tokens) is **BLOCKED** by
   the stable llama.cpp C API. It is a documented `NotImplementedError` stub in
   `src/structured_agents/llama_core/node_blend_live.py:190`. Verified against the
   pinned `llama.h`: there is no per-cell K/V read/write and no selective-attention
   recompute hook; the state blob is opaque and layout-versioned. It cannot be
   measured today without a fork/native shim.
2. The only *feasible* reuse path today is `blend_by_redecode` (restore the base
   chain's whole-seq blob, then honestly re-prefill only the pull-in + prompt
   tokens). Its economics are already characterised on the real target model
   (Ornith-1.0-9B Q4 on a 3060) by existing artifacts: **prefill is cheap
   (~2–2.8 ms/token) and any blob restore carries ~1.1–1.3 s of fixed cost**, so
   reuse only wins for *large* reused segments (hundreds of tokens). For the
   *small* partial segment this spike targets, plain re-prefill wins outright.

So the "elegant zero-prefill" version is not viable as-measurable; we live in the
partial-re-prefill world, and even that only pays off at scale, not for small
splices. Recommend reframing Project 19 economics around large KV objects, or
funding the native kernel (below) before committing.

## Hardware used

- 2× RTX 3060 (12 GB each). Both are **saturated by the production inference
  runners** right now: GPU0 10 479/12 288 MiB used (~1.4 GB free), GPU1
  9 741/12 288 MiB used (~2.1 GB free), from `nvidia-smi` at run time.
- llama-cpp-python 0.3.34 in `.scratch/projects/17-llama-cpp-inference-lab/.venv-spike`
  (CUDA build for 3060). Requires `LD_LIBRARY_PATH` = `/run/opengl-driver/lib`
  (driver-stub fix from memory) + a gcc-15 lib dir for `libstdc++.so.6`.

## What I measured

### 1. Existing authoritative GPU data (real model, GPU was free when captured)

`.scratch/projects/17-llama-cpp-inference-lab/artifacts/project17-prefix-restore-sweep-20260725T003640Z/prefix_restore_sweep.json`
(Ornith-1.0-9B-UD-Q4_K_XL, n_seq_max=4):

| prefix (per seq) | prefill_only ms | restore_only ms | ms/token prefill | prefill vs restore |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1432 | 1087 | ~2.8 | 1.32× |
| 256 | 2496 | 1145 | ~2.4 | 2.18× |
| 512 | 4674 | 1209 | ~2.3 | 3.87× |
| 1024 | 8598 | 1310 | ~2.1 | 6.56× |

Whole-seq **restore is ~flat ~1.1–1.3 s** regardless of length; **prefill is
linear at ~2–2.8 ms/token**. Restore only beats prefill above ~128 tokens (×4
seqs). Project 18 `BREAK_EVEN.md` corroborates: *persistent disk* whole-state
cache never beats cold prefill through 256 tokens; an *in-memory native*
checkpoint crosses cold prefill only at ≈256 tokens.

Implication for a splice of M cached tokens healed at fraction h: it saves
~(M − hM)·(2–2.8 ms) of prefill but must pay the RoPE-shift + selective-recompute
kernel cost. For small M (dozens of tokens) the saving is only tens of ms — below
plausible kernel + bookkeeping overhead. The saving is only material for large M.

### 2. Fresh runs today (confirm the hardware is blocked, not clean)

- GPU attempt (Qwen3.5-0.8B f16 on GPU1): **213 ms/token** at 32 tokens —
  ~100× slower than the free-GPU Ornith numbers, i.e. the model spilled to CPU /
  thrashed PCIe because only ~2 GB was free. Not a usable measurement; it proves
  a clean fresh GPU benchmark is impossible while the runners hold both cards.
- CPU-only scaling (Qwen3.5-0.8B f16, `n_gpu_layers=0`, saved to
  `/tmp/spike_b_prefill_cpu.json`): marginal ~8.85 ms/token with a large,
  noisy ~7.7 s per-call fixed overhead (graph reset each `eval`). Confirms
  prefill is linear in tokens; absolute numbers not decision-grade.

Bench script: `.scratch/projects/19-moe-moa-reactive-inference/spike_b_prefill_bench.py`.

## What was blocked and why

- **True `blend_by_reanchor`**: the stable API (`llama_state_seq_*`,
  `llama_memory_*`) exposes only opaque, layout-versioned blobs — no way to read
  a foreign KV object's per-cell K/V and write it into another sequence's cache at
  chosen positions, and no hook to recompute attention for only a subset of cached
  tokens (the CacheBlend "heal"). Confirmed in `llama.h`.
- **A clean fresh GPU benchmark**: both 3060s are held by the systemd runners
  (ports 8000/8001); <2.2 GB free per card, too little for a clean load. Spike
  ports 8002/8003 would need a card freed.

## Smallest unblock

- Partial credit is *already* reachable: `llama_memory_seq_add` +
  `llama_memory_can_shift` (both in `llama.h`, lines ~749/782) can RoPE-shift an
  **entire restored sequence** to new positions today. That gives "re-anchor a
  whole KV object" but **not** partial-position splice and **not** cross-attention
  healing — so it does not by itself deliver Spike B.
- The true kernel needs one of: (a) a native `ggml` shim over the cache K/V
  tensors in a llama.cpp fork (position-selective RoPE write + a masked
  attention-recompute pass over the heal set), or (b) an upstream
  `llama_kv_self_rope_shift`-style entry point plus a selective-recompute API.
  `node_blend_live.py` already exposes `base_position` as the wiring hook so a
  caller can dial position-delta + heal-fraction the moment the kernel lands.

## Recommendation

Do not build Project 19 on the assumption that small-segment splice beats
re-prefill — it does not on Ornith-9B/3060 as measurable today, and the elegant
version is blocked on a fork. Two viable directions: (1) reframe around **large**
reused KV objects (hundreds+ tokens), where even the correct `blend_by_redecode`
partial-prefill already wins; or (2) fund the native ggml heal kernel and
re-run this gate. Until then `blend_by_redecode` is the supported primitive.

## Command needed from the user (privileged — I did not run it)

To get clean fresh GPU numbers on the spike ports, free one 3060 by stopping one
production runner, e.g.:

    sudo systemctl stop <inference-runner-on-8001>.service   # frees GPU1

then re-run `spike_b_prefill_bench.py` with `CUDA_VISIBLE_DEVICES=1`,
`MODEL=~/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf`,
`LD_LIBRARY_PATH=/run/opengl-driver/lib:<gcc-lib>:$LD_LIBRARY_PATH`, and restart
the runner afterward. (Substitute the real unit name from `systemctl list-units
'*infer*'`.)
