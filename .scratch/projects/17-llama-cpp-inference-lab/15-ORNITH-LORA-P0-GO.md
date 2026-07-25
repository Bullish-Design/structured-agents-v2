# P0 Go/No-Go — LoRA application on Ornith's hybrid arch: **GO**

> 2026-07-24. Answers the first gate of `14-PER-SEQUENCE-LORA-GUIDE.md` §9/§12-P0:
> *does llama.cpp apply a LoRA to Ornith's hybrid GatedDeltaNet arch at all?*
> Verdict: **yes**, on the pinned CUDA build, GPU-synchronized, per-adapter distinct.

## Result

`artifacts/project17-ornith-lora-probe-20260724T211046Z/ornith_lora_probe.json`

| check | result |
| --- | --- |
| base reproducible (base == base_rerun, token-exact greedy) | ✅ true |
| delta applied (base != base+probe-a) | ✅ true (first divergence @ token 11) |
| per-adapter distinct (probe-a != probe-b) | ✅ true |

Base loops `…is Paris.\n` (`11751,13,198,760,…`). Each synthetic adapter breaks the
loop at token 11 into a *different* continuation. Deterministic greedy (temp 0,
top_k 1), `CUDA_VISIBLE_DEVICES=0`, `n_gpu_layers=-1`, GPU 1 idle.

## Method (reusable pipeline — clears §11 "zero adapters" blocker)

1. Synthetic large-perturbation PEFT adapters on the Ornith base
   (`benchmarks/project17/create_ornith_lora_probe.py`): rank 16, alpha 64
   (scale 4.0), `sigma=0.02`, targeting **k/v/o_proj on full-attention layer 3**.
   Two seeds → `probe-a`, `probe-b`. Quality-agnostic; proves *application*, not quality.
2. Convert each with the pinned `convert_lora_to_gguf.py --base <HF config snapshot>`
   (`--outtype f16`). Base **weights not needed** — only hparams/model-class.
3. Apply + greedy-diff (`benchmarks/project17/run_ornith_lora_probe.py`): base,
   base-rerun, base+probe-a, base+probe-b on the same base model.

## Key facts discovered

- **Converter supports the arch.** `Qwen3_5ForConditionalGeneration`
  (`model_type qwen3_5`) resolves for `ModelType.TEXT` to `Qwen3_5TextModel`
  (`conversion/qwen.py:623`, a `_LinearAttentionVReorderBase(Qwen3NextModel)`).
  LoRA keys `base_model.model.model.language_model.layers.N.self_attn.{proj}` →
  `blk.N.attn_{q,k,v,output}` (get_base_tensor_name strips `base_model.model.`,
  base.py:577 strips `language_model.`).
- **q_proj gate-fusion gotcha.** Ornith full-attention `attn_q.weight` is
  `(in=4096, out=8192)` in the GGUF — a gate is fused into q (verified via
  GGUFReader). A naive single-matrix q LoRA (out=4096) is **rejected**:
  `llama_adapter_lora_init: failed to apply lora adapter: tensor
  'blk.3.attn_q.weight' has incorrect shape (hint: maybe wrong base model?)` —
  and llama-cpp-python swallows it (adapters=0). k/v (out=1024) and o (4096×4096)
  are clean. **Real router adapters (P0) must handle q's gate fusion or avoid q.**
- **GPU env fix.** Without the real driver ahead of the CUDA stub, everything runs
  on CPU (`ggml_cuda_init: ... CUDA driver is a stub library`). Prepend
  `/run/opengl-driver/lib` (+ zlib for numpy/gguf-py) to `LD_LIBRARY_PATH`. Recorded
  in memory `llama-cpp-gpu-driver-stub-fix`.

## Caveats / scope

- Adapters are **synthetic** (random, large scale) — not behaviourally trained.
  This proves the *application path*, exactly the §9 gate; it is **not** the §11
  item-1 "coherent behaviour change from a real fine-tune."
- Only layer-3 attention k/v/o probed; FFN and linear-attention (GatedDeltaNet)
  projection targeting is still unverified (guide §8 "hybrid recurrent layers" row).

## Next (per guide §12)

- **P0 remainder:** fine-tune (or obtain) ≥2 *real* router adapters on Ornith and
  confirm coherent behaviour change; decide q-gate handling and `target_modules`.
- **P1:** context-pool router (no fork) using `llama_set_adapters_lora` + the proven
  per-sequence KV reuse.
