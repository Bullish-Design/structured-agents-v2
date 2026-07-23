# SP-03..SP-07 — SGLang Qwen3-4B-AWQ parity attempt (RTX 3060)

Date: 2026-07-22
Server: SGLang 0.5.14, torch 2.11.0+cu130, `Qwen/Qwen3-4B-AWQ`, GPU 1, `127.0.0.1:8002`,
`--enable-lora --max-loras-per-batch 2`, xgrammar grammar backend,
adapters `control-a` / `control-b` from `runtime/qwen3-loras-qkv` (q/k/v projections).

## Outcome: environment + load PROVEN; generation DEADLOCKS on this GPU

SGLang was brought fully online but **cannot complete a single generation** on the
RTX 3060 — the awq_marlin forward pass deadlocks. Parity with the vLLM run
(constraints + multi-LoRA under load) could therefore **not** be demonstrated.

### What was proven

- **Environment rebuilt** — the broken `.venv` was removed and restored via
  `devenv shell -- uv sync --locked` (sglang 0.5.14 + the ~615 MB kernel wheel).
  `import sglang, torch` OK (torch 2.11.0+cu130, CUDA available).
- **Server starts and serves HTTP** — weights load (`Load weight end`, awq 4-bit,
  2.58 GB), KV cache allocates (39554 tokens), Uvicorn comes up on 8002,
  `/v1/models` returns `base`, `control-a`, `control-b` (see `models.json`).
- **Multi-LoRA loads** — `Using csgmv as backend of LoRA kernels`; both adapters
  accepted once regenerated with q/k/v projections (SGLang's fused-QKV LoRA
  requires k_proj/v_proj present — a q_proj-only adapter that vLLM tolerates fails
  with `Failed to load LoRA adapter control-a: '...v_proj.lora_A.weight'`).

### The blocker (backend- and LoRA-independent)

- **flashinfer backend** (`serve-flashinfer.log`): first generation never returns;
  `Scheduler watchdog timeout (watchdog_timeout=300)` with `cur_batch.batch_size()=1`
  → `SIGQUIT` → process tree killed.
- **triton backend** (`serve-triton.log`, `--attention-backend triton
  --sampling-backend pytorch`): identical hang. A **base-model** (no LoRA)
  `/generate` ran 320 s with **GPU utilization 0%** before the client aborted —
  the forward pass is deadlocked, not merely slow.

Conclusion: SGLang 0.5.14's **awq_marlin** path for Qwen3-4B deadlocks on the
RTX 3060 (sm_86), independent of attention/sampling backend and independent of
LoRA. This is a runtime incompatibility, not a tuning gap. `/health` stays 503
throughout because SGLang's internal warmup generation is stuck in the same hang.

### Startup cost note

Startup is ~12 min: a ~6 min silent gap (awq_marlin weight repack, ninja-compiled)
then a ~6 min `Load weight` phase. Switching to triton did **not** shrink it, so
the bottleneck is the awq_marlin repack, not flashinfer JIT. `ninja` had to be put
on PATH (it ships in the venv bin) for the repack to compile at all.

### Suggested paths to actually qualify SGLang (future)

- Try a **non-marlin AWQ** or a different quant (e.g. GPTQ-marlin, fp8) that has a
  known-good SGLang kernel on sm_86.
- Try an **unquantized** small model on SGLang for the constraint/LoRA matrix
  (separates "SGLang works here" from "awq_marlin is broken here").
- Test on an sm_89/sm_90 GPU where awq_marlin is exercised upstream.

## Artifacts

- `models.json`, `health.txt` (503 — never ready)
- `serve-flashinfer.log` (watchdog-timeout hang), `serve-triton.log` (GPU-0% hang)
- `probe-abort-503.log` (probe aborted on the persistent 503)
- Runner: `../../probes/serve_qwen3_lora_sglang.sh`; probe: `../../probes/live_specialist_probe.py`
