# Deploying the vLLM endpoint on a Windows host (`tower`)

The runbook for standing up `deploy/vllm` on a **Windows + Docker Desktop (WSL2) + NVIDIA GPU**
box, reachable over Tailscale. Ordered so each step unblocks the next; most are one-time.

The container config itself is platform-agnostic and lives in [`../vllm`](../vllm) — see its
README for model/quantization tuning. This directory is just the **Windows host bootstrap**.

---

## 0. One-time bootstrap (must be done AT the machine)

There's no inbound shell until SSH is enabled, so this first step can't be done remotely.

1. **Enable SSH** — elevated PowerShell, at the box:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\enable-ssh.ps1
   ```
   Installs/starts OpenSSH Server, opens TCP 22, sets PowerShell as the default shell, and
   authorizes the deploy key (override with `-PublicKey '<key>'`, or `-NonAdminUser <name>` for a
   non-admin login). After this, `ssh <user>@tower` works from the dev box.

2. **Bootstrap prerequisites** — elevated PowerShell (can be over SSH once step 1 is done):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1 -RepoUrl https://github.com/Bullish-Design/structured-agents-v2.git
   ```
   Idempotent. Enables the WSL2 features, sets WSL default v2, checks for the NVIDIA driver and
   Docker, creates `E:\structured-agents-v2`, and (if `git` is present) clones the repo there. It
   prints a `[TODO]` checklist for anything that needs a human — typically:
   - a **reboot** to finish enabling WSL2 (then re-run `bootstrap.ps1`);
   - **Docker Desktop**: `winget install -e --id Docker.DockerDesktop`, then enable the *WSL 2
     based engine* and *WSL integration*;
   - the **NVIDIA Windows driver** if `nvidia-smi` is missing (WSL2 CUDA rides the Windows driver —
     never install a GPU driver *inside* WSL).

---

## 1. Confirm GPU reaches containers

Docker Desktop's WSL2 engine exposes the GPU via the NVIDIA Container Toolkit (bundled with recent
Docker Desktop). Verify from PowerShell or a WSL shell:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, `deploy/vllm`'s `deploy.resources.reservations.devices` (driver: nvidia)
will work. If it errors, update Docker Desktop and confirm the NVIDIA Windows driver is installed.

---

## 2. Configure + deploy

```powershell
cd E:\structured-agents-v2\deploy\vllm
copy .env.example .env      # set MODEL, VLLM_API_KEY, MAX_MODEL_LEN, quantization (see ../vllm/README)
docker compose up --build -d
docker compose logs -f      # watch the model load; first boot downloads weights into the hf-cache volume
```

Model weights live in **WSL2-backed named volumes** (`remora-vllm-hf-cache`, `remora-vllm-models`),
not on the `E:` NTFS drive — see [`../vllm/README.md`](../vllm/README.md#model-storage--wsl2-backed-named-volumes-not-a-host-bind-mount).
The code lives on `E:`; the weights do not.

Sizing note: the documented target is a **12 GB RTX 3060** — quantization is mandatory and a 3–4B
Qwen3.5 is recommended. `nvidia-smi` (or the `bootstrap.ps1` output) tells you the actual VRAM.

---

## 3. Verify (push-button)

From the **dev box** (has the repo + `devenv`), over Tailscale:

```bash
LLM_BASE_URL=http://tower:8000/v1 LLM_API_KEY=<VLLM_API_KEY> LLM_MODEL=base \
  deploy/vllm/verify.sh --pytest
```

`verify.sh` checks, in order: `/health` reachable → `/v1/models` lists the served model → a
`json_schema` completion returns schema-valid JSON → an XGrammar `regex` completion matches (the
part the llama.cpp box can't do) → optional LoRA round-trip (`LORA_NAME=<adapter>`). With
`--pytest` it also runs the library's `live`-marked tests. Exit code is non-zero if any hard check
fails.

---

## Gotchas seen on the real box (2026-07-01 bring-up)

The current `tower` is **not** the "clean 12 GB RTX 3060" the sizing notes assume — it's a busy
multi-service Windows host with an **8 GB Quadro RTX 4000 (Turing, sm_75)**. Things that bit us,
so the next person doesn't rediscover them:

- **SSH install stages behind a reboot.** `Add-WindowsCapability … OpenSSH.Server` can report
  `InstallPending` (no `sshd` service yet) if another servicing op is queued. Reboot, then
  `Start-Service sshd`.
- **Docker over SSH can't build.** Docker Desktop's credential helper (`docker-credential-desktop`)
  fails in a non-interactive SSH session ("A specified logon session does not exist"), which breaks
  BuildKit — so `docker compose build` (and thus the Dockerfile `COPY`) is impossible remotely.
  `--config <dir-with-{}>` fixes plain `docker pull` but **not** BuildKit. Workaround: don't build —
  pre-pull the base image and start with the opt-in [`../vllm/docker-compose.no-build.yml`](../vllm/docker-compose.no-build.yml),
  which mounts `entrypoint.sh` at runtime:
  ```powershell
  docker --config E:\dockercfg pull vllm/vllm-openai:v0.11.0
  cd E:\structured-agents-v2\deploy\vllm
  docker compose -f docker-compose.yml -f docker-compose.no-build.yml up -d
  ```
  (Keep a host-local `docker-compose.override.yml` copy if you want bare `docker compose up` to work —
  it's gitignored.)
- **Firewall:** open inbound TCP 8000 (`New-NetFirewallRule … -LocalPort 8000`) or `tower:8000` is
  unreachable over Tailscale even though the port publishes fine on `localhost`.
- **Turing needs two settings** (in `.env`): `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's sampler
  crashes on compute-cap 7.x → engine crash-loop), and `--quantization awq` **not** `awq_marlin`
  (Marlin needs Ampere+). FA2 is unavailable → vLLM falls back to FlexAttention automatically.
- **8 GB sizing:** a 4B fp16 model won't fit; use a pre-quantized **AWQ** repo (e.g.
  `Qwen/Qwen3-4B-AWQ`, ~3 GB) and `GPU_MEMORY_UTILIZATION=0.85` to leave room for the desktop.
- **vLLM 0.11.0 supports Gemma up to `gemma3n`, not `gemma4`.** Newer models may need a newer
  `VLLM_TAG` — but verify that tag still supports sm_75 before bumping (newer vLLM is dropping old archs).

## Files here

| File             | What it does                                                          |
|------------------|-----------------------------------------------------------------------|
| `enable-ssh.ps1` | One-time: enable OpenSSH Server + authorize the deploy key (at the box).|
| `bootstrap.ps1`  | Idempotent prereq setup: WSL2 features, driver/Docker checks, code dir.|
| `../vllm/verify.sh` | Endpoint verification (health → models → json_schema → xgrammar → lora).|
| `../vllm/bench.py` | Batched-throughput profile (diverse prompts; json + text workloads).|
| `../vllm/docker-compose.no-build.yml` | Opt-in no-build override for Docker-Desktop-over-SSH hosts.|
