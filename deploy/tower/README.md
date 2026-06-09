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

## Files here

| File             | What it does                                                          |
|------------------|-----------------------------------------------------------------------|
| `enable-ssh.ps1` | One-time: enable OpenSSH Server + authorize the deploy key (at the box).|
| `bootstrap.ps1`  | Idempotent prereq setup: WSL2 features, driver/Docker checks, code dir.|
| `../vllm/verify.sh` | Endpoint verification (health → models → json_schema → xgrammar → lora).|
