# Native vLLM on NixOS

This is the production deployment path for the private structured-agents endpoint. It runs one native vLLM process inside a locked devenv environment, binds the API only to `127.0.0.1:8000`, and publishes it only through Tailscale Serve on HTTPS.

## Table of contents

1. [Security boundary](#security-boundary) — Loopback and Tailscale ACL layers.
2. [Pinned first profile](#pinned-first-profile) — Exact model, revision, vLLM, and GPU limits.
3. [Install and service setup](#install-and-service-setup) — Locked environment and NixOS service.
4. [Start and local verification](#start-and-local-verification) — Health and constrained-output gates.
5. [Tailscale publication](#tailscale-publication) — Declarative Serve and remote checks.
6. [Operations and rollback](#operations-and-rollback) — Persistence, monitoring, rotation, cache ownership, and recovery.

## Security boundary

```text
authorized group:agents client
  -> https://server.tail770f47.ts.net:443
  -> Tailscale Serve
  -> http://127.0.0.1:8000
  -> native vLLM process on dedicated GPU 1
```

`serve.sh` hardcodes `--host 127.0.0.1 --port 8000`; no environment value can change that. Do not add a LAN/public port mapping, firewall exception, reverse proxy, or Tailscale Funnel. Tailscale policy admits only `group:agents` and tailnet admins to `tag:vllm:443`; that tailnet policy is the application access boundary.

The client base URL is:

```text
https://server.tail770f47.ts.net/v1
```

For a simultaneous llama.cpp comparison using the same GGUF, see
[`../../llama-cpp/native/README.md`](../../llama-cpp/native/README.md). It runs on
GPU 0 at local port 8001 (Tailnet HTTPS `:8443`); this vLLM service remains on GPU 1
at port 8000.

## Pinned first profile

| Setting | Value |
|---|---|
| Runtime | native devenv, Python 3.12, `vllm==0.25.0` plus `vllm-gguf-plugin` in `uv.lock` |
| Model | `unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL` |
| Model revision | `f18012b8f690e563b7f872cb764b4cb3de90b14a` |
| Quantization | Unsloth Dynamic Q4 GGUF (`--quantization gguf --dtype bfloat16`) |
| GPU profile | dedicated GPU 1; CUDA graphs; no LoRAs; CPU weight and KV offload disabled |
| Context | 16384 tokens (CUDA-graph all-GPU service limit on the 12 GiB GPU) |
| GPU memory utilization | 0.82 |
| Served name | `base` |

The selected Gemma model is Unsloth's 6.72 GiB Dynamic Q4 GGUF. Its revision is passed to vLLM, so a Hugging Face `main` update cannot silently alter the deployed weights. The vLLM GGUF plugin is required and the launcher uses the upstream Gemma tokenizer/config rather than converting GGUF metadata. CUDA graphs consume 0.79 GiB on this GPU, so the 0.82 memory profile reduces KV-cache reservation and leaves graph-warmup headroom. The launcher also sets `--cpu-offload-gb 0`; vLLM 0.25 has no CPU swap cache, and its KV offload is disabled unless a size is explicitly configured. If the requested context cannot fit in GPU KV cache, startup or the request must fail rather than spilling to host RAM.

## Install and service setup

These commands run on the NixOS host as the account that owns the checkout (`andrew` below).

1. Resolve and install the locked native environment before enabling the service. This download is large because it includes CUDA/PyTorch/vLLM dependencies:

   ```bash
   cd /home/andrew/Documents/Projects/structured-agents-v2/deploy/vllm/native
   devenv shell -- uv sync --locked --no-dev
   devenv shell -- uv run --locked --no-sync vllm --version
   devenv shell -- uv run --locked --no-sync python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
   ```

   The final command must print `True` and the RTX 3060 name. If it fails, stop; do not use CPU inference.

2. In the NixOS configuration repository, import `deploy/vllm/native/nixos-module.nix` and add:

   ```nix
   services.structuredAgentsVllm = {
     enable = true;
    repositoryPath = "/home/andrew/Documents/Projects/structured-agents-v2";
    user = "andrew";
    group = "users";
  };
   ```

   The module declares the model, immutable revision, context, GPU-memory limit, and persistent Hugging Face cache. Override those options in this block only after full verification.

3. Apply the NixOS configuration and start the service:

   ```bash
   sudo nixos-rebuild switch
   sudo systemctl enable --now structured-agents-vllm.service
   sudo journalctl -u structured-agents-vllm -f
   ```

The systemd unit restarts failures, survives reboot, and starts the locked environment with `uv run --no-sync`; it never resolves packages while serving.

## Start and local verification

Wait for model loading to finish, then run:

```bash
curl -fsS http://127.0.0.1:8000/health

LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_MODEL=base \
deploy/vllm/verify.sh --pytest
```

Check the listener before publishing it:

```bash
ss -ltnp 'sport = :8000'
```

It must show only `127.0.0.1:8000` (and optionally `[::1]:8000` if deliberately added later), never a LAN, public, or Tailscale address. The successful `verify.sh` result must include health, models, JSON Schema, XGrammar regex, and live pytest checks.

## Tailscale publication

The enabled NixOS module starts `structured-agents-vllm-tailscale-serve` at boot and maintains the HTTPS reverse proxy in Tailscale's background mode; this does not expose port 8000 or alter existing TCP Serve rules. After local verification succeeds, confirm its state:

```bash
systemctl status structured-agents-vllm-tailscale-serve
tailscale serve status --json
```

From an authorized `group:agents` device, use the HTTPS base URL:

```bash
LLM_BASE_URL=https://server.tail770f47.ts.net/v1 \
LLM_MODEL=base \
deploy/vllm/verify.sh --pytest
```

Also prove that direct `http://100.124.67.32:8000/health` and the server's LAN address on port 8000 fail from another tailnet/LAN device. Confirm a non-`group:agents` identity cannot reach HTTPS port 443. Never run `tailscale funnel`.

## Operations and rollback

- Run `systemctl status structured-agents-vllm` and `journalctl -u structured-agents-vllm` for service health/logs. Monitor `nvidia-smi`, cache growth under `/var/lib/structured-agents-vllm/hf`, and disk free space. Set the log-retention policy in the host's journald configuration.
- Run `BENCH_LEVELS=1,2,4,8,16 deploy/vllm/bench.py` through the HTTPS URL after warm-up. Record throughput, p95 latency, request errors, JSON validity, GPU memory behavior, the model revision, and vLLM version in a host-local operations record.
- Manage access by changing the tailnet ACL/grant membership for `group:agents`; restart is not required for a membership change. Record the policy owner and review date in the host-local operations record.
- Upgrade only by recording a candidate vLLM/model revision, creating its locked environment, completing local/remote verification and the benchmark, and then switching the service profile.
- Roll back by stopping the candidate (`sudo systemctl stop structured-agents-vllm`), restoring the prior known-good host-local environment file and lockfile, starting the service, and repeating remote verification. Do not delete the Hugging Face cache as part of rollback.
