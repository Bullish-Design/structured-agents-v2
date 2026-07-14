# vLLM + Tailscale Deployment Plan

## Table of contents

1. **Goal and delivery boundary** — Define the authenticated, tailnet-only vLLM endpoint and what this project excludes.
2. **Architecture and security model** — Document loopback-only vLLM, Tailscale Serve TLS termination, ACLs, and layered API-key authorization.
3. **Prerequisites and blocking gates** — Resolve GPU, Docker/NVIDIA runtime, Tailscale identity, storage, and access before deployment.
4. **Configuration and provenance** — Pin image/model settings, manage secrets, and preserve rollback metadata.
5. **Local deployment** — Bring up the container safely, initially on one RTX 3060.
6. **Tailnet publication** — Apply Serve and ACL configuration without exposing port 8000 to LAN or the public internet.
7. **Verification and capacity validation** — Validate health, JSON Schema, XGrammar, remote authorization, and throughput.
8. **Cutover, operations, and rollback** — Establish client configuration, persistence, monitoring, upgrade discipline, and recovery.
9. **Acceptance criteria** — State the completion gates for a production-ready first deployment.
10. **No-subagent rule** — Preserve the repository instruction for future work on this project.

## 1. Goal and delivery boundary

Deliver a private, authenticated OpenAI-compatible vLLM endpoint on this NixOS host for structured-agents-v2 clients. The service must be reachable only by authorized Tailscale identities over HTTPS and must prove JSON Schema and XGrammar constrained-output behavior before clients rely on it.

This first deployment excludes LoRA serving, the optional DBOS/Postgres dual-path subsystem, public internet access, and multi-GPU tensor parallelism. Those are follow-on changes after the baseline is stable.

## 2. Architecture and security model

```text
authorized Tailscale client
  -> HTTPS + Bearer API key
  -> Tailscale Serve :443
  -> http://127.0.0.1:8000
  -> vLLM Docker container
  -> one RTX 3060
```

- Docker publishes only `127.0.0.1:8000:8000`; vLLM may still listen on `0.0.0.0` inside its container.
- Tailscale Serve is the sole remote ingress and terminates HTTPS. Do not enable Tailscale Funnel.
- Tailnet ACLs limit `:443` on the serving node to the approved agents group/tag. The vLLM API key remains mandatory as independent application authorization.
- There is no LAN or public firewall opening for port 8000. Verify that direct LAN and Tailscale-IP access to the port fails.

## 3. Prerequisites and blocking gates

Do not pull a model or change Compose until all gates pass:

1. `nvidia-smi` works on the host and `/dev/nvidia*` exists for the deployment context.
2. Docker is usable by the deploying account and `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` sees a GPU.
3. Tailscale is logged in, has a stable MagicDNS name, and the operator can apply or request the required ACL/tag policy.
4. There is sufficient disk for the vLLM image, Hugging Face cache, and model; model-cache growth has an owner.

Observed during planning: the host has two RTX 3060 GPUs, 62 GiB RAM, and about 350 GiB free disk, but this session could not communicate with the NVIDIA driver or Docker/Tailscale sockets. Treat GPU/container access as the current hard blocker until verified directly on the host.

## 4. Configuration and provenance

- Copy `deploy/vllm/.env.example` to the gitignored `.env`; set mode `0600`.
- Generate a non-placeholder `VLLM_API_KEY`; record its owner and rotation date outside the repository.
- Start with a pinned 3–4B AWQ/HF model on a single card, `MAX_MODEL_LEN=8192`, `GPU_MEMORY_UTILIZATION=0.90`, no LoRAs, and `EXTRA_ARGS=--quantization awq --dtype half` where appropriate for the selected model.
- Record the image digest/tag, model revision, quantization, context limit, GPU-memory limit, model cache location, and known-good rollback profile in host-local operational notes.
- Keep `VLLM_TAG` synchronized with `docker-compose.no-build.yml` if the no-build path is used.

## 5. Local deployment

1. Update the Compose port binding to loopback-only.
2. Start with one GPU and `docker compose up --build -d` (or the documented no-build override only when necessary).
3. Watch `docker compose logs -f vllm`; accept no CUDA error, OOM, or restart loop.
4. Confirm `curl -fsS http://127.0.0.1:8000/health` and that `/v1/models` contains the configured served name.
5. Do not introduce LoRAs or a second GPU until the functional gates pass.

## 6. Tailnet publication

1. Assign the host a serving tag such as `tag:vllm`; restrict tag ownership to infrastructure administrators.
2. Add a tailnet ACL permitting only the approved agents group (and temporary infrastructure verification group) to access `tag:vllm:443`.
3. Publish the loopback service with `tailscale serve --bg --https=443 http://127.0.0.1:8000`.
4. Record `tailscale serve status` and the MagicDNS HTTPS URL. Clients use `https://<machine>.<tailnet>.ts.net/v1`.
5. From an authorized remote client, prove `/v1/models` works with the API key; prove an unauthorized identity cannot connect.

## 7. Verification and capacity validation

Run [deploy/vllm/verify.sh](../../../deploy/vllm/verify.sh) locally, then through the Tailscale HTTPS URL. It must pass health, models, strict JSON Schema, and XGrammar regex checks. Then run its `--pytest` mode to execute the repository's live markers.

Benchmark the remote client path using `deploy/vllm/bench.py` at concurrency 1, 2, 4, 8, and 16 after warm-up. Record throughput, p95 latency, individual-request error rate, JSON validity, and GPU-memory behavior. Do not raise concurrency or enable a second GPU without a stable baseline.

## 8. Cutover, operations, and rollback

- Update client configuration only after remote verification: `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.
- Define persistent service management for both Docker Compose and Tailscale Serve across reboot before declaring production readiness.
- Monitor container health, GPU memory/utilization, disk/cache growth, and logs; define owners and retention periods.
- Upgrade by deploying a separately recorded candidate image/model profile, verifying and benchmarking it, then switching clients. Preserve the last known-good profile.
- Roll back by removing/repointing Serve as necessary, stopping the candidate Compose service, restoring the known-good pinned profile, and repeating remote verification. Never delete named model volumes as part of rollback.

## 9. Acceptance criteria

The first deployment is complete only when all of the following are true:

1. GPU is visible to vLLM inside Docker and the container is healthy after restart.
2. Port 8000 is loopback-only; no LAN/public listener exists.
3. Tailscale Serve provides the HTTPS endpoint and ACLs deny unauthorized tailnet clients.
4. vLLM requires the configured API key.
5. Local and remote `verify.sh --pytest` pass, including JSON Schema and XGrammar regex checks.
6. A benchmark baseline and exact serving provenance are recorded.
7. Reboot persistence, monitoring, API-key ownership/rotation, and rollback are documented and tested.

## 10. No-subagent rule

Repository scratch rules prohibit subagents for work in this project. Future implementation must proceed directly, one task at a time, and keep `CONTEXT.md`, `PROGRESS.md`, and `ISSUES.md` current.
