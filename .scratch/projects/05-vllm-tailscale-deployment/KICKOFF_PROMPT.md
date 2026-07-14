# Clean-session kickoff prompt

```text
You are working in the `structured-agents-v2` repository. Your objective is to deploy a private, authenticated vLLM endpoint on this NixOS host for structured-agents-v2 clients, reachable only over Tailscale HTTPS.

Start by reading these files in full, in this order:

1. `.scratch/CRITICAL_RULES.md`
2. `.scratch/REPO_RULES.md`
3. `.scratch/projects/05-vllm-tailscale-deployment/ASSUMPTIONS.md`
4. `.scratch/projects/05-vllm-tailscale-deployment/DECISIONS.md`
5. `.scratch/projects/05-vllm-tailscale-deployment/CONTEXT.md`
6. `.scratch/projects/05-vllm-tailscale-deployment/PROGRESS.md`
7. `.scratch/projects/05-vllm-tailscale-deployment/ISSUES.md`
8. `.scratch/projects/05-vllm-tailscale-deployment/PLAN.md`
9. `deploy/vllm/README.md`, `deploy/vllm/docker-compose.yml`, `deploy/vllm/entrypoint.sh`, and `deploy/vllm/verify.sh`

Follow the repository scratch rules exactly: do not use subagents; work directly; keep `CONTEXT.md`, `PROGRESS.md`, and `ISSUES.md` current. Do not discard, overwrite, or reset unrelated user changes. Use `apply_patch` for repository edits.

## Target architecture

```text
authorized Tailscale client
  -> HTTPS + Bearer API key
  -> Tailscale Serve :443
  -> http://127.0.0.1:8000
  -> vLLM Docker container
  -> one RTX 3060 initially
```

The vLLM API must not be exposed on the LAN or public internet. Docker must bind the API only to loopback (`127.0.0.1:8000:8000`). Tailscale Serve is the only remote ingress; do not use Tailscale Funnel. Tailnet ACLs restrict which identities can reach the node on port 443. The vLLM API key remains required as independent application-level authorization.

## Known environment and current blockers

- Host: NixOS 26.11, approximately 62 GiB RAM and 350 GiB free disk.
- The loaded NVIDIA driver detects two RTX 3060 GPUs, but the prior planning session could not use `nvidia-smi` and could not see `/dev/nvidia*`. Treat working host GPU access as unverified until you prove it.
- Docker is installed, but the prior session could not access the Docker socket. Confirm the intended deployment account can use Docker.
- NVIDIA container-runtime support is unverified. Confirm GPU visibility inside Docker before pulling a model.
- Tailscale is installed, but the prior session could not query local status. Confirm login, MagicDNS hostname, and who can edit/request ACL and tag policy.
- The current repository Compose file maps `8000:8000`, which is insecure for this objective and must be changed before deployment.
- Do not deploy Postgres/DBOS as part of this work. It is unrelated to ordinary vLLM serving.

## Required order of work

1. **Host readiness — no deployment yet.** Run and record:

   ```bash
   nvidia-smi
   ls -l /dev/nvidia*
   docker version
   docker compose version
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   tailscale status
   tailscale version
   ```

   If GPU or Docker GPU access fails, do not work around it by using CPU inference. Diagnose safely, document the exact failure in `ISSUES.md` (and create an `ISSUE_<num>.md` after repeated attempts per scratch rules), and report the blocker.

2. **Confirm Tailscale policy prerequisites.** Determine the node’s stable MagicDNS name and the authorized client group/tag. The desired policy shape is illustrative:

   ```json
   {
     "tagOwners": {"tag:vllm": ["group:infra-admins"]},
     "acls": [
       {"action": "accept", "src": ["group:agents", "group:infra-admins"], "dst": ["tag:vllm:443"]}
     ]
   }
   ```

   Do not apply a tailnet ACL or tag policy without the appropriate authority. If it requires a tailnet administrator, prepare the exact requested change and stop for direction.

3. **Make safe repository changes.** Update the deployment configuration and documentation for NixOS plus Tailscale Serve. At minimum:

   - Change Compose to bind vLLM only to `127.0.0.1:8000:8000`.
   - Document Tailscale Serve and the HTTPS client base URL: `https://<machine>.<tailnet>.ts.net/v1`.
   - Document that direct port 8000 and Funnel are prohibited for this deployment.
   - Keep the API key in a gitignored, mode-0600 host-local `.env`; do not print secrets.
   - Add a NixOS/Linux deployment path without removing the existing Windows/tower guidance.
   - Ensure the no-build override cannot silently diverge from `VLLM_TAG`.

4. **Configure a conservative first profile.** Start with one RTX 3060, no LoRAs, a pinned 3–4B AWQ/Hugging Face model, `MAX_MODEL_LEN=8192`, and conservative GPU memory utilization (initially `0.90`, lower it if needed). Record exact image digest/tag, model revision, quantization, context limit, GPU memory setting, and rollback profile outside version control or in safe non-secret operational documentation.

5. **Bring up and validate locally.** Only after the host readiness gates pass, use Docker Compose to start vLLM. Require:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   LLM_BASE_URL=http://127.0.0.1:8000/v1 \
   LLM_API_KEY="$VLLM_API_KEY" \
   LLM_MODEL=base \
   deploy/vllm/verify.sh
   ```

   A successful health check alone is insufficient. JSON Schema and XGrammar regex checks must pass. Then run the repository live markers using `verify.sh --pytest` from a functional development environment.

6. **Publish through Tailscale Serve.** After local validation and approved ACL policy:

   ```bash
   tailscale serve --bg --https=443 http://127.0.0.1:8000
   tailscale serve status
   ```

   Verify from an authorized remote tailnet device using the HTTPS base URL and API key. Also verify unauthorized identities are blocked and that port 8000 is not reachable by LAN address, public address, or Tailscale IP.

7. **Benchmark and operationalize.** Run `deploy/vllm/bench.py` through the same Tailscale HTTPS URL clients will use at concurrency 1, 2, 4, 8, and 16 after warm-up. Record throughput, p95 latency, individual-request error rate, JSON validity, and GPU memory behavior. Establish reboot persistence for Docker Compose and Tailscale Serve, monitoring/log retention, API-key ownership/rotation, cache-capacity ownership, and a tested rollback path.

## Acceptance criteria

Do not call this complete until all conditions are met:

1. A GPU is visible inside the vLLM container and the container stays healthy after restart.
2. Port 8000 is loopback-only; there is no LAN/public listener.
3. Tailscale Serve provides HTTPS and ACLs deny unauthorized tailnet clients.
4. vLLM requires its configured API key.
5. Local and remote `deploy/vllm/verify.sh --pytest` pass, including JSON Schema and XGrammar regex checks.
6. A benchmark baseline and exact model/image configuration are recorded.
7. Operations, secret rotation, monitoring, persistence, and rollback are documented and tested.

Keep the work scoped to this deployment. Do not implement the optional dual-path/Postgres subsystem, LoRAs, or two-GPU tensor parallelism in the initial release. If a new authority, credential, ACL change, or destructive operation is needed, report the concrete requirement and ask for direction rather than assuming it.
```
