# Decisions

## D1 — Use Tailscale Serve rather than direct port 8000 access

The service will be published through Tailscale Serve on HTTPS port 443 while Docker binds vLLM only to loopback. This avoids the existing all-interface `8000:8000` exposure and supplies tailnet-scoped TLS. Tailscale ACLs constrain who can connect; the native deployment does not configure a separate vLLM API key.

## D2 — Start with one GPU and no LoRAs

The host has two RTX 3060 GPUs, but current GPU/container access is unverified and tensor-parallel serving adds failure modes. Establish a single-GPU, no-LoRA baseline first, then assess expansion using measured capacity data.

## D3 — Require functional constrained-output proof before cutover

Successful health checks alone are insufficient. JSON Schema and XGrammar regex validation, followed by the library live markers over the final Tailscale URL, are required cutover gates.

## D4 — Run vLLM natively in a dedicated devenv environment

The operator chose a direct host process managed by devenv rather than a Docker container. This removes Docker daemon, socket, and NVIDIA Container Toolkit/CDI dependencies, but does not relax the host-NVIDIA, loopback-only binding, Tailscale Serve, ACL, API-key, verification, provenance, persistence, or rollback requirements. Keep the native service definition in this repository's deployment area instead of creating an unrelated repository, so the service and its structured-output verification remain versioned together.

## D5 — Use `group:agents` for vLLM clients and tailnet admins for tag ownership

The operator selected `group:agents` as the authorized client identity set. Use `tag:vllm` on the serving node and `autogroup:admin` as its owner, the simplest standard tailnet-administration model. This is still a private endpoint: the policy must not retain a broad allow rule that gives other tailnet identities access to `tag:vllm:443`.
