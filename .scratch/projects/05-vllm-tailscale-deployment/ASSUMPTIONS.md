# Assumptions

- The target is this NixOS host, not the historical Windows/WSL2 `tower` host described in the existing deployment runbook.
- The endpoint is private to the Tailscale tailnet; it must not be publicly internet-accessible.
- Tailscale Serve and tailnet ACLs are available to the operator; Funnel is out of scope.
- The first serving profile uses one RTX 3060 and a 3–4B quantized model. Two-GPU tensor parallelism is a later optimization.
- Access is controlled by the loopback-only listener plus the `group:agents` Tailscale grant; the native vLLM launcher does not require an API key.
- Postgres/DBOS dual-path capture is not needed to serve ordinary structured agents.
- Deployment configuration and secrets stay host-local and gitignored; repository changes will contain only safe configuration defaults, documentation, and validation tooling.
