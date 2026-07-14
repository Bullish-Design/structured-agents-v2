# Progress

| Status | Task |
|---|---|
| done | Audit repository deployment scaffold and plan secure Tailscale architecture. |
| done | Inspect host readiness read-only and identify deployment blockers. |
| done | Confirm host NVIDIA device access. Operator reported successful host-terminal checks on 2026-07-13; native devenv serving replaces Docker (D4). |
| done | Confirm Tailscale login, hostname, and ACL/tag authority. `server.tail770f47.ts.net` is tagged `tag:vllm`; preview permits `group:agents`/admins to TCP 443. |
| done | Implement loopback-only Compose and native Linux/NixOS/Tailscale deployment configuration and documentation. |
| in-progress | Configure and start the locked, single-GPU native vLLM systemd service. Requires host-local secret and NixOS configuration-repository import. |
| pending | Verify local and remote constrained output, then benchmark and establish operations. |
