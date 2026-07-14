# Issues

## Active blockers

1. **GPU access was blocked only in this restricted workspace session.** Required probe on 2026-07-13:

   ```text
   $ nvidia-smi
   NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.

   $ find /dev -maxdepth 1 -name 'nvidia*'
   # no entries
   ```

   `/proc/driver/nvidia/version` is readable and reports NVIDIA open kernel module `595.84`, but that does not establish usable devices in this environment. The operator subsequently reported successful host-terminal NVIDIA checks. Docker GPU validation is no longer required because D4 selects native devenv serving. Do not use CPU inference as a workaround.
2. **Tailscale service access is unavailable only in this workspace session.** Required probe on 2026-07-13:

   ```text
   $ docker version
   Client: Version 29.6.0
   permission denied while trying to connect to the docker API at unix:///var/run/docker.sock

   $ docker compose version
   Docker Compose version 5.3.0

   $ tailscale status
   failed to connect to local tailscaled; it doesn't appear to be running (sudo systemctl start tailscaled ?)
   ```

   The Docker findings are no longer relevant to native serving. `systemctl` cannot access the system bus here, so Tailscale state cannot be queried from this session. `tailscale version` succeeds and reports `1.98.8`. Operator-supplied output confirms MagicDNS `server.tail770f47.ts.net`, `tag:vllm` assignment, and Preview Rules access from `group:agents, autogroup:admin` to TCP 443. This is resolved on the host; retain it only as an execution-environment limitation.

No repeated remediation attempts have been made, so no `ISSUE_<num>.md` incident log exists yet.
