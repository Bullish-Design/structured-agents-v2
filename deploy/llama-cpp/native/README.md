# Native llama.cpp comparison endpoint

This deployment runs the same immutable `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
as the native vLLM service, but in a separate `llama-server` process:

| Backend | GPU | Local URL | Tailnet URL |
|---|---:|---|---|
| vLLM | 1 | `http://127.0.0.1:8000/v1` | `https://server.tail770f47.ts.net/v1` |
| llama.cpp | 0 | `http://127.0.0.1:8001/v1` | `https://server.tail770f47.ts.net:8443/v1` |

The llama.cpp profile offloads all target and MTP drafter layers, uses Flash
Attention, keeps a q8 KV cache on GPU 0, and provides one generation slot. It
uses Unsloth's `mtp-gemma-4-12B-it.gguf` drafter with a four-token proposal
window; the target verifies every proposal, so MTP changes throughput rather
than output quality. This preserves a 16,384-token context on the 12 GiB GPU
without any CPU offload. Increase `parallelSlots` only after measuring its effect
on latency and available VRAM.

## Enable

Import both NixOS modules and add the second service configuration:

```nix
imports = [
  /home/andrew/Documents/Projects/structured-agents-v2/deploy/vllm/native/nixos-module.nix
  /home/andrew/Documents/Projects/structured-agents-v2/deploy/llama-cpp/native/nixos-module.nix
];

services.structuredAgentsLlamaCpp = {
  enable = true;
  repositoryPath = "/home/andrew/Documents/Projects/structured-agents-v2";
  user = "andrew";
  group = "users";
  # Override only if the existing local GGUF is not in vLLM's Xet cache.
  # modelPath = "/absolute/path/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
  # draftModelPath = "/absolute/path/mtp-gemma-4-12B-it.gguf";
};
```

Then apply and start it:

```bash
sudo nixos-rebuild switch
sudo systemctl enable --now structured-agents-llama-cpp.service
sudo systemctl enable --now structured-agents-llama-cpp-tailscale-serve.service
```

Verify both services independently:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8001/health
deploy/llama-cpp/native/verify.sh
```

`deploy/vllm/verify.sh`'s XGrammar-specific check is intentionally vLLM-only. The
llama.cpp verifier checks health, model discovery, and a chat completion. The
Tailscale listener is on `:8443`, so it
does not overwrite vLLM's `:443` Serve rule. Tailnet ACLs remain the network access
boundary; neither backend listens on a LAN or public address.

For a like-for-like generation-throughput run, warm up each service and write
separate CSVs with the existing profiler:

```bash
LLM_BASE_URL=http://127.0.0.1:8000/v1 BENCH_OUTPUT=/tmp/vllm.csv deploy/vllm/profile_tps.py
LLM_BASE_URL=http://127.0.0.1:8001/v1 BENCH_OUTPUT=/tmp/llama-cpp.csv deploy/vllm/profile_tps.py
```
