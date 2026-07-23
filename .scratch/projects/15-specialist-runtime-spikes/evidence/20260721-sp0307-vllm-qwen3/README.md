# SP-03..SP-07 — vLLM Qwen3-4B-AWQ + multi-LoRA + constraints (live)

Date: 2026-07-21
Server: vLLM 0.25.0, `Qwen/Qwen3-4B-AWQ`, GPU 1, `127.0.0.1:8003`, enforce-eager,
`--enable-lora --max-loras 2 --max-lora-rank 8`, xgrammar backend,
adapters `control-a` / `control-b` preloaded.

## Proven

- **Health** — `health.txt` = 200.
- **Model listing** — `models.json` lists `base`, `control-a` (parent `base`), `control-b`.
- **Constrained decode** — all three constraint modes validated per-item:
  - JSON Schema (`const` marker object)
  - regex (`marker-[0-9]{48}`)
  - EBNF grammar (fixed root production)
- **Adapter isolation** — base / control-a / control-b each requested and validated.
- **Concurrent mixed load** — 12 requests fanned out over base + both adapters with
  distinct constraints each; `summary.json` `all_validated: true`.
- **Engine metrics** (`metrics-samples.json`, 1147 samples):
  - peak `vllm:num_requests_running` = **8.0** (batch saturated at `max-num-seqs 8`)
  - `vllm:lora_requests_info` observed with `running_lora_adapters="control-a,control-b"`
    — two adapters resident and scheduled in the same batch.

## Artifacts

- `summary.json`, `outcomes.json`, `metrics-samples.json`, `models.json`, `health.txt`
- Probe: `../../probes/live_specialist_probe.py`
- Wall time: 149.4 s (enforce-eager, ~1-2 tok/s on RTX 3060).

## Not covered here

SGLang equivalent (port 8002) — pending env rebuild and llama.cpp GPU-0 handover.
