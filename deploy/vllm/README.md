# vLLM container — drop-in replacement for `remora-server`

A standalone, OpenAI-compatible vLLM server that replaces the current llama.cpp
`remora-server` while keeping the **same contract**: `http://<host>:8000/v1`.

This is what unblocks the parts of the xgrammar concept the request-path spike
*couldn't* verify against llama.cpp: actual **XGrammar** constrained decoding and
**per-agent LoRA** batched serving. See
`../../.scratch/projects/01-xgrammar-concept/spike/FINDINGS.md`.

## Why this exists

`remora-server:8000` today is **llama.cpp** serving a Qwen3.5-9B **GGUF** quant with
**no LoRA** and GBNF (not XGrammar) grammars. The library targets vLLM. This container
makes the swap a config change, not a code change — clients keep pointing at the same
base URL.

## Quick start

```bash
cp .env.example .env       # then edit MODEL, VLLM_API_KEY, LORA_MODULES, MODELS_HOST_DIR
docker compose up --build
curl -fsS http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

Point the library at it:

```bash
LLM_BASE_URL=http://<host>:8000/v1
LLM_API_KEY=<VLLM_API_KEY>
LLM_MODEL=base            # or a LoRA adapter name from LORA_MODULES
```

## Host model layout (bind-mounted read-only at `/models`)

```
$MODELS_HOST_DIR/
├── base/                 # HF-format (safetensors) base weights  — NOT GGUF
└── lora/
    ├── file-edit/        # one dir per fine-tuned adapter
    ├── git-ops/
    └── test-runner/
```

## How the pieces map to the verified request path

| Concept piece            | vLLM mechanism (set here)                                   |
|--------------------------|-------------------------------------------------------------|
| XGrammar constraint      | `--guided-decoding-backend xgrammar` (server-level)         |
| Per-agent output schema  | client sends `response_format: json_schema` (NativeOutput)  |
| Per-agent LoRA           | `--enable-lora --lora-modules name=path`; client `model=name` |
| Batched parallelism      | vLLM continuous batching (no flag; inherent)                |
| Auth                     | `--api-key $VLLM_API_KEY`                                    |

## Target: RTX 3060 (12 GB) — decisions & tuning

Confirmed constraints (2026-06-09): single **12 GB RTX 3060**, newest **Qwen3.5**,
**no LoRA adapters yet**.

- **Quantization is mandatory.** fp16 weights for a 9B (~18 GB) don't fit. The
  default `.env` uses on-the-fly 4-bit (`--quantization bitsandbytes`), which works
  from plain fp16 HF weights. Prefer a pre-quantized **AWQ** repo if one exists for
  the chosen Qwen3.5 (faster). Context is capped at `MAX_MODEL_LEN=8192` to leave
  VRAM for KV cache.
- **Size vs. the multi-agent goal.** On 12 GB a 4-bit 9B leaves little room for
  several concurrent LoRAs. A **3-4B-class Qwen3.5** fits comfortably and matches the
  "small models, constrained toolsets, batched" intent — `.env` defaults to a 4B; bump
  `MODEL` to the 9B only if you accept ~1-2 small-rank LoRAs and 8k context.
- **vLLM version + flag.** Confirm the structured-outputs flag against the pinned
  `VLLM_TAG` (`--guided-decoding-backend` vs `--structured-outputs-config.backend`).
- **LoRA.** `LORA_MODULES` empty until fine-tunes land; `MAX_LORA_RANK` must be ≥ the
  training rank, and verify LoRA + quantization compatibility on the pinned tag.

> The host's existing GGUF (`Qwen3.5-9B-UD-Q6_K_XL.gguf`) is **not** vLLM-loadable as a
> drop-in; vLLM's GGUF support is experimental and doesn't pair well with LoRA. Source
> HF safetensors for the chosen Qwen3.5 instead.
