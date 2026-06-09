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

## Run the library's live tests against this service

The `LLM_*` block in `.env` is the client-side mirror of the server settings
(`LLM_API_KEY == VLLM_API_KEY`, `LLM_MODEL == SERVED_MODEL_NAME`). Load it and run the
`live`-marked tests (skipped by default, so the normal suite stays GPU-free):

```bash
export $(grep -E '^LLM_' deploy/vllm/.env | xargs)

# json_schema round-trip (also works against the current llama.cpp box):
SAV_LIVE=1 devenv shell -- uv run --extra dev pytest -q -m live

# XGrammar bare-string (regex) round-trip — vLLM-only, proves constrained decoding:
SAV_LIVE_XGRAMMAR=1 devenv shell -- uv run --extra dev pytest -q -m live
```

The XGrammar test asserts the returned text actually matches the regex, so it only
passes once a real XGrammar backend (this container) is serving — that's the part the
llama.cpp box can't satisfy.

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
| XGrammar constraint      | structured-outputs backend flag, **auto-detected** from `vllm serve --help` (modern `--structured-outputs-config.backend xgrammar`, else `--guided-decoding-backend`, else vLLM's xgrammar default) |
| Per-agent output schema  | client sends `response_format: json_schema` (NativeOutput)  |
| Bare-string constraints  | client sends `structured_outputs: {regex\|choice\|grammar: …}` in the body (modern vLLM form) |
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
- **vLLM version + flag.** `VLLM_TAG` is pinned (default `v0.11.0`); bump it to the
  newest tag that supports your Qwen3.5. The structured-outputs flag is auto-detected
  at startup (see `entrypoint.sh`) — the boot log prints the resolved `vllm serve …`
  line, so you can confirm which flag was used. The tag must be new enough (vLLM 0.10+)
  to accept the client's modern `structured_outputs` request body.
- **LoRA.** `LORA_MODULES` empty until fine-tunes land; `MAX_LORA_RANK` must be ≥ the
  training rank, and verify LoRA + quantization compatibility on the pinned tag.

> The host's existing GGUF (`Qwen3.5-9B-UD-Q6_K_XL.gguf`) is **not** vLLM-loadable as a
> drop-in; vLLM's GGUF support is experimental and doesn't pair well with LoRA. Source
> HF safetensors for the chosen Qwen3.5 instead.
