# Spike: request-path verification

**Date:** 2026-06-09
**Goal:** verify exactly what PydanticAI puts on the wire for per-agent constrained,
schema-validated output, and whether concurrent agent runs batch — against a live
OpenAI-compatible server.

Run with: `devenv shell -- spike-run` (env in `devenv.nix`: `LLM_BASE_URL`, `LLM_MODEL`).

---

## TL;DR

The **client-side request path is fully verified** and PydanticAI 1.87.0 already
exposes every knob the xgrammar concept needs — no monkey-patching required.

But the biggest finding is about the *server*: **`remora-server:8000` is not vLLM.**

---

## Finding 0 (critical): the running server is llama.cpp, not vLLM

`GET /v1/models` → `"owned_by":"llamacpp"`, `"format":"gguf"`.
`GET /props` →
- model `Qwen3.5-9B-UD-Q6_K_XL.gguf` (Unsloth dynamic Q6 GGUF) at `/models/base/`
- `total_slots: 4` (continuous-batching slots), `n_ctx: 16384`
- `lora: []`, and `GET /lora-adapters` → `[]` → **no LoRA adapters loaded**

Implications for the concept (vLLM + XGrammar + per-agent LoRA, batched):
- **Constraint engine differs.** llama.cpp constrains with **GBNF grammars** (incl.
  json-schema→GBNF), *not* XGrammar. The `response_format: json_schema` path works,
  but XGrammar-specific behavior/perf is not what's running.
- **No multi-LoRA batched serving.** llama.cpp hot-swaps adapters with per-request
  scaling; it does not do vLLM-style batched per-request LoRA. None are loaded anyway.
- This is exactly why the next step is to **replace remora-server with a vLLM container.**

## Finding 1: native structured output works end-to-end ✅

`Agent(model, output_type=NativeOutput(FileEditPlan))` →
- on the wire: `response_format: {"type": "json_schema", "json_schema": {"name", "schema", "strict": false}}`
- the server constrained Qwen to valid JSON; PydanticAI validated it into the Pydantic model.

## Finding 2: `extra_body` passthrough works ✅ (the XGrammar hook)

`OpenAIChatModelSettings(extra_body={"structured_outputs": {...}, "guided_decoding_backend": "xgrammar"})`
→ both keys appear **verbatim at the top level** of the JSON body (see
`captured_request.json`). This is the mechanism for vLLM grammar/regex/choice modes
that `response_format` can't express.

Note: llama.cpp ignores these keys; the live call in step 2 hit a 512-token budget
before finishing (not a path failure — the capture proves the bytes were sent).

Guidance: with a real **vLLM** backend, prefer `NativeOutput`→`response_format`
json_schema and set XGrammar as the **server-level** backend
(`--structured-outputs-config.backend xgrammar`). Reserve `extra_body` for
grammar/regex/choice modes.

## Finding 3: per-agent model = LoRA selector ✅

The `model` field is set per `OpenAIChatModel` instance and lands as `"model": "..."`
on the wire. With vLLM `--enable-lora --lora-modules name=path`, that field selects the
adapter. (Couldn't exercise live: no adapters on this server.)

## Finding 4: concurrency batches ✅

4 × `agent.run()` via `asyncio.gather`: **10.1s concurrent vs 24.3s sequential = 2.4×**
on this 4-slot llama.cpp server. PydanticAI async dispatch parallelizes; the server
batches. (vLLM continuous batching should do better, esp. with paged-attention.)

---

## Verified PydanticAI 1.87.0 surface (for the library)

```python
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
```
- `ModelSettings` / `OpenAIChatModelSettings` both have first-class `extra_body` and `extra_headers`.
- `OpenAIProvider(base_url=, api_key=, http_client=)` — `http_client` lets us inject an
  httpx event hook to capture/inspect the wire (used by this spike).
- `max_tokens` is sent as `max_completion_tokens`.

## What is NOT verified (needs the real vLLM + GPU box)

- That XGrammar actually constrains decoding (vs llama.cpp GBNF).
- That a LoRA adapter loads and changes behavior.
- vLLM continuous-batching throughput / LoRA-swap overhead.

These unblock once the vLLM container (see `deploy/vllm/`) replaces remora-server.
