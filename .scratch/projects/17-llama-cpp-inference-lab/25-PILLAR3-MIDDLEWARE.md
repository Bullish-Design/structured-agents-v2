# Pillar 3 — composable decode middleware

## What shipped
- `src/structured_agents/llama_core/middleware.py`
  - `DecodeMiddleware` — `@runtime_checkable` Protocol with four optional stage
    methods: `on_prompt(tokens)`, `on_logits(logits)`, `on_token(token) -> bool|None`,
    `on_finish(outcome)`.
  - `BaseDecodeMiddleware` — no-op base so subclasses override only what they need.
  - `MiddlewarePipeline` — composes an ordered list and dispatches each stage.
    `on_logits` runs left-to-right (order matters for chained maskers); `on_token`
    returns True if *any* middleware requests a stop but still calls every
    middleware so observers never miss the final token.
  - `as_pipeline()` — normalizes `None | pipeline | iterable` to a pipeline.
  - `CallbackMiddleware` — adapter lifting the legacy `logits_hook`/`token_hook`.
  - `StopTokenMiddleware` — example control-flow middleware.
  - `GrammarMiddleware` — wraps `JsonSchemaGrammar.logits_hook`/`token_hook` (thin
    composition, not a reimplementation).
- `decode.py` — `generate_tokens`/`generate_text` accept an optional `middleware`
  (pipeline or iterable). It is dispatched **after** the legacy hooks at each
  stage, so the old `logits_hook`/`token_hook`/`benchmark`/`synchronize` params
  keep working unchanged. Middleware `on_token` stop reuses `FINISH_STOP`
  semantics (token discarded, `stop_token` set).
- `tests/test_llama_core_middleware.py` — pipeline ordering, mutable-view proof,
  `on_token` stop, callback bridge, and grammar masking (direct-hook equivalence
  + selection steering in the loop).

## Hot-path discipline
`on_logits` receives the zero-copy numpy view; `on_token` receives a plain int.
No Pydantic / native / numpy import at module import time (matches `decode.py`;
`importlib` allowlist untouched).

## Design tradeoffs / deferred
- **Additive, not replacement.** Legacy hooks were kept as live call sites rather
  than internally rewritten into `CallbackMiddleware`, so per-stage benchmark
  labels (`matcher_accept`, etc.) stay byte-identical and Gate 1's single
  `llama_sampler_accept` source check is untouched. `CallbackMiddleware` exists so
  callers *can* unify them, but the decoder does not force it.
- Middleware-requested stop maps to `FINISH_STOP`; a distinct finish reason for
  "middleware stop" was considered and deferred (would change `DecodeOutcome`).
- No batch-admission / KV-event stages yet (single-seq owned loop only); the
  Protocol is the extension point when the batched router grows hooks.
- Not exported from `llama_core/__init__.py` — imported directly like
  `decode`/`router` to keep the shared import path light.

## Tests
`pytest tests/` → **85 passed, 1 skipped**. Targeted trio
(`test_llama_core_middleware` + `test_owned_decode` + `test_llama_core_grammar`)
→ 19 passed. Ruff clean. GPU e2e not run (unit fakes cover the loop).
