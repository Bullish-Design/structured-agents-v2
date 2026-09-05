# DECISIONS — 25 inferference contract consumer

## D1. Capabilities replace the coarse `supports` flag for model naming

`Engine` gained `resolve_model(adapter, default)`. `"lora" in supports` could
only ever be a guess; an engine that negotiated with its server knows whether
a specific adapter is addressable, and can say why not.

`supports` keeps its narrower, honest meaning: which constraint dialects the
engine renders.

## D2. `LlamaCppEngine` still refuses adapters — on purpose

Its abilities are a constant, so it can establish nothing about a server. The
refusal now names the reason and points at `Backend.negotiate`. The existing
test (`tests/test_engine.py:33-37`) still passes unchanged.

## D3. Generation stays on the OpenAI transport

No custom `pydantic_ai.models.Model` in this lane. `llama-server` already does
chat templating, tool-call parsing, streaming and usage accounting. The defect
above is fixed by negotiation alone. Revisit when this library needs what
OpenAI cannot express — context-tree operations, per-sequence routing control.

## D4. The native decode surface is no longer re-exported

`structured_agents.MultiLoRARouter` exported a VRAM-holding native object from
the control plane, which invites a DBOS worker to construct one. Removed.
Import from `inferference` directly to get the runtime.

## D5. Core install is the control plane only

Measured after the change: `uv sync` drops torch, transformers, tokenizers,
triton, sympy, xgrammar, numpy and `llama_cpp`. torch arrived transitively
through xgrammar. `numpy`/`xgrammar` moved to `dev` (constraint.check() already
degrades to a no-op without them), `transformers` to a `training` extra, and
the ABI pair sits behind `runtime`, which forwards to `inferference[runtime]`.

The Mode A wheel pin stays in `[tool.uv.sources]`: it now applies only when
something pulls `llama-cpp-python` into the graph, and it must still win over
PyPI when it does.

## Out of scope, noted

`src/structured_agents/training/token_monitor.py` fails `ruff check` with 12
errors on `main`, before this lane. Untouched here. It is the sole importer of
transformers and is exported from nothing — a removal candidate, per the
concept doc's Phase 7.
