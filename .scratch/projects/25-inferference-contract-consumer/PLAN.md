# PLAN — 25 inferference contract consumer

## The defect this closes

`LlamaCppEngine.supports` is a constant that omits `lora`
(`src/structured_agents/engine/llama_cpp.py:22`), and `Backend.build` raised
whenever a spec named an adapter (`agent.py:57-60`). So **every adapter-bearing
spec was an unconditional error**, and the only test that would catch it needs
two opt-in environment variables (`tests/test_live.py:141-149`). The 1.6x-2.0x
mixed-adapter capability inferference measured was unreachable from here.

## Steps

1. `engine/inferference.py` — `InferferenceEngine`, abilities derived from a
   negotiated `Capabilities` instead of a constant. done
2. `Engine.resolve_model(adapter, default)` on the protocol; naming a model is
   the engine's call, not a coarse `"lora" in supports` flag. done
3. `LlamaCppEngine.resolve_model` keeps today's refusal, now with its reason.
   done
4. `Backend.negotiate(base_url, ...)` — probe, then build a backend that knows.
   done
5. Drop the inferference re-exports from `__init__.py`. done
6. Strip the core dependency set to the control plane. done

## Acceptance

- [x] An adapter-bearing spec builds when the server publishes the adapter.
- [x] An adapter the server loaded but does not publish is refused, with the
      reason.
- [x] `llama_cpp` engine still refuses an adapter it cannot prove.
- [x] `uv sync` (core only) installs no torch, transformers, xgrammar, numpy or
      `llama_cpp`.
- [x] 49 tests pass; `ruff check` clean on every changed file.

## REMINDER — NO SUBAGENTS

Do all work directly. Never use the Task tool.
