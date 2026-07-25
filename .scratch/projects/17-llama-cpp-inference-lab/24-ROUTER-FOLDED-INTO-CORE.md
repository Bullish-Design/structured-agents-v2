# Flagship folded into `llama_core` — router surface + demo (2026-07-25)

> Turns the GPU-validated research router (`benchmarks/project17/`) into a typed
> library API. Follows the audit in `23-LLAMA-CORE-AUDIT.md`.

## Landed

- **`llama_core/router.py`** — Pydantic surface (`AdapterSpec`, `RouterConfig`,
  `RouteRequest`, `RouteResult(GenerationResult)`) + `MultiLoRARouter`: shared base
  model, one pinned-adapter context per adapter (+ base), batched multi-seq routing,
  optional grammar via the existing `grammar.py` hooks. Reuses `EngineConfig`,
  `decode.py`'s finish constants, `grammar.apply_packed_bitmask_inplace`. Boundary
  Pydantic; plain tokens/numpy on the hot path (repo rule).
- **`examples/multi_lora_router.py`** — runnable demo (`--adapter NAME=PATH`,
  `--constrained`).
- **`tests/test_llama_core_router.py`** — boundary-model contracts (no GPU).

## Decoupling fix (prerequisite that surfaced)

Importing any `llama_core` submodule eagerly pulled the whole durable-agent stack
(dbos, pydantic_ai, the plane) via `structured_agents/__init__.py`, so the teaching
core was NOT importable standalone (Pillar 1). Converted the package `__init__` to
**PEP 562 lazy loading** (`__getattr__` + `TYPE_CHECKING` block): `from
structured_agents import Agent` still works (loaded on first access), but
`import structured_agents.llama_core.router` no longer imports dbos/pydantic_ai.
Verified: standalone import loads neither; lazy symbols are identical objects;
`test_config.py` importlib-allowlist updated (now includes `__init__.py`, sorted);
full suite green (70 passed + 4 new router tests).

## Validation (GPU, Ornith + probe-a/probe-b, constrained)

Demo output, one mixed batch through the library API:
```
t0 probe-a stop  {"tool":"calendar","confidence":"high"}
t1 probe-b length null                # truncated — now OBSERVABLE
t2 probe-a stop  {"tool":"smart_home","confidence":"high"}
t3 probe-b stop  {"tool":"search","confidence":"high"}
```

**Library defect found + fixed:** the router first hardcoded `finish_reason="length"`
for every result. Threaded per-sequence finish through `_generate_wave` (grammar
`is_terminated` or sampled EOS → `stop`; else `length`), and only parse/validate a
grammar result on `stop`. Now truncation is observable instead of a silent null —
matching `decode.py`'s discipline. (t1's truncation is a model/greedy artifact: the
strict-JSON grammar permits whitespace and the model burned the budget on spaces
after `"confidence":` — not a library bug; the library reports it honestly.)

## Still open (next increments)

- Wire `prefix_cache.py` policy layer to the live `llama_state_seq_*` restore (the
  mechanism is proven in benchmarks; the library policy layer isn't connected).
- Generalize the two decode hooks into a middleware pipeline (Pillar 3).
- Batching/admission layer (Pillar 4).
- Optional: expose the P2 fork's `run_seq_routed` behind a `RouterConfig` flag.
- Retire `engine/{sglang,vllm}.py` (D1).
