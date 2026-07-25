# 28 — Retire the legacy sglang/vLLM provider-abstraction engine (Decision D1)

## Verdict
Retired the sglang and vLLM engine plugins. Kept a single thin wire renderer
(`LlamaCppEngine`) plus the `Engine` protocol and `select()`, because the DBOS
durable-agent plane (`agent.py::Backend`) renders `Constraint` → `WireSpec` through
an engine and must keep working. After project 17's pivot to llama.cpp as the single
substrate, llama.cpp is the only wire dialect that remains meaningful, so the
abstraction collapses to one built-in rather than disappearing.

## Consumer audit
`structured_agents.engine` and its symbols are consumed in exactly these places:
- `src/structured_agents/agent.py` — `Backend` imports `Engine, select`; uses
  `engine.supports`, `engine.name`, `engine.render(constraint)`. **Load-bearing for
  the durable plane.** Migrated: default `engine="vllm"` → `engine="llama_cpp"`.
- `tests/test_engine.py` — exercised all three engines via `select(...)`.
- `tests/test_live.py` — opt-in (`SAV_LIVE=1`); `LLM_ENGINE` env default `"vllm"` → `"llama_cpp"`.

The many `from llama_cpp import ...` hits in `examples/`, `benchmarks/project17/`, and
`tests/test_prefix_cache_*` are the **llama-cpp-python PyPI package** used by the owned
`llama_core` decode path — unrelated to the engine abstraction, left untouched.
`config.py` / package `__init__.py` have no engine references; the PEP 562 lazy
`__init__` does not name engine symbols, so no change needed there.

## Removed
- `src/structured_agents/engine/sglang.py` (`SGLangEngine`)
- `src/structured_agents/engine/vllm.py` (`VLLMEngine`)
- `tests/test_engine.py::test_vllm_bytes_are_unchanged`
- `tests/test_engine.py::test_sglang_dialect` (and now-unused `NativeOutput` import)
- `_BUILTINS` entries for `"vllm"` and `"sglang"` in `engine/__init__.py`

## Kept (with reason)
- `engine/base.py` (`Engine` protocol) — the durable plane types against it.
- `engine/llama_cpp.py` (`LlamaCppEngine`) — sole remaining wire renderer; matches the
  llama.cpp substrate pivot.
- `engine/__init__.py::select` + `ConfigError` on unknown engine — still the resolution
  path used by `Backend`.

## Migrations
- `Backend.__init__` default engine `"vllm"` → `"llama_cpp"`.
- `tests/test_live.py` `LLM_ENGINE` default `"vllm"` → `"llama_cpp"`.

## Deliberately left in place
- `tests/live_crash_worker.py` `raw-vllm-*.json` artifact filenames — cosmetic output
  names in an opt-in crash-proof worker, not engine-abstraction references.

## Verification
- `PYTHONPATH=src python -m pytest tests/` → **90 passed, 2 skipped**.
- `import structured_agents` and `import structured_agents.llama_core.router` both OK.
- `select("llama_cpp").name == "llama_cpp"`; `select("vllm")`/`select("sglang")` now
  raise `ConfigError("Unknown engine ...")`.
- `ruff check` clean on touched files.
