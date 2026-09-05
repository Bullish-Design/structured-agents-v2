# CONTEXT — 25 inferference contract consumer

## State

Branch `feat/inferference-capability-engine` in structured-agents-v2.
49 tests green. `ruff check` clean on every changed file.

This repo is NOT gitman/testee-wired (no `gitman.toml`, no `testee.toml`), so
version control here is plain git and verification is
`uv run --extra dev pytest` / `ruff check src`. inferference IS wired, and its
side of this work landed there as lanes `sa-contract-layer` and
`contract-install-split` (project `009-structured-agents-contract`).

## What changed

- New `src/structured_agents/engine/inferference.py`.
- `Engine` protocol gained `resolve_model(adapter, default)`.
- `Backend.negotiate()`; `Backend.build` no longer gates adapters on a constant.
- `__init__.py` no longer re-exports inferference's native surface.
- Core dependencies reduced; torch and the ABI pair leave a default install.

## Open question — needs hardware

Is a LoRA adapter an addressable model id on the router? The live test
(`tests/test_live.py:119-120`) assumes yes; nothing here establishes it.
`Capabilities.resolve_model_field` refuses fail-closed until it is measured.
Run against a live router with adapters loaded, then record the answer in
`inferference/.scratch/projects/009-structured-agents-contract/DECISIONS.md` D5.

## Next

From the concept doc's revised order: swap `DBOSAgent` for `DBOSDurability`
(`agent.py:12,99-105` is still the nested-workflow model), then operation
identity and idempotency conflict detection.

## REMINDER — NO SUBAGENTS

Do all work directly. Never use the Task tool.
