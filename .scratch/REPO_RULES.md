# REPO_RULES.md — structured-agents

Repository-specific coding standards and conventions for structured-agents.

---

## Overview

structured-agents is a small library (~1,438 lines, 21 files) for running tool-using LLM agents. Its primary consumer is Remora — it is not a standalone tool.

---

## Directory Structure

```
src/structured_agents/
├── __init__.py           # Public API exports
├── types.py              # Core types: Message, ToolCall, ToolResult, etc.
├── exceptions.py         # Exception hierarchy
├── kernel.py             # AgentKernel — the main step loop
├── agent.py              # Agent class (DEAD CODE — removal target)
├── client/
│   ├── protocol.py       # LLMClient protocol
│   └── openai.py         # OpenAICompatibleClient (vLLM-only)
├── models/
│   ├── adapter.py        # ModelAdapter (DEAD CODE — removal target)
│   └── parsers.py        # ResponseParser, QwenResponseParser
├── grammar/
│   ├── config.py         # GrammarConfig
│   ├── pipeline.py       # ConstraintPipeline
│   └── models.py         # DecodingConstraint
├── events/
│   ├── types.py          # Event dataclasses
│   └── observer.py       # Observer protocol
└── tools/
    ├── protocol.py       # Tool protocol
    └── grail.py          # GrailTool (DEAD CODE — removal target)
```

---

## Coding Standards

### Typing

- All code must be fully typed (no untyped functions).
- Avoid `Any` unless unavoidable.
- Use `Protocol` for interfaces, not ABCs.
- Maintain mypy compatibility.

### Imports

- Preserve existing import order.
- Prefer explicit imports over `from x import *`.
- No circular imports.

### Formatting

- Follow existing formatting conventions in the file you're editing.
- Do not reformat unrelated code.
- Do not change whitespace unnecessarily.

### Edits

- Make surgical edits, not file rewrites.
- Do not refactor unrelated code.
- Do not rename symbols unless required by the task.

---

## Testing

- Tests live in `tests/`.
- Use pytest.
- Write focused tests covering only changed behavior.
- Keep tests deterministic.
- Run tests before marking a task complete: `pytest tests/`

---

## Dependencies

### Vendored Context

vLLM and xgrammar sources are vendored under `.context/`:

```
.context/vllm/
.context/xgrammar/
```

When behavior involves grammar-constrained decoding, sampling parameters, or engine internals, consult the vendored sources — not external documentation or memory.

### Runtime Dependencies

- `openai` — AsyncOpenAI client (current)
- `litellm` — Multi-provider routing (v0.4 target)
- `pydantic` — Type validation

---

## Key Files for v0.4 Refactor

| File | Relevance |
|------|-----------|
| `kernel.py` | Core step loop — receives response_parser and constraint_pipeline directly (no adapter) |
| `client/openai.py` | Replaced by `litellm_client.py` |
| `models/adapter.py` | Removed — adapter indirection eliminated |
| `models/parsers.py` | `QwenResponseParser` renamed to `DefaultResponseParser` |
| `events/types.py` | Dataclasses → Pydantic models |
| `agent.py` | Removed — Remora never uses it |
| `tools/grail.py` | Removed — Remora has its own tool system |

---

## Running Commands

```bash
# Run tests
pytest tests/

# Type check
mypy src/

# Lint
ruff check src/

# Format check
ruff format --check src/
```

---

## Active Project

The current active project is:

```
.scratch/projects/v04-refactor/
```

See `s-a_v04_concept.md` in that directory for the full architecture concept.
