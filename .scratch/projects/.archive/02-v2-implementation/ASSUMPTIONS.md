# ASSUMPTIONS — v2-implementation

Load this file before making any decision. Update it as assumptions change.

---

## Project Audience

- Developers building AI agents with structured outputs
- Users familiar with Pydantic and type-safe Python
- Internal team at Bullish Design and open-source community

## User Scenarios

- Define agents that return structured data (Pydantic models) from LLM calls
- Compose multiple agents into pipelines or workflows
- Validate and retry agent outputs against schemas
- Integrate with various LLM providers via Pydantic AI

## Constraints

- Python >= 3.13
- Must use Pydantic AI for LLM interactions
- Must be type-safe (mypy strict mode)
- Must have tests (pytest with coverage)
- Must follow TDD workflow

## Invariants

- All agent outputs are validated Pydantic models
- No untyped function definitions
- All public APIs have docstrings and type hints
- Configuration via Pydantic Settings or similar (TBD)

## Technical Context

- Build system: hatchling
- Package location: `src/structured_agents_v2/`
- Testing: pytest with coverage reporting
- Linting/formatting: ruff
- Type checking: mypy (strict)
