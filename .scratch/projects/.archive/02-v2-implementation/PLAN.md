# PLAN — v2-implementation

**CRITICAL: NEVER use subagents (the Task tool) under any circumstances. Do all work directly.**

---

## Goal

Implement the structured-agents-v2 library for building AI agents with structured outputs using Pydantic AI.

## Acceptance Criteria

- [ ] Library can define agents that return validated Pydantic models
- [ ] Basic agent execution with Pydantic AI works
- [ ] Test coverage >= 80%
- [ ] All mypy strict checks pass
- [ ] All ruff lint checks pass
- [ ] Documentation (README.md) updated with usage examples

## Implementation Steps (Ordered)

### Phase 1: Foundation
1. Define core `Agent` class interface
2. Implement basic agent using Pydantic AI
3. Write tests for basic agent execution
4. Set up project structure (`src/structured_agents_v2/`)

### Phase 2: Features
5. Add agent configuration (model, temperature, etc.)
6. Add output validation and retry logic
7. Add agent composition (pipelines)
8. Add error handling and custom exceptions

### Phase 3: Polish
9. Add docstrings and type hints to all public APIs
10. Update README.md with examples
11. Final test coverage check
12. Final linting and type checking pass

## Dependencies

- pydantic-ai (to be added via uv)
- pydantic (already in dependencies)
- pytest, mypy, ruff (dev dependencies)

---

**CRITICAL: NEVER use subagents (the Task tool) under any circumstances. Do all work directly.**
