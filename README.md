# structured-agents

Durable primitives for building crash-recoverable, exactly-once, observable
constrained-agent workflows on DBOS and OpenAI-compatible engines (vLLM, SGLang, llama.cpp) with XGrammar-family
constrained decoding.

The library provides typed constrained generation, authorization, durable
effects, and human approval primitives. Applications compose them in their own
DBOS workflows.

## Development

Run all project commands through devenv:

```bash
devenv shell -- pytest
devenv shell -- ty check src
devenv shell -- ruff check src tests
```

The initial package contains the shared error vocabulary. Constraint codecs,
durable agents, authority, approval, and plane services land in subsequent
phases.
