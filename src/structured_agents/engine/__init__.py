"""Selectable inference-engine plugins. Built-ins only; no out-of-tree discovery.

Since project 17 pivoted to llama.cpp as the single substrate, only the llama.cpp
wire dialect remains; the sglang/vLLM provider abstraction has been retired.
"""

from __future__ import annotations

from ..errors import ConfigError
from .base import Engine
from .llama_cpp import LlamaCppEngine

_BUILTINS: dict[str, Engine] = {
    "llama_cpp": LlamaCppEngine(),
}


def select(name: str) -> Engine:
    """Resolve a built-in engine by name."""
    try:
        return _BUILTINS[name]
    except KeyError:
        raise ConfigError(f"Unknown engine {name!r}.") from None


__all__ = ["Engine", "select"]
