"""Selectable inference-engine plugins. Built-ins only; no out-of-tree discovery."""

from __future__ import annotations

from ..errors import ConfigError
from .base import Engine
from .llama_cpp import LlamaCppEngine
from .sglang import SGLangEngine
from .vllm import VLLMEngine

_BUILTINS: dict[str, Engine] = {
    "vllm": VLLMEngine(),
    "sglang": SGLangEngine(),
    "llama_cpp": LlamaCppEngine(),
}


def select(name: str) -> Engine:
    """Resolve a built-in engine by name."""
    try:
        return _BUILTINS[name]
    except KeyError:
        raise ConfigError(f"Unknown engine {name!r}.") from None


__all__ = ["Engine", "select"]
