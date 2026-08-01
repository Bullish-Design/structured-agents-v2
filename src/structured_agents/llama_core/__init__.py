"""Shared boundary types and runtime facts for the llama.cpp teaching core."""

from .benchmark import BenchmarkRecord, BenchmarkTimer, write_benchmark_record
from .diagnostics import RuntimeDiagnostics, collect_runtime_diagnostics
from .fingerprint import (
    ArtifactIdentity,
    LlamaEngineFingerprint,
    file_identity,
    register_artifact,
)
from .grammar import GrammarCacheKey, GrammarCompilerCache, JsonSchemaGrammar
from .models import EngineConfig, GenerationRequest, GenerationResult
from .seq_routing import (
    NO_ADAPTER,
    SeqRoutingBinding,
    SeqRoutingUnavailable,
    library_supports_seq_routing,
)

# NOTE: the router (``router.py``) and owned decoder (``decode.py``) are heavy,
# native-backed modules; like ``decode``, ``router`` is imported directly by
# callers (``from ...llama_core.router import MultiLoRARouter``) so this shared
# import path stays lightweight (no numpy/llama_cpp import on ``import llama_core``).

__all__ = [
    "ArtifactIdentity",
    "BenchmarkRecord",
    "BenchmarkTimer",
    "EngineConfig",
    "GenerationRequest",
    "GenerationResult",
    "GrammarCacheKey",
    "GrammarCompilerCache",
    "JsonSchemaGrammar",
    "LlamaEngineFingerprint",
    "NO_ADAPTER",
    "RuntimeDiagnostics",
    "SeqRoutingBinding",
    "SeqRoutingUnavailable",
    "collect_runtime_diagnostics",
    "file_identity",
    "library_supports_seq_routing",
    "register_artifact",
    "write_benchmark_record",
]
