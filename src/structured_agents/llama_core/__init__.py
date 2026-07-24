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
    "RuntimeDiagnostics",
    "collect_runtime_diagnostics",
    "file_identity",
    "register_artifact",
    "write_benchmark_record",
]
