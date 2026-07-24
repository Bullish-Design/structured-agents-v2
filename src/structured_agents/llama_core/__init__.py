"""Shared boundary types and runtime facts for the llama.cpp teaching core."""

from .diagnostics import RuntimeDiagnostics, collect_runtime_diagnostics
from .fingerprint import (
    ArtifactIdentity,
    LlamaEngineFingerprint,
    file_identity,
    register_artifact,
)
from .models import BenchmarkRecord, EngineConfig, GenerationRequest, GenerationResult

__all__ = [
    "ArtifactIdentity",
    "BenchmarkRecord",
    "EngineConfig",
    "GenerationRequest",
    "GenerationResult",
    "LlamaEngineFingerprint",
    "RuntimeDiagnostics",
    "collect_runtime_diagnostics",
    "file_identity",
    "register_artifact",
]
