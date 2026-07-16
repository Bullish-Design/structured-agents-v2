"""structured-agents-v2: typed, constrained PydanticAI agents over a local vLLM backend."""

from __future__ import annotations

from .agent import AgentResult, StructuredAgent
from .backend import Backend, BackendCaps
from .capture import RequestCapture, RequestRecord
from .constrained import ConstrainedOutput
from .decoder import DecodeMode, DecoderApplication, DecoderSpec
from .errors import (
    BackendCapabilityError,
    ConfigError,
    ConstraintCompileError,
    ConstraintConfigError,
    ConstraintViolationError,
    FleetError,
    PolicyError,
    RoutingError,
    StructuredAgentsError,
)
from .executor import (
    AllowlistExecutor,
    BaseExecutor,
    Decision,
    DryRunExecutor,
    ExecResult,
    Executor,
    Policy,
)
from .fleet import AgentSet, RoutedExecution, RoutedResult, RoutingTable
from .profile import AgentProfile

__all__ = [
    "ConstrainedOutput",
    "DecoderSpec",
    "DecoderApplication",
    "DecodeMode",
    "Backend",
    "BackendCaps",
    "AgentProfile",
    "StructuredAgent",
    "AgentResult",
    "AgentSet",
    "RoutingTable",
    "RoutedResult",
    "RoutedExecution",
    "Executor",
    "BaseExecutor",
    "DryRunExecutor",
    "AllowlistExecutor",
    "Policy",
    "Decision",
    "ExecResult",
    "RequestCapture",
    "RequestRecord",
    "StructuredAgentsError",
    "ConfigError",
    "ConstraintConfigError",
    "ConstraintCompileError",
    "ConstraintViolationError",
    "BackendCapabilityError",
    "FleetError",
    "RoutingError",
    "PolicyError",
]
