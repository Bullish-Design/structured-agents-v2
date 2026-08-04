"""Durable primitives for constrained-agent workflows.

Public symbols are loaded lazily (PEP 562): ``from structured_agents import Agent``
still works, but merely importing a lightweight submodule — e.g.
``inferference.router`` — no longer eagerly pulls the durable-agent stack (dbos,
pydantic_ai, the plane). This keeps the llama.cpp teaching core usable
standalone without the heavy agent dependencies.

Step-7 refactor (003-reference-consumer-refactor): the reference no longer
carries a ``llama_core`` copy — the shared core is consumed from
``inferference``, and the top-level core symbols below resolve to the
inferference modules (lazy, exactly like the framework surface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# name -> submodule that defines it. Attribute access triggers a lazy import.
_LAZY: dict[str, str] = {
    # .agent
    "Agent": "agent", "AgentSpec": "agent", "Backend": "agent", "Settings": "agent",
    # .approval
    "Approval": "approval", "ApprovalClient": "approval", "PendingApproval": "approval",
    # .authority
    "Allowlist": "authority", "ApprovalEvidence": "authority", "AuthorityMode": "authority",
    "AuthorityRequest": "authority", "Authorizer": "authority", "AutomatedAuthorizer": "authority",
    "CommandBinding": "authority", "Decision": "authority", "DecisionKind": "authority",
    "Denied": "authority", "Effector": "authority", "Null": "authority",
    "ProcessResult": "authority", "Subprocess": "authority", "all_of": "authority",
    "any_of": "authority", "authorize": "authority", "execute": "authority",
    # .config
    "constraint_from_config": "config", "register_constraint": "config", "spec_from_config": "config",
    # .constraint
    "Choice": "constraint", "Constraint": "constraint", "Grammar": "constraint",
    "Regex": "constraint", "Schema": "constraint", "WireSpec": "constraint",
    # .errors
    "AuthorityError": "errors", "BackendCapabilityError": "errors", "ConfigError": "errors",
    "ConstraintCompileError": "errors", "ConstraintConfigError": "errors",
    "ConstraintViolation": "errors", "StructuredAgentsError": "errors",
    # .plane
    "Comparison": "plane", "Queue": "plane", "cancel": "plane", "compare": "plane",
    "configure": "plane", "fork": "plane", "launch": "plane", "schedule": "plane",
    "shutdown": "plane", "status": "plane", "workflows": "plane",
    # .llama_core — the shared llama.cpp teaching core, consumed from inferference
    # (step-7 refactor). These resolve to the inferference modules (absolute),
    # kept lazy like the framework surface. Additive convenience for the
    # reference's consumers; they may equally import inferference directly.
    "AdapterSpec": "inferference.router", "EngineConfig": "inferference.models",
    "GenerationRequest": "inferference.models", "GenerationResult": "inferference.models",
    "MultiLoRARouter": "inferference.router", "RouteRequest": "inferference.router",
    "RouterConfig": "inferference.router",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    if module.startswith("inferference."):
        # Absolute module: the shared core lives in the inferference dependency.
        value = getattr(importlib.import_module(module), name)
    else:
        value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # eager names for static analysis / IDEs only
    # The shared core (inferference) — step-7 refactor, lazy surface.
    from inferference.models import EngineConfig, GenerationRequest, GenerationResult
    from inferference.router import AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

    from .agent import Agent, AgentSpec, Backend, Settings
    from .approval import Approval, ApprovalClient, PendingApproval
    from .authority import (
        Allowlist,
        ApprovalEvidence,
        AuthorityMode,
        AuthorityRequest,
        Authorizer,
        AutomatedAuthorizer,
        CommandBinding,
        Decision,
        DecisionKind,
        Denied,
        Effector,
        Null,
        ProcessResult,
        Subprocess,
        all_of,
        any_of,
        authorize,
        execute,
    )
    from .config import constraint_from_config, register_constraint, spec_from_config
    from .constraint import Choice, Constraint, Grammar, Regex, Schema, WireSpec
    from .errors import (
        AuthorityError,
        BackendCapabilityError,
        ConfigError,
        ConstraintCompileError,
        ConstraintConfigError,
        ConstraintViolation,
        StructuredAgentsError,
    )
    from .plane import (
        Comparison,
        Queue,
        cancel,
        compare,
        configure,
        fork,
        launch,
        schedule,
        shutdown,
        status,
        workflows,
    )

__all__ = [
    "AuthorityError",
    "Agent",
    "AgentSpec",
    "Approval",
    "ApprovalClient",
    "ApprovalEvidence",
    "Backend",
    "BackendCapabilityError",
    "Allowlist",
    "Authorizer",
    "AuthorityMode",
    "AuthorityRequest",
    "AutomatedAuthorizer",
    "Choice",
    "CommandBinding",
    "Comparison",
    "ConfigError",
    "Constraint",
    "ConstraintCompileError",
    "ConstraintConfigError",
    "ConstraintViolation",
    "Decision",
    "DecisionKind",
    "Denied",
    "Effector",
    "Grammar",
    "Null",
    "PendingApproval",
    "ProcessResult",
    "Regex",
    "Queue",
    "Schema",
    "Settings",
    "Subprocess",
    "StructuredAgentsError",
    "WireSpec",
    "all_of",
    "any_of",
    "authorize",
    "cancel",
    "compare",
    "configure",
    "constraint_from_config",
    "execute",
    "fork",
    "launch",
    "register_constraint",
    "schedule",
    "shutdown",
    "status",
    "spec_from_config",
    "workflows",
    # The shared core (inferference) — step-7 refactor, lazy surface.
    "AdapterSpec",
    "EngineConfig",
    "GenerationRequest",
    "GenerationResult",
    "MultiLoRARouter",
    "RouteRequest",
    "RouterConfig",
]
