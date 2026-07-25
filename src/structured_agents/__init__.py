"""Durable primitives for constrained-agent workflows.

Public symbols are loaded lazily (PEP 562): ``from structured_agents import Agent``
still works, but merely importing a lightweight submodule — e.g.
``structured_agents.llama_core.router`` — no longer eagerly pulls the durable-agent
stack (dbos, pydantic_ai, the plane). This keeps the llama.cpp teaching core usable
standalone without the heavy agent dependencies.
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
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # eager names for static analysis / IDEs only
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
]
