"""Durable primitives for constrained-agent workflows."""

from .agent import Agent, AgentSpec, Backend, BackendCaps, Settings
from .approval import Approval, ApprovalClient, PendingApproval
from .authority import (
    Allowlist,
    Authorizer,
    Decision,
    Denied,
    Effector,
    Null,
    ProcessResult,
    Subprocess,
    all_of,
    any_of,
    execute,
)
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

__all__ = [
    "AuthorityError",
    "Agent",
    "AgentSpec",
    "Approval",
    "ApprovalClient",
    "Backend",
    "BackendCapabilityError",
    "BackendCaps",
    "Allowlist",
    "Authorizer",
    "Choice",
    "ConfigError",
    "Constraint",
    "ConstraintCompileError",
    "ConstraintConfigError",
    "ConstraintViolation",
    "Decision",
    "Denied",
    "Effector",
    "Grammar",
    "Null",
    "PendingApproval",
    "ProcessResult",
    "Regex",
    "Schema",
    "Settings",
    "Subprocess",
    "StructuredAgentsError",
    "WireSpec",
    "all_of",
    "any_of",
    "execute",
]
