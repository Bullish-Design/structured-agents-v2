"""Exception types for invalid configuration and failed constraints."""

from __future__ import annotations


class StructuredAgentsError(Exception):
    """Base exception for structured-agents failures."""


# Kept as a compatibility spelling for the Phase 0 import smoke test.
ConstricError = StructuredAgentsError


class ConfigError(StructuredAgentsError):
    """Raised when a configuration or specification is invalid."""


class ConstraintConfigError(ConfigError):
    """Raised when a constraint is constructed inconsistently."""


class ConstraintCompileError(StructuredAgentsError):
    """Raised when an optional grammar compile check fails."""


class BackendCapabilityError(StructuredAgentsError):
    """Raised when a backend lacks a capability required by an agent."""


class AuthorityError(StructuredAgentsError):
    """Raised when authority machinery is misconfigured."""


class ConstraintViolation(StructuredAgentsError):
    """Raised when backend output does not satisfy its constraint."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw
