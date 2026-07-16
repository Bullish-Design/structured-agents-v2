"""Library error hierarchy."""

from __future__ import annotations


class StructuredAgentsError(Exception):
    """Base class for all structured-agents-v2 errors."""


class ConfigError(StructuredAgentsError):
    """Invalid configuration supplied to the library."""


class ConstraintConfigError(ConfigError):
    """A decode constraint is internally inconsistent (e.g. regex mode with no regex)."""


class ConstraintCompileError(StructuredAgentsError):
    """A constraint failed to compile with XGrammar (optional dev-only check)."""


class BackendCapabilityError(StructuredAgentsError):
    """An agent requires a capability the configured backend does not provide."""


class FleetError(StructuredAgentsError):
    """An `AgentSet`/fleet is misconfigured (duplicate names, unknown agent, …)."""


class RoutingError(FleetError):
    """A `RoutingTable` is invalid, or a router emitted an unroutable value at runtime."""


class PolicyError(StructuredAgentsError):
    """An executor refused a command, or a policy/executor is misconfigured."""


class ConstraintViolationError(StructuredAgentsError):
    """A bare-string output did not satisfy its declared regex/choice constraint."""
