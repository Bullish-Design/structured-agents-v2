"""Error hierarchy for the dual-path layer."""

from __future__ import annotations

from ..errors import StructuredAgentsError


class DualPathError(StructuredAgentsError):
    """Base class for dual-path (capture/eval) errors."""


class DualPathConfigError(DualPathError):
    """Invalid dual-path configuration (bad pg url, non-json_schema reference, …)."""
