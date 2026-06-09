"""Dual-path inference + capture layer (optional, behind the `[dual-path]` extra).

Every *sampled* agent run executes against the local vLLM `Backend` **and** a frontier
OpenAI-compatible API; both outputs are validated against the same type and persisted as a versioned
`ComparisonRecord` (Postgres `jsonb`) for SFT data + local-vs-frontier evals.

**Phase 1:** the DBOS-free *data core* — records, comparators, store/export.
**Phase 2 (this module set adds):** the durable runtime — `DualPathRuntime`/`DualPathRunner`, which
import DBOS and run two `DBOSAgent` legs concurrently (Architecture C).

Importing this package requires the extra:  ``pip install 'structured-agents-v2[dual-path]'``.
"""

from __future__ import annotations

try:
    import dbos  # noqa: F401  (ships with the [dual-path] extra)
    import psycopg  # noqa: F401  (ships with the [dual-path] extra)
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "structured_agents_v2.dual_path requires the [dual-path] extra (pip install 'structured-agents-v2[dual-path]')."
    ) from exc

from .comparator import Comparator, ComparisonSignal, ExactFieldComparator
from .errors import DualPathConfigError, DualPathError
from .record import (
    ComparisonRecord,
    ModelIdentity,
    build_comparison_record,
    content_hash,
    lib_version,
    profile_version,
    schema_version,
)
from .runner import DualPathRunner
from .runtime import DualPathConfig, DualPathRuntime
from .store import ComparisonExport, ComparisonStore, EvalSummary, GroupEval

__all__ = [
    # records / versioning
    "ComparisonRecord",
    "ModelIdentity",
    "build_comparison_record",
    "content_hash",
    "profile_version",
    "schema_version",
    "lib_version",
    # runtime / runner
    "DualPathRuntime",
    "DualPathConfig",
    "DualPathRunner",
    # comparators
    "Comparator",
    "ComparisonSignal",
    "ExactFieldComparator",
    # store / export
    "ComparisonStore",
    "ComparisonExport",
    "EvalSummary",
    "GroupEval",
    # errors
    "DualPathError",
    "DualPathConfigError",
]
