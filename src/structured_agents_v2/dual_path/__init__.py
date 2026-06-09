"""Dual-path inference + capture layer (optional, behind the `[dual-path]` extra).

Every *sampled* agent run executes against the local vLLM `Backend` **and** a frontier
OpenAI-compatible API; both outputs are validated against the same type and persisted as a versioned
`ComparisonRecord` (Postgres `jsonb`) for SFT data + local-vs-frontier evals.

**Phase 1 (this module set):** the DBOS-free *data core* — records, comparators, store/export. The
durable runner (`DualPathRuntime`/`DualPathRunner`, which import DBOS) arrives in Phase 2.

Importing this package requires the extra:  ``pip install 'structured-agents-v2[dual-path]'``.
"""

from __future__ import annotations

try:
    import psycopg  # noqa: F401  (ships with the [dual-path] extra)
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "structured_agents_v2.dual_path requires the [dual-path] extra "
        "(pip install 'structured-agents-v2[dual-path]')."
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
