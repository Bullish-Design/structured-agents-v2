"""Comparators — how two validated outputs are turned into an agreement signal.

`ComparisonSignal` is the serializable result (it lives on every `ComparisonRecord`). A
`Comparator` maps two validated Pydantic outputs to a signal; `ExactFieldComparator` is the MVP
default (exact equality plus a field-level diff). Custom per-output-type comparators implement the
`Comparator` protocol — e.g. normalized paths, set-equality for lists, or semantic scoring.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ComparisonSignal(BaseModel):
    """The agreement signal between a primary and a reference output."""

    agreement_exact: bool
    field_diff: dict[str, list[Any]] = {}  # field -> [primary_value, reference_value]
    score: float | None = None  # optional 0..1 partial-credit


@runtime_checkable
class Comparator(Protocol):
    """Maps two validated outputs to a `ComparisonSignal`."""

    def compare(self, primary: BaseModel, reference: BaseModel) -> ComparisonSignal: ...


class ExactFieldComparator:
    """Exact equality on the validated objects, plus a per-field diff of what differs.

    `score` is the fraction of top-level fields that match (1.0 when exactly equal).
    """

    def compare(self, primary: BaseModel, reference: BaseModel) -> ComparisonSignal:
        p, r = primary.model_dump(), reference.model_dump()
        if p == r:
            return ComparisonSignal(agreement_exact=True, field_diff={}, score=1.0)
        keys = set(p) | set(r)
        diff = {k: [p.get(k), r.get(k)] for k in keys if p.get(k) != r.get(k)}
        matched = len(keys) - len(diff)
        score = matched / len(keys) if keys else 0.0
        return ComparisonSignal(agreement_exact=False, field_diff=diff, score=score)
