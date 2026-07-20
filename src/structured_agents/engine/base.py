"""Engine plugins translate a neutral Constraint onto one inference engine's wire shape."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..constraint import Constraint, WireSpec


@runtime_checkable
class Engine(Protocol):
    """One inference engine's constrained-decoding dialect. Internal; not part of the public API."""

    name: str
    supports: frozenset[str]

    def render(self, constraint: Constraint) -> WireSpec: ...
