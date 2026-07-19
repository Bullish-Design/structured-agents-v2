# S1(a): variadic TypeVarTuple unpacked into Literal. EXPECTED TO ERROR.
from __future__ import annotations

from typing import Any, Literal, Protocol, reveal_type


class Constraint[T](Protocol):
    def wire(self) -> Any: ...
    def parse(self, raw: Any) -> T: ...


class _C[T]:
    def __init__(self, opts: tuple[Any, ...]) -> None:
        self._opts = opts

    def wire(self) -> Any:
        return list(self._opts)

    def parse(self, raw: Any) -> T:
        return raw


# Candidate (a): TypeVarTuple unpacked inside Literal[...]
def Choice[*Opts](*options: *Opts) -> Constraint[Literal[*Opts]]:
    return _C(tuple(options))


ca = Choice("keep", "skip")
reveal_type(ca)
va = ca.parse("keep")
reveal_type(va)
