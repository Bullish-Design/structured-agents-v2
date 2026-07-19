# S1 edge: a truly opaque str (no literal to narrow) passed to Choice[S: str].
from __future__ import annotations

from typing import Any, cast, reveal_type


class _C[T]:
    def __init__(self, opts: tuple[str, ...]) -> None:
        self._opts = opts

    def parse(self, raw: Any) -> T:
        return cast(T, raw)


def Choice[S: str](*options: S) -> _C[S]:
    return _C(tuple(options))


def opaque() -> str:
    return "whatever"


c_opaque = Choice(opaque())
reveal_type(c_opaque)  # does it widen to _C[str]?

# opaque str mixed with a literal
c_opaque_mix = Choice("keep", opaque())
reveal_type(c_opaque_mix)
