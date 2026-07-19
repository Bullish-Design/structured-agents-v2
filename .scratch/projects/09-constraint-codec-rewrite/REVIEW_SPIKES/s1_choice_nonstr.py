# S1 edge: non-str and mixed-type options against Choice[S: str]. EXPECTED TO ERROR.
from __future__ import annotations

from typing import Any, cast, reveal_type


class _C[T]:
    def __init__(self, opts: tuple[Any, ...]) -> None:
        self._opts = opts

    def parse(self, raw: Any) -> T:
        return cast(T, raw)


def Choice[S: str](*options: S) -> _C[S]:
    return _C(tuple(options))


# edge: non-str ints against a str-bound TypeVar
c_int = Choice(1, 2)
reveal_type(c_int)

# edge: mixed str + int
c_mixed = Choice("a", 1)
reveal_type(c_mixed)
