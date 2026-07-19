# S1(b): single bounded TypeVar S: str. Plus edge cases the plan didn't test.
from __future__ import annotations

from typing import Any, cast, reveal_type


class _C[T]:
    def __init__(self, opts: tuple[str, ...]) -> None:
        self._opts = opts

    def wire(self) -> Any:
        return {"choice": list(self._opts)}

    def parse(self, raw: Any) -> T:
        if raw not in self._opts:
            raise ValueError(raw)
        return cast(T, raw)


def Choice[S: str](*options: S) -> _C[S]:
    return _C(tuple(str(o) for o in options))


# baseline: two string literals
cb = Choice("keep", "skip")
reveal_type(cb)
vb = cb.parse("keep")
reveal_type(vb)

# edge 1: single option
c_single = Choice("only")
reveal_type(c_single)

# edge 2: runtime str variable — does it widen to _C[str]?
x: str = "dynamic"
c_var = Choice(x)
reveal_type(c_var)

# edge 3: mix a literal and a runtime str
c_mix_str = Choice("keep", x)
reveal_type(c_mix_str)
