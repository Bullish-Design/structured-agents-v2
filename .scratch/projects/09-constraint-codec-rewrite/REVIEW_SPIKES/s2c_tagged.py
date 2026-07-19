# S2 extra: single tagged Result[T] dataclass (no subclasses) with methods.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import reveal_type


@dataclass(frozen=True)
class Result[T]:
    _value: T | None
    _reason: str
    ok: bool

    def unwrap(self) -> T:
        if not self.ok or self._value is None:
            raise RuntimeError(self._reason)
        return self._value

    def map[U](self, f: Callable[[T], U]) -> Result[U]:
        raise NotImplementedError

    def value_or[D](self, default: D) -> T | D:
        raise NotImplementedError


@dataclass
class Plan:
    argv: list[str]


def make() -> Result[Plan]:
    return Result(Plan(["git"]), "", True)


r = make()
reveal_type(r.unwrap())  # Plan?
reveal_type(r.map(lambda p: p.argv))  # Result[list[str]]?
reveal_type(r.value_or(None))  # Plan | None?
