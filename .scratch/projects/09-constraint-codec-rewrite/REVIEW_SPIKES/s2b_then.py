# S2(b) glossed-over: does `then` typecheck with `return self` for the Failed pass-through?
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, reveal_type


class Outcome[T]:
    def then[U](self, f: Callable[[T], Outcome[U]]) -> Outcome[U]:
        # For Ok, delegate; for Failed/Denied pass-through we want `return self`.
        # Does ty accept `return self` (an Outcome[T]) as Outcome[U]?
        return self


@dataclass(frozen=True)
class Ok[T](Outcome[T]):
    value: T


@dataclass(frozen=True)
class Failed(Outcome[Any]):
    error: Exception


@dataclass
class Plan:
    argv: list[str]


def make() -> Outcome[Plan]:
    return Ok(Plan(["git"]))


def step(p: Plan) -> Outcome[str]:
    return Ok(p.argv[0])


chained = make().then(step)
reveal_type(chained)  # Outcome[str]?
