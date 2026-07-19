# S2(b): generic base class Outcome[T] with method combinators.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, reveal_type


class Outcome[T]:
    def map[U](self, f: Callable[[T], U]) -> Outcome[U]:
        raise NotImplementedError

    def unwrap(self) -> T:
        raise NotImplementedError

    def value_or[D](self, default: D) -> T | D:
        raise NotImplementedError


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
    return Ok(Plan(["git", "log"]))


oc = make()
reveal_type(oc.unwrap())  # Plan?
reveal_type(oc.map(lambda p: p.argv))  # Outcome[list[str]]?
reveal_type(oc.value_or(None))  # Plan | None?

# construct Ok, see value + inherited unwrap
ok = Ok(Plan(["git"]))
reveal_type(ok)  # Ok[Plan]
reveal_type(ok.value)  # Plan
reveal_type(ok.unwrap())  # Plan (inherited)

# does a match on the base-class subclasses narrow at all?
match oc:
    case Ok(value=v):
        reveal_type(v)  # narrows? plan says no
