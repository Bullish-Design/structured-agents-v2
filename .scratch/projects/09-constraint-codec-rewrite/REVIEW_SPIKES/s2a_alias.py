# S2(a): bare union type alias Outcome[T]. Can ty recover T?
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeGuard, reveal_type


@dataclass(frozen=True)
class Ok[T]:
    value: T


@dataclass(frozen=True)
class Failed:
    error: Exception


type Outcome[T] = Ok[T] | Failed


@dataclass
class Plan:
    argv: list[str]


def make() -> Outcome[Plan]:
    return Ok(Plan(["git", "log"]))


oc = make()
reveal_type(oc)  # Outcome[Plan]

# 1) match narrowing
match oc:
    case Ok(value=v):
        reveal_type(v)  # T recovered?

# 2) isinstance narrowing
if isinstance(oc, Ok):
    reveal_type(oc.value)  # T recovered?

# 3) TypeGuard
def is_ok[T](o: Outcome[T]) -> TypeGuard[Ok[T]]:
    return isinstance(o, Ok)


oc2 = make()
if is_ok(oc2):
    reveal_type(oc2)  # Ok[Plan]?
    reveal_type(oc2.value)  # Plan?

# 4) free unwrap
def unwrap[T](o: Outcome[T]) -> T:
    if isinstance(o, Ok):
        return o.value
    raise o.error


u = unwrap(make())
reveal_type(u)  # Plan?

# 5) fold
def fold[T, R](o: Outcome[T], *, ok: Callable[[T], R], other: Callable[[Failed], R]) -> R:
    if isinstance(o, Ok):
        return ok(o.value)
    return other(o)


r = fold(make(), ok=lambda p: p.argv, other=lambda _f: [])
reveal_type(r)  # list[str]?
