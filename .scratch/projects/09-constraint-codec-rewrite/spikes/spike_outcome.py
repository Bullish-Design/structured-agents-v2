# Spike B: does the Outcome[T] sum type + then/match narrow correctly under ty?
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, reveal_type


@dataclass(frozen=True)
class Ok[T]:
    value: T


@dataclass(frozen=True)
class Denied:
    reason: str


@dataclass(frozen=True)
class Violated:
    reason: str
    raw: str


@dataclass(frozen=True)
class Failed:
    error: Exception


type Outcome[T] = Ok[T] | Denied | Violated | Failed


# `then` as a free function (bind). Can a type alias carry a method? No — so bind is a function.
def then[T, U](oc: Outcome[T], f: Callable[[T], Outcome[U]]) -> Outcome[U]:
    if isinstance(oc, Ok):
        return f(oc.value)
    return oc  # Denied | Violated | Failed pass through — does ty accept this as Outcome[U]?


class Plan:
    argv: list[str]


def make() -> Outcome[Plan]: ...


oc = make()
reveal_type(oc)  # Outcome[Plan] == Ok[Plan] | Denied | Violated | Failed

# match narrowing
match oc:
    case Ok(value=plan):
        reveal_type(plan)  # want: Plan
    case Denied(reason=r):
        reveal_type(r)  # str
    case Violated(reason=r2):
        reveal_type(r2)  # str
    case Failed(error=e):
        reveal_type(e)  # Exception

# bind chain
def step2(p: Plan) -> Outcome[str]: ...


chained = then(oc, step2)
reveal_type(chained)  # want: Outcome[str]

# unwrap: returns T or raises
def unwrap[T](oc: Outcome[T]) -> T:
    match oc:
        case Ok(value=v):
            return v
        case Denied(reason=r):
            raise RuntimeError(r)
        case Violated(reason=r):
            raise RuntimeError(r)
        case Failed(error=e):
            raise e


u = unwrap(make())
reveal_type(u)  # want: Plan
