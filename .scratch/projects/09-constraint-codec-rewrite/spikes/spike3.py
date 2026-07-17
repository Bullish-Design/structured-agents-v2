# Spike B2: which Outcome encoding lets ty recover T at the call site?
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Never, reveal_type


@dataclass(frozen=True)
class Ok[T]:
    value: T


@dataclass(frozen=True)
class Denied:
    reason: str


@dataclass(frozen=True)
class Failed:
    error: Exception


type Outcome[T] = Ok[T] | Denied | Failed


@dataclass
class Plan:
    argv: list[str]


def make(ok: bool) -> Outcome[Plan]:
    return Ok(Plan(["git", "log"])) if ok else Denied("nope")


# ---- Encoding 1: fold/visitor over the alias ----
def fold[T, R](oc: Outcome[T], *, ok: Callable[[T], R], other: Callable[[Denied | Failed], R]) -> R:
    if isinstance(oc, Ok):
        return ok(oc.value)
    return other(oc)


r1 = fold(make(True), ok=lambda p: p.argv, other=lambda _o: [])
reveal_type(r1)  # want list[str]


# ---- Encoding 2: method on Ok itself (map) ----
def use2() -> None:
    oc = make(True)
    if isinstance(oc, Ok):
        reveal_type(oc)  # Ok[?]
        reveal_type(oc.value)  # ?


# ---- Encoding 3: explicit typed local (annotate the Ok arm) ----
def use3() -> None:
    oc = make(True)
    match oc:
        case Ok() as ok_arm:
            reveal_type(ok_arm)  # Ok[?]
            reveal_type(ok_arm.value)  # ?


# ---- Encoding 4: value_or on a wrapper via helper that returns T | D ----
def value_or[T, D](oc: Outcome[T], default: D) -> T | D:
    return oc.value if isinstance(oc, Ok) else default


v4 = value_or(make(True), None)
reveal_type(v4)  # want Plan | None


# ---- Encoding 5: is_ok TypeGuard ----
from typing import TypeGuard


def is_ok[T](oc: Outcome[T]) -> TypeGuard[Ok[T]]:
    return isinstance(oc, Ok)


def use5() -> None:
    oc = make(True)
    if is_ok(oc):
        reveal_type(oc)  # want Ok[Plan]
        reveal_type(oc.value)  # want Plan
