# Spike B3: method-based encodings where T flows FORWARD from the class parameter.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, reveal_type


# ===== Encoding A: base generic class + subclasses; methods on the base carry T =====
class Outcome[T]:
    def map[U](self, f: Callable[[T], U]) -> Outcome[U]: ...
    def unwrap(self) -> T: ...
    def value_or[D](self, default: D) -> T | D: ...


@dataclass(frozen=True)
class Ok[T](Outcome[T]):
    value: T


@dataclass(frozen=True)
class Denied(Outcome[Any]):
    reason: str


@dataclass
class Plan:
    argv: list[str]


def make() -> Outcome[Plan]: ...


oc = make()
reveal_type(oc.unwrap())  # want Plan
reveal_type(oc.map(lambda p: p.argv))  # want Outcome[list[str]]
reveal_type(oc.value_or(None))  # want Plan | None

# Can we still construct Ok and see its value?
ok = Ok(Plan(["git"]))
reveal_type(ok)  # Ok[Plan]
reveal_type(ok.value)  # Plan
reveal_type(ok.unwrap())  # Plan (inherited)


# ===== Encoding B: single tagged class, no subclasses =====
@dataclass(frozen=True)
class Result[T]:
    _value: T | None
    _reason: str
    ok: bool

    def unwrap(self) -> T:
        if self._value is None:
            raise RuntimeError(self._reason)
        return self._value

    def map[U](self, f: Callable[[T], U]) -> Result[U]: ...


def make2() -> Result[Plan]: ...


r = make2()
reveal_type(r.unwrap())  # want Plan
reveal_type(r.map(lambda p: p.argv))  # want Result[list[str]]
