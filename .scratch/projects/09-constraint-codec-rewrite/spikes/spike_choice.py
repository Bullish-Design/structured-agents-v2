# Spike F: does `ty` synthesize Constraint[Literal[*opts]] from a variadic Choice?
# We test several candidate signatures and reveal_type the result.
from __future__ import annotations

from typing import Any, Literal, Protocol, reveal_type


class Constraint[T](Protocol):
    def wire(self) -> Any: ...
    def parse(self, raw: Any) -> T: ...


class _C[T]:
    def wire(self) -> Any: ...
    def parse(self, raw: Any) -> T: ...


# --- Candidate 1: the concept's proposal, TypeVarTuple unpacked into Literal ---
def Choice1[*Opts](*options: *Opts) -> Constraint[Literal[*Opts]]:  # type: ignore[valid-type]
    return _C()


# --- Candidate 2: plain *args: str -> Constraint[str] (honest, loses literal) ---
def Choice2(*options: str) -> Constraint[str]:
    return _C()


# --- Candidate 3: single TypeVar bound to the arg type ---
def Choice3[S: str](*options: S) -> Constraint[S]:
    return _C()


# --- Candidate 4: explicit Literal type param the caller supplies ---
def Choice4[L](*options: L) -> Constraint[L]:
    return _C()


c1 = Choice1("keep", "skip")
reveal_type(c1)  # want: Constraint[Literal["keep", "skip"]]

c2 = Choice2("keep", "skip")
reveal_type(c2)  # Constraint[str]

c3 = Choice3("keep", "skip")
reveal_type(c3)  # ?

c4: Constraint[Literal["keep", "skip"]] = Choice4("keep", "skip")
reveal_type(c4)  # Constraint[Literal["keep", "skip"]] via annotation

# Does the literal actually flow to a consumer?
v1 = c1.parse("x")
reveal_type(v1)  # want Literal["keep","skip"]
v4 = c4.parse("x")
reveal_type(v4)  # Literal["keep","skip"]
