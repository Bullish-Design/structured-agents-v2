# Spike F+B refined: real bodies so inference isn't polluted by empty-body Unknown.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, reveal_type


# ---------- Choice via single bounded TypeVar ----------
class _C[T]:
    def __init__(self, opts: tuple[str, ...]) -> None:
        self._opts = opts

    def wire(self) -> Any:
        return {"structured_outputs": {"choice": list(self._opts)}}

    def parse(self, raw: Any) -> T:
        if raw not in self._opts:
            raise ValueError(raw)
        return cast(T, raw)


def Choice[S: str](*options: S) -> _C[S]:
    return _C(tuple(options))


c = Choice("keep", "skip")
reveal_type(c)  # want _C[Literal["keep","skip"]]
val = c.parse("keep")
reveal_type(val)  # want Literal["keep","skip"]
# runtime:
print("choice runtime:", c.wire(), repr(c.parse("keep")))


# ---------- Outcome with real bodies ----------
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


def unwrap[T](oc: Outcome[T]) -> T:
    if isinstance(oc, Ok):
        return oc.value
    if isinstance(oc, Denied):
        raise RuntimeError(oc.reason)
    raise oc.error


u = unwrap(make(True))
reveal_type(u)  # want Plan

# match at runtime
oc = make(True)
match oc:
    case Ok(value=plan):
        reveal_type(plan)  # ty may say @Todo, but runtime must bind Plan
        print("match ok:", plan.argv)
    case Denied(reason=r):
        print("denied:", r)
    case Failed(error=e):
        print("failed:", e)

# isinstance-narrowing (the ty-friendly alternative to match)
oc2 = make(False)
if isinstance(oc2, Ok):
    reveal_type(oc2.value)  # want Plan
elif isinstance(oc2, Denied):
    reveal_type(oc2.reason)  # want str
