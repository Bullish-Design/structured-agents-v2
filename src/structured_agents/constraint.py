"""Pure codecs connecting constrained output wire shapes to typed values."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from .errors import ConstraintCompileError, ConstraintConfigError, ConstraintViolation


@dataclass(frozen=True)
class WireSpec:
    """The pydantic-ai output declaration and OpenAI-compatible extra body."""

    output_type: Any
    extra_body: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Constraint[T](Protocol):
    """A constrained-output codec."""

    def wire(self) -> WireSpec: ...

    def parse(self, raw: Any) -> T: ...

    def check(self) -> None: ...

    def to_config(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _Schema[M: BaseModel]:
    model: type[M]
    strict: bool

    def wire(self) -> WireSpec:
        from pydantic_ai.output import NativeOutput

        return WireSpec(output_type=NativeOutput(self.model, strict=self.strict))

    def parse(self, raw: Any) -> M:
        return cast(M, raw)

    def check(self) -> None:
        try:
            import xgrammar as xgr
        except ImportError:
            return
        try:
            xgr.Grammar.from_json_schema(self.model.model_json_schema())
        except Exception as exc:  # pragma: no cover - requires grammar-check extra
            raise ConstraintCompileError(f"Schema constraint for {self.model.__name__} did not compile: {exc}") from exc

    def to_config(self) -> dict[str, Any]:
        return {"kind": "schema", "ref": f"{self.model.__module__}:{self.model.__qualname__}", "strict": self.strict}


@dataclass(frozen=True)
class _Regex:
    pattern: str

    def __post_init__(self) -> None:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ConstraintConfigError(f"Invalid regex constraint {self.pattern!r}: {exc}") from exc

    def wire(self) -> WireSpec:
        return WireSpec(output_type=str, extra_body={"structured_outputs": {"regex": self.pattern}})

    def parse(self, raw: Any) -> str:
        if not isinstance(raw, str) or re.fullmatch(self.pattern, raw) is None:
            raise ConstraintViolation(f"Output does not match regex {self.pattern!r}.", raw=str(raw))
        return raw

    def check(self) -> None:
        return None

    def to_config(self) -> dict[str, Any]:
        return {"kind": "regex", "pattern": self.pattern}


@dataclass(frozen=True)
class _Choice[T: str]:
    options: tuple[T, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ConstraintConfigError("Choice constraint requires at least one option.")

    def wire(self) -> WireSpec:
        return WireSpec(output_type=str, extra_body={"structured_outputs": {"choice": list(self.options)}})

    def parse(self, raw: Any) -> T:
        if raw not in self.options:
            raise ConstraintViolation(f"Output is not one of {self.options!r}.", raw=str(raw))
        return cast(T, raw)

    def check(self) -> None:
        return None

    def to_config(self) -> dict[str, Any]:
        return {"kind": "choice", "options": list(self.options)}


@dataclass(frozen=True)
class _Grammar:
    ebnf: str

    def __post_init__(self) -> None:
        if not self.ebnf.strip():
            raise ConstraintConfigError("Grammar constraint requires non-empty EBNF.")

    def wire(self) -> WireSpec:
        return WireSpec(output_type=str, extra_body={"structured_outputs": {"grammar": self.ebnf}})

    def parse(self, raw: Any) -> str:
        if not isinstance(raw, str):
            raise ConstraintViolation("Grammar constraint expected string output.", raw=str(raw))
        return raw

    def check(self) -> None:
        try:
            import xgrammar as xgr
        except ImportError:
            return
        try:
            xgr.Grammar.from_ebnf(self.ebnf)
        except Exception as exc:  # pragma: no cover - requires grammar-check extra
            raise ConstraintCompileError(f"Grammar constraint did not compile: {exc}") from exc

    def to_config(self) -> dict[str, Any]:
        return {"kind": "grammar", "ebnf": self.ebnf}


def Schema[M: BaseModel](model: type[M], *, strict: bool = True) -> Constraint[M]:
    """Build a schema-backed constraint using pydantic-ai NativeOutput."""

    return _Schema(model, strict)


def Regex(pattern: str) -> Constraint[str]:
    """Build a full-match regex constraint."""

    return _Regex(pattern)


def Choice[S: str](*options: S) -> Constraint[S]:
    """Build a finite string-choice constraint preserving literal types."""

    return _Choice(options)


def Grammar(ebnf: str) -> Constraint[str]:
    """Build an EBNF grammar constraint."""

    return _Grammar(ebnf)
