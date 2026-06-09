"""`ConstrainedOutput` — the decode constraint as a built-in property of the output type.

Subclass it and the constraint travels with the model wherever it's used as an agent's
`output_type`. The default mode is `json_schema`; declare a different mode (and its
params) with class-level dunder attributes:

    class FileEditPlan(ConstrainedOutput):          # json_schema (default)
        action: Literal["edit_file", "refuse"]
        reason: str

    class GitCommandLine(ConstrainedOutput):        # bare-string, regex-constrained
        __decode_mode__ = "regex"
        __regex__ = r"git (status|diff|add|commit) [\\w./-]*"
        value: str

The constraint is enforced server-side by XGrammar. An *optional* dev-only check
(`check_compilable`, behind the `[grammar-check]` extra) can compile the constraint with
xgrammar at class-definition time to fail fast on an un-compilable schema.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from pydantic import BaseModel

from .decoder import DecodeMode, DecoderSpec
from .errors import ConstraintCompileError, ConstraintConfigError

_REQUIRED_FIELD: dict[str, str] = {
    "grammar": "__grammar__",
    "regex": "__regex__",
    "choice": "__choices__",
}


class ConstrainedOutput(BaseModel):
    """A Pydantic model that carries its own constrained-decoding contract."""

    __decode_mode__: ClassVar[DecodeMode] = "json_schema"
    __grammar__: ClassVar[str | None] = None
    __regex__: ClassVar[str | None] = None
    __choices__: ClassVar[list[str] | None] = None
    __strict__: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _validate_mode_fields(cls)
        if os.environ.get("SAV_GRAMMAR_CHECK") == "1":
            cls.check_compilable()

    @classmethod
    def decoder_spec(cls) -> DecoderSpec:
        """The serializable decode contract this model declares."""
        return DecoderSpec(
            mode=cls.__decode_mode__,
            grammar=cls.__grammar__,
            regex=cls.__regex__,
            choices=cls.__choices__,
            strict=cls.__strict__,
        )

    @classmethod
    def check_compilable(cls) -> None:
        """Compile this model's constraint with xgrammar; no-op if xgrammar is absent.

        Optional dev-only guard (`pip install structured-agents-v2[grammar-check]`).
        Raises `ConstraintCompileError` if the constraint cannot be compiled.
        """
        try:
            import xgrammar as xgr
        except ImportError:
            return
        try:
            mode = cls.__decode_mode__
            if mode == "json_schema":
                xgr.Grammar.from_json_schema(json.dumps(cls.model_json_schema()))
            elif mode == "regex":
                xgr.Grammar.from_regex(cls.__regex__ or "")
            elif mode == "grammar":
                xgr.Grammar.from_ebnf(cls.__grammar__ or "")
            # choice: a finite literal set, nothing to compile.
        except Exception as exc:  # noqa: BLE001 - surface any xgrammar failure uniformly
            raise ConstraintCompileError(f"{cls.__name__}: constraint is not XGrammar-compilable: {exc}") from exc


def _validate_mode_fields(cls: type[ConstrainedOutput]) -> None:
    mode = cls.__decode_mode__
    required = _REQUIRED_FIELD.get(mode)
    if required is not None and getattr(cls, required) in (None, "", []):
        raise ConstraintConfigError(f"{cls.__name__}: decode mode {mode!r} requires {required} to be set.")
