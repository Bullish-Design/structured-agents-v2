"""DecoderSpec.apply: mode -> (output_type, extra_body)."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from structured_agents_v2 import DecoderSpec
from structured_agents_v2.errors import ConstraintConfigError


class Plan(BaseModel):
    action: Literal["edit_file", "refuse"]
    reason: str


def test_json_schema_uses_native_output_no_extra_body() -> None:
    app = DecoderSpec(mode="json_schema").apply(Plan)
    # NativeOutput wraps the model; no vLLM-specific extra_body needed.
    assert app.extra_body == {}
    assert type(app.output_type).__name__ == "NativeOutput"


def test_json_schema_without_output_type_raises() -> None:
    with pytest.raises(ConstraintConfigError, match="requires a Pydantic model"):
        DecoderSpec(mode="json_schema").apply(None)


def test_regex_is_text_mode_with_extra_body() -> None:
    app = DecoderSpec(mode="regex", regex=r"git .*").apply(None)
    assert app.output_type is str
    assert app.extra_body == {"structured_outputs": {"regex": r"git .*"}}


def test_choice_is_text_mode_with_extra_body() -> None:
    app = DecoderSpec(mode="choice", choices=["a", "b"]).apply(None)
    assert app.output_type is str
    assert app.extra_body == {"structured_outputs": {"choice": ["a", "b"]}}


def test_grammar_is_text_mode_with_extra_body() -> None:
    app = DecoderSpec(mode="grammar", grammar='root ::= "x"').apply(None)
    assert app.output_type is str
    assert app.extra_body == {"structured_outputs": {"grammar": 'root ::= "x"'}}


@pytest.mark.parametrize("mode", ["grammar", "regex", "choice"])
def test_missing_param_raises(mode: str) -> None:
    with pytest.raises(ConstraintConfigError):
        DecoderSpec(mode=mode).apply(None)  # type: ignore[arg-type]
