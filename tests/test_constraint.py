from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel
from pydantic_ai.output import NativeOutput

from structured_agents import Choice, Grammar, Regex, Schema
from structured_agents.errors import ConstraintConfigError, ConstraintViolation


class Person(BaseModel):
    name: str
    age: int


@given(name=st.text(), age=st.integers())
def test_schema_round_trips_validated_models(name: str, age: int) -> None:
    value = Person(name=name, age=age)
    assert Schema(Person).parse(value) is value


@given(value=st.from_regex(r"[A-Z]{2}-[0-9]{4}", fullmatch=True))
def test_regex_round_trips_matching_text(value: str) -> None:
    assert Regex(r"[A-Z]{2}-[0-9]{4}").parse(value) == value


def test_string_constraints_reject_invalid_output() -> None:
    with pytest.raises(ConstraintViolation):
        Regex(r"yes|no").parse("maybe")
    with pytest.raises(ConstraintViolation):
        Choice("keep", "skip").parse("other")
    with pytest.raises(ConstraintViolation):
        Grammar('root ::= "a"').parse(1)


def test_wire_shapes_match_the_verified_table() -> None:
    schema_wire = Schema(Person).wire()
    assert isinstance(schema_wire.output_type, NativeOutput)
    assert schema_wire.output_type.outputs is Person
    assert schema_wire.output_type.strict is True
    assert schema_wire.extra_body == {}
    assert Regex(r"\d{4}-\d{2}-\d{2}").wire() == _wire(
        {"structured_outputs": {"regex": r"\d{4}-\d{2}-\d{2}"}}
    )
    assert Choice("keep", "skip").wire() == _wire({"structured_outputs": {"choice": ["keep", "skip"]}})
    assert Grammar('root ::= "a" | "b"').wire() == _wire(
        {"structured_outputs": {"grammar": 'root ::= "a" | "b"'}}
    )


def _wire(extra_body: dict[str, Any]):
    from structured_agents import WireSpec

    return WireSpec(output_type=str, extra_body=extra_body)


def test_invalid_constraint_configuration_is_clear() -> None:
    with pytest.raises(ConstraintConfigError, match="at least one"):
        Choice()
    with pytest.raises(ConstraintConfigError, match="Invalid regex"):
        Regex("[")
