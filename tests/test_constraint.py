from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from structured_agents import Choice, Grammar, Regex, Schema
from structured_agents.errors import ConstraintConfigError, ConstraintViolation


class Person(BaseModel):
    name: str
    age: int


@given(name=st.text(), age=st.integers())
def test_schema_round_trips_validated_models(name: str, age: int) -> None:
    value = Person(name=name, age=age)
    assert Schema(Person).parse(value) is value


def test_schema_parse_validates_raw_mappings_into_the_model() -> None:
    parsed = Schema(Person).parse({"name": "Ada", "age": 37})

    assert parsed == Person(name="Ada", age=37)
    assert isinstance(parsed, Person)


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


def test_constraints_carry_a_neutral_kind_and_config() -> None:
    assert Schema(Person).kind == "schema"
    assert Regex(r"\d{4}").kind == "regex"
    assert Choice("keep", "skip").kind == "choice"
    assert Grammar('root ::= "a"').kind == "grammar"
    assert Choice("keep", "skip").to_config() == {"kind": "choice", "options": ["keep", "skip"]}


def test_invalid_constraint_configuration_is_clear() -> None:
    with pytest.raises(ConstraintConfigError, match="at least one"):
        Choice()
    with pytest.raises(ConstraintConfigError, match="Invalid regex"):
        Regex("[")
