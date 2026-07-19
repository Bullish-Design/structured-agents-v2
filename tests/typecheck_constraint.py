from __future__ import annotations

from typing import Literal, assert_type

from pydantic import BaseModel

from structured_agents import Choice, Constraint, Grammar, Regex, Schema


class Person(BaseModel):
    name: str


choice = Choice("keep", "skip")
assert_type(choice, Constraint[Literal["keep", "skip"]])
assert_type(choice.parse("keep"), Literal["keep", "skip"])
assert_type(Schema(Person).parse(Person(name="Ada")), Person)
assert_type(Regex(".+"), Constraint[str])
assert_type(Grammar('root ::= "a"'), Constraint[str])
