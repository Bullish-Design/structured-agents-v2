"""SGLang engine: response_format json_schema; extra_body regex/ebnf; choice lowered to a regex.

UNVERIFIED: no constrained request is exercised against SGLang anywhere in this repo (deploy/sglang/
native/verify.sh only checks health/models/one chat; its README disclaims wire-shape compatibility).
Field names and choice lowering follow SGLang's published API, not a live run.
"""

from __future__ import annotations

import re
from typing import cast

from ..constraint import Constraint, WireSpec, _Choice, _Grammar, _Regex, _Schema
from ..errors import BackendCapabilityError


def _regex_alternation(options: tuple[str, ...]) -> str:
    return "(" + "|".join(re.escape(option) for option in options) + ")"


class SGLangEngine:
    name = "sglang"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})

    def render(self, constraint: Constraint) -> WireSpec:
        match constraint:
            case _Schema():
                from pydantic_ai.output import NativeOutput

                return WireSpec(output_type=NativeOutput(constraint.model, strict=constraint.strict))
            case _Regex():
                return WireSpec(output_type=str, extra_body={"regex": constraint.pattern})
            case _Grammar():
                return WireSpec(output_type=str, extra_body={"ebnf": constraint.ebnf})
            case _Choice():
                return WireSpec(
                    output_type=str, extra_body={"regex": _regex_alternation(cast(tuple[str, ...], constraint.options))}
                )
        raise BackendCapabilityError(f"sglang engine cannot render constraint {type(constraint).__name__}.")
