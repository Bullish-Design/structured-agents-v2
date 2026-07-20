"""llama.cpp engine: json_schema + GBNF grammar only. No regex, no LoRA over the OpenAI API.

UNVERIFIED: llama.cpp does not implement vLLM's XGrammar extension, and its verify.sh exercises no
grammar surface. GBNF is close to but not the same dialect as the EBNF that XGrammar accepts, so the
Grammar constraint's EBNF is passed through WITHOUT a parity claim.
"""

from __future__ import annotations

from typing import cast

from ..constraint import Constraint, WireSpec, _Choice, _Grammar, _Schema
from ..errors import BackendCapabilityError


def _gbnf_alternation(options: tuple[str, ...]) -> str:
    quoted = " | ".join('"' + option.replace("\\", "\\\\").replace('"', '\\"') + '"' for option in options)
    return f"root ::= {quoted}"


class LlamaCppEngine:
    name = "llama_cpp"
    supports = frozenset({"schema", "choice", "grammar"})

    def render(self, constraint: Constraint) -> WireSpec:
        match constraint:
            case _Schema():
                from pydantic_ai.output import NativeOutput

                return WireSpec(output_type=NativeOutput(constraint.model, strict=constraint.strict))
            case _Grammar():
                return WireSpec(output_type=str, extra_body={"grammar": constraint.ebnf})
            case _Choice():
                return WireSpec(
                    output_type=str,
                    extra_body={"grammar": _gbnf_alternation(cast(tuple[str, ...], constraint.options))},
                )
        raise BackendCapabilityError("llama_cpp engine does not support regex constraints.")
