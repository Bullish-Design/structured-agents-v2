"""vLLM engine: the reference dialect, byte-for-byte identical to the pre-refactor wire."""

from __future__ import annotations

from ..constraint import Constraint, WireSpec, _Choice, _Grammar, _Regex, _Schema
from ..errors import BackendCapabilityError


class VLLMEngine:
    name = "vllm"
    supports = frozenset({"schema", "regex", "choice", "grammar", "lora"})

    def render(self, constraint: Constraint) -> WireSpec:
        match constraint:
            case _Schema():
                from pydantic_ai.output import NativeOutput

                return WireSpec(output_type=NativeOutput(constraint.model, strict=constraint.strict))
            case _Regex():
                return WireSpec(output_type=str, extra_body={"structured_outputs": {"regex": constraint.pattern}})
            case _Choice():
                return WireSpec(
                    output_type=str, extra_body={"structured_outputs": {"choice": list(constraint.options)}}
                )
            case _Grammar():
                return WireSpec(output_type=str, extra_body={"structured_outputs": {"grammar": constraint.ebnf}})
        raise BackendCapabilityError(f"vllm engine cannot render constraint {type(constraint).__name__}.")
