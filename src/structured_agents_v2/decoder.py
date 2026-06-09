"""The constrained-decoding contract and how it applies to a PydanticAI agent.

`DecoderSpec` is the small, serializable description of *how* an agent's output should be
constrained. `apply()` turns it into the two concrete things a PydanticAI agent needs:

- the `output_type` to pass to `Agent(...)`, and
- the `extra_body` to merge into the OpenAI model settings.

The mapping is grounded in the verified wire shapes (see
`.scratch/projects/02-library-wrapper/VERIFICATION.md`):

| mode        | output_type                | extra_body                                  |
|-------------|----------------------------|---------------------------------------------|
| json_schema | NativeOutput(model)        | {} (XGrammar is the server-level backend)   |
| grammar     | str (text mode)            | {"structured_outputs": {"grammar": ...}}    |
| regex       | str (text mode)            | {"structured_outputs": {"regex": ...}}      |
| choice      | str (text mode)            | {"structured_outputs": {"choice": [...]}}   |

`json_schema` is the only mode that rides standard `response_format`; the bare-string
modes go through `extra_body` because `response_format` can't express them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import NativeOutput

from .errors import ConstraintConfigError

DecodeMode = Literal["json_schema", "grammar", "regex", "choice"]


@dataclass(frozen=True)
class DecoderApplication:
    """The concrete result of applying a `DecoderSpec` to an output type."""

    output_type: Any
    extra_body: dict[str, Any] = field(default_factory=dict)


class DecoderSpec(BaseModel):
    """A constrained-decoding contract. Usually produced by `ConstrainedOutput`."""

    mode: DecodeMode = "json_schema"
    grammar: str | None = None
    regex: str | None = None
    choices: list[str] | None = None
    strict: bool = True

    def apply(self, output_type: type[BaseModel] | None) -> DecoderApplication:
        """Resolve this spec against an output type into agent-ready pieces."""
        if self.mode == "json_schema":
            if output_type is None:
                raise ConstraintConfigError("json_schema mode requires a Pydantic model output_type.")
            return DecoderApplication(output_type=NativeOutput(output_type, strict=self.strict))
        if self.mode == "grammar":
            if not self.grammar:
                raise ConstraintConfigError("grammar mode requires `grammar`.")
            return DecoderApplication(output_type=str, extra_body={"structured_outputs": {"grammar": self.grammar}})
        if self.mode == "regex":
            if not self.regex:
                raise ConstraintConfigError("regex mode requires `regex`.")
            return DecoderApplication(output_type=str, extra_body={"structured_outputs": {"regex": self.regex}})
        if self.mode == "choice":
            if not self.choices:
                raise ConstraintConfigError("choice mode requires `choices`.")
            return DecoderApplication(
                output_type=str, extra_body={"structured_outputs": {"choice": list(self.choices)}}
            )
        raise ConstraintConfigError(f"Unknown decode mode: {self.mode!r}")  # pragma: no cover
