"""`AgentProfile` — the serializable binding of an output type + adapter + instructions.

A profile is pure data: it names an agent, points at an output type by dotted reference
(`"pkg.module:ClassName"`), and carries the instructions and model settings. The decode
contract is resolved from that output type:

- a `ConstrainedOutput` subclass supplies its own `decoder_spec()` (the constraint travels
  with the type), so `decoder` may stay `None`;
- a plain Pydantic model defaults to `json_schema`;
- `decoder` is the escape hatch for constraining a type you can't subclass, and is also
  used when there is no output type at all (bare-string grammar/regex/choice modes).

`Backend.build(profile)` turns a profile into a runnable `StructuredAgent`.
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel

from .constrained import ConstrainedOutput
from .decoder import DecoderSpec
from .errors import ConfigError

_BARE_STRING_MODES = frozenset({"grammar", "regex", "choice"})


class AgentProfile(BaseModel):
    """A declarative, serializable description of one constrained agent."""

    name: str
    adapter: str | None = None
    instructions: str
    output_type_ref: str | None = None
    decoder: DecoderSpec | None = None
    policy: str | None = None
    model_settings: dict[str, Any] = {}

    def resolve_output_type(self) -> type[BaseModel] | None:
        """Import and return the type named by `output_type_ref` (or `None` if unset).

        Raises `ConfigError` with a clear message if the reference is malformed, the module
        can't be imported, the attribute is missing, or it isn't a Pydantic model type.

        Note: this **executes an import** of the referenced module — a profile is therefore
        code-equivalent config, not inert data. That is fine for a personal library where
        profiles are authored in-tree; once profiles load from YAML/JSON (CONCEPT phase 5),
        gate importable module prefixes behind an allowlist so a config file can't import
        arbitrary modules.
        """
        ref = self.output_type_ref
        if ref is None:
            return None
        if ":" not in ref:
            raise ConfigError(f"{self.name!r}: output_type_ref must be 'module:ClassName', got {ref!r}.")
        module_path, _, attr = ref.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ConfigError(
                f"{self.name!r}: could not import module {module_path!r} from output_type_ref {ref!r}: {exc}"
            ) from exc
        try:
            obj = getattr(module, attr)
        except AttributeError as exc:
            raise ConfigError(
                f"{self.name!r}: module {module_path!r} has no attribute {attr!r} (from output_type_ref {ref!r})."
            ) from exc
        if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
            raise ConfigError(f"{self.name!r}: output_type_ref {ref!r} must point to a Pydantic model, got {obj!r}.")
        return obj

    def resolve(self) -> tuple[type[BaseModel] | None, DecoderSpec]:
        """Resolve the (output_type, decoder) pair this profile is built from.

        - `ConstrainedOutput` subclass → its own `decoder_spec()`.
        - explicit `decoder` → used as-is (override for a non-subclassable type, or a
          bare-string mode with no output type).
        - plain Pydantic model with no `decoder` → `json_schema`.

        A `ConstrainedOutput` subclass *and* an explicit `decoder` is a conflict (two sources
        of truth for the constraint) and raises `ConfigError` rather than silently ignoring one.
        """
        output_type = self.resolve_output_type()
        if isinstance(output_type, type) and issubclass(output_type, ConstrainedOutput):
            if self.decoder is not None:
                raise ConfigError(
                    f"{self.name!r}: output_type_ref points to a ConstrainedOutput (carries its own "
                    "decoder_spec) AND an explicit decoder is set — remove one; they conflict."
                )
            return output_type, output_type.decoder_spec()
        if self.decoder is not None:
            return output_type, self.decoder
        if output_type is not None:
            return output_type, DecoderSpec(mode="json_schema")
        raise ConfigError(
            f"{self.name!r}: no output_type_ref and no decoder — provide one of them "
            "(a ConstrainedOutput/Model output_type_ref, or a DecoderSpec for a bare-string mode)."
        )
