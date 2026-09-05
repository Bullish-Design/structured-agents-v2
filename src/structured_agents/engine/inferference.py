"""The llama.cpp serving plane, asked rather than assumed.

``LlamaCppEngine`` states its abilities as a constant. That constant omits
LoRA, so every adapter-bearing spec is refused before it reaches a server that
may well hold the adapter. This engine takes the same wire dialect but derives
its abilities from a negotiated ``Capabilities`` — what the running server said
about itself.

Build one with :meth:`structured_agents.Backend.negotiate`.
"""

from __future__ import annotations

from inferference.capabilities import Capabilities, CapabilityError, ConstraintKind

from ..constraint import Constraint, WireSpec
from ..errors import BackendCapabilityError
from .llama_cpp import LlamaCppEngine

# A neutral constraint kind, and the serving-plane dialect that enforces it.
# ``grammar`` maps to GBNF explicitly: GBNF is close to, but not the same as,
# the EBNF xgrammar accepts, and only a named dialect can be checked.
_DIALECT_BY_KIND = {
    "schema": ConstraintKind.SCHEMA,
    "choice": ConstraintKind.CHOICE,
    "grammar": ConstraintKind.GRAMMAR_GBNF,
}


class InferferenceEngine:
    """One inferference-served llama.cpp endpoint, with its abilities negotiated."""

    name = "inferference"

    def __init__(self, capabilities: Capabilities) -> None:
        self.capabilities = capabilities
        self.supports = frozenset(
            kind for kind, dialect in _DIALECT_BY_KIND.items() if capabilities.supports(dialect)
        )
        self._wire = LlamaCppEngine()

    def render(self, constraint: Constraint) -> WireSpec:
        """Render onto llama.cpp's wire; the transport is unchanged."""
        return self._wire.render(constraint)

    def resolve_model(self, adapter: str | None, default: str) -> str:
        """Return the OpenAI ``model`` field value, or refuse with the reason."""
        try:
            return self.capabilities.resolve_model_field(adapter or default)
        except CapabilityError as exc:
            raise BackendCapabilityError(str(exc)) from exc
