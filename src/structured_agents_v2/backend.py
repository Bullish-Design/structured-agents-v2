"""`Backend` — the OpenAI-compatible server + its capabilities, and the agent factory.

This is the **only** module that imports `pydantic_ai.models.openai`; callers build agents
through `Backend.build(profile)` and never touch the model client directly.

A `Backend` knows its `base_url`/`default_model` and what the server can do (`BackendCaps`:
XGrammar? LoRA?). `build()` resolves a profile's decode contract, **gates on capabilities**
(raising `BackendCapabilityError` at build time), selects the adapter as the wire `model`
field, applies the decoder, and returns a runnable `StructuredAgent`.

Capture is opt-in per backend (`Backend(capture=True)`); a test transport can be attached
with `attach_transport(...)` so the suite runs against the in-process mock with no network.
"""

from __future__ import annotations

import warnings
from typing import Any

import httpx
from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .agent import StructuredAgent
from .capture import RequestCapture
from .errors import BackendCapabilityError
from .profile import AgentProfile

_GATED_MODES = frozenset({"grammar", "regex", "choice"})

# Valid top-level keys for OpenAIChatModelSettings (covers inherited ModelSettings keys too).
# A TypedDict silently drops unknown keys, so a typo'd setting name would vanish without
# error — we warn at build time instead.
_ALLOWED_SETTINGS: frozenset[str] = frozenset(getattr(OpenAIChatModelSettings, "__annotations__", {}))


def _warn_unknown_settings(profile: AgentProfile, settings: dict[str, Any]) -> None:
    unknown = sorted(k for k in settings if k not in _ALLOWED_SETTINGS)
    if unknown:
        warnings.warn(
            f"{profile.name!r}: model_settings has key(s) not in OpenAIChatModelSettings {unknown}; "
            "a TypedDict silently drops these — check for typos.",
            stacklevel=3,
        )


class BackendCaps(BaseModel):
    """What an OpenAI-compatible server can do, gating which agents it can build."""

    xgrammar: bool = True  # honors XGrammar grammar/regex/choice via extra_body
    lora: bool = True  # selects adapters via the model field


class Backend(BaseModel):
    """An OpenAI-compatible inference server and the capabilities it advertises."""

    base_url: str
    api_key: str = "sk-none"
    default_model: str
    caps: BackendCaps = BackendCaps()
    capture: bool = False  # opt-in: wire a RequestCapture into every built agent

    _transport: httpx.AsyncBaseTransport | None = PrivateAttr(default=None)
    _client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _capture_obj: RequestCapture | None = PrivateAttr(default=None)
    _client_built: bool = PrivateAttr(default=False)

    def attach_transport(self, transport: httpx.AsyncBaseTransport) -> Backend:
        """Route this backend's HTTP through `transport` (e.g. the in-process mock). Returns self."""
        self._transport = transport
        return self

    def _capture(self) -> RequestCapture | None:
        """The single `RequestCapture` shared by every agent this backend builds (if capture on)."""
        if self.capture and self._capture_obj is None:
            self._capture_obj = RequestCapture()
        return self._capture_obj

    def _shared_client(self) -> httpx.AsyncClient:
        """One `httpx.AsyncClient` per `Backend`, built lazily and shared across built agents.

        N agents then reuse one connection pool against the single vLLM server (better for
        `run_batch`), and the whole backend closes with one `aclose()`. Capture stays correct
        via the per-run contextvar sink even though the client (and its hook) is shared.
        """
        if not self._client_built:
            self._client_built = True
            capture = self._capture()
            if capture is not None:
                self._client = capture.client(transport=self._transport)
            elif self._transport is not None:
                self._client = httpx.AsyncClient(transport=self._transport)
            else:
                self._client = httpx.AsyncClient()
        assert self._client is not None
        return self._client

    def model_for(self, adapter: str | None = None) -> OpenAIChatModel:
        """An `OpenAIChatModel` whose wire `model` field is the adapter (or the default)."""
        provider = OpenAIProvider(base_url=self.base_url, api_key=self.api_key, http_client=self._shared_client())
        return OpenAIChatModel(adapter or self.default_model, provider=provider)

    async def aclose(self) -> None:
        """Close the shared HTTP client. Call once during teardown; safe to call repeatedly."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._client_built = False

    def _check_caps(self, profile: AgentProfile, mode: str) -> None:
        if mode in _GATED_MODES and not self.caps.xgrammar:
            raise BackendCapabilityError(
                f"{profile.name!r}: decode mode {mode!r} needs XGrammar, but this backend has caps.xgrammar=False."
            )
        if profile.adapter is not None and not self.caps.lora:
            raise BackendCapabilityError(
                f"{profile.name!r}: adapter {profile.adapter!r} needs LoRA, but this backend has caps.lora=False."
            )

    def build(self, profile: AgentProfile) -> StructuredAgent:
        """Resolve, cap-check, and construct a runnable `StructuredAgent` from a profile."""
        output_type, spec = profile.resolve()
        self._check_caps(profile, spec.mode)

        capture = self._capture()
        model = self.model_for(profile.adapter)
        app = spec.apply(output_type)

        settings: dict[str, Any] = dict(profile.model_settings)
        if app.extra_body:
            # Merge, don't clobber: the profile may carry its own extra_body (e.g. vLLM
            # sampling extensions). Decoder keys win on conflict — the constraint is
            # non-negotiable.
            settings["extra_body"] = {**settings.get("extra_body", {}), **app.extra_body}
        _warn_unknown_settings(profile, settings)
        model_settings = OpenAIChatModelSettings(**settings)  # type: ignore[typeddict-item]

        agent: Agent[None, Any] = Agent(
            model,
            output_type=app.output_type,
            model_settings=model_settings,
            instructions=profile.instructions,
        )
        return StructuredAgent(profile, agent, spec=spec, capture=capture)
