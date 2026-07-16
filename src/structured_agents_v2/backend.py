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


class BackendCaps(BaseModel):
    """What an OpenAI-compatible server can do, gating which agents it can build."""

    xgrammar: bool = True  # honors XGrammar grammar/regex/choice via extra_body
    lora: bool = True  # selects adapters via the model field
    server_default_backend: bool = True  # XGrammar set via a server flag, not per-request


class Backend(BaseModel):
    """An OpenAI-compatible inference server and the capabilities it advertises."""

    base_url: str
    api_key: str = "sk-none"
    default_model: str
    caps: BackendCaps = BackendCaps()
    capture: bool = False  # opt-in: wire a RequestCapture into every built agent

    _transport: httpx.AsyncBaseTransport | None = PrivateAttr(default=None)

    def attach_transport(self, transport: httpx.AsyncBaseTransport) -> Backend:
        """Route this backend's HTTP through `transport` (e.g. the in-process mock). Returns self."""
        self._transport = transport
        return self

    def _http_client(self, capture: RequestCapture | None) -> httpx.AsyncClient | None:
        """Build the httpx client, if any, for capture and/or a test transport."""
        if capture is not None:
            return capture.client(transport=self._transport)
        if self._transport is not None:
            return httpx.AsyncClient(transport=self._transport)
        return None

    def model_for(self, adapter: str | None = None, *, capture: RequestCapture | None = None) -> OpenAIChatModel:
        """An `OpenAIChatModel` whose wire `model` field is the adapter (or the default)."""
        client = self._http_client(capture)
        provider_kwargs: dict[str, Any] = {"base_url": self.base_url, "api_key": self.api_key}
        if client is not None:
            provider_kwargs["http_client"] = client
        provider = OpenAIProvider(**provider_kwargs)
        return OpenAIChatModel(adapter or self.default_model, provider=provider)

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

        capture = RequestCapture() if self.capture else None
        model = self.model_for(profile.adapter, capture=capture)
        app = spec.apply(output_type)

        settings: dict[str, Any] = dict(profile.model_settings)
        if app.extra_body:
            # Merge, don't clobber: the profile may carry its own extra_body (e.g. vLLM
            # sampling extensions). Decoder keys win on conflict — the constraint is
            # non-negotiable.
            settings["extra_body"] = {**settings.get("extra_body", {}), **app.extra_body}
        model_settings = OpenAIChatModelSettings(**settings)  # type: ignore[typeddict-item]

        agent: Agent[None, Any] = Agent(
            model,
            output_type=app.output_type,
            model_settings=model_settings,
            instructions=profile.instructions,
        )
        return StructuredAgent(profile, agent, spec=spec, capture=capture)
