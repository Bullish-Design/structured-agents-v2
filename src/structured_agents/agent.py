"""Durable constrained-agent primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.durable_exec.dbos import DBOSAgent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .constraint import Constraint
from .engine import Engine, select
from .errors import BackendCapabilityError


@dataclass(frozen=True)
class Settings:
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec[T]:
    name: str
    constraint: Constraint[T]
    instructions: str
    adapter: str | None = None
    settings: Settings = field(default_factory=Settings)


class Backend:
    """The sole importer of pydantic_ai.models.openai. Builds durable agents against a selected engine."""

    def __init__(self, *, engine: str | Engine = "vllm", base_url: str = "http://localhost:8000/v1",
                 api_key: str = "sk-none", default_model: str = "test",
                 http_client: httpx.AsyncClient | None = None, model: Any | None = None) -> None:
        self.engine = engine if not isinstance(engine, str) else select(engine)
        self.base_url, self.api_key, self.default_model = base_url, api_key, default_model
        self.client, self.model = http_client, model

    def build[T](self, spec: AgentSpec[T]) -> Agent[T]:
        constraint = spec.constraint
        if constraint.kind not in self.engine.supports:
            raise BackendCapabilityError(
                f"Agent {spec.name!r} requires {constraint.kind} constraints, "
                f"unsupported by engine {self.engine.name!r}."
            )
        if spec.adapter and "lora" not in self.engine.supports:
            raise BackendCapabilityError(
                f"Agent {spec.name!r} requires LoRA, unsupported by engine {self.engine.name!r}."
            )
        constraint.check()
        wire = self.engine.render(constraint)
        settings = {k: v for k, v in spec.settings.__dict__.items() if v is not None and k != "extra_body"}
        settings["extra_body"] = {**spec.settings.extra_body, **wire.extra_body}
        model = cast(Model[Any], self.model) if self.model is not None else OpenAIChatModel(
            spec.adapter or self.default_model,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key, http_client=self.client),
        )
        return Agent(spec, DBOSAgent(PydanticAgent(  # type: ignore[no-matching-overload]
            model, output_type=wire.output_type, model_settings=settings,
            instructions=spec.instructions, name=spec.name), name=spec.name))

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()


class Agent[T]:
    def __init__(self, spec: AgentSpec[T], raw: DBOSAgent[Any, Any]) -> None:
        self.spec, self._raw = spec, raw

    async def run(self, prompt: str) -> T:
        return self.spec.constraint.parse((await self._raw.run(prompt)).output)

    @property
    def raw(self) -> DBOSAgent[Any, Any]:
        return self._raw
