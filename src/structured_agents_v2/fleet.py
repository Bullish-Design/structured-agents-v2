"""`AgentSet` + `RoutingTable` — a fleet of constrained agents, batched and routed.

A fleet is just a named collection of `StructuredAgent`s built from one `Backend`, plus an
optional, *validated* `RoutingTable`. Two capabilities sit on top:

- **Batching** (`run_batch`): fire many agent calls concurrently with a top-level
  `asyncio.gather`. Against a server with N decode slots (e.g. vLLM continuous batching) this
  is the throughput win — the requests overlap instead of serializing.
- **Routing-as-data** (`RoutingTable` + `route`/`route_and_run`): a router agent emits a route
  value, and a serializable table maps that value to a specialist agent. Dispatch stays
  **explicit** — `route()` returns the specialist's name and `route_and_run()` is thin sugar
  over `route()` + the specialist's `run()`. No hidden control flow; the library never owns a
  loop or branch over your agents, and never executes anything.

Routing is validated at `build()`: the router and every specialist must exist, and — when the
router's route field is a `Literal` we can introspect — every value it can emit must have a
table entry or an explicit `default`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel

from .errors import FleetError, PolicyError, RoutingError

if TYPE_CHECKING:
    from .agent import AgentResult, StructuredAgent
    from .backend import Backend
    from .executor import Decision, ExecResult, Executor
    from .profile import AgentProfile


def _literal_values(annotation: Any) -> list[str] | None:
    """Return the string values of a `Literal[...]` annotation, or `None` if it isn't one."""
    if get_origin(annotation) is Literal:
        return [str(a) for a in get_args(annotation)]
    return None


class RoutingTable(BaseModel):
    """Serializable router→specialist routing, validated against an `AgentSet` at build time."""

    router: str  # agent name whose output carries the route value
    routes: dict[str, str]  # route value -> specialist agent name
    default: str | None = None  # specialist used when an emitted value has no `routes` entry
    route_field: str | None = None  # field on the router's model output holding the route value;
    # if None, a str output is used directly and a single-field model uses its one field


@dataclass
class BatchResult:
    """The outcome of `run_batch`: per-item results in input order, failures surfaced as data.

    One call raising no longer discards its siblings. Each element of `results` is either an
    `AgentResult` or the `BaseException` that call raised. `BatchResult` is iterable, sized, and
    indexable over `results`, so happy-path `for r in batch: r.output` still reads naturally; use
    `.errors`/`.ok` to inspect failures explicitly.
    """

    results: list[AgentResult[Any] | BaseException]

    @property
    def ok(self) -> bool:
        """True when no item raised."""
        return not self.errors

    @property
    def errors(self) -> list[BaseException]:
        """The exceptions raised by failed items (in input order)."""
        return [r for r in self.results if isinstance(r, BaseException)]

    @property
    def outputs(self) -> list[AgentResult[Any]]:
        """Only the successful `AgentResult`s (failures dropped)."""
        return [r for r in self.results if not isinstance(r, BaseException)]

    def __iter__(self) -> Iterator[AgentResult[Any] | BaseException]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> AgentResult[Any] | BaseException:
        return self.results[index]


@dataclass
class RoutedResult:
    """The outcome of `route_and_run`: which route fired, the specialist, and its output."""

    route: str  # the route value the router emitted
    agent: str  # the specialist agent name it mapped to
    output: Any  # the specialist's validated output
    result: AgentResult[Any]  # the specialist's full AgentResult (escape hatch)


@dataclass
class RoutedExecution:
    """The outcome of `route_and_execute`: the routed output plus the authority decision/effect."""

    route: str  # the route value the router emitted
    agent: str  # the specialist that produced the command
    output: Any  # the specialist's validated command
    decision: Decision  # the executor's authority verdict
    result: ExecResult | None  # the side-effect result, or None when the command was denied


class AgentSet:
    """A fleet of `StructuredAgent`s built from one `Backend`, with optional validated routing."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.agents: dict[str, StructuredAgent] = {}
        self.routing: RoutingTable | None = None

    def build(self, profiles: list[AgentProfile], routing: RoutingTable | None = None) -> None:
        """Build every profile into an agent (keyed by `profile.name`); validate routing if given."""
        names = [p.name for p in profiles]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise FleetError(f"duplicate agent names in fleet: {dups}")
        self.agents = {p.name: self.backend.build(p) for p in profiles}
        if routing is not None:
            self._validate_routing(routing)
        self.routing = routing

    def __getitem__(self, name: str) -> StructuredAgent:
        return self.agents[name]

    def set_routing(self, routing: RoutingTable) -> None:
        """Assign (and validate) a routing table after build. Prefer this over assigning
        `self.routing` directly — a raw assignment bypasses the router/specialist/Literal-coverage
        checks that keep routing sound."""
        self._validate_routing(routing)
        self.routing = routing

    async def aclose(self) -> None:
        """Close the backend's shared HTTP client. Convenience for `self.backend.aclose()`."""
        await self.backend.aclose()

    async def run_batch(self, calls: list[tuple[str, str]]) -> BatchResult:
        """Run `(agent_name, prompt)` calls concurrently; results keep the input order.

        A single call raising does **not** lose its siblings: failures come back in the
        `BatchResult` as data (`return_exceptions=True`), so autonomous/batch callers can log
        refusals or retry per item. Unknown agent names are a configuration error and still
        raise `FleetError` up front.
        """
        unknown = sorted({name for name, _ in calls if name not in self.agents})
        if unknown:
            raise FleetError(f"run_batch: unknown agent(s) {unknown}")
        results = await asyncio.gather(
            *(self.agents[name].run(prompt) for name, prompt in calls), return_exceptions=True
        )
        return BatchResult(results=list(results))

    async def route(self, msg: str) -> str:
        """Run the router on `msg` and return the specialist agent name it maps to."""
        rt = self._require_routing()
        result = await self.agents[rt.router].run(msg)
        value = self._route_value(rt, result.output)
        specialist = rt.routes.get(value, rt.default)
        if specialist is None:
            raise RoutingError(f"router {rt.router!r} emitted route {value!r} with no `routes` entry and no default.")
        return specialist

    async def route_and_run(self, msg: str) -> RoutedResult:
        """Two-step sugar: route `msg` to a specialist, run it on `msg`, return both."""
        rt = self._require_routing()
        router_result = await self.agents[rt.router].run(msg)
        value = self._route_value(rt, router_result.output)
        specialist = rt.routes.get(value, rt.default)
        if specialist is None:
            raise RoutingError(f"router {rt.router!r} emitted route {value!r} with no `routes` entry and no default.")
        result = await self.agents[specialist].run(msg)
        return RoutedResult(route=value, agent=specialist, output=result.output, result=result)

    async def route_and_execute(self, msg: str, executor: Executor, *, policy: str | None = None) -> RoutedExecution:
        """Route `msg` to a specialist, then authorize + (if allowed) execute its command.

        The fully-automatic pipeline — still explicit (you call it and pass the `executor`), so
        nothing executes implicitly. The policy is `policy=` or the specialist's
        `AgentProfile.policy`. A denial is returned as **data** (`decision.allowed` False, `result`
        None), not raised, so autonomous/batch callers can log refusals instead of crashing.
        """
        routed = await self.route_and_run(msg)
        pol = policy or self.agents[routed.agent].profile.policy
        if pol is None:
            raise PolicyError(f"agent {routed.agent!r} has no policy; set its AgentProfile.policy or pass policy=.")
        decision = executor.authorize(pol, routed.output)
        # Run the (synchronous, possibly blocking — e.g. subprocess) action off the event loop
        # so it doesn't stall the concurrency run_batch is built to deliver. authorize is cheap
        # and stays inline.
        result = await asyncio.to_thread(executor.execute, pol, routed.output) if decision.allowed else None
        return RoutedExecution(
            route=routed.route, agent=routed.agent, output=routed.output, decision=decision, result=result
        )

    # --- internals ---------------------------------------------------------------------

    def _require_routing(self) -> RoutingTable:
        if self.routing is None:
            raise RoutingError("no RoutingTable on this AgentSet; pass one to build().")
        return self.routing

    @staticmethod
    def _route_value(rt: RoutingTable, output: Any) -> str:
        """Extract the route value from a router output (a bare str, or a field on a model)."""
        if isinstance(output, str):
            return output
        if isinstance(output, BaseModel):
            field = rt.route_field
            if field is None:
                fields = type(output).model_fields
                if len(fields) != 1:
                    raise RoutingError(
                        f"router output {type(output).__name__} has {len(fields)} fields; "
                        "set RoutingTable.route_field to name the route field."
                    )
                field = next(iter(fields))
            if not hasattr(output, field):
                raise RoutingError(f"route_field {field!r} not present on {type(output).__name__}.")
            return str(getattr(output, field))
        raise RoutingError(f"cannot derive a route value from router output of type {type(output).__name__}.")

    def _validate_routing(self, rt: RoutingTable) -> None:
        if rt.router not in self.agents:
            raise RoutingError(f"router {rt.router!r} is not in the agent set.")
        for value, specialist in rt.routes.items():
            if specialist not in self.agents:
                raise RoutingError(f"route {value!r} -> {specialist!r}: no such agent in the set.")
        if rt.default is not None and rt.default not in self.agents:
            raise RoutingError(f"default specialist {rt.default!r}: no such agent in the set.")
        self._check_route_coverage(rt)

    def _check_route_coverage(self, rt: RoutingTable) -> None:
        """If the router's route field is an introspectable `Literal`, require full coverage."""
        output_type = self.agents[rt.router].profile.resolve_output_type()
        if output_type is None:
            return  # bare-string router (choice/regex) — values can't be enumerated statically
        field_name = rt.route_field
        if field_name is None:
            if len(output_type.model_fields) != 1:
                return  # ambiguous which field; runtime extraction will surface a clear error
            field_name = next(iter(output_type.model_fields))
        field = output_type.model_fields.get(field_name)
        if field is None:
            raise RoutingError(f"route_field {field_name!r} is not a field of {output_type.__name__}.")
        literals = _literal_values(field.annotation)
        if literals is None:
            return  # not a Literal — can't enumerate the emittable values
        missing = [v for v in literals if v not in rt.routes]
        if missing and rt.default is None:
            raise RoutingError(f"router can emit {missing} with no `routes` entry and no default specialist.")
