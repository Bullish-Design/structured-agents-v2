"""`AgentSet`/`RoutingTable` — fleet batching + routing against the in-process mock."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import httpx
import pytest

from structured_agents_v2 import (
    AgentProfile,
    AgentSet,
    AllowlistExecutor,
    Backend,
    ConstrainedOutput,
    DecoderSpec,
    FleetError,
    Policy,
    RoutedExecution,
    RoutingError,
    RoutingTable,
)
from structured_agents_v2.errors import PolicyError


class Route(ConstrainedOutput):
    route: Literal["file_edit", "git_ops", "answer"]


class Answer(ConstrainedOutput):
    text: str


def _backend(transport: httpx.ASGITransport, **caps: bool) -> Backend:
    from structured_agents_v2 import BackendCaps

    kw: dict[str, Any] = {"base_url": "http://mock/v1", "default_model": "base", "capture": True}
    if caps:
        kw["caps"] = BackendCaps(**caps)
    return Backend(**kw).attach_transport(transport)


def _router_profile() -> AgentProfile:
    return AgentProfile(name="router", adapter="router", instructions="route", output_type_ref="test_fleet:Route")


def _specialist(name: str, adapter: str) -> AgentProfile:
    return AgentProfile(name=name, adapter=adapter, instructions="answer", output_type_ref="test_fleet:Answer")


def _keyed_responder(req: dict[str, Any]) -> str:
    """Return content keyed by the wire `model` (each agent has a distinct adapter)."""
    m = req.get("model")
    if m == "router":
        return '{"route": "git_ops"}'
    return json.dumps({"text": f"{m} handled"})


# --- batching --------------------------------------------------------------------------


def test_run_batch_preserves_order_and_outputs(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder
    fleet = AgentSet(_backend(transport))
    fleet.build([_specialist("file_edit", "fe"), _specialist("git_ops", "go")])

    results = asyncio.run(fleet.run_batch([("git_ops", "a"), ("file_edit", "b"), ("git_ops", "c")]))
    assert [r.output.text for r in results] == ["go handled", "fe handled", "go handled"]


def test_run_batch_unknown_agent_raises(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder
    fleet = AgentSet(_backend(transport))
    fleet.build([_specialist("file_edit", "fe")])
    with pytest.raises(FleetError, match="unknown agent"):
        asyncio.run(fleet.run_batch([("file_edit", "a"), ("nope", "b")]))


def test_run_batch_is_concurrent() -> None:
    """run_batch overlaps requests: a tracking server sees >1 in flight at once."""

    class Tracker:
        def __init__(self) -> None:
            self.inflight = 0
            self.max_inflight = 0

        async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
            while True:
                event = await receive()
                if not event.get("more_body", False):
                    break
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            await asyncio.sleep(0.05)  # hold the slot so concurrent calls overlap
            self.inflight -= 1
            payload = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "base",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": '{"text": "ok"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            data = json.dumps(payload).encode()
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": data})

    tracker = Tracker()
    fleet = AgentSet(_backend(httpx.ASGITransport(app=tracker)))
    fleet.build([_specialist(f"a{i}", f"a{i}") for i in range(4)])

    results = asyncio.run(fleet.run_batch([(f"a{i}", "go") for i in range(4)]))
    assert len(results) == 4
    assert tracker.max_inflight >= 2  # genuinely concurrent, not serialized


# --- routing-table validation ----------------------------------------------------------


def _full_routes() -> dict[str, str]:
    return {"file_edit": "file_edit", "git_ops": "git_ops", "answer": "git_ops"}


def _built_fleet(transport: httpx.ASGITransport, routing: RoutingTable | None = None) -> AgentSet:
    fleet = AgentSet(_backend(transport))
    fleet.build(
        [_router_profile(), _specialist("file_edit", "fe"), _specialist("git_ops", "go")],
        routing=routing,
    )
    return fleet


def test_duplicate_agent_names_raise(transport: httpx.ASGITransport) -> None:
    fleet = AgentSet(_backend(transport))
    with pytest.raises(FleetError, match="duplicate agent names"):
        fleet.build([_specialist("dup", "a"), _specialist("dup", "b")])


def test_routing_unknown_router_raises(transport: httpx.ASGITransport) -> None:
    with pytest.raises(RoutingError, match="router 'ghost'"):
        _built_fleet(transport, RoutingTable(router="ghost", routes=_full_routes()))


def test_routing_unknown_specialist_raises(transport: httpx.ASGITransport) -> None:
    with pytest.raises(RoutingError, match="no such agent"):
        _built_fleet(transport, RoutingTable(router="router", routes={"git_ops": "ghost"}, default="git_ops"))


def test_routing_literal_coverage_gap_raises(transport: httpx.ASGITransport) -> None:
    # Route can emit answer/file_edit/git_ops; omit two and provide no default.
    with pytest.raises(RoutingError, match="with no .*entry and no default"):
        _built_fleet(transport, RoutingTable(router="router", routes={"git_ops": "git_ops"}))


def test_routing_coverage_gap_ok_with_default(transport: httpx.ASGITransport) -> None:
    fleet = _built_fleet(transport, RoutingTable(router="router", routes={"git_ops": "git_ops"}, default="file_edit"))
    assert fleet.routing is not None


# --- routing dispatch ------------------------------------------------------------------


def test_route_returns_specialist_name(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder  # router emits "git_ops"
    fleet = _built_fleet(transport, RoutingTable(router="router", routes=_full_routes()))
    assert asyncio.run(fleet.route("do something")) == "git_ops"


def test_route_and_run_full_two_step(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder
    fleet = _built_fleet(transport, RoutingTable(router="router", routes=_full_routes()))
    routed = asyncio.run(fleet.route_and_run("do something"))
    assert routed.route == "git_ops"
    assert routed.agent == "git_ops"
    assert isinstance(routed.output, Answer)
    assert routed.output.text == "go handled"
    assert routed.result.request_body is not None  # full AgentResult escape hatch present


def test_route_uses_default_for_unmapped_value(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # router emits "git_ops" but we only map "file_edit"; default catches it.
    mock_openai.responder = _keyed_responder
    fleet = _built_fleet(
        transport, RoutingTable(router="router", routes={"file_edit": "file_edit"}, default="file_edit")
    )
    assert asyncio.run(fleet.route("x")) == "file_edit"


def test_route_without_table_raises(transport: httpx.ASGITransport) -> None:
    fleet = _built_fleet(transport)  # no routing
    with pytest.raises(RoutingError, match="no RoutingTable"):
        asyncio.run(fleet.route("x"))


def test_route_value_from_bare_string_choice_router(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    """A choice-mode router emits the route value as a bare string (no model field needed)."""
    mock_openai.responder = lambda _req: "git_ops"
    fleet = AgentSet(_backend(transport))
    chooser = AgentProfile(
        name="router", instructions="pick", decoder=DecoderSpec(mode="choice", choices=["file_edit", "git_ops"])
    )
    fleet.build(
        [chooser, _specialist("file_edit", "fe"), _specialist("git_ops", "go")],
        routing=RoutingTable(router="router", routes={"file_edit": "file_edit", "git_ops": "git_ops"}),
    )
    assert asyncio.run(fleet.route("x")) == "git_ops"


def test_getitem_returns_agent(transport: httpx.ASGITransport) -> None:
    from structured_agents_v2 import StructuredAgent

    fleet = _built_fleet(transport)
    assert isinstance(fleet["git_ops"], StructuredAgent)
    assert fleet["git_ops"].profile.name == "git_ops"


def test_route_and_run_unmapped_no_default_raises(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # router emits "git_ops"; only "file_edit" mapped and no default -> runtime RoutingError.
    mock_openai.responder = _keyed_responder
    fleet = _built_fleet(
        transport, RoutingTable(router="router", routes={"file_edit": "file_edit"}, default="file_edit")
    )
    fleet.routing = RoutingTable(router="router", routes={"file_edit": "file_edit"})  # drop the default
    with pytest.raises(RoutingError, match="no .*entry and no default"):
        asyncio.run(fleet.route_and_run("x"))


class MultiRoute(ConstrainedOutput):
    reason: str
    route: Literal["file_edit", "git_ops"]


def _multi_router() -> AgentProfile:
    return AgentProfile(name="router", adapter="router", instructions="route", output_type_ref="test_fleet:MultiRoute")


def test_multi_field_router_needs_route_field(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # A multi-field model output is ambiguous without route_field; build skips the static
    # coverage check, and route() surfaces a clear runtime error.
    mock_openai.responder = lambda _req: '{"reason": "x", "route": "git_ops"}'
    fleet = AgentSet(_backend(transport))
    fleet.build(
        [_multi_router(), _specialist("file_edit", "fe"), _specialist("git_ops", "go")],
        routing=RoutingTable(router="router", routes={"file_edit": "file_edit", "git_ops": "git_ops"}),
    )
    with pytest.raises(RoutingError, match="set RoutingTable.route_field"):
        asyncio.run(fleet.route("x"))


def test_route_field_names_the_field(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # With route_field set, the named field is used (and statically coverage-checked at build).
    mock_openai.responder = lambda _req: '{"reason": "x", "route": "git_ops"}'
    fleet = AgentSet(_backend(transport))
    fleet.build(
        [_multi_router(), _specialist("file_edit", "fe"), _specialist("git_ops", "go")],
        routing=RoutingTable(
            router="router",
            routes={"file_edit": "file_edit", "git_ops": "git_ops"},
            route_field="route",
        ),
    )
    assert asyncio.run(fleet.route("x")) == "git_ops"


# --- route_and_execute: router -> specialist -> executor (the autonomous pipeline) ------


def _specialist_with_policy(name: str, adapter: str, policy: str | None) -> AgentProfile:
    return AgentProfile(
        name=name, adapter=adapter, instructions="answer", output_type_ref="test_fleet:Answer", policy=policy
    )


def _routed_exec_fleet(transport: httpx.ASGITransport, *, policy: str | None) -> AgentSet:
    fleet = AgentSet(_backend(transport))
    fleet.build(
        [
            _router_profile(),
            _specialist_with_policy("file_edit", "fe", policy),
            _specialist_with_policy("git_ops", "go", policy),
        ],
        routing=RoutingTable(router="router", routes=_full_routes()),
    )
    return fleet


def test_route_and_execute_auto_runs_allowed_command(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder  # router -> git_ops; specialist -> Answer(text="go handled")
    sink: list[str] = []
    executor = AllowlistExecutor(
        [Policy("answer_v1", allow=lambda c: "handled" in c.text, action=lambda c: sink.append(c.text))]
    )
    fleet = _routed_exec_fleet(transport, policy="answer_v1")

    outcome = asyncio.run(fleet.route_and_execute("do something", executor))
    assert isinstance(outcome, RoutedExecution)
    assert outcome.agent == "git_ops"
    assert outcome.output.text == "go handled"
    assert outcome.decision.allowed
    assert outcome.result is not None and outcome.result.ok
    assert sink == ["go handled"]  # the side effect actually fired, no human in the loop


def test_route_and_execute_denial_is_data_not_effect(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder
    sink: list[str] = []
    executor = AllowlistExecutor([Policy("answer_v1", allow=lambda _c: False, action=lambda c: sink.append(c.text))])
    fleet = _routed_exec_fleet(transport, policy="answer_v1")

    outcome = asyncio.run(fleet.route_and_execute("do something", executor))
    assert outcome.decision.allowed is False
    assert outcome.result is None  # denied -> no effect
    assert sink == []


def test_route_and_execute_requires_a_policy(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    mock_openai.responder = _keyed_responder
    executor = AllowlistExecutor([Policy("answer_v1", allow=lambda _c: True)])
    fleet = _routed_exec_fleet(transport, policy=None)  # specialists carry no policy
    with pytest.raises(PolicyError, match="has no policy"):
        asyncio.run(fleet.route_and_execute("do something", executor))


def test_route_and_execute_policy_override(mock_openai: Any, transport: httpx.ASGITransport) -> None:
    # specialists have no policy, but an explicit policy= drives the executor anyway.
    mock_openai.responder = _keyed_responder
    sink: list[str] = []
    executor = AllowlistExecutor([Policy("answer_v1", allow=lambda _c: True, action=lambda c: sink.append(c.text))])
    fleet = _routed_exec_fleet(transport, policy=None)

    outcome = asyncio.run(fleet.route_and_execute("do something", executor, policy="answer_v1"))
    assert outcome.decision.allowed and sink == ["go handled"]
