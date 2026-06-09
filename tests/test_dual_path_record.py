"""Pure (no-Postgres) tests for the dual-path data core: hashing, comparator, record assembly."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

pytest.importorskip("psycopg")  # the dual_path package is gated on the [dual-path] extra

from structured_agents_v2 import AgentProfile  # noqa: E402
from structured_agents_v2.dual_path import (  # noqa: E402
    ComparisonRecord,
    ExactFieldComparator,
    ModelIdentity,
    build_comparison_record,
    content_hash,
    profile_version,
    schema_version,
)


class Widget(BaseModel):
    action: Literal["a", "b"]
    name: str


def _profile(instructions: str = "Emit one widget.") -> AgentProfile:
    return AgentProfile(name="w", instructions=instructions, output_type_ref="tests:Widget")


def _ids() -> tuple[ModelIdentity, ModelIdentity]:
    primary = ModelIdentity(kind="vllm", wire_model="adapter-x", base="base", adapter="adapter-x")
    reference = ModelIdentity(kind="frontier", wire_model="gpt", provider="openai", model_id="gpt-4o-mini")
    return primary, reference


def test_content_hash_is_order_independent() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_profile_version_tracks_instructions() -> None:
    assert profile_version(_profile()) == profile_version(_profile())
    assert profile_version(_profile("X")) != profile_version(_profile("Y"))


def test_schema_version_tracks_schema() -> None:
    class Other(BaseModel):
        action: Literal["a", "b"]
        name: str
        extra: int

    assert schema_version(Widget) == schema_version(Widget)
    assert schema_version(Widget) != schema_version(Other)


def test_exact_field_comparator_agreement() -> None:
    cmp = ExactFieldComparator()
    sig = cmp.compare(Widget(action="a", name="x"), Widget(action="a", name="x"))
    assert sig.agreement_exact is True
    assert sig.field_diff == {}
    assert sig.score == 1.0


def test_exact_field_comparator_diff() -> None:
    cmp = ExactFieldComparator()
    sig = cmp.compare(Widget(action="a", name="x"), Widget(action="a", name="y"))
    assert sig.agreement_exact is False
    assert sig.field_diff == {"name": ["x", "y"]}
    assert sig.score == 0.5  # 1 of 2 fields match


def test_build_record_both_valid() -> None:
    primary, reference = _ids()
    rec = build_comparison_record(
        run_id="r1",
        prompt="make a widget",
        profile=_profile(),
        output_type=Widget,
        primary_model=primary,
        reference_model=reference,
        primary_output=Widget(action="a", name="x"),
        reference_output=Widget(action="a", name="y"),
        primary_usage={"output_tokens": 5},
        reference_usage={"output_tokens": 6},
        primary_workflow_id="primary-r1",
        reference_workflow_id="reference-r1",
    )
    assert isinstance(rec, ComparisonRecord)
    assert rec.primary_valid and rec.reference_valid
    assert rec.signal is not None and rec.signal.field_diff == {"name": ["x", "y"]}
    assert rec.primary_output == {"action": "a", "name": "x"}
    assert rec.reference_model is not None and rec.reference_model.provider == "openai"
    assert rec.decode_mode == "json_schema"
    # round-trips through json (jsonb-ready)
    assert ComparisonRecord.model_validate(rec.model_dump()) == rec


def test_build_record_reference_skipped() -> None:
    primary, _ = _ids()
    rec = build_comparison_record(
        run_id="r2",
        prompt="make a widget",
        profile=_profile(),
        output_type=Widget,
        primary_model=primary,
        reference_model=None,
        primary_output=Widget(action="b", name="z"),
        reference_output=None,
        reference_skipped=True,
    )
    assert rec.primary_valid is True
    assert rec.reference_valid is False
    assert rec.reference_skipped is True
    assert rec.signal is None  # no signal without two valid outputs
    assert rec.reference_output is None
