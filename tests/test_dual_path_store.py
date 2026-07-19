"""Postgres tests for the dual-path jsonb store + export, using devenv's service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

pytest.importorskip("psycopg")
import psycopg  # noqa: E402

from structured_agents_v2 import AgentProfile  # noqa: E402
from structured_agents_v2.dual_path import (  # noqa: E402
    ComparisonExport,
    ComparisonStore,
    ModelIdentity,
    build_comparison_record,
)

pytestmark = pytest.mark.dual_path


class Widget(BaseModel):
    action: Literal["a", "b"]
    name: str


@pytest.fixture
def store(dual_path_isolated_pg_url: str) -> ComparisonStore:
    s = ComparisonStore(dual_path_isolated_pg_url)
    s.init_schema()
    return s


def _record(run_id: str, *, ref: Widget | None, skipped: bool = False, wire: str = "adapter-x"):
    primary = ModelIdentity(kind="vllm", wire_model=wire, adapter=wire)
    reference = None if ref is None else ModelIdentity(kind="frontier", wire_model="gpt", provider="openai")
    return build_comparison_record(
        run_id=run_id,
        prompt="make a widget",
        profile=AgentProfile(name="w", instructions="Emit one widget.", output_type_ref="tests:Widget"),
        output_type=Widget,
        primary_model=primary,
        reference_model=reference,
        primary_output=Widget(action="a", name="x"),
        reference_output=ref,
        reference_skipped=skipped,
    )


def test_save_and_query_roundtrip(store: ComparisonStore) -> None:
    rec = _record("r1", ref=Widget(action="a", name="y"))
    row_id = store.save(rec)
    assert row_id > 0

    got = store.query(profile_version=rec.profile_version)
    assert len(got) == 1
    # the schema_version filter narrows identically
    assert len(store.query(schema_version=rec.schema_version)) == 1
    assert store.query(schema_version="nope") == []
    # nested jsonb fields survive the round-trip
    assert got[0].primary_model.wire_model == "adapter-x"
    assert got[0].signal is not None and got[0].signal.field_diff == {"name": ["x", "y"]}
    assert got[0].signal.agreement_exact is False


def test_query_filter_by_agreement(store: ComparisonStore) -> None:
    store.save(_record("agree", ref=Widget(action="a", name="x")))  # agreement True
    store.save(_record("disagree", ref=Widget(action="b", name="z")))  # agreement False

    agree = store.query(agreement_exact=True)
    disagree = store.query(agreement_exact=False)
    assert {r.run_id for r in agree} == {"agree"}
    assert {r.run_id for r in disagree} == {"disagree"}


def test_jsonb_nested_query_is_available(store: ComparisonStore) -> None:
    store.save(_record("r1", ref=Widget(action="a", name="y")))
    with psycopg.connect(store.url) as conn:
        row = conn.execute(
            "select record->'primary_model'->>'wire_model', record->'signal'->>'agreement_exact' "
            "from comparison_records order by id desc limit 1"
        ).fetchone()
    assert row is not None
    assert row[0] == "adapter-x"
    assert row[1] == "false"


def test_export_sft_jsonl_gated_on_reference_valid(store: ComparisonStore, tmp_path: Path) -> None:
    store.save(_record("valid", ref=Widget(action="a", name="y")))
    store.save(_record("skipped", ref=None, skipped=True))  # reference invalid -> excluded
    out = tmp_path / "sft.jsonl"

    n = ComparisonExport(store).to_sft_jsonl(out)
    assert n == 1
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["target"] == {"action": "a", "name": "y"}
    assert lines[0]["prompt"] == "make a widget"


def test_export_only_agreement_filter(store: ComparisonStore, tmp_path: Path) -> None:
    store.save(_record("agree", ref=Widget(action="a", name="x")))
    store.save(_record("disagree", ref=Widget(action="b", name="z")))

    only_agree = ComparisonExport(store).to_sft_jsonl(tmp_path / "a.jsonl", only_agreement=True)
    only_disagree = ComparisonExport(store).to_sft_jsonl(tmp_path / "d.jsonl", only_agreement=False)
    assert only_agree == 1
    assert only_disagree == 1


def test_eval_view_rates(store: ComparisonStore) -> None:
    store.save(_record("a1", ref=Widget(action="a", name="x")))  # agree
    store.save(_record("a2", ref=Widget(action="b", name="z")))  # disagree
    store.save(_record("a3", ref=None, skipped=True))  # reference skipped

    summary = ComparisonExport(store).eval_view()
    assert summary.total == 3
    assert len(summary.groups) == 1  # all share wire_model "adapter-x"
    g = summary.groups[0]
    assert g.key == "adapter-x"
    assert g.n == 3
    assert g.primary_valid_rate == 1.0
    assert g.reference_valid_rate == pytest.approx(2 / 3)
    assert g.agreement_rate == 0.5  # 1 agree of 2 comparable (skipped excluded)
