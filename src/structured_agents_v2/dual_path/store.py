"""`ComparisonStore` — Postgres `jsonb` persistence + export, the dual-path *data* source of truth.

The full `ComparisonRecord` is stored in a `jsonb` column (GIN-indexed) with a few promoted columns
for cheap filtering. This is deliberately separate from DBOS's step store (which is the durable
*execution* record, coupled to pydantic-ai's internal types): training/eval data is exported from
here, never from DBOS.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from .record import ComparisonRecord

_DDL = """
create table if not exists comparison_records (
    id                    bigserial primary key,
    run_id                text not null,
    created_at            timestamptz not null default now(),
    primary_workflow_id   text,
    reference_workflow_id text,
    profile_version       text not null,
    schema_version        text not null,
    agreement_exact       boolean,
    record                jsonb not null
);
create index if not exists comparison_records_record_gin on comparison_records using gin (record);
create index if not exists comparison_records_profile_idx on comparison_records (profile_version);
"""


class ComparisonStore:
    """Stores/queries `ComparisonRecord`s in Postgres with the full object in a `jsonb` column."""

    def __init__(self, pg_url: str) -> None:
        self.url = pg_url

    def init_schema(self) -> None:
        """Create the table + indexes if absent (idempotent)."""
        with psycopg.connect(self.url) as conn:
            conn.execute(_DDL)
            conn.commit()

    def save(self, record: ComparisonRecord) -> int:
        """Persist one record; returns its row id."""
        agreement = record.signal.agreement_exact if record.signal is not None else None
        with psycopg.connect(self.url) as conn:
            row = conn.execute(
                """
                insert into comparison_records
                    (run_id, primary_workflow_id, reference_workflow_id,
                     profile_version, schema_version, agreement_exact, record)
                values (%s, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    record.run_id,
                    record.primary_workflow_id,
                    record.reference_workflow_id,
                    record.profile_version,
                    record.schema_version,
                    agreement,
                    Jsonb(record.model_dump()),
                ),
            ).fetchone()
            conn.commit()
            assert row is not None
            return int(row[0])

    def query(
        self,
        *,
        profile_version: str | None = None,
        schema_version: str | None = None,
        agreement_exact: bool | None = None,
        limit: int = 1000,
    ) -> list[ComparisonRecord]:
        """Read records back, filtered on the promoted columns (most-recent first)."""
        clauses: list[str] = []
        params: list[Any] = []
        if profile_version is not None:
            clauses.append("profile_version = %s")
            params.append(profile_version)
        if schema_version is not None:
            clauses.append("schema_version = %s")
            params.append(schema_version)
        if agreement_exact is not None:
            clauses.append("agreement_exact = %s")
            params.append(agreement_exact)
        # `where` is built only from hardcoded column-name literals (no user input), so the
        # interpolated query is injection-safe; ty can't prove LiteralString through the f-string.
        where = (" where " + " and ".join(clauses)) if clauses else ""
        params.append(limit)
        query = f"select record from comparison_records{where} order by id desc limit %s"
        with psycopg.connect(self.url, row_factory=dict_row) as conn:  # ty: ignore[invalid-argument-type]
            rows = conn.execute(query, params).fetchall()  # ty: ignore[invalid-argument-type]
        return [ComparisonRecord.model_validate(r["record"]) for r in rows]


class GroupEval(BaseModel):
    """Aggregate eval stats for one group (e.g. one primary model)."""

    key: str
    n: int
    primary_valid_rate: float
    reference_valid_rate: float
    agreement_rate: float  # over rows where both validated and the reference ran


class EvalSummary(BaseModel):
    """Local-vs-frontier eval snapshot across a set of records."""

    total: int
    groups: list[GroupEval]


class ComparisonExport:
    """Reads a `ComparisonStore` and emits SFT data + eval views (never touches DBOS)."""

    def __init__(self, store: ComparisonStore) -> None:
        self.store = store

    def _records(self, **filters: Any) -> list[ComparisonRecord]:
        return self.store.query(**filters)

    def to_sft_jsonl(
        self,
        path: str | Path,
        *,
        require_reference_valid: bool = True,
        only_agreement: bool | None = None,
        **filters: Any,
    ) -> int:
        """Write SFT/teacher rows (reference output as the label) to JSONL; returns the count.

        Each line: ``{profile_version, schema_version, instructions, prompt, target}`` where
        ``target`` is the validated reference output. Gated on `reference_valid` by default;
        `only_agreement=True/False` keeps only agreeing/disagreeing rows.
        """
        path = Path(path)
        written = 0
        with path.open("w") as fh:
            for rec in self._records(**filters):
                if require_reference_valid and not (rec.reference_valid and rec.reference_output is not None):
                    continue
                if only_agreement is not None:
                    agree = rec.signal.agreement_exact if rec.signal is not None else None
                    if agree is not only_agreement:
                        continue
                fh.write(
                    json.dumps(
                        {
                            "profile_version": rec.profile_version,
                            "schema_version": rec.schema_version,
                            "instructions": rec.instructions,
                            "prompt": rec.prompt,
                            "target": rec.reference_output,
                        }
                    )
                    + "\n"
                )
                written += 1
        return written

    def eval_view(self, *, by: str = "primary_model", **filters: Any) -> EvalSummary:
        """Validity + agreement rates, grouped (default: by primary model `wire_model`)."""
        records = self._records(**filters)
        groups: dict[str, list[ComparisonRecord]] = {}
        for rec in records:
            key = rec.primary_model.wire_model if by == "primary_model" else rec.profile_version
            groups.setdefault(key, []).append(rec)
        out = [self._group_eval(key, recs) for key, recs in sorted(groups.items())]
        return EvalSummary(total=len(records), groups=out)

    @staticmethod
    def _group_eval(key: str, recs: Iterable[ComparisonRecord]) -> GroupEval:
        recs = list(recs)
        n = len(recs)
        p_valid = sum(r.primary_valid for r in recs)
        r_valid = sum(r.reference_valid for r in recs)
        comparable = [r for r in recs if r.signal is not None and not r.reference_skipped]
        agree = sum(1 for r in comparable if r.signal and r.signal.agreement_exact)
        return GroupEval(
            key=key,
            n=n,
            primary_valid_rate=p_valid / n if n else 0.0,
            reference_valid_rate=r_valid / n if n else 0.0,
            agreement_rate=agree / len(comparable) if comparable else 0.0,
        )
