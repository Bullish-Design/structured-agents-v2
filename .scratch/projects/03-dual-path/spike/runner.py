"""Dual-path spike runtime — Architecture C.

Two **independent** ``DBOSAgent``s (each ``.run`` is its own DBOS workflow), joined by a
**top-level** ``asyncio.gather`` (no parent workflow), then a thin comparator that assembles a
versioned ``ComparisonRecord`` and persists it to Postgres as ``jsonb``.

Throwaway spike code — lives under ``.scratch/``, not the library.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from dbos import DBOS, DBOSConfig
from pydantic_ai.durable_exec.dbos import DBOSAgent, StepConfig
from structured_agents_v2.backend import Backend
from structured_agents_v2.profile import AgentProfile

# --- Postgres / DBOS config ------------------------------------------------------------

PG_URL = os.environ.get("DUAL_PATH_PG_URL", "postgresql://andrew@127.0.0.1:5433/dual_path")


def dbos_config(name: str = "dual-path-spike") -> DBOSConfig:
    """DBOS config backed by the spike Postgres (its workflow/step system DB)."""
    return {"name": name, "system_database_url": PG_URL, "run_admin_server": False}


@contextlib.contextmanager
def dbos_lifecycle(name: str = "dual-path-spike") -> Iterator[None]:
    """Init → launch → (yield) → destroy. Construct DBOSAgents BEFORE entering (see note).

    DBOS requires workflows to be registered before ``launch()``. ``DBOSAgent.__init__`` does the
    ``@DBOS.workflow`` registration, so all agents must be built between ``DBOS(config)`` and this
    context's ``launch()``. We therefore split init from launch: callers do ``DBOS(config=...)``,
    build their agents, then enter a *separate* ``launched()`` block. This helper is the simple
    all-in-one for the rare case where agents are built inside ``setup``.
    """
    DBOS(config=dbos_config(name))
    DBOS.launch()
    try:
        yield
    finally:
        DBOS.destroy()


# --- Versioned comparison record -------------------------------------------------------


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def profile_version(profile: AgentProfile) -> str:
    """Content hash of the serialized profile (instructions + ref + decoder + settings)."""
    return _hash(profile.model_dump(mode="json"))


def schema_version(output_type: type[BaseModel]) -> str:
    """Content hash of the resolved JSON schema (so the record is reproducible)."""
    return _hash(output_type.model_json_schema())


def _lib_version() -> str:
    try:
        return importlib.metadata.version("structured-agents-v2")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def compare(primary: BaseModel | None, reference: BaseModel | None) -> tuple[bool | None, dict[str, list[Any]] | None]:
    """Exact + field-level comparator. Returns (agreement_exact, field_diff)."""
    if primary is None or reference is None:
        return None, None
    p, r = primary.model_dump(), reference.model_dump()
    if p == r:
        return True, {}
    diff = {k: [p.get(k), r.get(k)] for k in set(p) | set(r) if p.get(k) != r.get(k)}
    return False, diff


def usage_dict(run_result: Any) -> dict[str, Any] | None:
    """Best-effort token usage extraction from a pydantic_ai AgentRunResult."""
    try:
        u = run_result.usage()
    except Exception:  # pragma: no cover
        return None
    return {k: getattr(u, k, None) for k in ("input_tokens", "output_tokens", "total_tokens", "requests")}


class ComparisonRecord(BaseModel):
    """The unit of training/eval data: identity + payload + signals."""

    # identity / versioning
    run_id: str
    profile_version: str
    schema_version: str
    primary_model: str
    reference_model: str
    decode_mode: str
    lib_version: str
    primary_workflow_id: str | None = None
    reference_workflow_id: str | None = None
    created_at: str

    # payload
    prompt: str
    instructions: str
    primary_output: dict[str, Any] | None = None
    reference_output: dict[str, Any] | None = None
    primary_error: str | None = None
    reference_error: str | None = None

    # signals
    primary_valid: bool
    reference_valid: bool
    agreement_exact: bool | None = None
    field_diff: dict[str, list[Any]] | None = None
    primary_usage: dict[str, Any] | None = None
    reference_usage: dict[str, Any] | None = None


def build_record(
    *,
    run_id: str,
    prompt: str,
    profile: AgentProfile,
    output_type: type[BaseModel],
    decode_mode: str,
    primary_model: str,
    reference_model: str,
    primary_result: Any | None,
    reference_result: Any | None,
    primary_error: str | None = None,
    reference_error: str | None = None,
    primary_workflow_id: str | None = None,
    reference_workflow_id: str | None = None,
) -> ComparisonRecord:
    """Assemble (validate, diff, version) a ComparisonRecord from two run results."""
    p_out = primary_result.output if primary_result is not None else None
    r_out = reference_result.output if reference_result is not None else None
    agreement, diff = compare(p_out, r_out)
    return ComparisonRecord(
        run_id=run_id,
        profile_version=profile_version(profile),
        schema_version=schema_version(output_type),
        primary_model=primary_model,
        reference_model=reference_model,
        decode_mode=decode_mode,
        lib_version=_lib_version(),
        primary_workflow_id=primary_workflow_id,
        reference_workflow_id=reference_workflow_id,
        created_at=datetime.now(UTC).isoformat(),
        prompt=prompt,
        instructions=profile.instructions,
        primary_output=p_out.model_dump() if isinstance(p_out, BaseModel) else None,
        reference_output=r_out.model_dump() if isinstance(r_out, BaseModel) else None,
        primary_error=primary_error,
        reference_error=reference_error,
        primary_valid=isinstance(p_out, BaseModel),
        reference_valid=isinstance(r_out, BaseModel),
        agreement_exact=agreement,
        field_diff=diff,
        primary_usage=usage_dict(primary_result) if primary_result is not None else None,
        reference_usage=usage_dict(reference_result) if reference_result is not None else None,
    )


# --- Postgres jsonb store --------------------------------------------------------------

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
"""


class ComparisonStore:
    """Stores ComparisonRecords in Postgres with the full object in a ``jsonb`` column."""

    def __init__(self, url: str = PG_URL) -> None:
        self.url = url

    def init_schema(self) -> None:
        with psycopg.connect(self.url) as conn:
            conn.execute(_DDL)
            conn.commit()

    def save(self, record: ComparisonRecord) -> int:
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
                    record.agreement_exact,
                    Jsonb(record.model_dump()),
                ),
            ).fetchone()
            conn.commit()
            assert row is not None
            return int(row[0])


# --- agent construction ----------------------------------------------------------------


def build_dbos_agent(
    backend: Backend,
    profile: AgentProfile,
    *,
    dbos_name: str,
    step_config: StepConfig | None = None,
) -> tuple[Any, DBOSAgent]:
    """Build a StructuredAgent via the real Backend factory, then wrap it as a DBOSAgent.

    Returns (StructuredAgent, DBOSAgent). The StructuredAgent retains the RequestCapture (if the
    backend was built with ``capture=True``) so the wire shape stays inspectable.
    """
    sa = backend.build(profile)
    dbos_agent = DBOSAgent(sa.agent, name=dbos_name, model_step_config=step_config or {})
    return sa, dbos_agent


def last_body(structured_agent: Any) -> dict[str, Any] | None:
    """Read the last captured request body off a StructuredAgent (spike escape hatch)."""
    capture = structured_agent._capture  # noqa: SLF001 - throwaway spike introspection
    return capture.last.body if (capture and capture.records) else None


def _to_plain(obj: Any) -> Any:
    """Best-effort: turn a StepInfo / dataclass / TypedDict-ish into a JSON-able dict."""
    if isinstance(obj, dict):
        return {k: str(v)[:200] for k, v in obj.items()}
    d = getattr(obj, "__dict__", None)
    if d:
        return {k: str(v)[:200] for k, v in d.items()}
    if hasattr(obj, "_fields"):  # NamedTuple
        return {k: str(getattr(obj, k))[:200] for k in obj._fields}
    return {"repr": str(obj)[:200]}


async def workflow_steps(workflow_id: str) -> list[dict[str, Any]]:
    """Read back the DBOS-persisted steps for a workflow (what DBOS stored).

    Uses the async API because DBOS forbids the sync variant while an event loop is running.
    """
    steps = await DBOS.list_workflow_steps_async(workflow_id)
    return [_to_plain(s) for s in steps]
