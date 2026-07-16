"""`DualPathRunner` — one registered dual-path agent (gather, validate, diff, persist).

A runner owns the two `DBOSAgent`s that `DualPathRuntime.register` built for one logical agent
(its `primary`, a local vLLM leg, and its `reference`, a frontier teacher leg), plus a handle to
the originating `StructuredAgent`s (for profile/output-type/identity) and the shared
`ComparisonStore`. `run()` is Architecture C: each leg's `.run` is its own DBOS workflow, joined by
a **top-level** `asyncio.gather` (no parent workflow). The reference leg fires only when sampled in
(`force_reference`, else `random() < sample_rate`); a failing leg is captured as an error rather
than failing the whole call. Every `run()` validates both outputs, diffs them via the `Comparator`,
assembles a versioned `ComparisonRecord`, persists it, and returns it — `record.primary_output` is
the user-facing answer.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from typing import TYPE_CHECKING, Any

from dbos import SetWorkflowID
from pydantic import BaseModel

from .record import ComparisonRecord, ModelIdentity, build_comparison_record

if TYPE_CHECKING:
    from pydantic_ai.durable_exec.dbos import DBOSAgent

    from ..agent import StructuredAgent
    from .comparator import Comparator
    from .store import ComparisonStore


def _identity(agent: StructuredAgent, *, kind: str) -> ModelIdentity:
    """Derive a `ModelIdentity` from a built `StructuredAgent` (read off its wire model)."""
    model = agent.agent.model
    wire = getattr(model, "model_name", None) or "unknown"
    if kind == "vllm":
        return ModelIdentity(kind="vllm", wire_model=wire, adapter=agent.profile.adapter)
    provider = getattr(model, "system", None)
    return ModelIdentity(kind="frontier", wire_model=wire, model_id=wire, provider=provider)


def _usage(result: Any) -> dict[str, Any] | None:
    """Best-effort token-usage extraction from a pydantic_ai run result."""
    try:
        u = result.usage  # pydantic-ai v2: property, not a method
    except Exception:  # pragma: no cover - defensive; .usage is a cheap attribute read
        return None
    return {k: getattr(u, k, None) for k in ("input_tokens", "output_tokens", "total_tokens", "requests")}


def _unpack(raw: Any) -> tuple[BaseModel | None, str | None, Any]:
    """Turn a gathered leg result (or exception) into (output, error, result)."""
    if isinstance(raw, BaseException):
        return None, str(raw), None
    output = raw.output if isinstance(raw.output, BaseModel) else None
    return output, None, raw


class DualPathRunner:
    """One dual-path agent: two `DBOSAgent` legs, run concurrently, compared, and persisted."""

    def __init__(
        self,
        *,
        name: str,
        primary_agent: StructuredAgent,
        reference_agent: StructuredAgent,
        primary_dbos: DBOSAgent[Any, Any],
        reference_dbos: DBOSAgent[Any, Any],
        store: ComparisonStore,
        sample_rate: float,
        comparator: Comparator | None = None,
    ) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self._primary_sa = primary_agent
        self._reference_sa = reference_agent
        self._primary_dbos = primary_dbos
        self._reference_dbos = reference_dbos
        self._store = store
        self._comparator = comparator
        output_type, _ = primary_agent.profile.resolve()
        assert output_type is not None  # guaranteed json_schema by DualPathRuntime.register
        self._output_type = output_type

    @property
    def primary(self) -> DBOSAgent[Any, Any]:
        """The primary (local vLLM) `DBOSAgent` (escape hatch)."""
        return self._primary_dbos

    @property
    def reference(self) -> DBOSAgent[Any, Any]:
        """The reference (frontier teacher) `DBOSAgent` (escape hatch)."""
        return self._reference_dbos

    async def run(self, prompt: str, *, force_reference: bool | None = None, **kw: Any) -> ComparisonRecord:
        """Run the primary (and, if sampled in, the reference), compare, persist, and return."""
        run_reference = force_reference if force_reference is not None else (random.random() < self.sample_rate)
        run_id = uuid.uuid4().hex[:12]
        pid = f"primary-{run_id}"
        rid = f"reference-{run_id}"

        async def _leg(agent: DBOSAgent, wid: str) -> Any:
            with SetWorkflowID(wid):
                return await agent.run(prompt, **kw)

        if run_reference:
            primary_raw, reference_raw = await asyncio.gather(
                _leg(self._primary_dbos, pid),
                _leg(self._reference_dbos, rid),
                return_exceptions=True,
            )
        else:
            primary_raw = await _leg(self._primary_dbos, pid)
            reference_raw = None

        primary_output, primary_error, primary_result = _unpack(primary_raw)
        if run_reference:
            reference_output, reference_error, reference_result = _unpack(reference_raw)
        else:
            reference_output, reference_error, reference_result = None, None, None

        record = build_comparison_record(
            run_id=run_id,
            prompt=prompt,
            profile=self._primary_sa.profile,
            output_type=self._output_type,
            primary_model=_identity(self._primary_sa, kind="vllm"),
            reference_model=_identity(self._reference_sa, kind="frontier") if run_reference else None,
            primary_output=primary_output,
            reference_output=reference_output,
            primary_error=primary_error,
            reference_error=reference_error,
            reference_skipped=not run_reference,
            primary_usage=_usage(primary_result) if primary_result is not None else None,
            reference_usage=_usage(reference_result) if reference_result is not None else None,
            primary_workflow_id=pid,
            reference_workflow_id=rid if run_reference else None,
            comparator=self._comparator,
        )
        self._store.save(record)
        return record
