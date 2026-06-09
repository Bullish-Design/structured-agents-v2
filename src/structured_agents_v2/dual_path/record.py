"""`ComparisonRecord` — the versioned, serializable unit of dual-path training/eval data.

DBOS-free: pure Pydantic + hashing. A record pairs a primary (local vLLM) output with a reference
(frontier) output for one prompt, carrying enough identity to be **reproducible** (content hashes of
the profile and the resolved JSON schema, structured model identities) and enough signal to be
**useful** (validity flags, an agreement signal, usage). It is persisted as `jsonb` by
`ComparisonStore` and read back for SFT export and evals.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

from ..profile import AgentProfile
from .comparator import Comparator, ComparisonSignal, ExactFieldComparator

DecodeMode = Literal["json_schema"]  # dual-path only teaches json_schema agents


def content_hash(obj: Any) -> str:
    """Stable sha256 of a JSON-serializable object (sorted keys; `str` fallback)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def profile_version(profile: AgentProfile) -> str:
    """Content hash of the serialized profile (instructions + ref + decoder + settings)."""
    return content_hash(profile.model_dump(mode="json"))


def schema_version(output_type: type[BaseModel]) -> str:
    """Content hash of the resolved JSON schema, so the record is reproducible."""
    return content_hash(output_type.model_json_schema())


def lib_version() -> str:
    """Installed structured-agents-v2 version, or 'unknown'."""
    try:
        return importlib.metadata.version("structured-agents-v2")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


class ModelIdentity(BaseModel):
    """Identifies which model produced an output, for reproducibility and per-model evals."""

    kind: Literal["vllm", "frontier"]
    wire_model: str  # the OpenAI `model` field actually sent
    base: str | None = None  # base checkpoint (vllm)
    adapter: str | None = None  # LoRA name (vllm)
    adapter_rev: str | None = None  # needs an adapter registry; best-effort for now
    vllm_tag: str | None = None  # container/image tag
    provider: str | None = None  # frontier provider id
    model_id: str | None = None  # frontier model id


class ComparisonRecord(BaseModel):
    """One primary-vs-reference comparison: identity + payload + signals."""

    # identity / versioning
    run_id: str
    profile_version: str
    schema_version: str
    primary_model: ModelIdentity
    reference_model: ModelIdentity | None = None
    decode_mode: DecodeMode = "json_schema"
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
    reference_skipped: bool = False
    signal: ComparisonSignal | None = None
    primary_usage: dict[str, Any] | None = None
    reference_usage: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None  # derived from usage × price table (later phase)


def build_comparison_record(
    *,
    run_id: str,
    prompt: str,
    profile: AgentProfile,
    output_type: type[BaseModel],
    primary_model: ModelIdentity,
    reference_model: ModelIdentity | None,
    primary_output: BaseModel | None,
    reference_output: BaseModel | None,
    primary_error: str | None = None,
    reference_error: str | None = None,
    reference_skipped: bool = False,
    primary_usage: dict[str, Any] | None = None,
    reference_usage: dict[str, Any] | None = None,
    primary_workflow_id: str | None = None,
    reference_workflow_id: str | None = None,
    comparator: Comparator | None = None,
) -> ComparisonRecord:
    """Assemble (validate, diff, version) a `ComparisonRecord` from two outputs.

    `output_type` is the shared Pydantic type both legs were validated against. The agreement
    `signal` is computed only when both outputs validated; otherwise it is `None`.
    """
    cmp = comparator or ExactFieldComparator()
    primary_valid = isinstance(primary_output, BaseModel)
    reference_valid = isinstance(reference_output, BaseModel)
    signal = (
        cmp.compare(primary_output, reference_output)
        if (primary_valid and reference_valid)
        else None
    )
    return ComparisonRecord(
        run_id=run_id,
        profile_version=profile_version(profile),
        schema_version=schema_version(output_type),
        primary_model=primary_model,
        reference_model=reference_model,
        lib_version=lib_version(),
        primary_workflow_id=primary_workflow_id,
        reference_workflow_id=reference_workflow_id,
        created_at=datetime.now(UTC).isoformat(),
        prompt=prompt,
        instructions=profile.instructions,
        primary_output=primary_output.model_dump() if primary_valid else None,
        reference_output=reference_output.model_dump() if reference_valid else None,
        primary_error=primary_error,
        reference_error=reference_error,
        primary_valid=primary_valid,
        reference_valid=reference_valid,
        reference_skipped=reference_skipped,
        signal=signal,
        primary_usage=primary_usage,
        reference_usage=reference_usage,
    )
