"""Gated tests for the dual-path spike (Architecture C).

These exercise the spike under `.scratch/projects/03-dual-path/spike/`. They are **skipped** in the
lean core suite because they need the optional `[dual-path]` extra (DBOS) and a running Postgres:

- `dbos` not installed            -> skipped via `importorskip`
- Postgres not on 127.0.0.1:5433  -> skipped via the socket probe
- live frontier leg               -> skipped unless `SAV_LIVE=1` + `REF_API_KEY`

Run: `devenv shell -- uv run --extra dev --extra dual-path pytest tests/test_dual_path_spike.py -q`
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest

SPIKE_DIR = Path(__file__).resolve().parents[1] / ".scratch" / "projects" / "03-dual-path" / "spike"


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pg_required = pytest.mark.skipif(
    not _port_open("127.0.0.1", 5433),
    reason="spike Postgres not running on 127.0.0.1:5433",
)


@pg_required
def test_dual_path_gates_1_to_3() -> None:
    """Gates 1-3 GPU-free: durable run, wire-shape survival, dual gather + ComparisonRecord."""
    pytest.importorskip("dbos")
    sys.path.insert(0, str(SPIKE_DIR))
    import run_spike

    assert asyncio.run(run_spike.main()) == 0


@pytest.mark.skipif(
    not (os.environ.get("SAV_LIVE") == "1" and os.environ.get("REF_API_KEY")),
    reason="live reference leg needs SAV_LIVE=1 and REF_API_KEY (REF_BASE_URL/REF_MODEL)",
)
def test_reference_leg_live() -> None:
    """Gate 4: a real frontier provider returns a schema-valid Command via the reference Backend.

    Records whether OpenAI-style strict `response_format: json_schema` round-trips (it 400s if our
    schema isn't strict-compliant) and that usage is populated. DBOS-free on purpose: this isolates
    the provider's structured-output support from the durability layer.
    """
    pytest.importorskip("dbos")
    sys.path.insert(0, str(SPIKE_DIR))
    import schemas

    from structured_agents_v2.backend import Backend, BackendCaps
    from structured_agents_v2.profile import AgentProfile

    backend = Backend(
        base_url=os.environ.get("REF_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["REF_API_KEY"],
        default_model=os.environ.get("REF_MODEL", "gpt-4o-mini"),
        caps=BackendCaps(xgrammar=False, lora=False),
        capture=True,
    )
    profile = AgentProfile(
        name="cmd",
        adapter=None,
        instructions="Emit one command object for the user's request.",
        output_type_ref="schemas:Command",
    )
    agent = backend.build(profile)
    result = agent.run_sync("Create a file notes.txt")
    assert isinstance(result.output, schemas.Command)
    assert result.usage.output_tokens is not None
