"""Live json_schema round-trip against $LLM_BASE_URL (reproduces the spike).

Skipped unless SAV_LIVE=1, so the default suite stays GPU-free and offline. Run with:

    SAV_LIVE=1 devenv shell -- uv run --extra dev pytest -q -m live
"""

from __future__ import annotations

import os
from typing import Literal

import pytest

from structured_agents_v2 import AgentProfile, Backend, ConstrainedOutput

pytestmark = pytest.mark.live

_LIVE = os.environ.get("SAV_LIVE") == "1"


class FileEditPlan(ConstrainedOutput):
    action: Literal["edit_file", "refuse", "needs_clarification"]
    reason: str


@pytest.mark.skipif(not _LIVE, reason="set SAV_LIVE=1 to run against the live server")
def test_live_json_schema_round_trip() -> None:
    backend = Backend(
        base_url=os.environ.get("LLM_BASE_URL", "http://remora-server:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "sk-no-key-required"),
        default_model=os.environ.get("LLM_MODEL", "Qwen3.5-9B-UD-Q6_K_XL.gguf"),
        capture=True,
    )
    profile = AgentProfile(
        name="file_edit",
        instructions="You produce file-edit plans only. Be terse.",
        output_type_ref="test_live:FileEditPlan",
    )
    result = backend.build(profile).run_sync("Change the retry timeout in config.py from 5s to 10s.")

    assert isinstance(result.output, FileEditPlan)
    # response_format (not the tool path) carried the schema
    assert result.request_body is not None
    assert result.request_body["response_format"]["type"] == "json_schema"
