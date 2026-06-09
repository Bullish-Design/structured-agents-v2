"""Live round-trips against $LLM_BASE_URL (reproduce + extend the spike).

Both tests are `live`-marked and skipped by default so the normal suite stays GPU-free:

- json_schema (SAV_LIVE=1): works against any OpenAI-compatible server, incl. the
  current llama.cpp box — reproduces the request-path spike.
- XGrammar regex (SAV_LIVE_XGRAMMAR=1): exercises a bare-string constraint via
  `structured_outputs`, and asserts the output actually matches the regex — so it only
  passes once a real XGrammar backend (the deploy/vllm container) is serving.

Run:
    SAV_LIVE=1          devenv shell -- uv run --extra dev pytest -q -m live
    SAV_LIVE_XGRAMMAR=1 devenv shell -- uv run --extra dev pytest -q -m live
"""

from __future__ import annotations

import os
import re
from typing import Literal

import pytest

from structured_agents_v2 import AgentProfile, Backend, ConstrainedOutput

pytestmark = pytest.mark.live

_LIVE = os.environ.get("SAV_LIVE") == "1"
_LIVE_XGRAMMAR = os.environ.get("SAV_LIVE_XGRAMMAR") == "1"

_GIT_CMD_RE = r"git (status|diff|add|commit) [\w./\- ]*"


def _backend() -> Backend:
    """A Backend pointed at $LLM_BASE_URL (the deploy/vllm service, when it's up)."""
    return Backend(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "sk-no-key-required"),
        default_model=os.environ.get("LLM_MODEL", "base"),
        capture=True,
    )


class FileEditPlan(ConstrainedOutput):
    action: Literal["edit_file", "refuse", "needs_clarification"]
    reason: str


class GitCommandLine(ConstrainedOutput):
    __decode_mode__ = "regex"
    __regex__ = _GIT_CMD_RE
    value: str


@pytest.mark.skipif(not _LIVE, reason="set SAV_LIVE=1 to run against the live server")
def test_live_json_schema_round_trip() -> None:
    profile = AgentProfile(
        name="file_edit",
        instructions="You produce file-edit plans only. Be terse.",
        output_type_ref="test_live:FileEditPlan",
    )
    result = _backend().build(profile).run_sync("Change the retry timeout in config.py from 5s to 10s.")

    assert isinstance(result.output, FileEditPlan)
    # response_format (not the tool path) carried the schema
    assert result.request_body is not None
    assert result.request_body["response_format"]["type"] == "json_schema"


@pytest.mark.skipif(not _LIVE_XGRAMMAR, reason="set SAV_LIVE_XGRAMMAR=1 to run against a vLLM/XGrammar server")
def test_live_xgrammar_regex_round_trip() -> None:
    profile = AgentProfile(
        name="git",
        instructions="Emit exactly one git command line for the request.",
        output_type_ref="test_live:GitCommandLine",
    )
    result = _backend().build(profile).run_sync("Show the repository status.")

    # text mode → plain string, sent via structured_outputs (not response_format)
    assert isinstance(result.output, str)
    assert result.request_body is not None
    assert result.request_body["structured_outputs"] == {"regex": _GIT_CMD_RE}
    # the payoff: XGrammar must have constrained decoding to the regex
    assert re.fullmatch(_GIT_CMD_RE, result.output) is not None
