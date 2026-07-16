"""`AgentProfile.output_type_ref` resolution and decoder derivation."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from structured_agents_v2 import AgentProfile, ConfigError, ConstrainedOutput, DecoderSpec


class PlainCmd(BaseModel):
    action: Literal["go", "stop"]


class ConstrainedCmd(ConstrainedOutput):
    value: str


class RegexCmd(ConstrainedOutput):
    __decode_mode__ = "regex"
    __regex__ = r"git (status|diff) [\w./-]*"
    value: str


def test_resolves_dotted_ref() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile:PlainCmd")
    assert profile.resolve_output_type() is PlainCmd


def test_missing_module_errors_clearly() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="no.such.module:Thing")
    with pytest.raises(ConfigError, match="could not import module"):
        profile.resolve_output_type()


def test_missing_attr_errors_clearly() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile:DoesNotExist")
    with pytest.raises(ConfigError, match="no attribute"):
        profile.resolve_output_type()


def test_malformed_ref_errors_clearly() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile.PlainCmd")
    with pytest.raises(ConfigError, match="module:ClassName"):
        profile.resolve_output_type()


def test_non_model_ref_errors() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile:test_resolves_dotted_ref")
    with pytest.raises(ConfigError, match="Pydantic model"):
        profile.resolve_output_type()


def test_constrained_output_derives_its_own_decoder() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile:ConstrainedCmd")
    output_type, spec = profile.resolve()
    assert output_type is ConstrainedCmd
    assert spec.mode == "json_schema"


def test_constrained_regex_mode_carries_pattern() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile:RegexCmd")
    _, spec = profile.resolve()
    assert spec.mode == "regex"
    assert spec.regex == r"git (status|diff) [\w./-]*"


def test_plain_model_defaults_to_json_schema() -> None:
    profile = AgentProfile(name="p", instructions="x", output_type_ref="test_profile:PlainCmd")
    output_type, spec = profile.resolve()
    assert output_type is PlainCmd
    assert spec.mode == "json_schema"


def test_explicit_decoder_for_bare_string_without_output_type() -> None:
    profile = AgentProfile(name="p", instructions="x", decoder=DecoderSpec(mode="regex", regex="x.*"))
    output_type, spec = profile.resolve()
    assert output_type is None
    assert spec.mode == "regex"


def test_no_ref_and_no_decoder_errors() -> None:
    profile = AgentProfile(name="p", instructions="x")
    with pytest.raises(ConfigError, match="no output_type_ref and no decoder"):
        profile.resolve()


def test_constrained_output_plus_explicit_decoder_conflicts() -> None:
    # Phase 5 item 3: a ConstrainedOutput carries its own decoder_spec; also passing an
    # explicit decoder is two sources of truth and must raise, not silently drop one.
    profile = AgentProfile(
        name="p",
        instructions="x",
        output_type_ref="test_profile:ConstrainedCmd",
        decoder=DecoderSpec(mode="regex", regex="x.*"),
    )
    with pytest.raises(ConfigError, match="conflict"):
        profile.resolve()
