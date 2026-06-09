"""ConstrainedOutput: decode-spec derivation and mode-field validation."""

from __future__ import annotations

from typing import Literal

import pytest

from structured_agents_v2 import ConstrainedOutput
from structured_agents_v2.errors import ConstraintConfigError


class FileEditPlan(ConstrainedOutput):
    action: Literal["edit_file", "refuse"]
    reason: str


class Route(ConstrainedOutput):
    __decode_mode__ = "choice"
    __choices__ = ["file_edit", "git_ops", "answer"]
    value: str


class GitCommandLine(ConstrainedOutput):
    __decode_mode__ = "regex"
    __regex__ = r"git (status|diff) [\w./-]*"
    value: str


def test_default_mode_is_json_schema() -> None:
    spec = FileEditPlan.decoder_spec()
    assert spec.mode == "json_schema"
    assert spec.strict is True


def test_choice_spec_carries_choices() -> None:
    spec = Route.decoder_spec()
    assert spec.mode == "choice"
    assert spec.choices == ["file_edit", "git_ops", "answer"]


def test_regex_spec_carries_regex() -> None:
    spec = GitCommandLine.decoder_spec()
    assert spec.mode == "regex"
    assert spec.regex == r"git (status|diff) [\w./-]*"


def test_missing_mode_field_raises_at_class_definition() -> None:
    with pytest.raises(ConstraintConfigError, match="requires __regex__"):

        class Bad(ConstrainedOutput):
            __decode_mode__ = "regex"  # no __regex__ supplied
            value: str


def test_choice_without_choices_raises() -> None:
    with pytest.raises(ConstraintConfigError, match="requires __choices__"):

        class Bad(ConstrainedOutput):
            __decode_mode__ = "choice"
            value: str


def test_check_compilable_is_noop_without_xgrammar() -> None:
    # In the default (no [grammar-check]) env, xgrammar is absent and the check no-ops.
    try:
        import xgrammar  # noqa: F401
    except ImportError:
        assert FileEditPlan.check_compilable() is None  # returns None, raises nothing
    else:  # pragma: no cover - only when the optional extra is installed
        FileEditPlan.check_compilable()
