"""ConstrainedOutput: decode-spec derivation and mode-field validation."""

from __future__ import annotations

import sys
import types
from typing import Any, Literal

import pytest

from structured_agents_v2 import ConstrainedOutput
from structured_agents_v2.errors import ConstraintCompileError, ConstraintConfigError


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


# --- check_compilable against a fake xgrammar (no heavy optional dep needed) ------------


def _fake_xgrammar(record: dict[str, Any] | None = None, *, raises: bool = False) -> types.ModuleType:
    """A stand-in `xgrammar` module: records which Grammar.from_* is called, or raises."""
    mod = types.ModuleType("xgrammar")

    def _mk(kind: str):
        def _fn(arg: str) -> str:
            if raises:
                raise ValueError(f"un-compilable {kind}")
            if record is not None:
                record.setdefault("calls", []).append((kind, arg))
            return f"grammar:{kind}"

        return staticmethod(_fn)

    mod.Grammar = type(  # type: ignore[attr-defined]
        "Grammar",
        (),
        {"from_json_schema": _mk("json_schema"), "from_regex": _mk("regex"), "from_ebnf": _mk("grammar")},
    )
    return mod


def test_check_compilable_dispatches_per_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    rec: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "xgrammar", _fake_xgrammar(rec))

    class GrammarOut(ConstrainedOutput):
        __decode_mode__ = "grammar"
        __grammar__ = 'root ::= "x"'
        value: str

    FileEditPlan.check_compilable()  # json_schema -> from_json_schema
    GitCommandLine.check_compilable()  # regex      -> from_regex
    GrammarOut.check_compilable()  # grammar    -> from_ebnf
    Route.check_compilable()  # choice     -> nothing to compile

    assert [c[0] for c in rec["calls"]] == ["json_schema", "regex", "grammar"]


def test_check_compilable_raises_on_uncompilable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "xgrammar", _fake_xgrammar(raises=True))
    with pytest.raises(ConstraintCompileError, match="not XGrammar-compilable"):
        FileEditPlan.check_compilable()


def test_grammar_check_env_compiles_at_class_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    # SAV_GRAMMAR_CHECK=1 makes __init_subclass__ compile the constraint eagerly.
    monkeypatch.setenv("SAV_GRAMMAR_CHECK", "1")

    rec: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "xgrammar", _fake_xgrammar(rec))

    class CompilesOk(ConstrainedOutput):
        __decode_mode__ = "regex"
        __regex__ = "x.*"
        value: str

    assert ("regex", "x.*") in rec["calls"]  # compiled at definition

    monkeypatch.setitem(sys.modules, "xgrammar", _fake_xgrammar(raises=True))
    with pytest.raises(ConstraintCompileError):

        class Uncompilable(ConstrainedOutput):
            __decode_mode__ = "regex"
            __regex__ = "x.*"
            value: str
