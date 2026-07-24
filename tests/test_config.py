from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from structured_agents import AgentSpec, Choice, ConfigError, Grammar, Regex, Schema, Settings
from structured_agents import config as config_module
from structured_agents.config import constraint_from_config, register_constraint, spec_from_config


class Plan(BaseModel):
    title: str


def test_builtin_constraint_configs_round_trip() -> None:
    constraints = [Schema(Plan, strict=False), Regex(r"[a-z]+"), Choice("keep", "skip"), Grammar('root ::= "ok"')]

    for constraint in constraints:
        restored = constraint_from_config(constraint.to_config(), allow_modules=frozenset({__name__}))
        assert restored.to_config() == constraint.to_config()


def test_spec_config_round_trips_the_supported_agent_fields() -> None:
    spec = spec_from_config(
        {
            "name": "planner",
            "constraint": Schema(Plan, strict=False).to_config(),
            "instructions": "Make a plan.",
            "adapter": "planner-lora",
            "settings": {"temperature": 0.2, "max_tokens": 100, "extra_body": {"trace": True}},
        },
        allow_modules=frozenset({__name__}),
    )

    assert spec == AgentSpec(
        "planner",
        Schema(Plan, strict=False),
        "Make a plan.",
        adapter="planner-lora",
        settings=Settings(temperature=0.2, max_tokens=100, extra_body={"trace": True}),
    )


def test_schema_reference_outside_the_allowlist_is_rejected() -> None:
    with pytest.raises(ConfigError, match="not allowed"):
        constraint_from_config({"kind": "schema", "ref": "pathlib:Path"}, allow_modules=frozenset({__name__}))


def test_custom_constraint_registration() -> None:
    kind = "phase-six-test-choice"
    register_constraint(kind, lambda d: Choice(*d["options"]))

    constraint = constraint_from_config({"kind": kind, "options": ["left", "right"]}, allow_modules=frozenset())

    assert constraint.parse("left") == "left"


def test_entry_points_are_discovered_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class EntryPoint:
        name = "entry-point-choice"

        def load(self) -> Any:
            calls.append("load")
            return lambda d: Choice(*d["options"])

    def entry_points(*, group: str) -> list[EntryPoint]:
        calls.append(group)
        return [EntryPoint()]

    monkeypatch.setattr(config_module.importlib.metadata, "entry_points", entry_points)
    monkeypatch.setattr(config_module, "_entry_points_discovered", False)
    monkeypatch.setattr(config_module, "_constraint_factories", dict(config_module._constraint_factories))

    assert calls == []
    constraint = constraint_from_config(
        {"kind": "entry-point-choice", "options": ["queued"]}, allow_modules=frozenset()
    )

    assert constraint.parse("queued") == "queued"
    assert calls == ["structured_agents.constraints", "load"]


def test_only_config_imports_importlib_in_package_source() -> None:
    package = Path(__file__).parents[1] / "src" / "structured_agents"
    importlib_users: list[Path] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(name.name.startswith("importlib") for name in node.names):
                importlib_users.append(path)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("importlib"):
                importlib_users.append(path)

    assert importlib_users == [package / "config.py", package / "llama_core" / "diagnostics.py"]
