"""The allowlist-gated boundary from serialized data to typed agent primitives."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from .agent import AgentSpec, Settings
from .constraint import Choice, Constraint, Grammar, Regex, Schema
from .errors import ConfigError

type ConstraintFactory = Callable[[dict[str, Any]], Constraint[Any]]


def _require_string(d: dict[str, Any], field: str, *, kind: str) -> str:
    value = d.get(field)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Constraint kind {kind!r} requires non-empty string field {field!r}.")
    return value


def _schema_from_config(d: dict[str, Any]) -> Constraint[Any]:
    ref = _require_string(d, "ref", kind="schema")
    strict = d.get("strict", True)
    if not isinstance(strict, bool):
        raise ConfigError("Schema constraint field 'strict' must be a boolean.")
    model = _resolve_schema(ref, allow_modules=_active_allow_modules())
    return Schema(model, strict=strict)


def _regex_from_config(d: dict[str, Any]) -> Constraint[Any]:
    return Regex(_require_string(d, "pattern", kind="regex"))


def _choice_from_config(d: dict[str, Any]) -> Constraint[Any]:
    options = d.get("options")
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ConfigError("Choice constraint field 'options' must be a list of strings.")
    return Choice(*options)


def _grammar_from_config(d: dict[str, Any]) -> Constraint[Any]:
    return Grammar(_require_string(d, "ebnf", kind="grammar"))


_constraint_factories: dict[str, ConstraintFactory] = {
    "schema": _schema_from_config,
    "regex": _regex_from_config,
    "choice": _choice_from_config,
    "grammar": _grammar_from_config,
}
_entry_points_discovered = False
_allow_modules_stack: list[frozenset[str]] = []


def register_constraint(kind: str, from_config: ConstraintFactory) -> None:
    """Register a config-only constraint factory under ``kind``."""
    if not kind:
        raise ConfigError("Constraint kind must be non-empty.")
    if not callable(from_config):
        raise ConfigError(f"Constraint factory for {kind!r} must be callable.")
    _constraint_factories[kind] = from_config


def _discover_constraint_entry_points() -> None:
    global _entry_points_discovered
    if _entry_points_discovered:
        return
    _entry_points_discovered = True
    try:
        entry_points = importlib.metadata.entry_points(group="structured_agents.constraints")
        for entry_point in entry_points:
            factory = entry_point.load()
            if not callable(factory):
                raise ConfigError(f"Constraint entry point {entry_point.name!r} must load a callable factory.")
            _constraint_factories.setdefault(entry_point.name, cast(ConstraintFactory, factory))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Could not discover constraint entry points: {exc}") from exc


def _active_allow_modules() -> frozenset[str]:
    if not _allow_modules_stack:  # pragma: no cover - internal factories are called through the public function
        raise RuntimeError("Constraint config factories must run through constraint_from_config().")
    return _allow_modules_stack[-1]


def _resolve_schema(ref: str, *, allow_modules: frozenset[str]) -> type[BaseModel]:
    module_name, separator, attribute_path = ref.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ConfigError(f"Schema ref must be 'module:Model', got {ref!r}.")
    if not any(module_name == allowed or module_name.startswith(f"{allowed}.") for allowed in allow_modules):
        raise ConfigError(f"Schema reference {ref!r} is not allowed by allow_modules.")
    try:
        value: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"Could not import allowed schema module {module_name!r}: {exc}") from exc
    try:
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise ConfigError(f"Schema reference {ref!r} does not exist.") from exc
    if not isinstance(value, type) or not issubclass(value, BaseModel):
        raise ConfigError(f"Schema reference {ref!r} must resolve to a Pydantic BaseModel class.")
    return value


def constraint_from_config(d: dict[str, Any], *, allow_modules: frozenset[str]) -> Constraint[Any]:
    """Build a constraint from trusted shape data under an explicit module allowlist."""
    kind = d.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ConfigError("Constraint config requires a non-empty string 'kind'.")
    _discover_constraint_entry_points()
    try:
        factory = _constraint_factories[kind]
    except KeyError as exc:
        raise ConfigError(f"Unknown constraint kind {kind!r}.") from exc
    _allow_modules_stack.append(allow_modules)
    try:
        return factory(d)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Invalid configuration for constraint kind {kind!r}: {exc}") from exc
    finally:
        _allow_modules_stack.pop()


def spec_from_config(d: dict[str, Any], *, allow_modules: frozenset[str]) -> AgentSpec[Any]:
    """Build an ``AgentSpec`` from serialized data without adding a configuration framework."""
    name = d.get("name")
    instructions = d.get("instructions")
    if not isinstance(name, str) or not name:
        raise ConfigError("Agent spec requires a non-empty string 'name'.")
    if not isinstance(instructions, str):
        raise ConfigError("Agent spec requires string 'instructions'.")
    constraint_config = d.get("constraint")
    if not isinstance(constraint_config, dict):
        raise ConfigError("Agent spec requires a constraint configuration mapping.")
    adapter = d.get("adapter")
    if adapter is not None and not isinstance(adapter, str):
        raise ConfigError("Agent spec field 'adapter' must be a string or null.")
    settings_config = d.get("settings", {})
    if not isinstance(settings_config, dict):
        raise ConfigError("Agent spec field 'settings' must be a mapping.")
    try:
        settings = Settings(**settings_config)
    except TypeError as exc:
        raise ConfigError(f"Invalid agent settings: {exc}") from exc
    return AgentSpec(name, constraint_from_config(constraint_config, allow_modules=allow_modules), instructions,
                     adapter=adapter, settings=settings)
