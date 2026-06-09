"""Spike output types — json_schema command models (the modes that get a teacher)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Command(BaseModel):
    """A tiny command object — exercises the NativeOutput / response_format json_schema path."""

    action: Literal["create", "delete", "noop"]
    target: str
    reason: str
