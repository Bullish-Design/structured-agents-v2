"""Optional fornix-backed durable effector."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

from dbos import DBOS
from pydantic import BaseModel

from ..authority import ProcessResult, validated_argv
from ..errors import BackendCapabilityError


class FornixEffector:
    """Run validated argv in a fornix check box without adding a Python dependency."""

    @DBOS.step()
    async def run(self, command: BaseModel) -> ProcessResult:
        fornix = shutil.which("fornix")
        if fornix is None:
            raise BackendCapabilityError("FornixEffector requires the 'fornix' executable.")
        completed = await asyncio.to_thread(
            subprocess.run,
            [fornix, "box", "--check", "--", *validated_argv(command)],
            capture_output=True,
            check=False,
            text=True,
        )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BackendCapabilityError("fornix did not return a JSON result") from exc
        if not isinstance(result, dict):
            raise BackendCapabilityError("fornix returned an invalid JSON result")
        return ProcessResult(
            int(result.get("returncode", completed.returncode)),
            str(result.get("stdout", completed.stdout)),
            str(result.get("stderr", completed.stderr)),
        )
