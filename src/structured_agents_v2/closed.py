"""A deliberately narrow, loopback-only structured-output client.

This module is intentionally separate from the legacy ``Backend`` API.  It
does not expose agents, SDK clients, profiles, tools, capture, or raw results.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

_MODEL_NAME = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_MAX_TIMEOUT_SECONDS = 600.0


def _validated_loopback_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    # Only literal loopback IPs are accepted; `localhost` is rejected deliberately so a
    # hostile DNS record cannot rebind the endpoint to an off-box address. Do not "fix"
    # this by adding "localhost" to the set.
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise ValueError("closed backend requires an http loopback endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("closed backend endpoint must not contain credentials, query, or fragment")
    return base_url.rstrip("/")


def _new_http_client(timeout: float) -> httpx.AsyncClient:
    """Create the only transport used by closed mode.

    Kept module-private so in-process ASGI tests can replace it without adding a
    transport or capture escape hatch to the public API.
    """
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=False, trust_env=False)


class ClosedBackend:
    """One fixed-model, one-request, JSON-schema-only structured call path."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        output_type: type[BaseModel],
        instructions: str,
    ) -> None:
        if not api_key:
            raise ValueError("closed backend API key must not be empty")
        if not _MODEL_NAME.fullmatch(model):
            raise ValueError("closed backend model is invalid")
        if not 0 < timeout <= _MAX_TIMEOUT_SECONDS:
            raise ValueError("closed backend timeout must be between 0 and 600 seconds")
        if not instructions or len(instructions.encode()) > 4096:
            raise ValueError("closed backend instructions must be nonempty and bounded")

        self._base_url = _validated_loopback_url(base_url)
        self._model = model
        self._api_key = api_key
        self._output_type = output_type
        self._instructions = instructions
        self._http_client = _new_http_client(timeout)

    async def run(self, prompt: str) -> BaseModel:
        """Return only validated structured output from one non-streaming call."""
        if not isinstance(prompt, str) or len(prompt.encode()) > 16_384:
            raise ValueError("closed backend prompt must be a bounded string")
        schema = self._output_type.model_json_schema()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._instructions},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "closed_output", "strict": True, "schema": schema},
            },
        }
        response = await self._http_client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            if response.status_code != 200:
                raise ClosedBackendError()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ClosedBackendError()
            return self._output_type.model_validate_json(content)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ClosedBackendError() from error

    async def aclose(self) -> None:
        """Release the owned HTTP client during application teardown."""
        await self._http_client.aclose()


class ClosedBackendError(Exception):
    """A deliberately detail-free transport or output-validation failure."""
