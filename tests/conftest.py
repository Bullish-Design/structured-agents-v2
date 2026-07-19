from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from dbos import DBOS, DBOSConfig

_directory = Path(tempfile.mkdtemp(prefix="structured-agents-tests-"))
DBOS(
    config=DBOSConfig(
        name="structured-agents-tests",
        system_database_url=f"sqlite:///{_directory / 'system.sqlite'}",
        use_listen_notify=False,
        scheduler_polling_interval_sec=0.05,
    )
)


@pytest.fixture(scope="session", autouse=True)
def dbos_lifecycle() -> Generator[None]:
    DBOS.launch()
    yield
    DBOS.destroy(destroy_registry=True)
