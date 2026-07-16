"""`DualPathRuntime` — the DBOS lifecycle owner (register → launch → shutdown).

DBOS requires every workflow to be registered *before* `DBOS.launch()`, and is a process-global
singleton. This runtime makes that the API shape: construct it (which constructs the one `DBOS(...)`),
`register()` each dual-path agent (wrapping both `StructuredAgent.agent`s as uniquely-named
`DBOSAgent`s — the registration), then `launch()` once. Each `register()` returns a `DualPathRunner`.
`shutdown()` (or the context manager's exit) destroys the singleton.

Only `json_schema` agents get a reference teacher (decision #4), so `register()` refuses any leg
whose resolved decode mode is not `json_schema`, and refuses any `register()` after `launch()`.

**Required shape** (register BEFORE the `with`): because `__enter__` launches, and `register()`
refuses to run after launch, you must register first and use the context manager only for the
launch/shutdown lifecycle::

    rt = DualPathRuntime(cfg)
    runner = rt.register("agent", primary=..., reference=...)   # all registration first
    with rt:                                                     # __enter__ launches
        result = await runner.run(prompt)                       # __exit__ shuts down

The tempting `with DualPathRuntime(cfg) as rt: rt.register(...)` does NOT work — `__enter__`
has already launched, so the `register()` raises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbos import DBOS, DBOSConfig
from pydantic import BaseModel
from pydantic_ai.durable_exec.dbos import DBOSAgent, StepConfig

from .errors import DualPathConfigError
from .runner import DualPathRunner
from .store import ComparisonStore

if TYPE_CHECKING:
    from ..agent import StructuredAgent
    from .comparator import Comparator


class DualPathConfig(BaseModel):
    """Configuration for a `DualPathRuntime` (DBOS app + the shared Postgres)."""

    app_name: str = "dual-path"
    pg_url: str  # DBOS system DB + ComparisonStore (same Postgres)
    run_admin_server: bool = False
    default_sample_rate: float = 1.0  # fraction of runs that also fire the reference leg


class DualPathRuntime:
    """Owns the process-global DBOS singleton and the dual-path runners registered against it."""

    def __init__(self, config: DualPathConfig) -> None:
        self.config = config
        self.store = ComparisonStore(config.pg_url)
        self._launched = False
        self._runners: dict[str, DualPathRunner] = {}
        DBOS(
            config=DBOSConfig(
                name=config.app_name,
                system_database_url=config.pg_url,
                run_admin_server=config.run_admin_server,
            )
        )

    def register(
        self,
        name: str,
        *,
        primary: StructuredAgent,
        reference: StructuredAgent,
        sample_rate: float | None = None,
        comparator: Comparator | None = None,
        model_step_config: StepConfig | None = None,
    ) -> DualPathRunner:
        """Wrap both legs as `DBOSAgent`s (before launch) and return their `DualPathRunner`."""
        if self._launched:
            raise DualPathConfigError(f"register({name!r}) called after launch(); register all agents before launch().")
        if name in self._runners:
            raise DualPathConfigError(f"duplicate dual-path runner name {name!r}.")
        self._require_json_schema(name, "primary", primary)
        self._require_json_schema(name, "reference", reference)

        step_config = model_step_config if model_step_config is not None else StepConfig()
        primary_dbos = DBOSAgent(primary.agent, name=f"{name}@primary", model_step_config=step_config)
        reference_dbos = DBOSAgent(reference.agent, name=f"{name}@reference", model_step_config=step_config)

        runner = DualPathRunner(
            name=name,
            primary_agent=primary,
            reference_agent=reference,
            primary_dbos=primary_dbos,
            reference_dbos=reference_dbos,
            store=self.store,
            sample_rate=sample_rate if sample_rate is not None else self.config.default_sample_rate,
            comparator=comparator,
        )
        self._runners[name] = runner
        return runner

    @staticmethod
    def _require_json_schema(name: str, leg: str, agent: StructuredAgent) -> None:
        _, spec = agent.profile.resolve()
        if spec.mode != "json_schema":
            raise DualPathConfigError(
                f"{name!r} {leg} agent has decode mode {spec.mode!r}; dual-path only teaches "
                "json_schema agents (decision #4)."
            )

    def launch(self) -> None:
        """Launch the DBOS singleton (after all `register()` calls) and ensure the store schema."""
        if not self._launched:
            DBOS.launch()
            self.store.init_schema()
            self._launched = True

    def shutdown(self) -> None:
        """Destroy the DBOS singleton (process-global; one per process)."""
        DBOS.destroy()
        self._launched = False

    def __enter__(self) -> DualPathRuntime:
        # Launches immediately — all register() calls must already be done (see class docstring).
        self.launch()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.shutdown()
        return False
