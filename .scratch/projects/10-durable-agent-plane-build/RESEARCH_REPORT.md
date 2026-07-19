# Phase 3 DBOS test-lifecycle note — 2026-07-18

## Evidence

- Environment: DBOS 2.23.0, Python 3.13.13, SQLite system database, pytest-asyncio 1.4.0.
- First focused authority run (before the lifecycle adjustment):
  `devenv shell -- pytest tests/test_authority.py tests/test_fornix.py`
  passed the first async tests, then failed later workflow invocations with
  `RuntimeError: cannot schedule new futures after shutdown` from
  `concurrent.futures.thread.ThreadPoolExecutor.submit`.
- The same B5 test alone passed and surfaced its intended `RuntimeError("effect failed")`.
  This ruled out an effect-retry or `SetWorkflowID` semantic failure.
- Cause: DBOS is process-global and launched once in the session fixture, while pytest-asyncio
  had been creating a fresh event loop per async test. Its worker executor was associated with a
  loop that pytest subsequently closed.

## Minimal fix

Set `asyncio_default_test_loop_scope = "session"` in `pyproject.toml`, matching the existing
session-scoped DBOS singleton lifecycle. No DBOS, pydantic-ai, model, or test behavior was changed.

## Reverification

- `devenv shell -- pytest` → 12 passed
- `devenv shell -- ty check src tests` → all checks passed
- `devenv shell -- ruff check src tests` → all checks passed

## Scope

This proves the SQLite Phase 3 authorization/effect boundary and keyed replay behavior. It does not
prove crash recovery across a separate process; that remains the DBOS contract exercised in later
durability work.

---

# Phase 4 approval messaging evidence — 2026-07-18

## Evidence

- Environment: DBOS 2.23.0, Python 3.13.13, pytest-asyncio 1.4.0, and the
  session-global SQLite system database from `tests/conftest.py`.
- Installed DBOS signatures confirm the async APIs used by this phase:
  `set_event_async(key, value)`, `recv_async(topic, timeout_seconds)`,
  `send_async(destination_id, message, topic)`, and
  `list_workflows_async(status="PENDING")`. `get_all_events_async(workflow_id)`
  exposes an existing workflow's events without waiting for a missing key.
  [DBOS workflow messaging documentation](https://docs.dbos.dev/python/tutorials/workflow-messaging)
  describes `send`/`recv` as durable workflow messaging.
- Focused runtime command: `devenv shell -- pytest tests/test_approval.py -q`.
  Result: 4 passed. A workflow blocked in `recv_async` was observed as `PENDING`;
  its raw `pending_command` event was retrieved; an external async send resumed
  it; an explicit denial returned `Denied` data and made zero effect calls; and
  a receive timeout returned `Decision(False, "timeout")`.

## Minimal implementation

- `Approval.request` publishes the raw command as `pending_command` and the
  recipient as `pending_to`, then receives only on its configured topic.
- `ApprovalClient.pending` filters `PENDING` workflows by these two published
  events, preventing unrelated pending DBOS workflows from being reported as
  approvals.
- The public timeout remains optional. When it is absent the implementation
  calls `recv_async` without `timeout_seconds`, preserving DBOS's own default;
  DBOS's typed API does not accept `None` for that parameter.

## Scope

This proves the in-process SQLite durable-message transition from PENDING to
SUCCESS and the data-denial composition boundary. It does not independently
prove recovery after a separate-process crash/restart; that remains the DBOS
durability contract and Phase 7's recovery verification scope.

---

# Phase 5 plane-service evidence — 2026-07-18

## Installed DBOS 2.23 contract

- `Queue(name, concurrency=None, limiter=None, ...)` takes its rate limit as
  a `QueueRateLimit` mapping with `{"limit": int, "period": float}`; durable
  submission is `await queue.enqueue_async(workflow, *args)`. The public
  `Queue` wrapper translates its `(limit, period_seconds)` API to that mapping.
- `@DBOS.scheduled(cron)` is the 2.23 scheduling decorator. Its scheduler
  invokes the workflow with two UTC `datetime` arguments: the scheduled time
  and the actual dispatch time. The installed implementation parses cron with
  `second_at_beginning=True`, so the SQLite test uses `*/2 * * * * *` and
  `scheduler_polling_interval_sec=0.05` rather than sleeping through a minute.
- Async observability APIs are `list_workflows_async`,
  `get_workflow_status_async`, `fork_workflow_async(workflow_id, start_step)`,
  and `cancel_workflow_async`. There is no `get_workflow_handle_async` in this
  release.

## Focused runtime evidence

- `devenv shell -- pytest tests/test_plane.py::test_queue_caps_concurrency_and_isolates_batch_failures tests/test_plane.py::test_compare_is_keyed_and_durable -q`
  passed. A DBOS queue with concurrency 2 kept peak concurrent runs at or below
  2; a failed item surfaced only through its own handle while its three siblings
  completed. A pair of real `Agent` legs used two keyed durable runs; repeating
  the same comparison key made exactly two model requests total, not four.
- `devenv shell -- pytest tests/test_plane.py::test_scheduled_workflow_fires -vv`
  passed in 4.15s. The scheduled workflow fired and appeared as SUCCESS in the
  SQLite workflow store.
- The observability focused test passed: a blocked `Approval` was listed as
  PENDING, status lookup returned PENDING, a workflow fork from step 0 replayed
  its meaningful step, and a blocked receive transitioned to CANCELLED.

## First failures and minimal corrections

- The first collection attempt assumed a private `dbos._workflow` module for
  handle types. DBOS 2.23 exports `WorkflowHandleAsync` and `WorkflowStatus`
  from `dbos`; the wrapper now imports those public names.
- The first scheduler run failed with
  `TypeError: scheduled_workflow() takes 0 positional arguments but 2 were given`.
  Installed source confirms the two-datetime callback contract, so the test
  now accepts those arguments. No DBOS lifecycle, model, or durable semantics
  were changed to hide the failure.

## Scope

This proves the Phase 5 SQLite operational surface in-process: queue limits and
failure isolation, cron dispatch, workflow-store observability, fork/cancel,
and keyed dual-agent idempotency. It does not independently prove recovery
across a separate process, which remains Phase 7 scope.

---

# Phase 6 config-edge evidence — 2026-07-18

## Implementation and safety boundary

- `config.py` is the sole module that imports `importlib`. It resolves a
  `schema` reference only after its module name exactly matches, or is a child
  of, an explicit `allow_modules` entry. A disallowed reference raises
  `ConfigError` before any import is attempted.
- Built-in factories cover the canonical `schema`, `regex`, `choice`, and
  `grammar` forms. `spec_from_config` reconstructs only the existing
  `AgentSpec` fields (`name`, `constraint`, `instructions`, `adapter`, and
  `Settings`), preserving the code-first toolkit rather than adding a config
  framework.
- Entry points are discovered on the first `constraint_from_config` call via
  the `structured_agents.constraints` group. An entry point's name is its
  constraint kind and its loaded callable is the factory; explicitly
  registered factories take precedence.

## Focused evidence

- `devenv shell -- pytest tests/test_config.py` → `6 passed in 2.21s`.
  The tests prove built-in constraint serde round-trips, complete `AgentSpec`
  reconstruction, pre-import allowlist rejection, custom factory registration,
  lazy entry-point discovery, and the source-level importlib boundary.
- Initial `devenv shell -- pytest` baseline before edits → `20 passed in
  10.22s` on the session SQLite DBOS harness.

## Final verification

- `devenv shell -- pytest` → `26 passed in 10.22s`.
- `devenv shell -- ty check src tests` → `All checks passed!`.
- `devenv shell -- ruff check src tests` → `All checks passed!`.

## Scope

This proves the Phase 6 serialized-data boundary and its in-process plugin
discovery behavior. It does not validate third-party packaged entry points in
a separately installed distribution; the mocked entry-point test verifies the
standard-library discovery and load contract without adding a package fixture.

---

# Phase 7 live-cutover attempt — 2026-07-19

## Implementation prepared for cutover

- Added tests/test_live.py, a module-level SAV_LIVE=1 gate. Without that exact
  opt-in it skips before constructing a client or contacting tower, so ordinary
  pytest remains SQLite-only.
- The live checks require /health and /v1/models success with the configured
  model id, then run the real durable Backend for JSON-schema, regex, choice,
  and grammar constraints. Each assertion is deterministic: a typed
  LiveCommand, phase7-2048, phase7-allow, and phase7-grammar, respectively.
- The durable pipeline test generates a typed LiveCommand, authorizes only
  ("echo", "phase7-live"), and executes a DBOS @step through
  execute(..., key="phase7-live-effect") twice. It requires both returned
  values to be 1 and the effect counter to remain 1.
- Adapted deploy/vllm/verify.sh without changing library API: beyond its
  established health, model, schema, and regex checks it now makes exact
  XGrammar choice and grammar assertions. If LORA_NAME is configured, its
  adapter request is also choice-constrained and must return phase7-lora,
  rather than merely HTTP 200.

## Offline revalidation

- devenv shell -- pytest → 26 passed, 1 skipped in 10.21s; the one skip is the
  opt-in live module.
- devenv shell -- ty check src tests → All checks passed!.
- devenv shell -- ruff check src tests → All checks passed!.
- devenv shell -- bash -n deploy/vllm/verify.sh passed. The grammar request was
  also parsed offline as JSON and yielded root ::= "phase7-grammar".

## Live evidence and blocker

The requested fixed target was http://tower:8000/v1 with LLM_MODEL=qwen3-4b
and no API key present in this environment. Every attempt received a fresh UTC
artifact directory with command, git state, environment, versions,
stdout/stderr, and before/after state:

- artifacts/20260719T022946Z-phase7-live-probe/: raw curl probes of /health
  and /v1/models both returned HTTP 000, curl exit 6, and Could not resolve
  host: tower.
- artifacts/20260719T023314Z-phase7-live-pytest/: the explicit SAV_LIVE=1
  pytest -x tests/test_live.py attempt failed its first health prerequisite
  with httpx.ConnectError: [Errno -2] Name or service not known. DBOS launched
  and shut down its SQLite test database cleanly; no model request or durable
  effect was reached.
- artifacts/20260719T023345Z-phase7-verify-script/: the adapted verifier
  failed at its first /health check (000) and correctly skipped all dependent
  checks. Its dns-diagnostics.txt shows no IPv4 result for tower;
  /etc/resolv.conf points at the configured Tailscale resolvers, while
  resolvectl is unavailable on this host.

Classification: external DNS/network availability failure before HTTP service
reachability, model selection, authentication, or vLLM wire handling. No
library compatibility defect is evidenced, and no substitute endpoint or
weakened assertion was used. Therefore Phase 7 is not claimed complete and no
completion commit should be made until tower resolves and the same SAV_LIVE=1
checks can run through the pipeline.

---

# Phase 7 local-server correction and live result — 2026-07-19

The operator clarified that server is this GPU host. The native deployment
documentation establishes the actual local verification endpoint as
http://127.0.0.1:8000/v1, not http://server:8000/v1: the service deliberately
binds only 127.0.0.1. The live-test defaults now follow that documented endpoint
and its served model name, base.

## Confirmed local runtime state

- artifacts/20260719T024324Z-phase7-localhost-probe/: /health and /v1/models
  returned 200. The listener is vllm on 127.0.0.1:8000, and the model listing
  contains only base.
- The process is the enabled system service structured-agents-vllm, running
  vLLM 0.25.0 with the Gemma-4 GGUF profile and
  --structured-outputs-config.backend xgrammar. Its deployment profile has no
  LoRA adapters, so the optional LoRA assertion is correctly skipped.
- artifacts/20260719T024551Z-phase7-local-plain-chat/: an unconstrained
  /v1/chat/completions request returned 200 and exactly
  "phase7 plain chat works.". Disk capacity was 349 GiB free. This rules out
  general connectivity, serving, model loading, and disk exhaustion as the
  current request-path failure.

## Constrained-output blocker

- artifacts/20260719T024405Z-phase7-local-verify-script/: the adapted verifier
  passed health and models, then JSON Schema, XGrammar regex, choice, and
  grammar each returned HTTP 500.
- artifacts/20260719T024451Z-phase7-local-constraint-500/: verbatim direct
  requests for each of those four supported output forms all returned the same
  body: {"error":{"message":"","type":"InternalServerError","param":null,"code":500}}.
- artifacts/20260719T024708Z-phase7-local-live-pytest/: the actual
  SAV_LIVE=1 durable library suite passed health/model identity, then
  Schema(LiveCommand) failed in the pydantic-ai DBOS durable request step with
  openai.InternalServerError. The client retried the same request three times;
  vLLM logged all three as 500. The pipeline cannot begin until generation
  works.

The service journal records requests but no exception detail beyond the 500.
Because plain chat works while all constrained wire forms fail before any
library parsing or authorization/effect code runs, this is an external vLLM
structured-output runtime incompatibility or deployment defect. Changing the
active system service, model, vLLM version, or XGrammar package would be a
material external-state change and was not performed. Phase 7 remains blocked;
no completion commit was made.

---

# Phase 7 structured-output diagnosis continuation — 2026-07-19

## Fresh live probe

- Fresh artifact:
  `artifacts/20260719T031426Z-phase7-xgrammar-choice-diagnosis/`. It contains
  the exact request, raw response headers/body, curl status, service state,
  journal slices, and git state before and after the attempt.
- The smallest deterministic XGrammar request,
  `structured_outputs.choice=["phase7-choice"]`, again returned HTTP 500 with
  the same empty `InternalServerError` body. The service remained active and
  its journal recorded only the HTTP 500; it emitted no Python traceback or
  compiler diagnostic for the request.
- This confirms the pre-generation structured-output failure boundary with a
  single supported constraint form. It does not test or alter a library
  fallback, because accepting unconstrained output would violate Phase 7's
  server-side-enforcement requirement.

## Local deployment mechanism and provenance

- The enabled `structured-agents-vllm.service` starts
  `deploy/vllm/native/run.sh`; its live process arguments pin vLLM 0.25.0,
  `--quantization gguf`, the Gemma-4 12B GGUF model, and
  `--structured-outputs-config.backend xgrammar`. The native lock resolves
  XGrammar 0.2.3. Thus every failing request shares the XGrammar compiler/
  bitmask path; plain chat does not enter that path.
- The active API and engine processes still refer to
  `deploy/vllm/native/.venv/...`, including the XGrammar shared library, but
  that environment no longer exists on disk. The process has retained mapped
  libraries, but its exact installed Python source cannot now be inspected or
  relaunched reproducibly. This is concrete deployment drift, although it is
  not alone proof that the missing environment caused the current 500.
- Current upstream material documents Gemma 4 structured outputs as a vLLM
  guided-decoding capability and demonstrates `response_format=json_schema`
  with a Gemma 4 server. [Gemma 4 recipe](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md)
  The same recipe documents a supported quantized serving path, but not this
  out-of-tree GGUF-plugin combination. The vLLM project also describes
  XGrammar/guidance structured generation as a supported engine feature.
  [vLLM project](https://github.com/vllm-project/vllm)

## Diagnosis and required operator action

**Classification:** external runtime/deployment failure in the local vLLM
structured-output path, after HTTP request acceptance and before constrained
generation. It is not a DNS, listener, generic chat, pydantic-ai request-wire,
DBOS, authorization, or exactly-once-effect defect.

No narrow repository-side compatibility patch is evidenced. The normal client
wire matches the pinned pydantic-ai capture provenance, and the same raw vLLM
wire fails without this library.

The next justified action requires explicit operator authorization because it
changes runtime state:

1. Recreate `deploy/vllm/native/.venv` from its locked native environment
   (`uv sync --locked` in the native deployment's `devenv` shell), then restart
   `structured-agents-vllm.service` from that reconciled environment.
2. Run the preserved deterministic choice probe first, with request-time
   exception logging enabled if the failure persists, so the XGrammar/vLLM
   traceback is captured rather than collapsed to the blank API 500.
3. Only after that passes, run JSON Schema, regex, grammar, the gated durable
   pipeline, and applicable LoRA checks in order with fresh artifacts.

This recommendation does not authorize those actions and no service restart,
model change, package upgrade, or configuration change was performed here.

## Offline revalidation

- `devenv shell -- pytest` → `26 passed, 1 skipped`.
- `devenv shell -- ty check src tests` → clean.
- `devenv shell -- ruff check src tests` → clean.
- `devenv shell -- bash -n deploy/vllm/verify.sh` → clean.

---

# Phase 7 native-environment reconciliation — 2026-07-19

## Completed safely

- The previous live service was found to reference a missing/invalid native
  `.venv`. The directory was validated to contain no Python executable before
  it was removed.
- A fresh native environment was then recreated with the deployment's own
  `deploy/vllm/native/devenv.nix` and exact lock:
  `cd deploy/vllm/native && NIXPKGS_ALLOW_UNFREE=1 devenv shell --impure -- uv sync --locked`.
  The successful lock resolution installed `vllm==0.25.0`, `xgrammar==0.2.3`,
  and the pinned local `vllm-gguf-plugin` 0.0.4.
- Evidence: `artifacts/20260719T032113Z-phase7-native-env-recreate/` records
  the previously invalid environment, removal status, complete sync output,
  resolved package versions, and before/after service state.

## Current blocker

The rebuilt environment is not active yet. `systemctl restart` requires an
interactive polkit authorization on this host, and `sudo -n` correctly failed
with `sudo: a password is required`; the old service remains active and was not
interrupted. Evidence: `artifacts/20260719T032321Z-phase7-authorized-restart/`.

An operator with the local authorization should now run:

```sh
sudo systemctl restart structured-agents-vllm.service
```

Then wait for the pinned service to report ready on `127.0.0.1:8000`. No model,
version, or service configuration change is needed for this step: it activates
the already rebuilt locked environment. The next agent action is a fresh
health/model probe followed by the deterministic choice constraint, then the
remaining Phase 7 acceptance chain.

---

# Phase 7 rebuilt-runtime acceptance — 2026-07-19

## Recovery result

After the operator restarted `structured-agents-vllm.service`, the new service
process ran from the reconciled `deploy/vllm/native/.venv`. The native
deployment environment reports the pinned `vllm=0.25.0` and `xgrammar=0.2.3`.
No model, server flag, package version, or library wire change was made; the
successful action was restoring the missing locked environment and activating
it.

- `artifacts/20260719T033726Z-phase7-rebuilt-ready-check/`: `/health` and
  `/v1/models` both returned 200 and the server identifies the served model as
  `base`.
- `artifacts/20260719T033756Z-phase7-rebuilt-choice/`: the exact deterministic
  XGrammar choice request returned 200 and `phase7-choice`.
- `artifacts/20260719T033834Z-phase7-rebuilt-all-constraints/`: raw requests
  and responses prove HTTP 200 for JSON Schema, regex, choice, and grammar.
  The schema response is valid against the supplied closed object schema; the
  regex response full-matches the requested pattern; choice and grammar are
  exact required strings.
- `artifacts/20260719T033910Z-phase7-rebuilt-verify-script/`:
  `deploy/vllm/verify.sh` reports `6 passed, 0 failed` for health, model,
  JSON Schema, regex, choice, and grammar. LoRA is not configured in this
  native profile and remains correctly inapplicable.
- `artifacts/20260719T033946Z-phase7-rebuilt-live-pytest/`: the real
  `SAV_LIVE=1 LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=base devenv
  shell -- pytest -x tests/test_live.py` suite reports `6 passed, 1 skipped`.
  It covers health/model identity, all four constrained modes through the
  durable `Backend`, and the live generate → authorize → keyed exactly-once
  effect pipeline. The only skip is the no-adapter LoRA branch.

## Conclusion

The prior blank 500s were caused by the active service running after its
native virtual environment had been removed, not by this repository's
constraint codec, pydantic-ai wire construction, DBOS durability plumbing, or
authority/effect implementation. Restoring the exact locked native environment
resolved all constrained-output paths without weakening a constraint or adding
a repository-side fallback.

## Remaining Phase 7 boundary

The successful keyed pipeline proves the live generation, authorization, and
exactly-once replay path under normal service operation. It does **not**
independently prove the plan's optional/disruptive mid-workflow service-kill
crash-recovery demonstration. That experiment would intentionally interrupt
`structured-agents-vllm.service` and needs separate explicit operator
authorization and a recovery plan. Therefore no full Phase 7 completion commit
has been made despite all requested non-disruptive live acceptance checks now
passing.

---

# Phase 7 authorized service-crash recovery experiment — 2026-07-19

The operator authorized a disruptive vLLM service-kill experiment.

- `artifacts/20260719T035010Z-phase7-service-kill-recovery/` records a long
  JSON-Schema request started against the ready service, `SIGKILL` sent to the
  active vLLM API child, the request's `curl: (52) Empty reply from server`,
  and systemd's service failure/restart records. The engine's own journal says
  it aborted one in-flight request, confirming the kill happened during active
  generation rather than during an idle period.
- Systemd restarted the unchanged unit after its configured delay and launched
  a new vLLM API/engine pair from the rebuilt locked environment. The initial
  30/45-second probes correctly observed the model still loading; subsequent
  `/health` returned 200.
- `artifacts/20260719T035350Z-phase7-service-kill-postrecovery-pipeline/`
  records a passing real `test_live_durable_pipeline_executes_keyed_effect_once`
  after recovery. Its assertion requires two keyed invocations to return the
  cached effect result while the effect counter remains one.

This demonstrates recovery of the active inference service after an
in-flight constrained request is aborted and proves that the real durable
generation → authorization → keyed exactly-once-effect pipeline remains
operational after recovery. It does not claim that the interrupted raw HTTP
request itself was transparently retried; that client request was intentionally
aborted by the service kill and its captured failure is part of the evidence.

---

# Phase 7 DBOS workflow-process crash recovery — 2026-07-19

## Scope and implementation

Added an opt-in `SAV_LIVE_CRASH=1` test plus a separate, test-owned worker
program. The ordinary `pytest` path remains SQLite-only and does not construct
or contact vLLM because `tests/test_live.py` retains its existing
`SAV_LIVE=1` module gate; the crash proof additionally skips unless
`SAV_LIVE_CRASH=1` is explicitly set.

Each run creates a fresh UTC artifact directory and dedicated SQLite DBOS
system database. The initial worker configures DBOS, builds the actual live
`Backend` Schema agent, registers the user-authored workflow and durable
effect before `launch()`, then starts the workflow under a fixed business id.
The workflow performs a real constrained generation, asserts the typed
`LiveCommand(argv=("echo", "phase7-live"))`, authorizes it, invokes the
keyed `execute(...)` durable effect, records an append-only effect line, and
only then blocks durably in `DBOS.recv_async`.

After the parent observes DBOS `PENDING` plus exactly one effect line, it
SIGKILLs only that spawned worker process. It neither signals nor restarts
`structured-agents-vllm.service`. A replacement process registers the same
workflows against the same SQLite DBOS file, starts the existing keyed workflow
handle, sends the durable resume message, and waits for the recovered result.

## Passing evidence

- Command: `SAV_LIVE=1 SAV_LIVE_CRASH=1
  LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=base devenv shell -- pytest
  -x tests/test_live.py::test_live_durable_workflow_recovers_after_worker_crash
  -vv` → `1 passed in 13.19s`.
- Fresh artifact:
  `artifacts/20260719T042222Z-phase7-dbos-worker-crash-recovery/`.
  `versions.txt` records Python 3.13.13, DBOS 2.23.0, and pydantic-ai-slim
  2.11.0; `environment.json`, command files, git state before/after, separate
  worker stdout/stderr and return codes are retained.
- `worker-start-before-kill.json` records `{"event":"pending",
  "status":"PENDING","effect_lines":1}`. This is the required durable
  checkpoint after the live constrained generation and effect, not a mere
  process-start checkpoint.
- `raw-vllm-request.json`, `raw-vllm-response.json`, and
  `raw-vllm-status.txt` preserve a fresh direct JSON-Schema request and its
  HTTP 200 response. `recovered-constrained-output.json` records the workflow
  agent's recovered typed output as `{"argv":["echo","phase7-live"]}`.
- `effect-before-replacement.txt` and `effect-after-recovery.txt` each contain
  the single line `phase7-worker-crash`. The replacement did not re-execute the
  external append-only effect.
- `worker-after-recovery.json` records DBOS `SUCCESS`, recovered result
  `{"argv":["echo","phase7-live"]}`, and `effect_lines: 1`.

## Conclusion and limitation

This proves DBOS workflow-process recovery across an actual process crash with
the real constrained vLLM agent, authorization, keyed durable effect, and
durable pause. It proves the original workflow reaches `SUCCESS` after the
replacement worker resumes it, while the effect executes exactly once and the
constrained result is preserved. It does not claim recovery from a simultaneous
SQLite database failure, a vLLM service outage, or a process crash in the
middle of an individual external effect; those are distinct fault boundaries.

## Final Phase 7 revalidation

- `devenv shell -- pytest` → `26 passed, 1 skipped`.
- `devenv shell -- ty check src tests` → `All checks passed!`.
- `devenv shell -- ruff check src tests` → `All checks passed!`.
- `devenv shell -- bash -n deploy/vllm/verify.sh` → clean.
- Fresh `artifacts/20260719T042410Z-phase7-final-live-pytest/` records the
  non-crash `SAV_LIVE=1` suite: `6 passed, 2 skipped` (the configured profile
  has no LoRA adapter, and the separate crash proof is intentionally not
  enabled by `SAV_LIVE` alone).
- Fresh `artifacts/20260719T042431Z-phase7-final-verify/` records
  `deploy/vllm/verify.sh`: `6 passed, 0 failed` for health, served model,
  JSON Schema, regex, choice, and grammar.
