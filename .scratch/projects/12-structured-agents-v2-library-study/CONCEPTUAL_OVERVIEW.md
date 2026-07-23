# `structured-agents` — Conceptual Overview

**Review date:** 2026-07-21  
**Version studied:** 0.3.0  
**Commit studied:** `90725a5` (`main`)  
**Primary package:** [`src/structured_agents`](../../../src/structured_agents/)  
**License:** MIT  
**Runtime floor:** Python 3.13

## 1. What this library is

`structured-agents` is a small, code-first Python toolkit for building durable LLM workflows whose outputs have explicit structural constraints and whose external side effects pass through explicit authorization boundaries.

Its central idea is not to provide a monolithic agent application. It provides a set of composable primitives:

- A typed description of acceptable model output.
- A translation layer from that neutral description to one of several OpenAI-compatible inference-engine dialects.
- A PydanticAI agent wrapped in DBOS durable execution.
- DBOS-backed queueing, scheduling, workflow inspection, cancellation, forking, and paired comparison.
- Synchronous authorization policies and durable effectors.
- Durable human approval using DBOS events and workflow messaging.
- A narrow serialized-configuration boundary with explicit schema-module allowlisting.
- An optional external command sandbox integration through the `fornix` executable.

The package is best understood as a **durable constrained-agent plane**, not as an autonomous-agent framework. Applications are expected to define their own commands, workflows, policies, approval UX, storage, and business orchestration around these primitives.

## 2. The problem it is trying to solve

A model-driven application has several distinct failure domains:

1. The model can return data in the wrong shape.
2. Different inference servers express constrained decoding through different request fields.
3. A worker can crash after an expensive model call or external operation.
4. Retrying can duplicate effects.
5. Valid model output is not the same thing as authorized intent.
6. A human decision may take minutes or days, during which a process should not have to remain alive.
7. Operators need to inspect, cancel, replay, compare, and rate-limit work.

This library addresses those domains with deliberately separate abstractions. Constraints describe output; engines render wire syntax; agents generate; authorizers decide; effectors act; approvals pause; DBOS records and resumes.

That separation is the strongest aspect of the design. In particular, model generation is not itself an authority to execute a process.

## 3. System map

```text
                           build time

  AgentSpec[T] ──contains──> Constraint[T]
       │                          │
       │                    check/compile
       │                          │
       └──────────────> Engine.render()
                              │
                              v
                         WireSpec
                    (output type + body)
                              │
                              v
                 PydanticAI OpenAIChatModel
                              │
                              v
                         DBOSAgent
                              │
                              v
                           Agent[T]

                            run time

  prompt ──> durable model workflow ──> constrained output ──> parse ──> T
                                                                    │
                                                                    v
  command T ──> Authorizer.decide ──denied──> Denied data
                         │
                       allowed
                         │
                         v
                 durable Effector.run
                         │
                         v
                        result

  optional human gate:
  workflow ──publish command/to──> PENDING ──message──> Decision
```

## 4. The core abstractions

### 4.1 Constraints are codecs, not provider requests

[`constraint.py`](../../../src/structured_agents/constraint.py) defines the generic protocol `Constraint[T]`. A constraint has four responsibilities:

- `kind`: a neutral capability name.
- `check()`: an optional build-time compilation check.
- `parse(raw)`: the final runtime conversion or validation step.
- `to_config()`: a serializable declaration.

The public constructors return private, frozen dataclasses:

| Constructor | Result type | Neutral meaning | Local post-check |
|---|---|---|---|
| `Schema(Model)` | `Model` | JSON conforming to a Pydantic model | Currently trusts the PydanticAI result and casts it |
| `Regex(pattern)` | `str` | A complete regex match | Uses `re.fullmatch` |
| `Choice(*options)` | a literal-like string type | One finite string option | Membership test |
| `Grammar(ebnf)` | `str` | Text accepted by an EBNF grammar | Only checks that the result is a string |

`Schema` and `Grammar` optionally call XGrammar during `check()`. If `xgrammar` is not installed, the check silently becomes a no-op. The normal development environment does not install it; it is exposed as the `grammar-check` extra.

The important design decision is that a constraint does **not** know whether the server is vLLM, SGLang, or llama.cpp. It describes semantics; the engine owns transport syntax.

### 4.2 Engine plugins render backend dialects

[`engine/base.py`](../../../src/structured_agents/engine/base.py) defines the internal `Engine` protocol:

- `name`
- `supports`
- `render(constraint) -> WireSpec`

[`engine/__init__.py`](../../../src/structured_agents/engine/__init__.py) contains a closed registry of three built-ins. One is selected for each `Backend`; this is selection among in-tree implementations, not third-party engine discovery.

The dialect differences are:

| Constraint | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| Schema | PydanticAI `NativeOutput` / OpenAI `response_format` | Same | Same |
| Regex | `extra_body.structured_outputs.regex` | `extra_body.regex` | Unsupported |
| Choice | `extra_body.structured_outputs.choice` | Lowered to escaped regex alternation | Lowered to GBNF alternation |
| Grammar | `extra_body.structured_outputs.grammar` | `extra_body.ebnf` | Passed as `extra_body.grammar` |
| LoRA adapter | Model name, advertised supported | Model name, advertised supported | Unsupported |

The vLLM path is the reference path and has golden wire-shape tests. The SGLang and llama.cpp modules explicitly label themselves unverified. In particular, the public `Grammar` abstraction says EBNF, while llama.cpp expects GBNF; the current implementation passes the text through without claiming dialect parity.

### 4.3 `Backend` is the construction boundary

[`agent.py`](../../../src/structured_agents/agent.py) contains four important types:

- `Settings`: sampling and request options.
- `AgentSpec[T]`: name, constraint, instructions, optional adapter, and settings.
- `Backend`: an inference-engine connection and agent factory.
- `Agent[T]`: the narrow durable run wrapper returned to callers.

`Backend.build(spec)` performs this sequence:

1. Check that the selected engine advertises the constraint kind.
2. Check LoRA capability when `spec.adapter` is set.
3. Run the constraint's optional compiler check.
4. Ask the engine to render a `WireSpec`.
5. Merge user settings with the constraint wire body. Engine-generated constraint fields win on key collisions.
6. Construct an `OpenAIChatModel`, unless a model object was injected.
7. Construct a PydanticAI `Agent` with the output type and settings.
8. Wrap it in PydanticAI's `DBOSAgent` under the agent name.
9. Return the library's `Agent[T]` wrapper.

At runtime, `Agent.run(prompt)` awaits `DBOSAgent.run(prompt)`, extracts `.output`, and applies the constraint's `parse()` method.

PydanticAI therefore performs the principal schema conversion; the local codec provides an additional postcondition for regex and choice outputs.

### 4.4 DBOS supplies the durable substrate

DBOS is process-global in this design. [`plane.py`](../../../src/structured_agents/plane.py) exposes its lifecycle and operational surface:

- `configure()` creates the DBOS singleton, defaulting to a SQLite file in the current working directory.
- Applications build agents and register workflows.
- `launch()` starts DBOS.
- `shutdown()` destroys the singleton.

Registration order matters. Agents contain named DBOS workflows and must be built before launch for recovery registration to be complete.

The durable plane also provides:

- `Queue`: DBOS queue submission with optional concurrency and rate limits.
- `schedule(cron)`: DBOS scheduled-workflow decoration.
- `workflows()` and `status()`: workflow observability.
- `fork()`: replay from a recorded step.
- `cancel()`: cancellation.
- `compare()`: two independently keyed agent runs gathered into a `Comparison[T]`.

An optional `key` becomes a DBOS workflow ID. Reusing that key retrieves the already recorded workflow outcome instead of executing the workflow body again. `compare()` derives `:primary` and `:reference` workflow IDs from one comparison key.

This is durable execution, not general conversation/session management. The library does not expose message-history APIs, tools, dependencies, streaming, or the rest of PydanticAI's full surface through `Agent.run`.

### 4.5 Authorization is separate from generation

[`authority.py`](../../../src/structured_agents/authority.py) divides authority into two protocols:

- `Authorizer[C]`: synchronously maps a command to a `Decision`.
- `Effector[C, R]`: asynchronously performs an operation and returns `R`.

`execute(authorizer, effector, command, key=...)` is the blessed composition path:

1. Evaluate policy.
2. Return `Denied` as ordinary data if not allowed.
3. Otherwise enter the named DBOS effect workflow.
4. Run the effector, normally as a DBOS step.

Included policy tools are:

- `Allowlist`, which evaluates named predicates and fails closed if a rule raises.
- `all_of`, for conjunctive policy composition.
- `any_of`, for disjunctive policy composition.

Included effectors are:

- `Null`, a recorded dry run.
- `Subprocess`, which extracts a strict non-empty `argv` from a Pydantic command model and invokes `subprocess.run` without a shell.
- `FornixEffector`, which invokes `fornix box --check -- ...` and decodes its JSON result.

The command model is application-defined. The intended pattern is for a schema-constrained agent to return a Pydantic command, for application policy to inspect that command, and only then for an effector to act.

### 4.6 Human approval is a durable workflow pause

[`approval.py`](../../../src/structured_agents/approval.py) uses DBOS events and messaging.

Inside an application-authored DBOS workflow, `Approval.request(command, to=...)`:

1. Publishes the command as a workflow event.
2. Publishes the intended recipient.
3. Waits durably on a message topic.
4. Returns a normal `Decision` for approval, denial, timeout, or malformed data.

Outside the workflow, `ApprovalClient` can:

- List pending workflows that expose the two approval events.
- Send an approval message.
- Send a denial message.

This leaves UI, identity, access control, notification delivery, and audit presentation to the application. The library supplies the durable pause and message mechanics.

### 4.7 Serialized configuration is intentionally narrow

[`config.py`](../../../src/structured_agents/config.py) reconstructs constraints and `AgentSpec` objects from dictionaries.

Schema constraints contain a reference of the form `module:QualifiedModel`. Resolving that reference requires an explicit `allow_modules` set. A module must exactly equal an allowed module or be its descendant before import occurs.

Constraint kinds have a global factory registry. Built-ins cover schema, regex, choice, and grammar. Applications can register a factory directly, and installed packages can expose factories through the `structured_agents.constraints` entry-point group.

This is not a full configuration system. There is no file loader, environment merger, migration layer, settings model, or `AgentSpec.to_config()` method. It is a controlled boundary from already-parsed mapping data into the library's typed objects.

## 5. Principal execution stories

### 5.1 Constrained generation

```python
class Plan(BaseModel):
    title: str
    steps: list[str]

backend = Backend(engine="vllm", default_model="base")
planner = backend.build(
    AgentSpec(
        "planner",
        Schema(Plan),
        "Create a concise executable plan.",
        settings=Settings(temperature=0, max_tokens=512),
    )
)

plan: Plan = await planner.run("Prepare the release")
```

The model call is a DBOS workflow. The server is asked to constrain decoding, PydanticAI validates the returned JSON, and the application receives a `Plan`.

### 5.2 Constrained intent followed by authorized effect

```text
prompt
  -> Schema(Command)
  -> Command(argv=[...])
  -> Allowlist.decide(Command)
       -> Denied(reason, command)
       or
       -> execute(..., key="business-operation-id")
            -> durable effector step
            -> ProcessResult
```

The business key is what makes replay converge on one recorded workflow outcome. It must be stable, unique in DBOS's workflow-ID namespace, and derived from the application's real operation identity.

### 5.3 Human-gated effect

An application workflow typically requests approval, converts a negative `Decision` to `Denied`, and calls `execute()` only after approval. The approval and effect mechanisms are intentionally not fused, so an application can compose human review with additional policy.

### 5.4 Paired comparison

`compare(primary, reference, prompt, key=...)` creates two deterministic workflow IDs, runs both agents concurrently, and returns the two values with their IDs. It is an operational comparison primitive; it does not score, persist, export, or adjudicate the pair.

## 6. Durability and "exactly once"—the precise mental model

The library's durability comes from DBOS workflow and step recording:

- A completed model workflow can be replayed without another model request when the same workflow ID is reused.
- A completed effect workflow returns its recorded result when the same key is reused.
- A pending receive can survive process loss and resume when a message arrives.
- Queue items, schedules, status, cancellation, and forks are DBOS operations.

There are three important boundaries:

1. **No key means no business-level idempotency.** DBOS assigns invocation identity, but two application calls are distinct operations.
2. **Authorization runs before the durable effect workflow.** The policy decision itself is not recorded under the effect key by this library.
3. **Arbitrary external effects still need idempotent design.** A subprocess or remote operation can complete immediately before a worker dies and before the durable step result is committed. The safe design is to give the external system the same idempotency key or make the operation naturally idempotent.

The phrase "exactly once" should therefore be read as keyed durable workflow replay for recorded completions, not as a universal transactional guarantee over every external system.

## 7. Type model

The library uses Python 3.13's PEP 695 generic syntax throughout:

- `Constraint[T]`
- `AgentSpec[T]`
- `Agent[T]`
- `Authorizer[C]`
- `Effector[C, R]`
- `Comparison[T]`

`Choice("keep", "skip")` preserves literal-like typing, and `Schema(Plan)` carries `Plan` through `AgentSpec` and `Agent.run`.

The type story is strongest on the direct agent path. Some dynamic boundaries deliberately erase precision (`Constraint[Any]` during config reconstruction), and some current casts claim more than runtime behavior guarantees. The most consequential example is the current queue handle, discussed in the code review.

## 8. Operational topology in this repository

The distributable library is about 1,000 lines of Python. The repository is much larger because it also contains deployment and experimental infrastructure:

- Native and container vLLM deployment.
- Native llama.cpp deployment.
- An isolated SGLang/GGUF compatibility spike.
- NixOS modules and `devenv` environments.
- Runtime artifacts, model compatibility investigations, and benchmark output.
- A vendored vLLM GGUF plugin under the deployment tree.

The intended service shape is an OpenAI-compatible endpoint selected per `Backend`. Deployment documents favor loopback listeners exposed through Tailscale rather than public/LAN binding.

Only the vLLM constraint wire is treated as the established reference path. Repository history shows substantial runtime work around GGUF, Gemma 4, XGrammar, and backend compatibility. Those artifacts provide useful evidence, but they are not part of the Python package's runtime API.

## 9. Test strategy and observed evidence

The normal test harness creates one session-scoped DBOS singleton backed by a temporary SQLite database. Tests cover:

- Property-based schema and regex round trips.
- Constraint rejection and config serialization.
- Config allowlisting, direct registration, and mocked entry-point discovery.
- Exact vLLM wire shapes and expected SGLang/llama.cpp renderings.
- Mocked OpenAI-compatible schema output through a real PydanticAI/DBOS agent.
- Policy composition, denial-before-effect, keyed effect replay, and subprocess results.
- Pending approval inspection, resume, denial, and timeout.
- Queue limits with a test double, cron scheduling, workflow inspection, fork, cancellation, and comparison replay.
- An optional live inference suite.
- An optional separate-process crash-recovery proof.

At the reviewed commit:

- `pytest`: **32 passed, 1 skipped**.
- `ruff check src tests`: **passed**.
- `ty check src tests/typecheck_constraint.py`: **passed**.
- `ruff format --check src tests`: **failed; 8 files would be reformatted**.
- Wheel and source distribution: **built successfully**.
- Live inference tests: **not run** in this review.
- XGrammar compile checks: **not run**, because XGrammar is not installed in the normal environment.

The code review adds focused probes that reveal gaps the green suite does not cover.

## 10. What the library deliberately does not provide

The public surface does not attempt to provide:

- A hosted service or CLI.
- A conversation/session abstraction.
- General PydanticAI tools, dependencies, streaming, or message history.
- Automatic runtime capability negotiation with inference servers.
- A scoring or persistence system for comparisons.
- Authentication or role enforcement for approval clients.
- A policy language.
- Transactional integration with arbitrary external effects.
- General engine entry-point plugins.
- A complete application configuration framework.

These omissions keep the core small. They also mean callers must understand the lifecycle and security boundaries rather than assuming the library has solved them implicitly.

## 11. Current maturity

The architecture is coherent and promising. The direct constrained-agent path, keyed durable effect path, and DBOS operational primitives are implemented compactly and backed by meaningful tests.

The reviewed 0.3.0 implementation is not yet ready to be treated as a production authority boundary without remediation. The detailed review identifies confirmed fail-open approval parsing, cross-thread allowlist contamination, a broken real-agent queue path, overbroad durability claims, unverified backend behavior, and release/documentation gaps.

The best one-sentence mental model is:

> `structured-agents` is a thin typed membrane around constrained OpenAI-compatible generation and DBOS durability, with explicit policy and approval hooks—but application authors still own identity, idempotency, lifecycle discipline, and external-effect safety.

