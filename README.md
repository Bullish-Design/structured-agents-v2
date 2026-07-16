# structured-agents-v2

Typed, constrained [PydanticAI](https://ai.pydantic.dev/) agents over a local
**vLLM** backend — XGrammar for constrained decoding, per-agent LoRA adapters,
batched throughput.

## Thesis

Small local models are unreliable at *free-form* structured output but excellent
when the decoder is **constrained to a grammar**. This library leans on that: an
agent's output type *is* its decoding contract. XGrammar (server-side, in vLLM)
guarantees the model can only emit syntactically valid output; Pydantic then
validates it; and — for command-style agents — an explicit **executor** decides
whether the validated command is *authorized* to run. Syntax, validity, and
authority are three separate guarantees, enforced at three separate layers.

Per-agent **LoRA adapters** let one served base model host many specialists
cheaply (the adapter name rides the request's `model` field), and requests are
batched for throughput against a single vLLM server.

## Decode modes

A `DecoderSpec` (usually carried by a `ConstrainedOutput` subclass) says *how* an
agent's output is constrained. Each mode maps to a verified wire shape:

| mode          | output_type           | wire mechanism                                       |
|---------------|-----------------------|------------------------------------------------------|
| `json_schema` | `NativeOutput(model)` | standard `response_format` (XGrammar is server-side) |
| `grammar`     | `str`                 | `extra_body["structured_outputs"]["grammar"]`        |
| `regex`       | `str`                 | `extra_body["structured_outputs"]["regex"]`          |
| `choice`      | `str`                 | `extra_body["structured_outputs"]["choice"]`         |

`json_schema` returns a validated Pydantic model. The bare-string modes
(`grammar`/`regex`/`choice`) return a guarded `str` — the subclass acts as a
*spec carrier*, and the run validates the string against its declared
`regex`/`choices` client-side (see `ConstrainedOutput`).

## Quickstart

```python
from structured_agents_v2 import Backend, AgentProfile, AgentSet, RoutingTable
# Route, FileEditPlan, GitCommand are your own ConstrainedOutput subclasses.

backend = Backend(
    base_url="http://tower:8000/v1",
    api_key="...",
    default_model="base",
)

profiles = [
    AgentProfile(name="router", adapter="router",
                 instructions="Route to exactly one specialist.",
                 output_type_ref="myapp.schemas:Route"),
    AgentProfile(name="file_edit", adapter="file-edit",
                 instructions="Produce file-edit plans only.",
                 output_type_ref="myapp.schemas:FileEditPlan", policy="repo_file_edit_v1"),
    AgentProfile(name="git_ops", adapter="git-ops",
                 instructions="Translate to a single safe git command.",
                 output_type_ref="myapp.schemas:GitCommand", policy="git_safe_v1"),
]
routing = RoutingTable(router="router",
                       routes={"file_edit": "file_edit", "git_ops": "git_ops"})

fleet = AgentSet(backend=backend)
fleet.build(profiles, routing=routing)

routed = await fleet.route_and_run(user_msg)   # RoutedResult(route=..., output=...)

# The escape hatch stays open: the underlying pydantic_ai.Agent is always reachable.
raw_agent = fleet.agents["file_edit"].agent
```

Executors turn a validated command into a *decision* and then an effect. The
allowlist is **default-deny**: only commands matching a policy's `allow` rule run.

```python
from structured_agents_v2 import AllowlistExecutor, Policy

def run_git(cmd):
    ...  # the policy's effect; runs only for authorized commands

executor = AllowlistExecutor([
    Policy("git_safe_v1", allow=lambda c: c.value.split()[1] in {"status", "diff", "log"},
           action=run_git),
])
routed = await fleet.route_and_execute(user_msg, executor)   # authorize -> execute; denials are data
```

## Optional extras

| extra           | pulls in            | for                                                        |
|-----------------|---------------------|------------------------------------------------------------|
| `grammar-check` | `xgrammar`          | compile-check a constraint at class-definition time (heavy: torch + CUDA) |
| `dual-path`     | `dbos`, `psycopg`   | dual-path capture (local ‖ frontier) for fine-tuning / evals via DBOS      |

```bash
uv sync --extra dev                      # core + test toolchain
uv sync --extra dev --extra dual-path    # + dual-path capture
```

The lean core stays DBOS/Postgres-free and depends only on `pydantic` +
`pydantic-ai-slim[openai]`.

## Live backend verification

Point a real vLLM server (XGrammar + LoRA) and verify the whole wire path —
health → models → `json_schema` → xgrammar regex/choice → LoRA — with:

```bash
deploy/vllm/verify.sh
```

Live tests in the suite are gated behind `SAV_LIVE=1`.

## Development

Everything runs inside the devenv shell:

```bash
devenv shell -- pytest        # test suite
devenv shell -- ty check src  # type-check the library
```
