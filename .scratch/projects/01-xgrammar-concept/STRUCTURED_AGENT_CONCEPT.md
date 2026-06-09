# STRUCTURED_AGENT_CONCEPT

## Purpose

This concept describes an agent architecture where each small, task-specific Pydantic AI agent is bound to a specific vLLM structured-decoding configuration, ideally backed by XGrammar, and optionally to a task-specific LoRA adapter.

The goal is to make operational agents emit constrained, validated command objects rather than free-form prose. This is intended for small agentic use cases such as file editing, git operations, test execution, and similar developer tooling tasks.

## Core Thesis

A safe and efficient self-hosted agent stack should separate four concerns:

1. **Agent contract**: Pydantic AI defines the agent, dependencies, tools, output type, and validation.
2. **Behavior specialization**: vLLM serves a shared base model with task-specific LoRA adapters.
3. **Decoding constraints**: vLLM applies a per-agent structured decoding configuration, preferably with XGrammar.
4. **Authority and execution**: a hardened executor validates and performs actions with explicit permissions.

In short:

```text
Pydantic AI agent
  -> OpenAI-compatible request
  -> vLLM server over Tailscale
  -> base model + selected LoRA adapter
  -> per-agent XGrammar / structured output config
  -> validated Pydantic command object
  -> restricted executor
```

## Relevant Current Capabilities

### Pydantic AI

Pydantic AI supports OpenAI-compatible providers through `OpenAIChatModel` and configurable provider settings. This makes it suitable for calling a self-hosted vLLM OpenAI-compatible server.

Pydantic AI also supports structured outputs using Pydantic models, native output modes, tool-output modes, and runtime validation. This is useful for representing final agent decisions as typed Python objects.

Reference:
- https://pydantic.dev/docs/ai/models/openai/
- https://pydantic.dev/docs/ai/core-concepts/output/

### vLLM

vLLM provides an OpenAI-compatible server and supports structured outputs in that server. Structured output backends include XGrammar and Guidance, with the backend configurable through vLLM server flags.

vLLM also supports serving LoRA adapters through the OpenAI-compatible server with `--enable-lora` and `--lora-modules`.

Reference:
- https://docs.vllm.ai/en/latest/features/structured_outputs/
- https://docs.vllm.ai/en/latest/features/lora/
- https://docs.vllm.ai/en/latest/serving/online_serving/

### XGrammar

XGrammar is a structured generation engine for constrained decoding. It is intended to enforce JSON, regex, or context-free grammar constraints at the decoding layer. Its value here is not just validation after the fact, but preventing invalid syntax during generation.

Reference:
- https://github.com/mlc-ai/xgrammar
- https://arxiv.org/abs/2411.15100

## Design Goal

Each Pydantic AI agent should be linked to a specific decoder configuration.

For example:

```text
file_edit_agent
  model: "file-edit"
  LoRA adapter: file-edit
  decoder config: JSON schema for FileEditPlan
  executor scope: repository file edits only

git_agent
  model: "git-ops"
  LoRA adapter: git-ops
  decoder config: JSON schema for GitCommand
  executor scope: safe git subcommands only

test_runner_agent
  model: "test-runner"
  LoRA adapter: test-runner
  decoder config: grammar or JSON schema for allowed test invocations
  executor scope: test commands only
```

The important property is that the Pydantic AI agent is not merely “prompted” to behave in a constrained way. The agent is bound to a structured decoding contract that vLLM enforces during token generation.

## V1 Architecture

### Components

```text
┌─────────────────────────────┐
│ Application / Orchestrator  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Pydantic AI Agent           │
│ - instructions              │
│ - deps_type                 │
│ - output_type               │
│ - tool policy               │
│ - decoder config reference  │
└──────────────┬──────────────┘
               │ OpenAI-compatible request
               ▼
┌─────────────────────────────┐
│ vLLM Server on Tailscale    │
│ - base model                │
│ - LoRA adapters             │
│ - XGrammar backend          │
│ - structured_outputs        │
└──────────────┬──────────────┘
               │ constrained JSON / command object
               ▼
┌─────────────────────────────┐
│ Pydantic Validation         │
│ - model_validate_json       │
│ - business validators       │
└──────────────┬──────────────┘
               │ validated object
               ▼
┌─────────────────────────────┐
│ Hardened Executor           │
│ - path allowlists           │
│ - command allowlists        │
│ - permission checks         │
│ - approval gates            │
│ - audit logs                │
└─────────────────────────────┘
```

### vLLM Server Sketch

Example server shape:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 100.x.y.z \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules \
    file-edit=/models/lora/file-edit \
    git-ops=/models/lora/git-ops \
    test-runner=/models/lora/test-runner \
  --structured-outputs-config.backend xgrammar
```

Notes:

- The exact vLLM flag surface can change across versions.
- Current vLLM docs describe structured outputs as supported by the OpenAI-compatible server.
- Current docs also describe backend selection for structured outputs.
- The server should still require an API key even on a private Tailscale network.

## Agent-to-Decoder Binding

The main concept is a small wrapper object that binds:

1. a Pydantic AI model name or LoRA adapter name,
2. a Pydantic output schema,
3. a vLLM structured output mode,
4. optional generation settings,
5. executor policy metadata.

### Example Configuration Model

```python
from typing import Any, Literal
from pydantic import BaseModel, Field


class StructuredDecoderConfig(BaseModel):
    mode: Literal["json_schema", "choice", "regex", "grammar"]
    backend: Literal["xgrammar"] = "xgrammar"
    schema: dict[str, Any] | None = None
    choices: list[str] | None = None
    regex: str | None = None
    grammar: str | None = None
    strict: bool = True


class AgentRuntimeConfig(BaseModel):
    name: str
    model_name: str
    lora_adapter: str | None = None
    decoder: StructuredDecoderConfig
    temperature: float = 0.0
    max_tokens: int = 1024
    executor_policy: str
```

### File Edit Example

```python
from typing import Literal
from pydantic import BaseModel, Field


class FilePatch(BaseModel):
    op: Literal["replace", "insert_after", "delete"]
    path: str
    target: str | None = None
    content: str | None = None


class FileEditPlan(BaseModel):
    action: Literal["edit_file", "refuse", "needs_clarification"]
    patches: list[FilePatch] = Field(default_factory=list)
    reason: str
```

Associated runtime config:

```python
file_edit_config = AgentRuntimeConfig(
    name="file_edit_agent",
    model_name="file-edit",
    lora_adapter="file-edit",
    decoder=StructuredDecoderConfig(
        mode="json_schema",
        schema=FileEditPlan.model_json_schema(),
    ),
    temperature=0.0,
    max_tokens=2048,
    executor_policy="repo_file_edit_v1",
)
```

The model name maps to the vLLM-served LoRA adapter. The schema maps to the structured decoding request.

## Request Shape to vLLM

For a direct OpenAI-compatible client call, the request can attach vLLM-specific structured output parameters.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://100.x.y.z:8000/v1",
    api_key="your-vllm-api-key",
)

schema = FileEditPlan.model_json_schema()

response = client.chat.completions.create(
    model="file-edit",
    messages=[
        {"role": "system", "content": "Emit only a valid FileEditPlan."},
        {"role": "user", "content": "Change the retry timeout from 5s to 10s."},
    ],
    temperature=0,
    max_tokens=2048,
    extra_body={
        "structured_outputs": {
            "json": schema,
        }
    },
)

plan = FileEditPlan.model_validate_json(response.choices[0].message.content)
```

This direct form is the most explicit V1 implementation. It proves the relationship between:

```text
Pydantic model -> JSON Schema -> vLLM structured_outputs -> XGrammar -> Pydantic validation
```

## Pydantic AI Integration Pattern

Pydantic AI can be the orchestrator and type-validation layer, while a custom model/client wrapper handles the vLLM-specific `extra_body`.

### V1 Recommendation

Use a thin internal wrapper before trying to fully abstract it into Pydantic AI internals.

```python
class VllmStructuredClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def run_json_schema(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        output_type: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> BaseModel:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "structured_outputs": {
                    "json": output_type.model_json_schema(),
                }
            },
        )

        content = response.choices[0].message.content
        return output_type.model_validate_json(content)
```

Then bind that wrapper to each agent config.

```python
result = structured_client.run_json_schema(
    model=file_edit_config.model_name,
    messages=messages,
    output_type=FileEditPlan,
    temperature=file_edit_config.temperature,
    max_tokens=file_edit_config.max_tokens,
)
```

### Why Not Make This Too Abstract in V1?

Pydantic AI may already support native structured output through OpenAI-compatible mechanisms depending on provider and model settings. However, the required capability here is more specific:

```text
Each individual Pydantic AI agent must be bound to a specific vLLM/XGrammar decoder config.
```

That is stronger than simply saying `output_type=FileEditPlan`.

V1 should therefore own the request shape directly, confirm vLLM behavior, and only then wrap it into a cleaner Pydantic AI abstraction.

## V2: Pydantic Wrapper for XGrammar Decoder Config

V2 can introduce a first-class wrapper that makes the decoder binding explicit at the agent definition site.

### Possible API Shape

```python
class XGrammarOutput(BaseModel):
    output_type: type[BaseModel]
    mode: Literal["json_schema", "grammar", "regex", "choice"] = "json_schema"
    grammar: str | None = None
    regex: str | None = None
    choices: list[str] | None = None
    backend: Literal["xgrammar"] = "xgrammar"
    strict: bool = True

    def to_vllm_extra_body(self) -> dict:
        if self.mode == "json_schema":
            return {
                "structured_outputs": {
                    "json": self.output_type.model_json_schema(),
                }
            }

        if self.mode == "grammar":
            return {
                "structured_outputs": {
                    "grammar": self.grammar,
                }
            }

        if self.mode == "regex":
            return {
                "structured_outputs": {
                    "regex": self.regex,
                }
            }

        if self.mode == "choice":
            return {
                "structured_outputs": {
                    "choice": self.choices,
                }
            }

        raise ValueError(f"Unsupported mode: {self.mode}")
```

Then each agent can declare its decoder contract:

```python
file_edit_decoder = XGrammarOutput(
    output_type=FileEditPlan,
    mode="json_schema",
)

test_runner_decoder = XGrammarOutput(
    output_type=TestRunRequest,
    mode="grammar",
    grammar=TEST_RUNNER_GRAMMAR,
)
```

### Desired Pydantic AI Binding

The desired agent declaration would look roughly like:

```python
file_edit_agent = StructuredAgent(
    name="file_edit_agent",
    model="file-edit",
    instructions="Produce file edit plans only.",
    output_type=FileEditPlan,
    decoder=file_edit_decoder,
    executor_policy="repo_file_edit_v1",
)
```

Where `StructuredAgent` is an internal wrapper around Pydantic AI plus the vLLM request adapter.

## Agent Examples

### Router Agent

The router agent should have a very small output space.

```python
class RouteDecision(BaseModel):
    route: Literal[
        "file_edit",
        "git_ops",
        "test_runner",
        "answer",
        "refuse",
    ]
    reason: str
```

This can use either JSON schema or `choice`.

For pure route selection, `choice` may be faster and simpler:

```python
router_decoder = StructuredDecoderConfig(
    mode="choice",
    choices=["file_edit", "git_ops", "test_runner", "answer", "refuse"],
)
```

### Git Agent

```python
class GitCommand(BaseModel):
    action: Literal[
        "status",
        "diff",
        "add",
        "commit",
        "checkout_new_branch",
        "refuse",
    ]
    paths: list[str] = Field(default_factory=list)
    message: str | None = None
    reason: str
```

The executor should still prevent dangerous actions such as force pushes, arbitrary shell execution, deleting branches, or modifying remotes unless explicitly allowed.

### Test Runner Agent

```python
class TestRunRequest(BaseModel):
    action: Literal["run_tests", "run_single_test", "refuse"]
    command: Literal["pytest", "python -m pytest"]
    path: str | None = None
    test_name: str | None = None
    reason: str
```

A stricter version could use an EBNF grammar instead of JSON schema, especially if the final output is intended to be a shell-like command string. For most operational safety cases, a JSON object plus executor-side command construction is preferable.

## Security Model

XGrammar restricts syntax. It does not grant or deny authority.

The executor must still enforce:

- path allowlists,
- repository boundaries,
- read/write separation,
- command allowlists,
- max runtime limits,
- network restrictions,
- approval gates for destructive actions,
- audit logging,
- per-agent credentials or authorization scopes.

Bad pattern:

```text
model emits shell command -> shell executes it
```

Better pattern:

```text
model emits typed command object
  -> Pydantic validates it
  -> executor checks policy
  -> executor constructs safe command
  -> executor runs in sandbox
```

## What Should Be Measured

The design should be benchmarked across:

1. total wall-clock latency,
2. first-token latency,
3. tokens per second,
4. structured decoding overhead,
5. retry rate,
6. invalid-output rate,
7. tool/action error rate,
8. executor rejection rate,
9. LoRA adapter switching overhead,
10. schema complexity sensitivity.

The expected benefit is not only raw decoding speed. The larger benefit is avoiding invalid outputs, retries, ambiguous prose, and unsafe action interpretation.

## V1 Implementation Checklist

1. Start vLLM on the Tailscale network.
2. Enable LoRA serving.
3. Register task-specific LoRA adapters.
4. Enable or select XGrammar as the structured output backend.
5. Define one Pydantic output model per agent.
6. Generate JSON Schema from each Pydantic model.
7. Send requests with vLLM `structured_outputs`.
8. Validate outputs using Pydantic.
9. Execute only through hardened executors.
10. Log the selected agent, adapter, decoder config, schema hash, and final validated command.

## V2 Implementation Checklist

1. Create `XGrammarOutput` or equivalent wrapper.
2. Add per-agent decoder config as a first-class field.
3. Support JSON schema, grammar, regex, and choice.
4. Add schema hashing and decoder config versioning.
5. Add compatibility tests against vLLM versions.
6. Add latency benchmarks comparing:
   - unconstrained output,
   - prompted JSON,
   - provider-native JSON schema,
   - vLLM structured outputs with XGrammar.
7. Add failure-mode tests for invalid paths, invalid commands, malformed patches, and unauthorized actions.

## Open Questions

1. Should the router use a separate LoRA adapter or the base model?
2. Should grammar mode be used only for command strings, with JSON schema preferred for typed command objects?
3. Should Pydantic AI remain the public agent interface while a lower-level vLLM client owns the structured output request?
4. How should schema versioning be represented in logs and audit records?
5. Should each agent have separate Tailscale ACLs/API keys, or should authorization live entirely in the executor service?
6. What is the target fallback behavior if vLLM structured output fails or returns a transport error?

## Recommended V1 Position

For the first implementation, keep the contract explicit:

```text
One Pydantic output model
+ one LoRA model name
+ one vLLM structured output config
+ one executor policy
= one operational agent
```

This is the simplest way to ensure that each Pydantic AI agent is linked to a specific XGrammar-backed decoder configuration.

Avoid designing a general abstraction too early. First prove:

1. vLLM receives the expected structured output request,
2. XGrammar is the active backend,
3. the LoRA adapter is selected correctly,
4. the output validates as the intended Pydantic model,
5. the executor enforces authority independently of the model.

Once those are stable, build the V2 Pydantic wrapper that makes the decoder binding declarative.
