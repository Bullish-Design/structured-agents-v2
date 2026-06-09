# Phase 2 Kickoff — structured-agents-v2 library wrapper

You are starting **Phase 2** of the library. This doc is self-contained: read it fully,
confirm the green baseline, then implement. Everything it references is in the repo.

---

## 0. How to operate (read first)

- **Run ALL commands inside the devenv shell:** `devenv shell -- <cmd>`. The system
  Python is 3.12; the devenv shell gives Python 3.13.13 + uv. Never use bare `uv run`
  outside devenv or the system python.
- Type check: `devenv shell -- uv run --extra dev ty check src`  (type checker is **ty**, Astral's — not mypy)
- Lint + format: `devenv shell -- uv run --extra dev ruff check src tests` and `... ruff format src tests`
- Tests: `devenv shell -- uv run --extra dev pytest -q`
- **Baseline is GREEN on `main`** (19 tests, 90% cov, ty + ruff clean). Run the tests
  first and confirm green before changing anything.
- **Git:** work on a branch off `main` (e.g. `phase-2-backend-agent`); when done + green,
  fast-forward `main` and push. Remote is `origin`
  (github.com/Bullish-Design/structured-agents-v2). **No AI-attribution trailers** in
  commit messages (user global rule: no "Co-Authored-By", no "Generated with").
- If you need the user to run an interactive command, ask them to use `! <cmd>`.

## 1. Authoritative context (read these, in order)

1. `.scratch/projects/02-library-wrapper/CONCEPT.md` (rev 2) — the design. Focus on
   **§4 (abstractions), §7 (package layout), §8 (build phases), §9 (MVP), §10 (open Qs).**
2. `.scratch/projects/02-library-wrapper/VERIFICATION.md` — verified PydanticAI wire shapes.
3. `.scratch/projects/01-xgrammar-concept/spike/FINDINGS.md` — request-path verification.
4. `.scratch/projects/01-xgrammar-concept/STRUCTURED_AGENT_CONCEPT.md` — the thesis.

Auto-memory (`~/.claude/projects/-home-andrew-Documents-Projects-structured-agents-v2/memory/MEMORY.md`)
is loaded each session and indexes the same facts.

## 2. The project in one paragraph

A thin, declarative binding over **PydanticAI** where the **output type carries its own
decoding constraint**. Goal: many small, specialized agents — each a constrained output
model + a per-agent LoRA adapter — run **batched** against one local **vLLM** server with
**XGrammar** constrained decoding, emitting validated command objects (not prose). The
library is the binding/composition layer + an authority boundary; PydanticAI owns the
runtime, model client, and validation. NOT a custom agent loop, model client, or
orchestration engine.

## 3. Backend reality (important)

- The server at `remora-server:8000` (Tailscale 100.123.9.95) is currently **llama.cpp**
  (Qwen3.5-9B GGUF, 4 batch slots, **no LoRA**, GBNF not XGrammar), NOT vLLM. It speaks
  the OpenAI-compatible API and honors `response_format: json_schema`, so the client path
  is fully testable against it.
- `deploy/vllm/` holds the drop-in vLLM container that will replace it (same `:8000/v1`
  contract). XGrammar + LoRA can only be exercised once that's up (needs the 12GB 3060 box).
- Env vars are set by `devenv.nix`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

## 4. What Phase 1 delivered (already on `main`)

`src/structured_agents_v2/`:
- `constrained.py` — **`ConstrainedOutput`** base: decode mode via class dunders
  (`__decode_mode__`, `__regex__`, `__choices__`, `__grammar__`, `__strict__`), validated
  at class-definition; `.decoder_spec()`; optional `.check_compilable()` (gated xgrammar).
- `decoder.py` — **`DecoderSpec`** + `.apply(output_type) -> DecoderApplication(output_type, extra_body)`.
  json_schema → `NativeOutput(model)`; grammar/regex/choice → `str` + `extra_body`.
- `capture.py` — **`RequestCapture`**: httpx event hook → `RequestRecord` (`.model`,
  `.response_format`, `.tools`, `.extra_body_keys`, `.body`). `.client(transport=...)`.
- `errors.py` — `StructuredAgentsError`, `ConfigError`, `ConstraintConfigError`,
  `ConstraintCompileError`, **`BackendCapabilityError`** (defined, used in Phase 2).
- `__init__.py` exports the public surface; `py.typed` present.

`tests/`:
- `conftest.py` — **`MockOpenAI`** in-process ASGI server (GPU-free) + `transport` fixture.
  Set `mock_openai.responder = lambda req: <content str>` per test.
- `test_wire_shapes.py` — builds an agent inline (`_build_agent`) and asserts the wire
  shape. **In Phase 2, this inline builder becomes `StructuredAgent`'s real build path.**

## 5. Verified facts that constrain Phase 2 design

- **PydanticAI defaults a Model output to the function-calling tool**, not `response_format`.
  You MUST wrap in `NativeOutput` for clean `response_format: json_schema`. `DecoderSpec.apply`
  already does this — agent build must use `app.output_type` from `apply()`.
- `output_type=str` = text mode (no rf/tools); the substrate for grammar/regex/choice.
- `extra_body` (in `OpenAIChatModelSettings`) lands verbatim top-level → the vLLM hook.
- The OpenAI `model` field is the **LoRA adapter selector** (one `OpenAIChatModel` per adapter).
- `OpenAIProvider(base_url=, api_key=, http_client=)` accepts an httpx client → inject capture.
- Concurrent `asyncio.gather(agent.run(...))` batches (≈2.4× on the 4-slot server). [Phase 3]
- `max_tokens` is sent as `max_completion_tokens`. `NativeOutput(model, strict=...)` exists.

## 6. Phase 2 scope (per CONCEPT §8 phase 2)

Build **Backend + AgentProfile + StructuredAgent** (Fleet/RoutingTable is Phase 3,
Executor is Phase 4). New files (CONCEPT §7 layout):

### `backend.py` (the ONLY module importing `pydantic_ai.models.openai`)
```python
class BackendCaps(BaseModel):
    xgrammar: bool = True              # honors XGrammar grammar/regex/choice via extra_body
    lora: bool = True                  # selects adapters via the model field
    server_default_backend: bool = True  # XGrammar set via server flag, not per-request

class Backend(BaseModel):
    base_url: str
    api_key: str = "sk-none"
    default_model: str
    caps: BackendCaps = BackendCaps()
    # runtime helpers (PrivateAttr / methods, not serialized):
    #   model_for(adapter: str | None) -> OpenAIChatModel  (adapter or default_model)
    #   optional capture: hold/return a RequestCapture; wire its .client() as http_client
```
**Capability gating** — raise `BackendCapabilityError` at agent-build time when:
- `decoder.mode in {"grammar","regex","choice"}` and `not caps.xgrammar`
- `profile.adapter is not None` and `not caps.lora`
(json_schema works on any OpenAI-compatible server, so it is never gated.)

### `profile.py`
```python
class AgentProfile(BaseModel):
    name: str
    adapter: str | None = None            # LoRA name; None -> backend.default_model
    instructions: str
    output_type_ref: str | None = None    # dotted "pkg.module:ClassName"
    decoder: DecoderSpec | None = None     # only to override a non-ConstrainedOutput type
    policy: str | None = None             # carried through; used in Phase 4
    model_settings: dict[str, Any] = {}   # temperature, max_tokens, seed, ...
```
- Resolve `output_type_ref` via importlib (`module:Name`); error clearly if missing.
- If the resolved type is a `ConstrainedOutput`, use its `.decoder_spec()`; else require
  `profile.decoder`. If neither, default to json_schema only when the type is a BaseModel.

### `agent.py`
```python
@dataclass
class AgentResult(Generic[OutputT]):
    output: OutputT
    usage: Any                 # from the PydanticAI result (RunUsage)
    request_body: dict | None  # last captured request, when capture is enabled
    raw: Any                   # the AgentRunResult (escape hatch)

class StructuredAgent:
    profile: AgentProfile
    @property
    def agent(self) -> pydantic_ai.Agent: ...   # escape hatch
    async def run(self, prompt: str, **kw) -> AgentResult: ...
    def run_sync(self, prompt: str, **kw) -> AgentResult: ...
```
Build path (mirror the test's `_build_agent`, now real):
`Backend.model_for(profile.adapter)` + `DecoderSpec.apply(output_type)` →
`Agent(model, output_type=app.output_type, model_settings=OpenAIChatModelSettings(extra_body=app.extra_body, **profile.model_settings), instructions=profile.instructions)`.

A factory ties it together (e.g. `Backend.build(profile) -> StructuredAgent`, doing the
cap checks), so callers don't import pydantic_ai directly.

## 7. Tests to add (keep GPU-free + one live)

- `test_backend.py` — cap gating raises `BackendCapabilityError` for grammar/regex/choice
  when `caps.xgrammar=False`, and for an adapter when `caps.lora=False`; json_schema never gated.
- `test_profile.py` — `output_type_ref` resolution (success + missing module/attr errors);
  ConstrainedOutput → decoder auto-derived; plain Model needs `decoder` or defaults to json_schema.
- `test_agent.py` — `StructuredAgent.run` against `MockOpenAI`: json_schema returns a
  validated model; regex returns a str; `AgentResult.request_body` captured; `.agent` exposed.
- Mark a **`live`** test (skipped by default) that runs the json_schema round-trip against
  `$LLM_BASE_URL` (reproduces the spike). Gate with an env check or pytest marker.
- Refactor `test_wire_shapes.py` to build via `StructuredAgent`/`Backend` instead of the
  inline `_build_agent`, proving the real path emits the same wire shapes.

## 8. Open questions (decide or defer in Phase 2)

- **AgentResult.usage**: expose PydanticAI's `result.usage()` as-is (lean: yes, keep raw too).
- **Capture wiring**: opt-in per `Backend` (e.g. `Backend(..., capture=True)` builds providers
  with a `RequestCapture` client). Default off for live use.
- **Strict-mode schema rewriting** (CONCEPT §10 open #1): pass `strict=spec.strict` to
  `NativeOutput` for now; full `additionalProperties:false`/all-required rewrite can wait.
- **choice → Literal coercion** (open #2): defer to Phase 3 (routing) unless trivial.
- **vLLM structured-outputs flag** (cutover, Phase 5): out of scope for Phase 2.

## 9. Phase 2 acceptance criteria

1. `Backend.build(profile)` returns a `StructuredAgent` whose `.run` yields a validated
   command object from the live server via `response_format` (json_schema), not the tool path.
2. grammar/regex/choice agents emit `output_type=str` + correct `extra_body` (wire-asserted via capture).
3. A non-None `adapter` sets the wire `model` field; `default_model` used when adapter is None.
4. A backend lacking a capability raises `BackendCapabilityError` at build time.
5. `AgentProfile.output_type_ref` resolves; a `ConstrainedOutput` auto-supplies its decoder.
6. `AgentResult` carries `output`, `usage`, optional `request_body`, and `raw`; `.agent` exposed.
7. `ty check src`, ruff (check+format), and `pytest` all green; coverage not regressed.

## 10. Suggested task order

1. Confirm green baseline. 2. `backend.py` + `test_backend.py`. 3. `profile.py` + `test_profile.py`.
4. `agent.py` + `test_agent.py`. 5. Refactor `test_wire_shapes.py` onto the real build path.
6. Add the `live` test. 7. Update `__init__.py` exports. 8. ty/ruff/pytest green →
   branch, commit (no attribution), fast-forward `main`, push. 9. Update CONCEPT §10
   resolved/open and the memory index if anything material changed.

## 11. Commands cheat-sheet

```bash
devenv shell -- uv run --extra dev pytest -q
devenv shell -- uv run --extra dev ty check src
devenv shell -- uv run --extra dev ruff check src tests
devenv shell -- uv run --extra dev ruff format src tests
devenv shell -- uv run python .scratch/projects/01-xgrammar-concept/spike/run_spike.py   # live sanity
```
