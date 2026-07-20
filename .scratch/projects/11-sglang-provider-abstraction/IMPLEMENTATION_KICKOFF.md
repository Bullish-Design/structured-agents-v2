# Kickoff Prompt — Implement the Multi-Engine Plugin Refactor

> Paste everything below the line into a fresh session (working dir:
> `/home/andrew/Documents/Projects/structured-agents-v2`). It is self-contained.

---

## Role & task

You are implementing a well-specified, already-reviewed refactor of the `structured-agents` library. This
is **not** a design task — the design is settled. Your job is to execute `REFACTORING_GUIDE.md`
faithfully, verify each phase against the real toolchain, and land a series of green commits. Follow the
guide's code; do not redesign it. If you believe a step is wrong, stop and flag it rather than silently
diverging.

## The goal (one sentence)

Make the library support **three OpenAI-compatible inference engines — vLLM, SGLang, llama.cpp — as
selectable in-tree plugin modules**, chosen per `Backend` (e.g. `Backend(engine="sglang")`), with vLLM
as the default and its structured-output wire bytes preserved **exactly**.

Confirmed scope (do not expand it):
- **Select one engine per `Backend`.** NOT three engines running simultaneously.
- **In-tree built-ins only.** NO out-of-tree/third-party plugin discovery.

## Authoritative documents (read in this order, all under `.scratch/projects/11-sglang-provider-abstraction/`)

1. `REFACTORING_GUIDE.md` — **your primary spec.** Step-by-step, with exact code and per-phase verify
   commands. Implement this.
2. `ARCHITECTURE_REVIEW.md` — the *why* behind the design (what was rejected and the reasons). Read it so
   you don't "helpfully" reintroduce rejected machinery.
3. `SGLANG_ANALYSIS.md` — background evidence (vLLM coupling inventory, per-engine wire gap, the SGLang
   GGUF serving blocker). Reference, not instructions.

## Critical trap — read this before touching anything

The current library is package **`structured_agents` at v0.3.0** (`src/structured_agents/`). There is a
file in the repo root, `CODE_REVIEW_FINAL_REFACTOR_GUIDE.md`, that describes a **different, superseded**
codebase — package `structured_agents_v2` at v0.2.0, with classes `ClosedBackend`, `AgentProfile`,
`AgentSet`, `DualPathRuntime`, `StrictConfig`, `provider_extra`, `SecretStr` api_key. **None of those
exist in the current tree** (`grep -rn "ClosedBackend\|provider_extra\|structured_agents_v2" src/`
returns nothing). That guide is stale. **Ignore it.** Do not import from it, do not try to "compose" with
it, do not add `provider_extra`. Everything you build lands directly on the current `Backend`.

## The design you are implementing (summary — the guide has the code)

**Seam:** move the wire-shape decision out of the constraint codecs and into per-engine modules.

- `constraint.py`: the four codecs (`_Schema`/`_Regex`/`_Choice`/`_Grammar`) **lose `wire()`** and **gain
  a `kind` tag** (`ClassVar[str]`; also add `kind: str` to the `Constraint` protocol). `WireSpec` stays
  here. `parse()`/`check()`/`to_config()` are untouched.
- New `engine/` package: `base.py` (internal `Engine` protocol: `name`, `supports: frozenset[str]`,
  `render(constraint) -> WireSpec`), plus `vllm.py`, `sglang.py`, `llama_cpp.py`, and `__init__.py` with
  an internal `_BUILTINS` dict + `select(name)`. Each engine `match`es on the concrete constraint and
  reads its real fields.
- `agent.py`: `Backend` gains `engine: str | Engine = "vllm"`; the capability gate becomes
  `constraint.kind not in self.engine.supports` (and a LoRA check). **Delete `BackendCaps`** (and the now
  unused `from pydantic import BaseModel` import). Adapter→model-name mapping is unchanged.
- `__init__.py`: drop the `BackendCaps` export; keep `WireSpec`. Add **no** new public names.

**Public API delta = exactly one thing:** the `engine=` argument to `Backend`. Do NOT export or create
`Provider`, `Capabilities`, `ConstraintSpec`, `load_provider`, or `register_provider`.

## Hard guardrails (violating any of these means stop and reconsider)

- **Preserve vLLM bytes byte-for-byte.** `VLLMEngine.render` must reproduce today's exact `WireSpec`s:

  | Constraint | `output_type` | `extra_body` |
  |---|---|---|
  | `Schema(M)` | `NativeOutput(M, strict=True)` | `{}` |
  | `Regex(P)` | `str` | `{"structured_outputs": {"regex": P}}` |
  | `Choice(a,b)` | `str` | `{"structured_outputs": {"choice": [a, b]}}` |
  | `Grammar(G)` | `str` | `{"structured_outputs": {"grammar": G}}` |

  The golden test (`test_engine.py::test_vllm_bytes_are_unchanged`) is the guard. If it ever fails, the
  dialect drifted — fix it, don't update the expected value.
- **Do NOT reintroduce rejected machinery:** no `ConstraintSpec` second IR, no entry-point discovery, no
  public engine registry, no `Provider`/`Capabilities` public protocol, no `adapter_wire` method. (See
  `ARCHITECTURE_REVIEW.md` §Q1/§Q2/§Q5 for why.)
- **Do NOT touch** `plane.py`, `authority.py`, `approval.py`, `errors.py`, or `integrations/fornix.py` —
  they are verified LLM-neutral. (You *may* read `errors.py`; you will use `BackendCapabilityError` and
  `ConfigError` from it — both already exist.)
- **Default must stay `engine="vllm"`** so every existing caller, serialized `AgentSpec`, and the
  `MockTransport` test (`tests/test_agent.py`) behaves identically with no change.
- **SGLang and llama.cpp wire shapes are doc-derived and UNVERIFIED in this repo.** Ship them, but label
  them unverified in their module docstrings (the guide has the wording). Do NOT flip any default to
  them. Do NOT enable SGLang live tests — the target GGUF does not load in SGLang yet
  (`08-unsloth-gemma4-gguf-compatibility/`; risk R1).
- **`ty` override is moved, not deleted.** The `unresolved-import` override currently on `constraint.py`
  (`pyproject.toml:51-55`) must be repointed to `engine/**`, because the `from pydantic_ai.output import
  NativeOutput` import relocates there. Only remove it if `ty check src` is green without it.

## Toolchain & verification (run through devenv; do not skip)

```bash
devenv shell -- ruff check src tests
devenv shell -- ty check src
devenv shell -- pytest              # live tests auto-skip without SAV_LIVE=1
```

Conventions (`pyproject.toml`): line length 120, double quotes, `from __future__ import annotations`,
Python 3.13, ruff `E,F,I,UP,B`. Match the surrounding terse, dataclass/protocol-driven style.

Start by capturing a green baseline, then work phase by phase, running the phase's verify command before
moving on:

```bash
git switch -c 11-sglang-provider-abstraction
devenv shell -- pytest              # baseline must be green before you change anything
```

## Suggested commit sequence (from the guide)

1. Phase 1 (codecs drop `wire()`, gain `kind`) + Phase 2 (engine package) — engines unit-testable;
   `agent.py` temporarily red.
2. Phase 3 (`Backend` rewire) + Phase 4 (exports) — full suite green.
3. Phase 5 (tests, incl. the golden guard).
4. Phase 6 (live parametrize by `LLM_ENGINE`) + Phase 7 (README/marker/`ty` override).

Commit only when the user asks, or if you're told to proceed, branch first (already on a feature branch
above). End commit messages with the required `Co-Authored-By` trailer.

## Acceptance criteria (definition of done)

- `devenv shell -- ruff check src tests`, `devenv shell -- ty check src`, and `devenv shell -- pytest`
  are all green.
- `test_engine.py::test_vllm_bytes_are_unchanged` passes (vLLM wire preserved).
- `tests/test_agent.py` passes **unchanged**.
- `Backend(engine="sglang")` and `Backend(engine="llama_cpp")` construct; `Backend(engine="nope")` raises
  `ConfigError("Unknown engine ...")`.
- Capability gating works: llama.cpp + `Regex` raises `BackendCapabilityError`; llama.cpp + `adapter`
  raises `BackendCapabilityError`.
- `BackendCaps` is gone from `src/` and `__all__`; no new public names beyond the `engine=` argument.
- No `ConstraintSpec`, no `providers/` package, no `[project.entry-points]` table.
- `git grep -n "ClosedBackend\|provider_extra\|structured_agents_v2" src tests` is empty (you didn't drag
  in stale concepts).

## When to stop and ask

- If preserving vLLM bytes turns out to require touching `WireSpec`'s shape, or if pydantic-ai's
  `NativeOutput` signature differs from what the guide assumes.
- If `ty` or `ruff` flags the `match`-on-private-codec pattern in the engine modules in a way the guide
  didn't anticipate.
- If any guardrail above appears to conflict with making tests pass. Do not resolve such a conflict by
  weakening a guardrail on your own.

Deliverable: the implemented refactor on branch `11-sglang-provider-abstraction`, green across ruff/ty/
pytest, with the commit sequence above. Report what you ran and its output; state plainly if anything is
skipped or unverified.
