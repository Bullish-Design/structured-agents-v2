# SPIKES — empirical findings that ground the v3 plan

All spikes run in-devenv against the repo's real toolchain:
**Python 3.13.13 · ty 0.0.46 · pydantic 2.13.3 · pydantic-ai 2.11.0.**
Source in `spikes/`. Every wire-facing or type-facing claim in DECISIONS.md/DESIGN.md
traces to one of these results, not to a guess. `pyright`/`mypy` are **not** in the
devenv — `ty` is the sole checker and therefore the sole authority for the quality bar.

---

## S1 — `Choice` variadic generics (Decision F) → **RESOLVED, concept signature rejected**

**Question:** does `ty` synthesize `Constraint[Literal[*opts]]` from a variadic `Choice`?

**Files:** `spike_choice.py`, `spike2.py`.

| Candidate signature | `reveal_type(Choice("keep","skip"))` | verdict |
|---|---|---|
| `Choice[*Opts](*options: *Opts) -> Constraint[Literal[*Opts]]` *(the CONCEPT §4.1 form)* | `Constraint[Unknown]` + hard error `invalid-type-form: Type arguments for Literal must be … a literal value` | **REJECTED** |
| `Choice(*options: str) -> Constraint[str]` | `Constraint[str]` | honest but drops the literal |
| **`Choice[S: str](*options: S) -> Constraint[S]`** | **`Constraint[Literal["keep","skip"]]`** | **WINNER** |
| `Choice[L](*options: L) -> Constraint[L]` (caller supplies `L`) | correct only with explicit annotation | redundant for the caller |

**Finding.** `Literal[*Opts]` is a **category error** — `Literal` takes *values*, a `TypeVarTuple`
binds *types*; `ty` rejects it outright. The CONCEPT's own proposed constructor does not compile.
But a **single bounded `TypeVar` `S: str`** makes `ty` infer the *join* of the literal argument
types automatically: `Choice("keep","skip")` ⇒ `Constraint[Literal["keep","skip"]]`, and
`parse()` returns `Literal["keep","skip"]` — **no `TypeVarTuple`, no explicit type param, no
`cast`.** Runtime confirmed: `wire() == {"structured_outputs": {"choice": ["keep","skip"]}}`,
`parse("keep") == "keep"`.

**Consequence for the design:** the CONCEPT's variadic-`Literal` idea is replaced by
`def Choice[S: str](*options: S) -> Constraint[S]`. This is *cleaner* than the concept and closes
v2's deferred choice-coercion (open question #2) for free — the literal is real, statically.

---

## S2 — `Outcome[T]` typed consumption (Decision B) → **the bare-union encoding is broken; the class-with-methods encoding is correct**

**Question:** does the CONCEPT's flagship promise —
```python
match oc:
    case Ok(value=cmd):   # "cmd: GitCommand ← ty knows this"  (CONCEPT §15)
        print(cmd.argv)
```
— actually deliver a typed `cmd` under `ty`?

**Files:** `spike_outcome.py`, `spike2.py`, `spike3.py`, `spike4.py`.

### S2a — bare union alias `type Outcome[T] = Ok[T] | Denied | Violated | Failed` (the CONCEPT code block)

Every path that must recover `T` **backward out of the union** degrades:

| consumption form | `reveal_type` of the value | verdict |
|---|---|---|
| `match oc: case Ok(value=cmd)` | `cmd : @Todo` | **fails** |
| `if isinstance(oc, Ok): oc.value` | `object` | **fails** |
| `def is_ok(o) -> TypeGuard[Ok[T]]` then `.value` | `Ok[Unknown]`, `.value: Unknown` | **fails** |
| `fold(oc, ok=lambda p: p.argv, …)` | `Unknown` | **fails** |
| free `unwrap[T](oc: Outcome[T]) -> T` | `Unknown` | **fails** |

What *does* work on the union: the **outer** type of a forward bind — `then(oc, step2)` where
`step2: Plan -> Outcome[str]` reveals `Ok[str] | Denied | Violated | Failed` (correct), because
`U` flows *forward* from `step2`'s signature, never backward from `oc`. Non-generic arms narrow
fine (`Denied.reason: str`). Runtime `match` binds correctly in every case — **this is a static
inference gap in ty 0.0.46, not a runtime bug.**

### S2b — generic **base class** `Outcome[T]` with method combinators (`spike4.py`)

```python
class Outcome[T]:
    def map[U](self, f: Callable[[T], U]) -> Outcome[U]: ...
    def unwrap(self) -> T: ...
    def value_or[D](self, default: D) -> T | D: ...
class Ok[T](Outcome[T]): value: T
class Denied(Outcome[Any]): reason: str
```

| consumption form | `reveal_type` | verdict |
|---|---|---|
| `oc.unwrap()` | **`Plan`** | ✓ |
| `oc.map(lambda p: p.argv)` | **`Outcome[list[str]]`** (and `p` is `Plan` inside the lambda) | ✓ |
| `oc.value_or(None)` | **`Plan \| None`** | ✓ |
| `Ok(Plan(...)).value` | `Plan` | ✓ |
| inherited `ok.unwrap()` | `Plan` | ✓ |

**Finding.** When `Outcome[T]` is a **class** and consumption goes through **methods**, `T` flows
*forward* from the class parameter into every method signature — no union-narrowing is required, so
`ty` delivers the fully-typed value, including inside `map`'s lambda. The single-tag variant
(`spike4.py` Encoding B) works identically.

**Consequence for the design (a real departure from CONCEPT):** `Outcome[T]` is modeled as a
**generic base class with `Ok[T]`/`Denied`/`Violated`/`Failed` subclasses**, and the combinators
(`then`/`map`/`unwrap`/`value_or`) are **methods**, not free functions and not a bare `type` alias.
Runtime `match` on the subclasses stays available for humans (it works at runtime); the *typed*
path is the method API. This resolves the concept's internal contradiction (§5.1 writes `Outcome`
as a `class` with `.then`, but the code block above it writes `type Outcome[T] = …` — only the
class form typechecks). The "no-cast, typed end-to-end" promise **holds**, but *only* via this
encoding.

---

## S3 — pydantic-ai 2.11.0 codec surface (Decisions A, I) → **confirmed**

**File:** inline probe.

- `from pydantic_ai import NativeOutput` — importable in 2.11.0.
- `Agent(output_type=str)` **and** `Agent(output_type=NativeOutput(M))` are both accepted — the
  `WireSpec.output_type` split (str for bare-string modes, `NativeOutput(M)` for schema) is real.
- **`AgentRunResult.usage` is a `property`** (`isinstance(AgentRunResult.usage, property) is True`).
  So v3, pinned to `pydantic-ai>=2.11,<3`, accesses `raw.usage` **without calling it**. The v2 A1
  bug (`raw.usage()`) becomes structurally impossible: there is exactly one supported access shape,
  and it is the property. No `callable()` shim needed if the version floor is 2.11.

This grounds Decision A (keep pydantic-ai as the Layer-2 loop) and Decision I (pin `>=2.11,<3`).

---

## What the spikes did **not** need to test

- The **wire bodies** (`response_format` / `extra_body structured_outputs`) are already empirically
  captured in `../02-library-wrapper/VERIFICATION.md` and encoded in v2's `decoder.py`; they carry
  over verbatim (see SALVAGE.md). No re-capture needed to plan; a wire-shape regression test
  (TESTS.md) re-grounds them against the ASGI mock at build time.
- **Live vLLM** (XGrammar/LoRA) behavior is unchanged from v2's verified cutover
  (`deploy/vllm/verify.sh`); v3 cooperates with it identically.
