# DECISIONS — structured-agents v3 (the Constraint-Codec rewrite)

Every decision the planning session must resolve, each as **recommendation · rationale ·
rejected alternatives**. The success metric is architectural coherence; implementation cost is
explicitly *not* a tie-breaker (per the kickoff). Decisions grounded in a spike cite it; see
`SPIKES.md`.

Ordering: concept's §18 open questions (A–G) first, then repo/strategy questions (H–K), then
decisions this session **surfaced** that the kickoff didn't enumerate (L–R).

---

## A. Depth of pydantic-ai coupling → **Keep pydantic-ai as the Layer-2 model loop.**

**Recommendation.** Confine `pydantic_ai` to `agent.py` (Layer 2) as the sole importer, exactly as
v2 does. `Constraint.wire()` produces `output_type` + `extra_body`; pydantic-ai runs the model
loop (retries, message handling, `NativeOutput` response_format enforcement + validation);
`Constraint.parse()` normalizes the result. Do **not** own the loop on `wire/`.

**Rationale.**
- The layering already makes the dependency *swappable*: everything below (`wire/`, `constraint`)
  and above (`authority`, `fleet`, `observe`) is framework-agnostic. The coupling is one module.
- Re-implementing retry/message/tool-calling machinery on `wire/` buys *uniformity of the parser*
  (every mode parsed by `Constraint.parse`) at the cost of re-deriving a large, subtle, well-tested
  subsystem. That is a worse trade even under an "elegance-over-effort" mandate, because the loop is
  **not part of this library's thesis** — constrained decoding + authority is. Owning the loop
  dilutes the concept.
- Spike S3 confirms 2.11.0 gives exactly the two `output_type` shapes the codec needs and a single
  `.usage` property access — the coupling surface is small and stable.
- **The `closed` path already proves we can go pydantic-ai-free where privacy demands it** (Layer
  0+1 only). So we get the uniform-parser benefit *exactly where it matters* (closed) without
  paying for it everywhere.

**Rejected.** *Own the loop on `wire/`, drop pydantic-ai.* Maximal uniformity, but re-implements
retries/streaming/message coalescing/tool-approval — a second framework masquerading as a parser
change. Revisit only if pydantic-ai's output-typing story regresses or a second runtime is needed;
the layer boundary keeps that option open (Layer 2 is replaceable without touching any other layer).

---

## B. The `Outcome[T]` spine → **USER DECISION (2026-07-17): the lighter `Ok`/`Failed` variant for `run`; `Denied` added only on the executed pipeline. Encode as a generic base class with method combinators (spike S2 — orthogonal to variant count).**

> **Resolved by the user**, overriding this session's original "commit to the full four-variant
> spine" recommendation. The user chose the pragmatic, more idiomatic-Python variant (the concept's
> own §18-B option (b)). Rationale below reflects the *decided* design.

**Decided design.**
- **Generation spine:** `Outcome[T] = Ok[T] | Failed` — `Agent.run`/`run_batch` return this. A run
  either produced a valid typed `T` (`Ok`) or it didn't (`Failed`).
- **`Violated` folds into `Failed`.** When `parse()` rejects the raw output (the backend didn't
  enforce the constraint), that is a `Failed` carrying a distinct error type
  `ConstraintViolation(Exception)` — so "backend not enforcing" stays *diagnosable* (catch/inspect
  the error type) without being its own top-level variant. One binary result at the common call site.
- **`Denied` lives only on the executed pipeline.** `fleet.execute(...)` returns
  `Ok[Effect] | Denied | Failed` — because an authority **denial is not a failure** (the safety
  boundary working as intended); collapsing it into `Failed` would be a lie. Denial stays *data*
  exactly where it occurs (concept invariant #4 preserved *where it matters*), and `run` stays binary.
- **Encoding (unchanged from the spike finding):** `Outcome[T]` is a **generic base class** with `Ok`/
  `Failed` (and `Denied` on the execute path) as subclasses; combinators (`then`/`map`/`unwrap`/
  `value_or`/`is_ok`) are **methods**. This is required regardless of variant count — see S2.

**Rationale (empirical — S2, still load-bearing).** Under the repo's checker (`ty` 0.0.46), a bare
`type Outcome[T] = Ok[T] | Failed` alias **still fails** to deliver typed consumption via `match`/
`isinstance`/`TypeGuard` (all degrade `T` to `@Todo`/`object`/`Unknown`). The **class-with-methods**
encoding types perfectly — `oc.unwrap() → Plan`, `oc.map(lambda p: p.argv) → Outcome[list[str]]`
with `p` correctly `Plan` — because `T` flows *forward* from the class parameter. The spike finding
is **independent of how many variants there are**; it governs the *encoding*, which the user's
lighter variant keeps.

**Why the lighter variant is coherent (not the v2 wart).** v2's wart was declining *the same
domain event* two different ways (raise in `BaseExecutor.run`, data in `route_and_execute`). This is
different: `run` has exactly one way to not-succeed (`Failed`), and `execute` adds *one more axis*
(authority) with exactly one representation for it (`Denied`, as data). There is no
same-event-two-ways split — each event has one canonical shape. `Violated`-as-a-`Failed`-subtype
keeps the "backend didn't enforce" signal without a fourth top-level variant.

**Rejected (now).** *(a) Full four-variant spine* — the session's original recommendation; the user
chose fewer variants for idiom/ergonomics. *(b) Fold `Denied` into `Failed`* — would misrepresent a
policy denial as an error. *(c) Bare union alias* — breaks typed consumption under ty (S2).

**Rejected.** *(a) Bare union `type` alias* — breaks typed consumption under ty (S2a). *(b) Lighter
Ok/Failed only* — resurrects the raise-vs-data split. *(c) Exceptions throughout* — the v2 status
quo the review faulted.

---

## C. Heterogeneous-fleet typing → **`Fleet` is `Agent[Any]`-valued by nature; re-narrow via `fleet[name] -> Agent[T]`. No typed router machinery.**

**Recommendation.** A fleet holds specialists with *different* `T`; there is no single `OutputT` for
the collection, so `Agent[Any]` is **honest, not a defect**. Concreteness returns the instant you
pull a named agent: `fleet["git_ops"]` is typed `Agent[GitCommand]`. Do not build a `Router[Enum]`
that narrows dispatch to specialist types.

**Rationale.** A typed router would need a dependent map from route-literal → specialist type — a
lot of variadic-generic machinery to type a control-flow edge whose payload is `Any` anyway once it
crosses the routing boundary (the router's *output* is a route string, not the specialist's `T`).
The value is marginal and the machinery is exactly the "typing-around" the kickoff warns against.
`fleet[name]` re-narrowing gives real static types at every site that has a concrete type to give.

**Mechanism for `fleet[name] -> Agent[T]`.** `__getitem__` returns `Agent[Any]`; provide a typed
accessor `fleet.get(name, constraint=Schema(GitCommand)) -> Agent[GitCommand]` (or
`fleet.typed(name, GitCommand)`) for call sites that want the narrow type without a `cast`. The
plain `fleet[name]` stays `Agent[Any]` for dynamic use. See DESIGN.md §fleet.

**Rejected.** *`Router[Enum]` typed dispatch.* Machinery ≫ payoff; the routed payload is `Any` at
the boundary regardless.

---

## D. Streaming → **Out of scope for v3.0; `Outcome[T]` stays batch-shaped. Reserve the seam.**

**Recommendation.** Constrained decoding for these agents is batch-shaped (a plan, a command, a
choice — consumed whole). Do not add a streaming sibling to `Outcome[T]` now. Keep the door open:
`Agent.run` is the only method; a future `Agent.stream(prompt) -> AsyncIterator[Delta]` that folds
to `Outcome[T]` is an additive method, not a redesign.

**Rationale.** Streaming a *constrained* output whose value is validated/guarded as a whole
(`parse`) has little product value and complicates the codec (partial parses). Nothing in the three
axes or the authority pipeline needs it. Adding it later costs one method + one `Delta` type; the
`Outcome` spine is unaffected.

**Rejected.** *Streaming sibling now.* Speculative; complicates `parse`/`Violated` semantics for no
current consumer.

---

## E. Tool/function-calling agents → **Out of scope. `Constraint` expresses `response_format`/`extra_body` only, never a tool schema.**

**Recommendation.** v3 uses the `NativeOutput`/`str`+`extra_body` substrate exclusively (the
verified `response_format` path, VERIFICATION.md §1). A `Constraint` is *not* a tool definition.
Tool-using agents are a different abstraction; if ever needed, they attach at Layer 2 as a separate
spec kind, not by overloading `Constraint`.

**Rationale.** The whole wire-grounded design rests on "PydanticAI's default is the function-calling
*output tool*, which we deliberately avoid by applying `NativeOutput`" (VERIFICATION.md). Letting a
`Constraint` mean "a tool schema" re-opens the exact ambiguity the library was built to close, and
tools bring approval/multi-step-loop semantics orthogonal to constrained decoding.

**Rejected.** *`Constraint` doubles as a tool schema.* Category confusion; conflates constrained
output with tool-calling control flow.

---

## F. `Choice` variadic generics → **`def Choice[S: str](*options: S) -> Constraint[S]`. The concept's `Literal[*Opts]` form is rejected.** (spike S1)

**Recommendation.** Use a single bounded `TypeVar` `S: str`. `ty` infers the join of the argument
literals: `Choice("keep","skip") : Constraint[Literal["keep","skip"]]`, and `parse` returns the
literal — no `TypeVarTuple`, no explicit param, no `cast`.

**Rationale (S1).** The concept's `Choice[*Opts](*options: *Opts) -> Constraint[Literal[*Opts]]` is
a **category error** `ty` rejects with `invalid-type-form` (Literal wants values, not a type-tuple)
and yields `Constraint[Unknown]`. The `S: str` form is *simpler and stronger* — it closes v2's
deferred choice→Literal coercion (v2 open question #2) statically and for free.

**Rejected.** *(a) TypeVarTuple + `Literal[*Opts]`* — does not compile (S1). *(b) `*options: str ->
Constraint[str]`* — honest but discards the literal, needlessly. *(c) explicit `Choice[L](...)`* —
forces a redundant annotation on the caller.

---

## G. Multi-turn sessions & the context axis → **Ship the neutral per-segment `Context` model (settled, §22.1/§22.8) but NOT a `Session` sibling in v3.0. `Context` is single-shot; `Session` is a recorded deferral with a reserved seam.**

**Recommendation.** Land `Context`/`Segment`/`Reuse` as the input axis (per-segment cache policy;
`PREFIX` default, `CHUNK` opt-in, `NONE` for the query). A lone instruction string is a one-segment
`PREFIX` context (migration-compatible). Do **not** build `Session`/`Conversation` (growing-history
KV reuse) in v3.0 — flag it (concept §22.8) as the next axis feature.

**Rationale.** The neutral per-segment model is the load-bearing anti-trap decision and is already
settled (concept §18-8). A `Session` is a genuinely bigger abstraction (identity of a *growing*
prefix, turn threading) whose payoff (PIC on conversation history) is real but orthogonal to the
single-shot `Agent.run` that every other axis composes with. Shipping `Context` now with `id` as a
latent per-segment slot makes `Session` an additive assembler + a `Reuse`/threading policy later,
not a redesign.

**Rejected.** *Build `Session` in v3.0.* Scope creep that would delay the codec spine (the actual
thesis) for a feature with no current consumer; the seam is preserved so nothing is lost by waiting.

---

## H. Repo & name → **New repository, new name. v2 stays frozen at v0.2.0 for Lodestar; v3 is a clean fleet-sibling.** Proposed name: **`constric`** (fallback: `gridiron`, `xgrammar-agents`→ no).

**Recommendation.** A **new repo**, not a branch of `structured-agents-v2`. v0.2.0 remains the
frozen, tagged, Lodestar-pinned artifact; v3 is a greenfield sibling in the fleet. The public import
package is `structured_agents` (dropping the `_v2`).

**Naming.** The fleet encodes role in the name (AGENTS.md families). This library is a *library
primitive* about constrained decoding — the `*dantic` family is "Pydantic-based building blocks."
Candidates evaluated:
- **`constric`** *(recommended)* — "constrain" + the `-ic` primitive feel; short, unclaimed in the
  fleet, evokes the thesis (constraint is the linchpin). Not a `*dantic`/`*man` role word, which is
  correct — it is neither a manager tool nor a generic pydantic block; it is *the constrained-agent
  library*.
- `codex`/`codec-*` — collides with unrelated products; avoid.
- keep `structured-agents` (drop `-v2`) — clear and descriptive; **acceptable fallback** if the
  fleet prefers descriptive over coined names. Import stays `structured_agents` either way.

**Coexistence & Lodestar migration.**
1. v2 repo: tag `v0.2.0` is the last release; put it in maintenance (security-only). Lodestar stays
   pinned there.
2. v3 repo: implements `closed_backend(...)` with **byte-for-byte the same guarantees** (H-invariant
   #8). Lodestar's migration is: swap `from structured_agents_v2.closed import ClosedBackend` →
   `from structured_agents.closed import closed_backend`, adjust the constructor call, re-run
   Lodestar's own closed-path tests (which assert the guarantees). Because Lodestar consumes *only*
   `closed`, the blast radius is one import + one constructor shape (see SALVAGE.md → closed).
3. The v3 `closed` preset ships a **thin compatibility shim** `ClosedBackend` class matching the v2
   constructor keyword signature, so Lodestar can migrate with a near-zero diff and drop the shim
   later. (Recorded as a build item in PHASES.md Phase 4.)

**Rejected.** *(a) Rewrite on a v2 branch.* Muddies the frozen Lodestar artifact and invites
accidental API drift into the pinned line; a clean repo makes the "v2 is frozen" guarantee physical.
*(b) Reuse the `_v2` package name.* The rewrite is a new major identity; the `_v2` suffix was always
a placeholder.

> **User decision (2026-07-17): HELD — the user is choosing the name themselves.** Phase 0 (repo
> genesis) is **blocked** until the name is settled. `constric` remains the session's suggestion, not
> a derivation. Everything else in the plan is name-agnostic: the import package is `structured_agents`
> regardless, and no design decision depends on the distribution name. New-repo (not a v2 branch)
> stands as the recommendation unless the user says otherwise.

---

## I. Package / extras layout → **Lean core = `pydantic` + `pydantic-ai-slim[openai]` (pinned `>=2.11,<3`). Everything optional is an extra.**

**Recommendation.**
```
[project.dependencies]         pydantic>=2.11 ; pydantic-ai-slim[openai]>=2.11,<3 ; httpx  (transitive, but declare)
[project.optional-dependencies]
  grammar-check = [xgrammar]           # client-side compile check; pulls torch/CUDA — dev/CI only
  observe       = [dbos>=0.26, psycopg[binary]>=3]   # dual-path/eval capture; Postgres+DBOS
  fornix        = []                    # no dep — stdlib subprocess integration; extra exists only to name the seam
```
- **Core installs pydantic-ai-slim[openai], not full pydantic-ai** — v3 uses only the OpenAI model
  path (single-importer, VERIFICATION.md).
- `psycopg` is **declared explicitly** in `observe` (v2 finding D2: it was arriving transitively via
  dbos — a latent break).
- `grammar-check` stays optional (VERIFICATION.md §2: xgrammar pulls ~2 GB torch/CUDA).
- Pin `pydantic-ai` to `>=2.11,<3` (S3 + v2 finding A1/D3: unbounded range is what bit Lodestar).

**Rationale.** Most consumers want the agent path; making them opt into an `[agent]` extra is
friction for a benefit (below) that is mostly theoretical. The load-bearing privacy guarantee is
**import isolation** — `closed.py` never imports pydantic-ai — which holds regardless of packaging
(test-enforced, T8). So pydantic-ai in core does **not** weaken any closed guarantee; it only means a
closed-only consumer has pydantic-ai *installed but unused*.

### I.2 — Should pydantic-ai be an `[agent]` extra so `closed` installs without it? → **USER DECISION (2026-07-17): No. pydantic-ai in core by default.**

> **Resolved by the user**, overriding this session's original "yes, `[agent]` extra" recommendation.

**Decided design.** pydantic-ai-slim[openai] is a **core dependency**. There is no `[agent]` extra.
The lean core is `pydantic + httpx + pydantic-ai-slim[openai]`. `closed` still imports **no**
pydantic-ai (import isolation, T8) — that is the real guarantee and it is unchanged. What we give up
is only the marginal *installed-footprint* win for a closed-only consumer (Lodestar), which does not
justify taxing every agent-path consumer with an opt-in extra.

**Reconciling with the closed thesis.** The concept's "attack-surface/dependency-minimization" goal
(§8) is satisfied at the level that matters — closed's *code path* pulls in none of pydantic-ai, so
none of it can execute for a closed request. Whether pydantic-ai is *present on disk* for a
closed-only install is a packaging nicety, not a security property. If a truly minimal closed-only
distribution is ever needed, the import isolation means it remains *possible* later (a `[minimal]`
packaging could exclude it) without changing any code — but it is **not** the default and not a v3.0
requirement.

**Rejected (now).** *`[agent]` extra as default* — the session's original recommendation; the user
chose ergonomics (pydantic-ai available out of the box) over a marginal minimal-install win.

---

## J. Naming finalization → **Lock the public vocabulary; resolve the `Agent` collision by import discipline, not renaming the concept.**

**Recommendation — public vocabulary (locked):**
`Constraint[T]`, `Schema/Regex/Choice/Grammar`, `WireSpec`; `AgentSpec[T]`, `Backend`, `Agent[T]`;
`Outcome[T]` (`Ok`/`Failed`; `+Denied` on the executed pipeline; `ConstraintViolation` error subtype; `.then/.map/.unwrap/.value_or/.is_ok`); `Fleet`,
`Router`; `Authorizer`/`Effector`, `Decision`/`Effect`, `Allowlist`, `authorize`, effectors
(`Null`, `Subprocess`, `Fornix`, `DbosStep`); `Context`/`Segment`/`Reuse`/`Role`, fidelity
`EXACT/BLENDED`, `Adapter`, `AdapterProvider`, `ContextProvider`; `Retention`; `Observer`.

**The `Agent` collision with `pydantic_ai.Agent`.** v3's public `Agent[T]` is the runnable wrapper;
`pydantic_ai.Agent` is the loop underneath. They never share a namespace *except inside `agent.py`*
(the sole importer). Resolve by **import discipline, not concept-renaming**: in `agent.py`, import
the dependency as `from pydantic_ai import Agent as PydanticAgent` (or `import pydantic_ai; …
pydantic_ai.Agent`). The public name stays `Agent[T]` because it is the right domain word and users
never import `pydantic_ai.Agent`. The escape hatch is `Agent[T].raw -> pydantic_ai.Agent`
(unchanged from v2's `.agent`).

**Rationale.** Renaming the public runnable to avoid an import collision that only exists in one
private module would tax every user to spare one module a qualified import. Import discipline is the
proportionate fix and matches v2's single-importer rule.

**Rejected.** *Rename to `Runner`/`Runnable`/`StructuredAgent`.* `StructuredAgent` was v2's name and
is verbose; `Runner` loses the domain noun. The collision is contained to one file.

---

## K. Config/plugin registration → **One tiny registry per seam, localized to `config.py`, plus Python entry points. Import-execution gated by an explicit `allow_modules` allowlist at the single resolution function.**

**Recommendation.**
- **In-code plugins are registration-free** (structural typing — implement the `Protocol`, use it).
- **Config-loaded plugins** resolve a `kind` string → constructor via a per-seam registry:
  `register_constraint(kind, from_config)` etc., *and* a Python entry-point group
  (`[project.entry-points."constric.constraints"]`) so third-party packages extend without editing
  core.
- **The import-execution vector is localized to exactly one function** `constraint_from_config(d, *,
  allow_modules: frozenset[str])` (and its siblings), which refuses any dotted ref whose module
  prefix isn't in `allow_modules`. No `importlib`-on-data anywhere else (v2 finding C:
  `output_type_ref` was a latent import vector in the hot path).

**Rationale.** This satisfies "open to extension by addition, closed to modification" (concept §23):
no central `if kind ==` ladder in the core, seams are Protocols, and the *one* place strings become
code is explicit, gated, and out of the hot path (principle #7). Entry points let plugin *packages*
register without a central edit; the allowlist makes YAML-loaded config safe by construction.

**Rejected.** *(a) Global mutable registry touched from anywhere.* Re-creates the hot-path import
vector. *(b) No entry points, `register_*` only.* Forces every plugin consumer to import-for-side-
effect; entry points are the idiomatic seam.

---

## L. *(surfaced)* Do `Backend` and `ClosedBackend` merge or stay separate? → **Stay separate types; share the `wire/`+`constraint` primitives. `closed_backend()` is a preset factory, not a subclass.**

**Recommendation.** Keep two *types* with two *threat models*: `Backend` (rich: agents, capture,
adapters, all four constraints, capability gating, pydantic-ai) and the closed path (loopback-only,
json-schema-only, no capture/retention, detail-free, **no pydantic-ai**). They **share** the Layer-0
`wire/` transport+request+retention primitives and the Layer-1 `Constraint` codec — eliminating v2's
*duplicated* loopback/bounded-input/`response_format` logic — but they are not a subclass hierarchy.

**Rationale.** The concept is right that the *duplication* is the smell and the *dependency instinct*
(closed avoids pydantic-ai) is correct. A shared base class would leak the rich path's capabilities
(capture, raw, escape hatches) into the closed type's surface — exactly what `closed`'s scope
discipline forbids (the v2 test asserts `ClosedBackend` has **no** `agent`/`run_sync`/`build`/
`attach_transport`). Composition over shared primitives gives DRY without surface leakage.

**Rejected.** *`ClosedBackend(Backend)` subclass.* Would inherit escape hatches the closed threat
model must not expose.

---

## M. *(surfaced)* Where does client-side vs server-enforced validation live, and how is the distinction visible? → **A `Constraint`'s tier is visible in what `wire()` returns; `parse()` always runs.**

**Recommendation.** Make the server-enforced/client-validated distinction (concept §23) a
*structural* property: a constraint whose `wire()` emits an enforced key (regex/grammar/choice/
json_schema) is true constrained decoding; one whose `wire()` returns `output_type=str,
extra_body={}` and does all its work in `parse()` is client-side validation (`SemVer`, `IsoDate`,
`Json[T]`). **`parse()` always runs regardless** — it is the safety net when the server *claims* to
enforce but doesn't (v2 finding B4: unconstrained text must never reach the executor unchecked).

**Rationale.** This turns v2's "the guard that was never implemented, scattered from the spec" into
a first-class, always-present half of every codec, and makes the enforcement tier a readable
property of the value rather than tribal knowledge.

**Rejected.** *Trust the server for enforced modes, skip `parse` there.* Re-opens B4 — any backend
that silently ignores `extra_body` (a frontier API, a mis-capped vLLM, the test mock) floods the
authority boundary with unvalidated text.

---

## N. *(surfaced)* Capture delivery → **In the `Ok` outcome (`Ok.wire`), not a `ContextVar`. Keep the httpx event-hook *technique* to grab the bytes.**

**Recommendation.** Capture is opt-in per `Backend`; when on, the request record rides
`Ok.wire: RequestRecord | None`. Drop the ambient `ContextVar` sink from the public data-flow — the
record is data on the result, like everything else. Keep the httpx `event_hooks={"request": …}`
technique to obtain the exact on-the-wire bytes, and keep a **bounded** internal buffer, but the
*delivery* to the caller is the return value, not global state.

**Rationale.** v2's A5 (misattribution under concurrency, unbounded growth) was a *consequence of
ambient delivery*. Threading the record through `Ok` makes per-run attribution structural (the
record is built for *this* run and returned by *this* run) — the concurrency race cannot exist
because there is no shared "last" to read. The event-hook technique still needs a per-run correlation
to know which request belongs to the awaited call; a request-scoped correlation (httpx request
`extensions` tag, or a per-run sink still used *internally* but never read cross-run) is an
implementation detail behind the `Ok.wire` result. See RISKS.md R4.

**Rejected.** *Keep the `ContextVar`-sink as the public mechanism.* It is the source of A5 and the
"ambient global state opposite to the library's explicit character" the concept calls out.

---

## O. *(surfaced)* Where do capability errors sit vs `Outcome` variants? → **Capability/config mismatches are exceptions (`BackendCapabilityError`, `ConfigError`) at `build`; only *runtime* domain outcomes are `Outcome` variants.**

**Recommendation.** Keep the invariant "decisions are data; exceptions are for bugs" sharp by
partitioning: a mis-capped backend asked for `Grammar`, an adapter the server doesn't serve, a
config that names an un-importable module → **exceptions at `build`/config time** (programmer/config
error). `Denied`/`Failed` (incl. a `ConstraintViolation` inside `Failed`) are **runtime** outcomes of a *correctly built* pipeline.

**Rationale.** This is the concept's §9 stance, made explicit as a rule. It prevents the opposite
error (turning genuine programmer mistakes into silently-handled data), which would erode the "fail
fast on config" discipline v2 got right.

**Rejected.** *Everything is an `Outcome`, including config errors.* Would make `constric` swallow
bugs as data — the inverse of v2's raise-happy wart, equally wrong.

---

## P. *(surfaced)* `check()` on `Constraint` — required or optional, and when does it run? → **Optional `check()` with a no-op default; runs at `build` time when the `grammar-check` extra is present; `__pydantic_init_subclass__` timing bug is structurally gone because there is no subclass.**

**Recommendation.** `Constraint` has `check() -> None` defaulting to no-op. `Schema` implements it as
the xgrammar compile check (VERIFICATION.md §2); `Regex`/`Grammar` compile their pattern/EBNF;
`Choice` is a no-op. `Backend.build` calls `spec.constraint.check()` when the `grammar-check` extra
is importable. Absent the extra, it is skipped (server enforces anyway).

**Rationale.** v2's A2 (the check validated the *parent's* empty schema because `__init_subclass__`
fired before pydantic built the subclass core schema) is **structurally impossible in v3**: there is
no `ConstrainedOutput` subclass whose schema is built lazily — `Schema(FileEditPlan)` holds a
*fully-formed* model class and calls `FileEditPlan.model_json_schema()` on a complete type. The
timing hazard cannot recur. `check` at `build` (not class-definition) is also the correct time — it
is a backend-capability precondition, co-located with capability gating.

**Rejected.** *Eager check at constraint construction, mirroring v2's class-definition check.*
Re-imports the timing question and forces the xgrammar/torch import earlier than needed.

---

## Q. *(surfaced)* Adapter as `str` vs `Adapter` value, and where provisioning lives → **`AgentSpec.adapter: Adapter | str | None`; provisioning is an `AdapterProvider` on the `Backend`, never on the `Constraint`.**

**Recommendation.** Bare `str` is the simple case (becomes the wire `model` field). An `Adapter`
value (`name`, `source`, `base_model`) carries what providers need to resolve/provision and what the
cache namespace needs (`base_model + adapter` folds into `cache_salt`, invariant #10). Resolution
(logical name → served name; ensure loaded via e.g. vLLM `/v1/load_lora_adapter`) is an
`AdapterProvider.resolve` on the `Backend`. A runtime "unknown model" 404 becomes a
`BackendCapabilityError` at `build` (concept §21.4).

**Rationale.** Keeps the three axes orthogonal (concept §20): the adapter rides the `model` field,
the constraint rides `response_format`/`extra_body`, the context is the messages — switching any one
is a full cache hit for the others. Putting provisioning on the `Backend` (not the `Constraint`)
keeps the codec pure.

**Rejected.** *Adapter-as-`Constraint`, or provisioning on the codec.* The exact axis-conflation the
design exists to prevent (invariant #3).

---

## R. *(surfaced)* Does `Fleet.run_batch` keep the shared client + results-as-data? → **Yes to both; they are v0.2.0 fixes promoted to invariants.** One shared `httpx.AsyncClient` per `Backend` (with `aclose()`), `run_batch -> list[Outcome[T]]` (never `gather` that raises and loses siblings).

**Recommendation.** `Backend` owns exactly one `httpx.AsyncClient` (shared pool → real `run_batch`
concurrency), closed by `Backend.aclose()`/`Fleet.aclose()`. `run_batch` returns `list[Outcome[T]]`
— per-item failure is a `Failed` variant, never a raised exception that discards siblings
(generalizes v2's `BatchResult` fix; the `Outcome` spine makes it uniform).

**Rationale.** These were v2 section-C fixes shipped in v0.2.0; the `Outcome` spine makes
results-as-data the *only* shape, and the shared client is already the formalized lifecycle. Nothing
to re-litigate — recorded so the plan doesn't regress them.

**Rejected.** *Per-agent clients / raising `gather`.* The v2 pre-fix state (N pools, lost siblings).

---

## Decision index (one-line each)

| # | Decision | Verdict |
|---|---|---|
| A | pydantic-ai coupling | keep as Layer-2 loop, sole importer |
| B | Outcome spine | **USER: `Ok`/`Failed` for `run`** (+ `Denied` on execute; `Violated`→`Failed` subtype); class-with-methods encoding (S2) |
| C | fleet typing | `Agent[Any]` + `fleet[name]` re-narrow |
| D | streaming | out of scope; seam reserved |
| E | tools | out of scope; `Constraint` ≠ tool schema |
| F | Choice generics | `Choice[S: str](*o: S) -> Constraint[S]` (S1) |
| G | context/session | ship `Context`; defer `Session` |
| H | repo & name | new repo; **name HELD (user picking)** — Phase 0 blocked until settled |
| I | extras | **USER: pydantic-ai in core by default** (no `[agent]` extra); closed still imports none of it |
| J | naming | lock vocab; `Agent` collision → import discipline |
| K | config/plugins | per-seam registry + entry points; `allow_modules` allowlist at one function |
| L | Backend vs Closed | separate types, shared `wire/` primitives, no subclass |
| M | validation tier | visible in `wire()`; `parse()` always runs |
| N | capture | delivered in `Ok.wire`, not a ContextVar |
| O | cap errors | exceptions at build; Outcome variants only at runtime |
| P | check() | optional, runs at build under `grammar-check`; A2 structurally gone |
| Q | adapter | `Adapter\|str`; provisioning on `Backend`, not codec |
| R | run_batch | shared client + results-as-data (promoted v0.2.0 fixes) |
