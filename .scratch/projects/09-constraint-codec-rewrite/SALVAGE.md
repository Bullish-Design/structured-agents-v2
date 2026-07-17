# SALVAGE — what carries over verbatim vs is rewritten vs is dropped

v3 is greenfield, but v2 (v0.2.0) is a well-built, 124-test-green, review-hardened codebase. Some of
it is *empirically verified truth* that must survive byte-for-byte; some is a good idea in a shape the
codec makes obsolete; some is dead weight. This ledger is explicit so nothing valuable is lost and
nothing obsolete is cargo-culted.

Legend: **VERBATIM** (copy the exact bytes/logic) · **REWRITTEN** (same intent, new shape) ·
**DROPPED** (gone, with why).

---

## Load-bearing facts & techniques — VERBATIM

| Asset | v2 location | v3 home | Why verbatim |
|---|---|---|---|
| **Wire mode→body table** (`response_format` json_schema; `extra_body {"structured_outputs": {"regex"/"choice"/"grammar": …}}`) | `decoder.py` `DecoderSpec.apply` | body of `Constraint.wire()` (`constraint.py`) | Empirically captured (VERIFICATION.md); the crown jewel. Any drift breaks server enforcement. |
| **`response_format` json_schema body** (`{"type":"json_schema","json_schema":{"name","strict","schema"}}`) | `closed.py` + `decoder.py` | `wire/request.response_format` | Verified shape XGrammar constrains directly. |
| **Loopback-URL guard** (http + `{127.0.0.1, ::1}` only; **reject `localhost`**; no creds/query/fragment) + the "do not add localhost" comment | `closed._validated_loopback_url` | `wire/transport.LoopbackTransport` | Deliberate DNS-rebinding defense with an explicit "do not fix this" comment. Security-critical. |
| **Bounded-input limits** (`model` `[A-Za-z0-9._:-]{1,128}`; instructions ≤4096B; prompt ≤16_384B; `0<timeout≤600`) | `closed.py` inline guards | `wire/request` `BoundedStr` value objects | Exact limits Lodestar's threat model depends on. |
| **httpx wire-capture technique** (`event_hooks={"request":[hook]}` to grab exact on-wire bytes) | `capture.py` | `agent.py` (delivered via `Ok.wire`, DECISION N) | The mechanism that keeps the design wire-grounded. Only the *delivery* changes (ContextVar → return value). |
| **In-process ASGI mock** (tiny OpenAI-shaped app asserting method/path) | `tests/conftest.py` | `tests/conftest.py` | The test seam the whole suite rides; no network, deterministic. |
| **In-flight concurrency proof** (tracking ASGI app proving real overlap in `run_batch`) | `tests/test_fleet.py` | `tests/test_fleet.py` | Proves shared-client concurrency + per-run capture attribution — a technique, not just a test. |
| **`deploy/vllm/verify.sh`** (health→models→json_schema→xgrammar→lora) + `deploy/tower/` bootstrap | `deploy/` | `deploy/` | Live cutover already verified on tower (Turing/FlashInfer + Docker-over-SSH fixes). Cooperation is unchanged. |
| **`VERIFICATION.md`** (captured wire shapes; NativeOutput-vs-tool insight) | `02-library-wrapper/` | referenced by `constraint.py` docstrings + T2 | The empirical grounding document; not code, but load-bearing truth. |
| **`live` marker gated on `SAV_LIVE=1`** | `pyproject`/`test_live.py` | same | Keeps the live tower dependency out of default CI. |
| **Error-message discipline** (every error names the offending agent/spec + the remedy) | `errors.py` messages | `errors.py` | A real v2 strength; carry the style. |

---

## Good ideas, obsolete shape — REWRITTEN

| v2 asset | Becomes | Why the shape changed |
|---|---|---|
| `ConstrainedOutput` (BaseModel subclass + `__decode_mode__`/`__regex__`/`__choices__`/… dunders) | `Schema/Regex/Choice/Grammar` constructors returning `Constraint[T]` | The constraint becomes a **value**, not an emergent property of dunders on a subclass with dead fields (concept §4). No subclassing; model classes stay plain `BaseModel`. |
| `DecoderSpec` + `DecoderApplication` | `Constraint[T]` + `WireSpec` | The outbound half of the codec, now on the same value as the inbound half. |
| `StructuredAgent._guard` (bare-string regex/choice validation) | `Constraint.parse` | The inbound half; now co-located with `wire`, honest (returns the literal for `Choice`), and **always run** (kills B4). |
| `AgentProfile` (string `output_type_ref` + `importlib`) | `AgentSpec[T]` (carries `Constraint[T]`) + `config.spec_from_config` | Typed code path primary; the string→code vector localized to one allowlist-gated function (DECISION K). |
| `AgentProfile.instructions: str` | `AgentSpec.context: Context` (a lone str = one `PREFIX` segment via `Context.of`) | Input becomes the cache-cooperative axis (concept §22); backward-compatible sugar. |
| `AgentProfile.adapter: str` | `AgentSpec.adapter: Adapter \| str` | Gains `source`/`base_model` for provisioning + cache-namespacing (DECISION Q). |
| `Backend.model_settings: dict` + `_warn_unknown_settings` | typed `Settings` dataclass | Setting typos become a **type error**, not a runtime warning on a silently-dropping TypedDict (kills the v2 finding). |
| `Executor`/`DryRunExecutor`/`AllowlistExecutor`/`Policy` | `Authorizer × Effector` + `Allowlist` + effectors + `authorize(a) >> e` | Decision × effect decomposed; `DryRun = a >> Null`, `Fornix = a >> FornixEffector` — compositions, not subclasses (kills B1/B2/B5, answers "where does fornix go"). |
| `AgentResult` / `BatchResult` / `RoutedResult` / `RoutedExecution` (four result types) | `Outcome[T]` (one spine, class + subclasses + methods) | Decisions-as-data, uniformly; one `then`-composable result (concept §5, encoding per S2). |
| `RequestCapture` + module `ContextVar` sink (public data-flow) | `Ok.wire: RequestRecord \| None` | Capture delivered as data on the result; ambient global state removed (kills A5 misattribution/unbounded — DECISION N). The event-hook technique stays; only delivery changes. |
| `closed.py` (bespoke: re-implements loopback/bounded/response_format/one-shot/detail-free) | `closed_backend()` preset over shared `wire/`+`constraint` | Same guarantees byte-for-byte, zero duplicated validation (DECISION L). Adds a v2-signature shim for Lodestar (DECISION H). |
| `dual_path/` (parallel DBOS runner) | `observe/` (pipeline `Observer`s) | Reuses the `Agent[T]`/`Outcome[T]` spine; the persisted artifact is an `Outcome`, idempotent in a DBOS step keyed on `run_id` (fixes v2 §C durability gap). |
| `RoutingTable` | `Router` | Same spirit (serializable, `Literal`-coverage-validated at build); slimmer name, `set_routing` validation kept. |
| `BackendCaps{xgrammar, lora}` | `+ chunk_cache` | New capability for position-independent KV reuse (concept §22). Dead `server_default_backend` cap **dropped** (v2 §C: declared, never read). |

---

## Dead weight — DROPPED

| Dropped | Why |
|---|---|
| **`grail` dependency** | Zero imports in v2 `src/`/`tests/`; a leftover from the archived 01/02 direction; drags a git dep + a pre-release native Rust interpreter (review §F). Never enters v3's `pyproject`. |
| `ConstraintViolationError` (as an **exception**) | Becomes the `Violated` **outcome** variant (DECISION B/M) — the whole point of the spine. |
| `DecodeMode = Literal[...]` central union + the `if mode ==` ladder | Replaced by the `Constraint` Protocol seam — open-to-extension-by-addition, no central enum (concept §23; kills the "closed to new modes" defect). |
| `server_default_backend` cap | Declared, documented, never read (v2 §C). Removed; if per-request `guided_decoding_backend` is ever needed it rides `Settings.extra_body`. |
| `DualPathDecodeMode = Literal["json_schema"]` narrowing | Observers record whatever `Outcome` they see; no mode-narrowing needed. |
| The `AgentProfile.decoder` vs `ConstrainedOutput` conflict path (v2 raised `ConfigError`) | No longer expressible — a `Constraint` is one value; there is no second decoder to conflict with it. |
| v2's `raw.usage()` **call** | `raw.usage` is a property in the pinned 2.11.0 (S3); accessed directly. The `callable()` shim is unnecessary at floor `>=2.11` — A1 structurally gone. |

---

## Fornix — the settled disposition (review §G)

- **Not a dependency, not an executor subclass.** `FornixEffector` (in `integrations/fornix.py`) is a
  stdlib-`subprocess` boundary that serializes a validated command to **argv** (never a shell string —
  fornix's `Item.cmd` is validated argv) and shells `fornix box --check … -- <argv>`, parsing the JSON
  `Result` into `Effect`. Net dependency delta: **zero**.
- Composes as `authorize(Allowlist(...)) >> FornixEffector()` — the full pipeline **XGrammar constrains
  syntax → pydantic validates the command → allowlist authorizes → fornix contains + captures the diff
  → `apply` promotes under a lock**. For non-repo commands, plain `Subprocess` remains the right path;
  fornix's fork/diff/apply machinery is dead weight there.
- Fornix's own in-repo `CODE_REVIEW.md` is **stale** (reviews a demolished Phase-2 codebase); do not
  inherit its conclusions. Fornix is effectively NixOS-only with a provisioned substrate — hence an
  optional app-layer integration behind the `[fornix]` seam, never in the lean core.

---

## Net dependency ledger v2 → v3

- **Out:** `grail` (+ its git-dep + native Rust interpreter transitive pin).
- **In:** nothing new to the *core* (fornix never enters; it's a subprocess boundary).
- **Moved to extras:** `pydantic-ai-slim[openai]` → `[agent]` (so `closed` installs pydantic-ai-free,
  DECISION I.2); `xgrammar` → `[grammar-check]`; `dbos`+**`psycopg`** (now declared) → `[observe]`.
- **Pinned:** `pydantic-ai-slim >=2.11,<3` (S3; kills the unbounded-range A1/D3 bite).
- **Lean core:** `pydantic + httpx` only.
