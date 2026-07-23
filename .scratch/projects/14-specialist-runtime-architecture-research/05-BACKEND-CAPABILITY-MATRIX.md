# Backend Capability Matrix

Date: 2026-07-21

## Status rules

- **Verified**: this exact version, model format/quantization, plugin/launcher, hardware class, client mapping, and feature was exercised through the stated boundary; a dated artifact exists.
- **Experimental**: documented upstream or partially demonstrated, but the exact deploy profile or combined behavior lacks live evidence.
- **Unsupported**: explicitly unavailable, contradicted by evidence, intentionally outside scope, or blocked by the selected profile.

Statuses expire when any material profile dimension changes. A source link proves upstream documentation only; it does not promote a local cell to Verified.

## Configuration-specific status

| Profile | Base chat | Structured/XGrammar | Per-agent LoRA | Different LoRAs overlapping | Mixed constraint + LoRA | Operational evidence | Overall |
|---|---|---|---|---|---|---|---|
| Active vLLM 0.25.0 + XGrammar 0.2.3 + custom GGUF plugin + Gemma-4-12B QAT GGUF + current launcher | **Verified** (local health/model; preserved generation evidence) | **Verified** for schema/regex/choice/grammar in project-10 artifacts | **Unsupported** by current launcher | **Unsupported** by current launcher | **Unsupported** | Health/model current; prior generation artifacts; no LoRA/metrics load proof | Experimental specialist profile |
| Candidate vLLM 0.25.0 native model/adapter profile | **Experimental** | **Experimental**; [documented upstream](https://docs.vllm.ai/en/v0.25.0/features/structured_outputs/) | **Experimental**; [documented upstream](https://docs.vllm.ai/en/v0.25.0/features/lora/) | **Experimental**; depends on `max_loras` and memory | **Experimental** | Metrics are [documented upstream](https://docs.vllm.ai/en/v0.25.0/usage/metrics/); no exact artifact | First live qualification target |
| Historical SGLang 0.5.14 + attempted GGUF/Gemma-4 profile | **Unsupported** for this profile; startup failed | **Unsupported** absent base startup | **Unsupported** absent base startup | **Unsupported** | **Unsupported** | Failure artifacts only | Contradicted profile |
| Candidate pinned SGLang native model profile | **Experimental** | **Experimental**; XGrammar is [documented upstream](https://docs.sglang.io/docs/advanced_features/structured_outputs) | **Experimental**; [documented upstream](https://docs.sglang.io/docs/advanced_features/lora) | **Experimental**; same-batch adapters documented | **Experimental** | Metrics [documented upstream](https://docs.sglang.io/docs/references/production_metrics/); no local service | Required second target |
| Active llama.cpp local profile | **Verified** base service/state | **Unsupported** for XGrammar; GBNF is a different implementation | **Experimental** generally | **Unsupported** for required contract: upstream says different LoRA configs are not batched | **Unsupported** | One-slot current profile | Comparison/fallback only |

## Feature semantics by engine

| Concern | Neutral library contract | vLLM 0.25.0 | SGLang current docs | llama.cpp current docs |
|---|---|---|---|---|
| JSON Schema | `ConstraintKind.JSON_SCHEMA`, strictness explicit | `structured_outputs` JSON | OpenAI `response_format` | JSON-schema-to-GBNF subset |
| Regex | Neutral pattern plus declared dialect/profile | `structured_outputs.regex`; dialect backend-sensitive | top-level `regex`; one constraint/request | grammar conversion is not XGrammar-equivalent |
| Choice | Neutral finite values | `structured_outputs.choice` | no equivalence assumed until tested | no equivalence assumed |
| Grammar | Neutral grammar language identity | `structured_outputs.grammar`; backend selection may be auto | top-level `ebnf`; XGrammar default | native GBNF |
| Structural tags | Separate neutral kind | documented | do not claim without a pinned live test | unsupported for target |
| Adapter selection | Logical `AdapterIdentity` | rendered into `model` selector | native `lora_path` or OpenAI base/adapter selector | per-request adapter/scale list |
| Multiple LoRAs | Profile limits and evidence required | `max_loras` per batch; default 1 | `max_loras_per_batch`; docs default 8 | different configurations not batched together |
| Runtime adapter load | Trusted control plane only | upstream warns against production dynamic loading | dynamic load/eviction documented; still control-plane constrained | preloading/control endpoints are deployment concerns |
| Continuous batching | Proven using engine metrics | engine scheduler; metrics include running/waiting and tokens/iteration | scheduler; running/queued/throughput metrics documented | slot/batch behavior does not satisfy mixed-LoRA target |

## Three different meanings of “batch”

```text
logical library batch
  = many independently identified invocations + fan-in results

client concurrency
  = more than one HTTP/model call in flight, under backpressure

engine continuous batching
  = backend scheduler combines compatible active sequences/token steps
```

Only the third produces the intended GPU scheduling effect. Offline provider batch APIs are a fourth concept and are not required by this design. BAT-04 requires correlated client and engine metrics; throughput alone is insufficient because it cannot distinguish concurrency, caching, or shorter outputs.

## Required capability record

Every profile record must include:

```text
profile_id, status, evidence_timestamp, expiry
engine name/version/commit
Python/CUDA/driver/GPU class
base model identity + revision/digest
model format + quantization + plugin identity
adapter identities, ranks, target modules, load/eviction settings
constraint engine/version/backend and strictness
max running requests, token/batch budgets, LoRA limits
client request rendering version
artifact manifest and known limitations
```

The public API should return the record or a concise derived view, not a mutable set of backend names.

## Qualification grid

Each vLLM and SGLang candidate must pass all applicable cells independently.

| Dimension | Cases |
|---|---|
| Output mode | unconstrained, JSON Schema, regex, choice where supported, EBNF/grammar |
| Adapter | base, adapter A, adapter B, invalid adapter, wrong-base adapter |
| Load shape | 1 request; homogeneous concurrent; mixed adapters; mixed constraints; mixed adapters + constraints |
| Lifecycle | preloaded; controlled load; eviction/pinning if supported; restart |
| Failure | validation failure, backend 4xx/5xx, timeout, cancellation, saturation, adapter unavailable |
| Durability | first execution, same-key attach, different-input conflict, worker crash/replay |
| Observability | client in-flight, engine running/queued, tokens/iteration or throughput, queue/model latency, adapter identity |

A backend can be Verified for schema and Unsupported for multi-LoRA in the same profile. There is no single engine-wide boolean.

## Planning implications

1. Preserve the active vLLM/GGUF profile as a constrained-output control; do not retrofit arbitrary adapter paths into it.
2. Qualify a separate native vLLM profile for LoRA and mixed batching first, because the version is already pinned and upstream behavior is version-addressable.
3. Pin one SGLang version and a model/adapter artifact that fits available hardware before claiming any support. The historical 0.5.14 GGUF failure is a negative control, not a reason to infer current failure.
4. Keep llama.cpp outside the combined XGrammar/mixed-LoRA acceptance path. It remains useful for GBNF and GGUF comparisons.
