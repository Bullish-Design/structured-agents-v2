# Upstream source register — 2026-07-21

Access date: 2026-07-21. These are upstream/primary sources. A source proves the documented upstream capability only; it does not prove that this repository's exact model, quantization, plugin, launcher, GPU, or client path implements it correctly.

## vLLM 0.25.0 (the locally pinned deployment version)

- [Structured outputs](https://docs.vllm.ai/en/v0.25.0/features/structured_outputs/): the OpenAI-compatible server documents `choice`, `regex`, JSON Schema, context-free grammar, and structural tags under `structured_outputs`; backend `auto` may select among implementations. The legacy `guided_*` fields were removed in v0.12.0. Regex dialect depends on the chosen backend.
- [LoRA adapters](https://docs.vllm.ai/en/v0.25.0/features/lora/): an adapter is selected through the OpenAI-compatible `model` field; base and distinct adapter requests may run in parallel when `max_loras` is high enough. Runtime load/unload is explicitly warned as unsafe outside an isolated, fully trusted environment.
- [Server arguments](https://docs.vllm.ai/en/v0.25.0/cli/serve/): `max_loras` is the maximum number of LoRAs in one batch (default 1); `max_cpu_loras` must be at least that value; `max_lora_rank`, scheduling policy, maximum sequences, and token budget are deployment controls.
- [Production metrics](https://docs.vllm.ai/en/v0.25.0/usage/metrics/): `/metrics` exposes running/waiting requests, per-LoRA request state, tokens per engine step, queue time, latency, preemptions, and KV-cache utilization. These are the minimum signals for proving actual concurrency/continuous batching.

Assessment: **Verified upstream for vLLM 0.25.0**. The repository's exact active GGUF/plugin launcher remains a distinct qualification target; local launcher evidence currently excludes LoRA.

## SGLang (current upstream documentation)

- [Structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs): XGrammar is documented as the default grammar backend and supports JSON Schema, regex, and EBNF. The OpenAI-compatible examples use `response_format` for schema and top-level extra request fields `regex` and `ebnf`; only one constraint parameter is allowed per request.
- [LoRA serving](https://docs.sglang.io/docs/advanced_features/lora): SGLang documents different adapters for different sequences in one batch, `max_loras_per_batch` (default 8), `max_loaded_loras`, adapter eviction/pinning, dynamic loading, native batched `lora_path`, and OpenAI-compatible `base-model:adapter-name` selection. Overlap loading can reduce batching efficiency and is disabled by default.
- [Server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments): concurrency and scheduling controls include maximum running/queued requests, chunked-prefill and prefill batch budgets, priority scheduling, and overlap controls.
- [Production metrics](https://docs.sglang.io/docs/references/production_metrics): `--enable-metrics` exposes running and queued requests, token usage, throughput, TTFT, TPOT, and end-to-end latency.

Assessment: **Verified upstream, version-sensitive**. These current docs do not establish that the repository's historical SGLang 0.5.14 exact GGUF/Gemma-4 profile works. Local evidence says that profile failed before tensor loading and the service is currently inactive.

## XGrammar

- [Python Grammar API](https://xgrammar.mlc.ai/docs/api/python/grammar.html): current 0.2.4 docs define JSON Schema, regex, GBNF-style EBNF, Lark, and structural-tag grammars. Strict JSON mode rejects unspecified properties/items. `any_order=True` weakens presence and uniqueness checks, so it is not an interchangeable strictness setting.
- [Advanced topics](https://xgrammar.mlc.ai/docs/tutorials/advanced_topics.html): grammar compilation is multithreaded and cacheable, and `BatchGrammarMatcher` generates per-sequence masks in a batch.
- [Serialization](https://xgrammar.mlc.ai/docs/xgrammar_features/serialization.html): grammar/compiler serialization is versioned and rejects mismatched serialization versions.
- [Project repository](https://github.com/mlc-ai/xgrammar): the project documents integrations with both SGLang and vLLM and distinguishes XGrammar-2/current development from earlier releases.

Assessment: **Verified upstream**. The active vLLM environment has XGrammar 0.2.3; the library's main environment has no XGrammar unless the `grammar-check` extra is installed. Exact engine integration still needs live tests.

## PydanticAI and DBOS

- [PydanticAI DBOS integration](https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/): `DBOSAgent` wraps runs as workflows and model/MCP calls as steps. Custom tools and event handlers are not automatically durable; I/O tools require explicit DBOS steps. Agent/toolset identities must remain stable. Inputs/outputs use pickle, should stay small, and streaming has specific restrictions. Runtime MCP/dynamic toolsets are rejected; function toolsets can execute inline.
- [PydanticAI multi-agent patterns](https://pydantic.dev/docs/ai/guides/multi-agent-applications/): delegation, programmatic hand-off, and graph control are separate composition patterns. Usage limits are recommended for bounded agent loops.
- [PydanticAI message history](https://pydantic.dev/docs/ai/core-concepts/message-history/): histories can cross agents/models and have JSON adapters, but client-supplied history is trusted-state input that must be sanitized.
- [DBOS workflows](https://docs.dbos.dev/python/tutorials/workflow-tutorial): recovery resumes at completed checkpoints; steps are attempted at least once and are not rerun after completion; DB transactions commit exactly once. This does **not** make an arbitrary non-transactional external effect exactly once if a crash occurs after the effect but before step completion is recorded.
- [DBOS queues](https://docs.dbos.dev/python/reference/queues): queues are database-backed and expose global/per-worker concurrency, rate limiting, priority, partitioning, and runtime reconfiguration.
- [DBOS workflow/step decorators](https://docs.dbos.dev/python/reference/decorators): workflow serialization/argument validation and step retry/preemption behavior are explicit configuration.
- [DBOS FAQ](https://docs.dbos.dev/faq): reusing an existing workflow ID is a no-op returning the existing execution. Therefore the library must detect same-key/different-input conflicts itself if that distinction matters.
- [DBOS serialization](https://docs.dbos.dev/python/reference/contexts): Python defaults to pickle plus Base64. Persisted values must not be treated as a safe interchange format across untrusted boundaries.

Assessment: **Verified upstream and locally inspected at installed versions**. The current library queues the wrong callable and places authorization outside the keyed workflow, so it does not yet expose these primitives with the required semantics.

## llama.cpp boundary

- [GBNF guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md): llama.cpp constrains output with its own GBNF implementation and a subset of JSON Schema; this is not evidence of XGrammar.
- [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): current upstream supports preloaded LoRAs and per-request LoRA scale lists, but explicitly states that requests with different LoRA configurations are not batched together. Its OpenAI-compatible endpoints make no strong compatibility guarantee.

Assessment: **Verified upstream boundary**. The local active deployment is a useful GGUF/GBNF comparison backend, not a target for the required XGrammar plus heterogeneous multi-LoRA batching contract.
