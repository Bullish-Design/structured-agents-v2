# Decisions — Project 17

## D1. Repo strategy: in-place refactor of `src/`
- Rip out the engine-provider abstraction (sglang/vLLM, project 11) and the
  OpenAI-compat engine framing.
- KEEP + repurpose: constraint codecs (01, 09), DBOS durable plane (10) for
  benchmark orchestration / runpod fan-out.
- Grow the llama.cpp core inside existing `src/`.

## D2. Flagship demo: multi-LoRA agent-router fleet
**The story:** many small **fine-tuned LoRA adapters**, each a custom router for
a different agent, served over ONE base model with **massively parallel batched
generation** — small specialized routers beating a big general model on
tool/function-call routing (latency + always-valid via xgrammar).

This single demo exercises every pillar:
- Grammar (xgrammar): routing output is valid-by-construction tool calls.
- Custom batching: batch requests across DIFFERENT LoRA adapters in flight.
- KV reuse: shared base-model prefix / per-adapter KV.
- Workflow hooks: per-request adapter selection + intercept.
- Benchmarking: router accuracy + tok/s + TTFT vs. a big general model.

### NEW KEY RISK — per-sequence LoRA in llama.cpp batching
llama.cpp applies LoRA adapters at the **context** level
(`llama_set_adapter_lora`, scale per context). Batching *different adapters in a
single `llama_decode` call* is NOT trivially supported the way vLLM's
multi-LoRA (punica/SGMV) is. Must verify current llama.cpp capability:
- Option (a): per-adapter contexts, our scheduler batches WITHIN an adapter and
  round-robins across adapters (simplest, still a great teaching artifact).
- Option (b): true mixed-adapter batch if/when llama.cpp exposes per-seq
  adapters — verify against the pinned llama.cpp version.
- This choice defines what "massively parallel multi-LoRA" can actually mean.
  **Verify BEFORE committing the batching design.** See ISSUES.

## Standing rules (from interrogation)
- Teaching wins on conflict with raw perf; perf is the subject, not the SLA.
- Pydantic at boundaries only; hot path uses plain/typed structs + numpy views.
- xgrammar means we own the logits→mask→sample loop (bypass llama's sampler).
- Batching win = scheduling/control/understanding, not beating C++ throughput.
- Reuse lm-eval-harness for industry benches; build only bespoke inference-opt
  metrics runner. Reuse DBOS for parallel fan-out.
