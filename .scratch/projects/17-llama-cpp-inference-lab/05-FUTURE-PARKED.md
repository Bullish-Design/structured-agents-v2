# Parked / out-of-scope for the MVP

Everything here is deliberately deferred. It is preserved because the intern's
XGRAMMAR + LMCACHE concept docs contain good, real research (all low-level API
claims verified true — see 04-INTERN-CONCEPT-REVIEW) and we do not want to lose
it. Nothing in this file is part of the teaching-library MVP. Each item lists
**why parked** and the **trigger** that would revive it.

Guiding rule: the MVP is a *teaching* library on a single-user 2×3060 rig. Ship
the Phase-0/Phase-1 core of each pillar; park production/ops/distributed/native
scope until a measured need appears.

---

## From the XGRAMMAR concept

### P-X1. Native C++ `llama_sampler` bridge (Appendix A)
- **What:** a small C++ shared lib implementing XGrammar as a real
  `llama_sampler`, removing Python from the per-token grammar path; single-
  `libllama` build discipline, C ABI, apply/accept/reset/clone/free.
- **Why parked:** big ABI/build burden; only justified once the Python-first
  path is proven insufficient by profiling.
- **Trigger:** measured grammar overhead is a material % of token latency, OR
  the router-fleet demo needs continuous batching / GPU-resident masking, OR we
  want torch/Python out of the worker. NOTE: our flagship (small fast routers,
  high tok/s) makes this *probable*, not hypothetical — but still measure first.

### P-X2. Speculative decoding + jump-forward decoding
- **What:** `traverse_draft_tree`, `rollback`, `fork`, `batch_rollback`,
  `find_jump_forward_string` for draft verification and deterministic-span skips.
- **Why parked:** unverified against an installed xgrammar; needs the native
  bridge and correct rollback semantics first.
- **Trigger:** after native sampler + batching land and correctness is locked.

### P-X3. GPU/backend-resident bitmask application
- **What:** apply the grammar mask before logits are materialized to a host
  candidate array, via a device bitmask kernel.
- **Why parked:** requires the pinned sampler backend ABI + device mask
  integration; advanced.
- **Trigger:** host-logit sync per token shows up as a real cost on the 3060s.

### P-X4. Structural tags / mixed free-text+constrained regions, BatchGrammarMatcher
- **What:** XGrammar-2 structural-tag features and batched matcher fill.
- **Why parked:** MVP starts with a single whole-response strict JSON Schema.
- **Trigger:** demo needs interleaved prose + constrained spans, or static
  multi-sequence batching is in place.

### P-X5. GGUF-derived `TokenizerInfo` (VocabType.RAW) as the primary path
- **What:** build TokenizerInfo from `llama_token_to_piece` bytes, dropping the
  `transformers` + HF-repo dependency.
- **Why parked:** self-described experimental; special/control/byte-fallback
  edge cases unproven, especially for a merged/finetuned Ornith GGUF.
- **Trigger:** MVP proves correctness via `from_huggingface`, THEN we validate
  the GGUF path against it as a second backend to shed the heavy dep.

---

## From the LMCACHE concept

### P-L1. Real LMCache integration (opaque-blob Level C, native connector Level D)
- **What:** store llama.cpp sequence states in LMCache; ultimately a native
  block-level KV export/import connector.
- **Why parked:** llama.cpp lacks block-addressable KV export/import APIs; a
  native connector is a C++ fork project. Opaque-blob use gives ~none of
  LMCache's real value (chunk dedup, CacheBlend, block injection). The MVP cache
  is a homegrown persistent *prefix* snapshot cache, NOT LMCache.
- **Trigger:** long-context reuse is proven valuable AND whole-state duplication
  is a real bottleneck AND we can maintain/upstream llama.cpp C changes.

### P-L2. Distributed / multi-tier / shared cache (LMCACHE Phases 2–3)
- **What:** RAM+NVMe+remote tiers, per-host daemon, distributed controller,
  prefix-aware routing, replication/promotion, single-flight leases.
- **Why parked:** production ops scope irrelevant to a single-user rig.
- **Trigger:** multiple hosts/workers and a proven cross-process economic case.

### P-L3. Upstream llama.cpp KV-block API proposal (LMCACHE §33)
- **What:** stable sequence-memory export/import abstraction, ownership/position/
  partial-state semantics, storage format, backend transfer hooks, scheduler
  integration.
- **Why parked:** only meaningful alongside P-L1 Level D.
- **Trigger:** committing to the native connector path.

### P-L4. Checkpoint-policy machinery (radix tree, cost-based admission, TinyLFU)
- **What:** trie/radix indexes, geometric+semantic checkpointing, cost-model
  admission, eviction tiers.
- **Why parked:** MVP uses a simple checkpoint-boundary hash map + strict
  fingerprint. Sophisticated indexing is premature.
- **Trigger:** workload shows enough irregular shared prefixes to justify it.

### P-L5. Multi-tenant security hardening (LMCACHE §27)
- **What:** tenant isolation, encryption at rest/in transit, cache-probe
  defenses, audit.
- **Why parked:** off-threat-model for a single user. MVP keeps only namespace +
  checksum + no user-controlled paths.
- **Trigger:** shared/multi-tenant deployment.

### P-L6. Compression / lossy KV (LMCACHE §20)
- **What:** LZ4/Zstd/dedup and lossy KV quantization for state blobs.
- **Why parked:** benchmark raw first; lossy is quality-sensitive.
- **Trigger:** I/O is the measured bottleneck (lossless only).

### P-L7. Non-text / hybrid model cache support
- **What:** SWA, recurrent/state-space, hybrid, multimodal state handling.
- **Why parked:** the MVP cache targets conventional full-attention text models.
  CRITICAL: Ornith-1.0-9B is hybrid (GatedDeltaNet) — so the cache demo may need
  a plain-transformer base even though Ornith stays the router-fleet base. This
  item tracks resolving Ornith-hybrid KV specifically.
- **Trigger:** an Ornith-hybrid-state validation spike (must precede trusting any
  restore against Ornith).

---

## Shared infra the MVP WILL build (not parked — noted here to prevent the
## two docs from each reinventing it)

- One engine-fingerprint / pinned-version-tuple module (used by grammar, cache,
  and the multi-LoRA scheduler).
- One benchmark harness (grammar overhead, TTFT/tok-s, KV break-even, router
  accuracy) — reuse for all pillars; reuse DBOS for any future parallel fan-out.
- One per-sequence runtime + lifecycle (matcher, sampler chain, KV/state,
  adapter selection) rather than two parallel package stacks.
