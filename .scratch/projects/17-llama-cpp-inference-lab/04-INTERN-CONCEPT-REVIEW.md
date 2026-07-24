# Interrogation — intern concept docs (XGRAMMAR + LMCACHE)

Reviewed 2026-07-23. Verified every low-level API claim against the pinned
`llama-cpp-python 0.3.34` in `.venv-spike`. **All checked claims are TRUE** —
`llama_state_seq_*`, the `_ext` variants, `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE=2`,
the sampler/vocab APIs, and the argtypes in the sample code all match. The docs
are well-researched. The problems are framing, scope, and buried assumptions —
not feasibility.

## Cross-cutting issues (both docs)

1. **Disconnected from the repo's actual decision.** Both are standalone
   product proposals (separate packages `llama-lmcache/`,
   `llama-xgrammar-runtime/`). Our decision (02-DECISIONS) is an in-place `src/`
   refactor, one runtime, teaching-first. They should share ONE runtime, ONE
   engine fingerprint, ONE bench harness — not two parallel stacks that both
   reinvent fingerprinting, the pinned-version tuple, per-sequence lifecycle,
   and benchmarking.
2. **Teaching-library altitude ignored.** Both are written as production/ops
   systems (SQLModel catalogs, Prometheus, multi-tenant security, distributed
   controllers, C++ forks). The stated goal is a *teaching* library on a
   single-user 2×3060 rig. ~80–90% of both docs is scope that will never ship
   and isn't the lesson.
3. **The flagship model breaks both happy paths.** Ornith-1.0-9B is a HYBRID /
   linear-attention model (GatedDeltaNet — see the project-16 GDN transform
   work, [[ornith-gguf-sglang-progress]]). Both docs assume a conventional
   full-attention transformer and explicitly flag hybrid/recurrent state as an
   unsolved danger zone (LMCACHE §23.3–23.4). So the chosen base model sits
   exactly in the failure region neither doc resolves. This must be confronted
   before either pillar is scoped against Ornith.

## XGRAMMAR doc — sharpest points

- **Double-accept bug in the hot path — CONFIRMED (Gate 1, 07-doc).** The loop
  edits raw logits, then calls `llama_sampler_sample(chain, ctx, idx)` AND then
  `llama_sampler_accept(chain, token)`. The pinned `llama.h:1488` documents
  `llama_sampler_sample` as already calling `llama_sampler_accept` internally, so
  the extra explicit accept double-advances stateful samplers
  (penalties/repetition). Fix: drop the manual chain-accept (keep only
  `matcher.accept_token`), or use the Option-B owned-candidate-array loop.
- **The recommended path is worst-suited to the flagship.** Host-logit masking
  forces a device→host sync every token and blocks GPU-resident sampling
  (§12.3). The doc's own §12.4 says the Python-first path degrades exactly under
  "small model, hundreds of tok/s, continuous batching" — which is precisely the
  multi-LoRA router-fleet demo. So the recommended starting architecture is the
  weakest fit for our actual workload. Fine to start there for clarity, but name
  it and plan the native sampler (Appendix A) as a real, likely-needed step, not
  a "maybe."
- **Torch pulled into a GGUF runtime.** `apply_token_bitmask_inplace` via torch
  drags multi-GB torch into an otherwise lightweight llama.cpp stack. xgrammar
  has a numpy path — prefer it; don't hard-depend on torch.
- **Tokenizer equivalence is unsolved for our model.** §6 is correctly called
  "the most important correctness issue," but BOTH proposed paths are caveated:
  `from_huggingface` needs `transformers` (heavy) + the original HF tokenizer
  repo (a merged/finetuned Ornith GGUF may not have a clean match); the
  GGUF-derived `VocabType.RAW` path is self-described as experimental. Neither is
  proven for Ornith. This is a Phase-0 gate, not a detail.
- **Specific commit hashes / dates unverified.** `e3546c79…`, `c0bc8591…`,
  "131 commits ahead", the release dates — plausible and 0.3.34 is real
  (installed), but the vendored-commit and "ahead" claims are exactly the kind
  of precise assertion to confabulate. Don't enshrine them as fact without
  checking. `xgrammar` and `transformers` are NOT installed here, so the entire
  xgrammar path is paper — unrun end-to-end.
- **Appendix A speculative/jump-forward APIs** (`traverse_draft_tree`,
  `find_jump_forward_string`, `BatchGrammarMatcher.batch_rollback`) are
  unverified against an installed xgrammar. Treat as aspirational.

## LMCACHE doc — sharpest points

- **The title oversells; it is not an LMCache integration.** The doc's OWN
  analysis (§7 table, §8, Level C/D, §34, conclusion) establishes that a real
  LMCache connector needs block-level KV export/import primitives that DO NOT
  exist in llama.cpp, and that using LMCache as an opaque blob store yields
  ~none of LMCache's actual value (chunk dedup, CacheBlend non-prefix reuse,
  block injection). What's actually proposed is a **homegrown persistent
  exact-prefix KV snapshot cache**; LMCache is at best a low-value optional
  backend later. Honest title: "Persistent prefix-KV cache for llama.cpp." The
  admission is buried across many sections instead of leading.
- **Load-bearing assumption #1 filed as an "open question": cross-process
  portability.** The whole persistence value prop (survive restart, share across
  workers) depends on `llama_state_seq_save/load_file` bytes restoring correctly
  in a *different process*. Listed in §38 open questions, yet Phase 1 depends on
  it. It must be the Phase-0 gate #1. (My spike confirmed the funcs EXIST; nobody
  has tested an actual round-trip restore for correctness.)
- **Load-bearing assumption #2: break-even is unproven and may be negative on
  our rig.** On a 3060, GPU prefill is fast and KV state is huge (their own
  example: 2 GB for 16K tokens). Reading/transferring GBs may lose to just
  re-prefilling. Their single quantitative example is internally inconsistent
  (2 GB "read_ms: 121" ⇒ ~16 GB/s, faster than NVMe — only a RAM tier hits
  that). The entire justification hinges on a number they haven't measured.
- **Checkpoint duplication is inherent, not incidental.** Whole-sequence
  snapshots at 2K/4K/8K each re-store all earlier state; geometric spacing only
  softens it. This is the fundamental cost of having whole-sequence serialize
  instead of real block-addressable KV — reinforces that "LMCache-style" is
  aspirational.
- **Scope explosion.** 40 sections, 5 phases, radix trees, single-flight leases,
  multi-tier/distributed, encryption, a proposed upstream llama.cpp C API. The
  genuinely teachable core is Phase 0 + Phase 1 only:
  capture → persist → restore → suffix-only prefill, guarded by a strict engine
  fingerprint, with a break-even benchmark. Ship that; treat the rest as
  "someday / out of scope for a teaching lib."
- **Security §27** (multi-tenant KV-probing, encryption-at-rest) is off-threat-
  model for a single-user teaching rig. Cut to: namespace + checksum + no
  user-controlled paths.

## What both docs get RIGHT (keep)

- The strict engine-fingerprint / invalidation discipline (both). This is the
  correct safety spine and should be a single shared module.
- XGRAMMAR: the sampler ORDER (mask before top-k/top-p), the vocab-padding
  warning (mask must cover full logits dim, not tokenizer vocab), one matcher
  per sequence, compiled-grammar reuse, the zero-copy host-logits view mechanic.
- LMCACHE: the honest §34 conclusion that arbitrary module composition is
  impossible for causal transformers; the benchmark-gate-before-each-phase
  discipline; the fallback-never-fails-inference rule.

## Recommendation

Fold both into ONE runtime aligned to 02-DECISIONS, teaching-first:
1. Shared core: engine fingerprint + pinned-version tuple + bench harness (used
   by grammar, cache, and the multi-LoRA scheduler alike).
2. XGrammar pillar MVP = Phase 0 + Phase 1 (own the decode loop, numpy mask, one
   schema, tokenizer-equivalence gate on Ornith). Resolve the double-accept
   question first. Plan the native sampler as probable, not optional.
3. Cache pillar MVP = LMCACHE Phase 0 + Phase 1 ONLY, renamed to "persistent
   prefix-KV cache," with cross-process restore + break-even as the Phase-0
   gates, and an explicit Ornith-hybrid-state validation before trusting any
   restore.
4. Drop for now: LMCache-as-blob-store, distributed tiers, native C++ block
   connector, multi-tenant security, radix trees. Park in a "future" doc.
5. Confront the Ornith-hybrid-KV question up front — it may force a
   conventional-transformer base model for the cache demo even if Ornith stays
   the router-fleet base.
```
