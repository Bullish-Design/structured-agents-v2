# Project 18 — Persistent Exact-Prefix Cache Runtime

## Objective

Turn Project 17's tested shared-interface contracts into the smallest real
persistent exact-prefix llama.cpp state cache that is correct before it is
measured.  The cache must be able to restore a compatible checkpoint, decode
the uncached suffix to obtain fresh logits, and otherwise fall back without
affecting inference correctness.

This is a homegrown teaching cache for llama.cpp whole-state snapshots.  It is
not an LMCache connector or a general distributed cache.

## Established evidence

- `Llama.save_state` / `load_state` are bit-exact for Ornith on the same
  instance and a newly constructed instance.
- Saved state does not include current output logits.  The only valid flow is:
  restore cached prefix → decode one or more uncached suffix tokens → consume
  fresh logits.
- `LlamaEngineFingerprint`, exact token IDs, namespace, format version, state
  size, and SHA-256 checksum are already represented by the Project 17 prefix
  cache contracts.
- Whole Ornith state is large (about 3.5 MB per token in the CPU gate); no
  break-even claim is valid until the CUDA path is measured.

## Non-goals

- No LMCache integration, block-level KV interface, compression, encryption,
  remote tier, distributed coordination, eviction policy, or radix tree.
- No adapter scheduler or grammar-pillar changes.
- No inference-throughput claim before a controlled CUDA measurement.
- No cache lookup based only on prompt text or a prompt hash.

## Workstream A — Reproducible state-API research spike

### Questions

1. Do `llama_state_seq_get_data` / `llama_state_seq_set_data` round-trip
   Ornith state exactly for a single sequence in the pinned build?
2. Can a checkpoint at token count `K` restore and then decode a suffix of one
   token and of several tokens, producing the same greedy continuation as a
   no-cache baseline?
3. What are the API's byte-size, allocation, return-value, and sequence-ID
   semantics on success and malformed input?
4. Does this path remain correct across separately constructed contexts, or is
   whole-context `save_state` / `load_state` the appropriate initial codec?

### Method

- Use the pinned llama-cpp-python and known Ornith GGUF with fixed greedy
  settings, token IDs, and context configuration.
- Record model/build identity, state size, checkpoint length, suffix length,
  byte checksum, baseline continuation, and restored continuation.
- Test checkpoint positions including `K = 1`, a middle prefix, and
  `K = len(prompt) - 1`; use one-token and multi-token suffixes.
- Use a new context for cross-instance validation where API semantics permit.
- Preserve stdout/stderr and raw token IDs under an ignored timestamped
  artifact directory.  Update a dated research report with the first failure
  boundary or the passing evidence.

### Decision gate

- If per-sequence state APIs are exact and portable, use them for the initial
  codec.
- If they are not portable or have unclear sequence-position semantics, use
  the already-proven whole-context save/load path for the teaching MVP and
  record per-sequence reuse as a follow-up.

## Workstream B — State codec and restore integration

### Design

Add a narrow codec behind the existing `PrefixCacheEntry`, `PrefixCacheKey`,
and `RestorePlan` contracts:

1. Capture a checkpoint after decoding exactly the cached prefix token IDs.
2. Calculate byte length and SHA-256 before publication.
3. On lookup, reject mismatched namespace, format, frozen engine fingerprint,
   checkpoint token count, exact prefix IDs, size, or checksum.
4. Restore only an accepted blob.
5. Decode `RestorePlan.uncached_suffix_token_ids` before reading logits.
6. Treat all cache failures as a cache miss and perform normal prefill; do not
   make inference fail because caching is unavailable or corrupt.

The decoder interface should expose the restore path as an explicit operation,
not silently substitute cached state mid-loop.  Keep Pydantic at external
configuration/request boundaries; state bytes and token IDs stay plain typed
values on the hot path.

### Required tests

- Same-process and fresh-context continuation equals no-cache greedy baseline.
- One-token suffix and multi-token suffix plans both produce fresh, matching
  continuation.
- A zero-token suffix is rejected before decode/logit consumption.
- Fingerprint, token ID, namespace, version, size, and checksum failures do
  not call restore and fall back to full prefill.
- A partial-prefix checkpoint restores only when the requested token sequence
  starts with the saved exact prefix.
- No state from one sequence/request is observable in another.

## Workstream C — Minimal durable storage

### Scope

Implement only a local filesystem blob store and checkpoint-boundary index
using the Project 17 `PrefixCacheBlobStore` and `PrefixCacheIndex` protocols.

### Requirements

- Derive paths from the deterministic key; never accept a caller-provided path.
- Write a temporary blob, fsync as appropriate, verify its metadata, then atomically
  publish it with `os.replace`.
- Persist index metadata atomically in a simple local format appropriate for
  the one-user teaching MVP.  Do not add SQLite unless the required lookup and
  restart tests cannot be met without it.
- Missing, malformed, or corrupt metadata/blob is a cache miss, not an
  inference failure.
- Exercise process restart by constructing a new store and a new model context.

## Workstream D — Break-even benchmark

Run only after correctness and durable restart pass.

1. Use the Project 17 benchmark timer and pinned CUDA library/model tuple.
2. Sweep prompt/checkpoint lengths that cover short, medium, and long shared
   prefixes on the 3060 configuration.
3. Measure baseline prefill separately from index lookup, blob read, checksum,
   restore, suffix decode, and first fresh-logit availability.
4. Publish raw state sizes and median/p95 timings; explicitly state the fixed
   model, context, backend, storage medium, suffix length, and repetition
   count.
5. Report the first observed break-even point if one exists; otherwise report
   that whole-state persistence did not pay off under the measured conditions.

## Implementation sequence

1. Establish clean baseline and run Workstream A against Ornith.
2. Write failing codec/restore regression tests from the spike's observed API
   semantics.
3. Implement the smallest state codec and in-memory test double needed to pass
   those tests.
4. Re-run the exact model-backed continuation checks.
5. Add the atomic filesystem store plus restart/corruption tests.
6. Run focused tests, Ruff, formatter, static type checks, and the relevant
   full suite after each meaningful change.
7. Run the CUDA benchmark only after all correctness gates pass.

## Completion criteria

- Process-restart cache restore produces the same deterministic continuation as
  baseline for a compatible engine and exact token prefix.
- Incompatible/corrupt entries never reach restore and never alter the output.
- Every restore decodes at least one uncached suffix token before logits are
  used.
- Persistent storage publication/read paths are atomic enough for a single
  local process and safely degrade to cache misses.
- A reproducible benchmark record states whether this cache breaks even on the
  target CUDA configuration.
