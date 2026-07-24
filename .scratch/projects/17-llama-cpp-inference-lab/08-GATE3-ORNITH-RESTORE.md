# Gate 3 — Ornith hybrid-KV restore — RESOLVED ✅

**Question (PLAN Phase 0.f):** does Ornith-1.0-9B's hybrid (GatedDeltaNet /
linear-attention) sequence state survive a save/restore round-trip? Decides
whether the cache pillar can use Ornith or needs a plain-transformer base.

**Answer: YES — restore is bit-exact for deterministic greedy continuation,
same-instance AND cross-instance (restart scenario).** Empirically verified on
CPU with llama-cpp-python 0.3.34 (`gate3_ornith_restore.py`).

## Prerequisite cleared first (`gate3_ornith_load.py`)
Ornith loads and generates COHERENTLY in llama.cpp on CPU — "capital of France?
A: Paris. ... capital of Germany? A: Berlin." This overturns the prior sglang
result (gibberish, see [[ornith-gguf-sglang-progress]]): **llama.cpp handles
Ornith's hybrid architecture correctly out of the box.** Load 3.3s, ~2.7 tok/s
CPU, n_vocab=248320.

## Result
```
[state] n_tokens=15  llama_state_size=53183582 bytes   (~3.5 MB/token)
[baseline]        [264, 37550, 421, 369, 3841, 11, 13477, 11, ...]
[A same-instance] match=True
[B cross-instance] match=True
[GATE 3] same-instance=PASS  cross-instance=PASS
```
Cross-instance = save_state on one `Llama`, `load_state` on a freshly constructed
one → identical continuation. This is the survive-restart / cross-worker path the
cache pillar needs.

## Two load-bearing implementation notes for Phase 2
1. **Restore pattern = restore prefix KV, then DECODE the suffix.** The saved
   state does NOT include the output logits buffer, so `get_logits()` immediately
   after `load_state()` is STALE (returns the prior decode's logits). Correct
   continuation checkpoints at `prefix = tokens[:-1]` and decodes the last/suffix
   token to produce fresh logits — exactly the LMCACHE data-flow (restore prefix
   → eval uncached suffix). My first harness run FAILED purely from this bug
   (read stale logits with zero suffix decode); fixing it → PASS. Bake this into
   the cache codec's contract and its tests.
2. **State is large (~3.5 MB/token here).** Hybrid recurrent+attention state.
   At n_ctx=1024 that's ~3.6 GB. Implication for break-even (Phase 2): restore
   moves multi-GB; it wins only when prefill is expensive enough (CPU: likely
   yes; 3060 GPU with fast prefill: TBD — measure with the CUDA build).

## Consequence for decisions
- Cache-pillar base model open question (PLAN "Open decisions") → **can stay
  Ornith**; a plain-transformer fallback is NOT forced for correctness. (Break-
  even economics may still argue for a smaller state model, but that's a perf
  choice, not a correctness one.)
- Update 05-FUTURE-PARKED P-L7: the Ornith-hybrid-state validation it demanded
  before trusting restore is DONE and green for the save/restore path.

## Not yet tested (future, non-blocking)
- Per-sequence `llama_state_seq_get_data`/`set_data` specifically (this gate used
  high-level whole-context `save_state`/`load_state`, which exercises the same
  serializer). Confirm the seq-level API for multi-sequence cache reuse in Phase 2.
- Partial-prefix restore (checkpoint at K < prefix, decode longer suffix).
- SWA/window edge cases at long context near n_ctx.
