# Gate 1 — sampler double-accept — RESOLVED

**Question (PLAN Phase 0.d):** does `llama_sampler_sample` already call
`llama_sampler_accept` on the chain internally? If so, the intern's XGRAMMAR hot
path double-accepts.

**Answer: YES — `llama_sampler_sample` accepts internally.** Authoritative source:
the pinned `llama.h` shipped with llama-cpp-python 0.3.34
(`include/llama.h:1488`) documents `llama_sampler_sample` verbatim as:

```
/// @details Sample and accept a token from the idx-th output of the last evaluation
// Shorthand for:
//    const auto * logits = llama_get_logits_ith(ctx, idx);
//    llama_token_data_array cur_p = { ... init from logits ... };
//    llama_sampler_apply(smpl, &cur_p);
//    auto token = cur_p.data[cur_p.selected].id;
//    llama_sampler_accept(smpl, token);   // <-- internal accept
//    return token;
```

## The bug this confirms (intern XGRAMMAR doc §8.4 / §8.5)
Their loop does BOTH:
```
token = llama.llama_sampler_sample(sampler_chain, context, idx)   # accepts internally
...
llama.llama_sampler_accept(sampler_chain, token)                  # DOUBLE accept
```
=> stateful samplers in the chain (penalties / repetition / any accept-tracking
sampler) are advanced twice per token. Grammar-only chains without stateful
members won't visibly break, which is why it can pass casual testing — exactly
the kind of silent bug to kill now.

Note: `matcher.accept_token(token)` (XGrammar) is SEPARATE and still required —
it advances the grammar matcher, not the llama sampler chain. Only the second
`llama_sampler_accept` is wrong.

## Canonical decode-loop contract (adopt for Phase 1)

Two correct shapes. Both call the grammar-matcher accept exactly once and the
llama-chain accept exactly once.

**Option A — use `llama_sampler_sample` (simplest):**
1. edit raw logits in place to apply the grammar mask (numpy view over
   `llama_get_logits_ith` — `sample()` re-reads that same buffer, so the mask is
   seen);
2. `token = llama_sampler_sample(chain, ctx, idx)`  (accepts the chain
   internally — do NOT call `llama_sampler_accept` again);
3. `matcher.accept_token(token)`.

**Option B — own the candidate array (more control, recommended for teaching):**
1. build `cur_p` from the logits;
2. apply grammar mask, then `llama_sampler_apply(chain, &cur_p)`;
3. select `token = cur_p.data[cur_p.selected].id`;
4. `llama_sampler_accept(chain, token)` — exactly once, WE own it;
5. `matcher.accept_token(token)`.
Option B avoids the "does sample() re-read my edited buffer" subtlety entirely
and makes the accept ownership explicit — better as the taught reference loop.

**Rule:** never mix — if `llama_sampler_sample` is used, the chain is already
accepted; never call `llama_sampler_accept` after it.

## Confidence / follow-up
Resolved from the pinned header's documented contract (authoritative for this
exact version). A belt-and-suspenders empirical check (chain with a penalty
sampler; assert its state advances once per token) is cheap to add once the
Phase 1 loop exists — not required to unblock.
