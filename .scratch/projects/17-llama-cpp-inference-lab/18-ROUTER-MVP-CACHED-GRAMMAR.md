# Path-(a) Router MVP: Cached Prefix + Grammar-Constrained Routing — **GO**

> 2026-07-25. Composes the two proven halves into the complete no-fork teaching
> MVP. `context_pool_router.py` `run_constrained_cached()`, driver
> `run_router_grammar_cached.py`. Closes the Decision-D2 path-(a) story.

## What it is

One shared base model; one pinned-adapter llama_context per adapter. A long shared
router prompt is prefilled ONCE per adapter and its per-seq KV blob restored into
every seq slot; each request decodes only its short suffix and generates a
grammar-constrained JSON decision (per-sequence xgrammar matcher, independent
termination). All batched within a context and multiplexed across contexts. Zero
C++ changes.

## Result (`artifacts/project17-router-grammar-cached-20260725T015949Z/`)

Schema `Route{tool: Literal[search|calculator|calendar|smart_home|none],
confidence: Literal[low|medium|high]}`. 259-token shared prefix, 4 requests each
on base + probe-a.

| gate | result |
| --- | --- |
| all outputs schema-valid JSON | ✅ 8/8 |
| cached+constrained == cold+constrained (token-exact) | ✅ true |

Decisions (both adapters): book meeting -> `calendar`, 17x23 -> `calculator`,
lights -> `smart_home`, mars news -> `search`; every output
`{"tool": ..., "confidence": "high"}`. Throughput cached-vs-cold 1.15-1.18x at a
259-token prefix (grows with prefix length per `16-...P1.md` sweep: 4.75x @1536).

## Why it matters

This is the full flagship path-(a): the two optimizations that could have
interacted (KV-state restore vs grammar masking) compose with NO interaction bug
— cached output is token-identical to cold. So a production router can cache a
long shared system prompt once, serve many adapters' requests in parallel with
per-request suffixes, and guarantee every decision is valid JSON — all on stock
llama.cpp.

## Next

- Swap the synthetic probes for the real Qwen3.5-0.8B task adapters (acrouter
  emits routing JSON natively) for a semantic demo; grammar makes it inviolable.
- Package a single `demo` entrypoint (load base + adapters, cache prefix, serve a
  mixed grammar-constrained batch, print decisions).
- P2 mixed-batch fork remains the flagship stretch (guide §7).
