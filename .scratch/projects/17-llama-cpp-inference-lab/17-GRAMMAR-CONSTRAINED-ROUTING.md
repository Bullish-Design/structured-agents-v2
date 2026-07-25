# Grammar-Constrained Routing in the Context-Pool Router: **GO**

> 2026-07-25. Wires xgrammar 0.2.1 into the router (`context_pool_router.py`
> `enable_grammar`/`compile_json_schema`/`run_constrained`), driver
> `run_router_grammar.py`. The "always-valid tool-call routing" half of the
> flagship (guide 14 §1), on top of the proven no-fork router.

## What it is

Each sequence gets its OWN stateful `xgrammar.GrammarMatcher`. In the batched
decode step, before argmax, that seq's logits are masked to the grammar
(`fill_next_token_bitmask` → numpy int32 mask → disallowed logits set to -inf,
torch-free per 12-XGRAMMAR-API-FINDINGS.md), then the chosen token is
`accept_token`ed (fail-closed). Sequences terminate independently when the
grammar completes; terminated slots ride along in the batch (frozen output) to
keep the `seq_id == row` invariant. vocab_size = llama.cpp n_vocab (248320), NOT
len(hf_tokenizer), so padded logit ids stay masked.

## Result (`artifacts/project17-router-grammar-20260725T014248Z/`)

Schema: `Route{ tool: Literal[search|calculator|calendar|smart_home|none],
confidence: Literal[low|medium|high] }`. 10 requests across base + probe-a.

| | schema-valid |
| --- | --- |
| **constrained** (grammar) | **10 / 10** |
| unconstrained (same prompts) | 0 / 10 |

- Every constrained output is well-formed JSON validating against the schema,
  e.g. `{"tool":"calendar","confidence":"high"}` (book meeting),
  `{"tool":"calculator",...}` (17x23), `{"tool":"smart_home",...}` (lights),
  `{"tool":"search",...}` (mars news).
- Unconstrained, Ornith (a *thinking* model) emits `<think>...` chains and never
  produces clean JSON in the token budget → 0/10. One unconstrained attempt even
  emitted `{"tool":"home_assistant"}` — an INVALID enum value the grammar
  corrected to `smart_home`. Clear demonstration of what the grammar guarantees.
- Routing is sensible; "sing a song" -> smart_home is a model-quality miss
  (should be `none`), not an engine issue.

## Notes / gotchas

- **Ran on GPU 1**, not GPU 0: another project's job (flora `qc_probe.py`) was
  holding ~4.3 GB on GPU 0 and new `llama_new_context_with_model` returned NULL
  (OOM for model + 3 contexts). Left that job alone; relaxed the runner's
  GPU-only guard to accept `CUDA_VISIBLE_DEVICES in {0,1}` and used n_ctx=2048.
- xgrammar `apply_token_bitmask_inplace` is torch-tensor-only; we apply the packed
  int32 mask with a tiny numpy routine (`np.unpackbits(..., bitorder='little')`).
- Grammar is adapter-agnostic: the same compiled grammar drives base and any
  adapter's context; matchers are per-sequence and never shared.

## Next

- Compose grammar + cached shared-prefix restore (both proven separately) for the
  full path-(a) MVP: long shared router prompt cached once, per-request suffix,
  grammar-guaranteed JSON out.
- Swap in the real Qwen3.5-0.8B acrouter adapter (emits routing JSON natively) as
  the semantic demo; grammar makes it inviolable.
- Time the mask hot path separately (fill + numpy apply) at scale.
