# XGrammar API findings — 2026-07-24

## Decision

Use the locked `xgrammar==0.2.1` API surface for the Phase-1 MVP.  The local
uv cache also contains `0.2.3`; inspection shows the compiler, matcher, and
tokenizer methods used below are compatible.  Do not float this dependency:
the xgrammar API and compiled-grammar serialization are versioned integration
boundaries.

Required runtime resolution tuple:

```text
xgrammar==0.2.1
transformers>=4.38,<5
torch>=1.10        # xgrammar's mandatory installation dependency; idle at inference time
numpy              # host bitmask and llama.cpp logits mutation
```

The repository lock currently resolves xgrammar `0.2.1`, but resolves
`transformers 5.10.2`.  Upstream xgrammar 0.2.1/0.2.3 package metadata only
declares `transformers>=4.38.0`; it does **not** enforce `<5`.  Pinning
`<5` remains the conservative compatibility choice until the actual Ornith
smoke proves the `from_huggingface` path under a newer transformers release.
This is an intentional pin recommendation, not a claim about the published
metadata.

## Exact MVP construction

After loading the already-proven equivalent HF Ornith tokenizer and obtaining
`n_vocab=248320` from llama.cpp:

```python
tokenizer_info = xgr.TokenizerInfo.from_huggingface(
    hf_tokenizer,
    vocab_size=llama_n_vocab,  # 248320, not len(hf_tokenizer)
)
compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
compiled = compiler.compile_json_schema(
    pydantic_model.model_json_schema(), strict_mode=True
)
matcher = xgr.GrammarMatcher(compiled)  # one fresh matcher per sequence
bitmask = np.zeros(xgr.get_bitmask_shape(1, llama_n_vocab), dtype=np.int32)

need_apply = matcher.fill_next_token_bitmask(bitmask)
if need_apply:
    apply_xgrammar_numpy_mask_inplace(llama_logits, bitmask[0], llama_n_vocab)
```

`GrammarCompiler` owns an in-memory compiled-grammar cache.  The library layer
should additionally key its compatible compiler/cache by the engine
fingerprint plus canonical schema/options and xgrammar version.  A matcher is
stateful and must never be shared by sequences.  After each selected token,
call `matcher.accept_token(token)` exactly once and fail closed if it returns
false.

## Numpy boundary: resolved API detail

`GrammarMatcher.fill_next_token_bitmask` accepts CPU DLPack array-likes, which
includes a modern NumPy `int32` array.  This gives the desired torch-free *hot
path* for filling masks.  The public `xgr.apply_token_bitmask_inplace` helper,
however, is torch-tensor-only in both inspected versions: its signature and
CPU implementation require `torch.Tensor` and use `.device` / `.data_ptr()`.

Therefore Phase 1 must **not** call that helper with llama-cpp-python's NumPy
logit view.  Apply the packed int32 mask directly with a tiny NumPy routine:
each zero bit means set the corresponding logit to `-np.inf`, bounded by the
true `llama_n_vocab`.  This keeps torch installed only because xgrammar imports
it, not on the token loop.  Time this separately as mask application.

The mask has `ceil(248320 / 32) == 7760` int32 words.  Passing the full model
dimension is mandatory: the HF tokenizer has 248077 entries, leaving 243
padded logit IDs that must remain masked.

## Validation added

`tests/test_xgrammar_api_contract.py` is model/download-free.  Because xgrammar
is a hard runtime dependency, absence is a test/setup failure.  It checks
the `TokenizerInfo.from_huggingface(..., vocab_size=...)` / compiler / matcher
surface and exercises a compiler → matcher → NumPy bitmask fill on a toy
padded vocabulary.  It is deliberately a dependency/API contract test; the
Ornith tokenizer-equivalence gate and GPU JSON smoke remain integration tests.

## Current blocker

The project environment must install this tuple before any test or application
run.  Re-run Gate 2 after resolution, then run the owned-loop Ornith smoke with
the NumPy mask adapter.
