# Gate 2 — tokenizer equivalence (Ornith) — RESOLVED ✅

**Question (PLAN Phase 0.e):** does llama.cpp (GGUF) tokenization agree
token-ID-for-token-ID with the HF reference tokenizer? If not, xgrammar masks
built over one tokenizer mask the wrong IDs for the other → silent grammar
corruption. Correctness prerequisite for the whole grammar pillar.

**Answer: YES — exact agreement.** `gate2_tokenizer_equiv.py`, llama-cpp-python
0.3.34 vs `transformers` (tokenizers-only, no torch):
```
[probes] 26/26 match  ALL MATCH     # incl. <|im_start|> <|im_end|> <tool_call>
[fuzz]   600/600 match ALL MATCH     #      <function= <parameter= </think> etc.
[GATE 2] PASS
```
Compared with llama.cpp `tokenize(add_bos=False, special=True)` vs HF
`encode(add_special_tokens=False)` over the xgrammar boundary probes,
Ornith/Qwen3.5 special + tool markers, and a deterministic unicode/json/
whitespace fuzz corpus.

## Reference tokenizer provenance (from GGUF metadata)
- `general.architecture = qwen35`; `general.base_model.0.repo_url =`
  `https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B` → the reference.
- Tokenizer: `tokenizer.ggml.model = gpt2` (BPE), `tokenizer.ggml.pre = qwen35`.
- The Unsloth GGUF carries "Unsloth fixes" in its chat template, so divergence was
  plausible — but tokenization matches the pristine base exactly on this corpus.
- Fetched tokenizer files only (no weights) from the base repo.

## KEY implementation note — vocab padding (xgrammar doc §6.4 confirmed)
```
llama n_vocab = 248320   (model logits / lm_head dim)
hf vocab_size = 248044   len(hf) = 248077   (tokenizer vocab)
```
The model head is WIDER than the tokenizer vocab (~243 padded slots,
248077..248319). Every ID either tokenizer *produces* agrees, but:
- **The grammar bitmask MUST be sized to `n_vocab` (248320), not the tokenizer
  vocab.** Otherwise trailing padded IDs stay unmasked and could be sampled.
- Pass the model's real logits dim to `xgr.TokenizerInfo(..., vocab_size=n_vocab)`
  and allocate the bitmask at 248320. Bake into the grammar-pillar contract +
  a test asserting padded IDs are always masked.

## Scope / not-yet
- This validates the ID-equivalence prerequisite. The xgrammar `TokenizerInfo`
  itself (from_huggingface path) is the next step and simply consumes this same
  HF tokenizer — deferred with the xgrammar install (pulls torch; keep to the
  numpy mask backend per standing rules).
- Only the base tokenizer was compared; if an Unsloth-specific tokenizer repo is
  later used as the runtime reference, re-run this gate against it.

All three Phase-0 gates (1 double-accept, 2 tokenizer, 3 hybrid restore) now GREEN.
