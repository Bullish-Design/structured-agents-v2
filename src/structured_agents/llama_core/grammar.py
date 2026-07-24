"""XGrammar integration helpers that keep torch out of the decode hot path."""

from __future__ import annotations

from typing import Any


def apply_packed_bitmask_inplace(logits: Any, packed_mask: Any, vocab_size: int) -> None:
    """Set masked logits to ``-inf`` from XGrammar's packed int32 bitmask.

    Bit ``i`` is one when token ``i`` is allowed.  ``vocab_size`` must be the
    model logit width, not the Hugging Face tokenizer length: padded model IDs
    must be explicitly masked too.  NumPy is imported locally to preserve a
    lightweight shared-core import path.
    """
    import numpy as np

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if len(logits) < vocab_size:
        raise ValueError("logits is smaller than vocab_size")
    required_words = (vocab_size + 31) // 32
    if len(packed_mask) < required_words:
        raise ValueError("packed_mask is smaller than vocab_size")
    ids = np.arange(vocab_size, dtype=np.int64)
    words = np.asarray(packed_mask[:required_words], dtype=np.uint32)
    allowed = ((words[ids >> 5] >> (ids & 31)) & 1).astype(bool)
    logits[:vocab_size][~allowed] = -np.inf


class JsonSchemaGrammar:
    """One compiled grammar and one fresh XGrammar matcher per sequence."""

    def __init__(self, compiler: Any, compiled: Any, *, vocab_size: int) -> None:
        self.compiler = compiler
        self.compiled = compiled
        self.vocab_size = vocab_size

    @classmethod
    def from_huggingface(
        cls,
        tokenizer: Any,
        schema: dict[str, Any],
        *,
        vocab_size: int,
    ) -> JsonSchemaGrammar:
        """Compile strict JSON Schema with the true llama.cpp vocabulary width."""
        import xgrammar as xgr

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
        return cls(compiler, compiler.compile_json_schema(schema, strict_mode=True), vocab_size=vocab_size)

    def new_matcher(self) -> Any:
        import xgrammar as xgr

        return xgr.GrammarMatcher(self.compiled)

    def logits_hook(self, matcher: Any) -> Any:
        """Return an owned-loop hook which fills and applies the next-token mask."""
        import numpy as np
        import xgrammar as xgr

        bitmask = np.zeros(xgr.get_bitmask_shape(1, self.vocab_size), dtype=np.int32)

        def apply(logits: Any) -> None:
            bitmask.fill(0)
            if matcher.fill_next_token_bitmask(bitmask):
                apply_packed_bitmask_inplace(logits, bitmask[0], self.vocab_size)

        return apply
