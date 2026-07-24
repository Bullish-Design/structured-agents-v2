"""Contract checks for the optional XGrammar integration surface.

These tests deliberately need neither an Ornith GGUF nor a Hugging Face download.
They are skipped in the base development environment, where grammar support is an
optional dependency.
"""

from __future__ import annotations

import inspect

import pytest


def test_xgrammar_compiler_matcher_and_numpy_bitmask_contract() -> None:
    """Keep the Phase-1 construction path honest across XGrammar upgrades."""
    xgr = pytest.importorskip("xgrammar")
    numpy = pytest.importorskip("numpy")

    assert "vocab_size" in inspect.signature(xgr.TokenizerInfo.from_huggingface).parameters
    assert "strict_mode" in inspect.signature(xgr.GrammarCompiler.compile_json_schema).parameters
    assert "debug_print" in inspect.signature(xgr.GrammarMatcher.fill_next_token_bitmask).parameters

    # The model's logits dimension, rather than the tokenizer's number of
    # entries, controls the mask shape.  The 65-entry toy model leaves padded
    # entries just like Ornith's 248320-wide lm_head.
    tokenizer = xgr.TokenizerInfo(
        ["<eos>", "{", "}", '"', "ok", ":", "true", "false", ",", " "],
        xgr.VocabType.RAW,
        vocab_size=65,
        stop_token_ids=[0],
    )
    compiler = xgr.GrammarCompiler(tokenizer)
    compiled = compiler.compile_json_schema(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        strict_mode=True,
    )
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = numpy.zeros(xgr.get_bitmask_shape(1, 65), dtype=numpy.int32)

    assert matcher.fill_next_token_bitmask(bitmask) is True
    assert bitmask.shape == (1, 3)
    assert bitmask.dtype == numpy.int32
