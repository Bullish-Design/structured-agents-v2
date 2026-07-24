"""GPU-free checks for the LlamaState blob composition model.

These lock the arithmetic that the Project 18 investigation uses to attribute
the persistent whole-state cache blob size to the ``LlamaState`` score buffer
versus the native llama.cpp state.  The reference numbers are the exact byte
sizes recorded in ``artifacts/project17-prefix-cache-20260724T175312Z`` and
``...191154Z`` for Ornith (``n_vocab=248320``, ``n_batch=128``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module() -> object:
    path = Path(__file__).parents[1] / "benchmarks" / "project17" / "state_blob_model.py"
    spec = importlib.util.spec_from_file_location("state_blob_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the frozen slots dataclasses resolve their module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORNITH_N_VOCAB = 248320
ORNITH_N_BATCH = 128


def test_scores_dominate_growth_and_saturate_at_n_batch() -> None:
    mod = _module()
    # Recorded blob totals (bytes) and the implied native state per the report.
    observed = {16: 69111237, 96: 151196677, 256: 188227718}
    for n_tokens, total in observed.items():
        native = total - min(n_tokens, ORNITH_N_BATCH) * ORNITH_N_VOCAB * 4 - n_tokens * 4
        comp = mod.llama_state_blob_composition(
            n_tokens=n_tokens,
            n_vocab=ORNITH_N_VOCAB,
            n_batch=ORNITH_N_BATCH,
            native_state_bytes=native,
        )
        assert comp.total_bytes == total
        # Native state stays ~53-61 MB (recurrent-dominated), never the driver of growth.
        assert 52_000_000 <= comp.native_state_bytes <= 62_000_000
    # Scores saturate: 256 tokens copy the same rows as 128 (n_batch cap).
    cap = mod.llama_state_blob_composition(
        n_tokens=256, n_vocab=ORNITH_N_VOCAB, n_batch=ORNITH_N_BATCH, native_state_bytes=0
    )
    assert cap.scored_rows == ORNITH_N_BATCH


def test_overhead_fraction_and_native_savings() -> None:
    mod = _module()
    comp = mod.llama_state_blob_composition(
        n_tokens=96, n_vocab=ORNITH_N_VOCAB, n_batch=ORNITH_N_BATCH, native_state_bytes=55_800_000
    )
    # At 96 tokens the wrapper score/input arrays are the majority of the blob.
    assert comp.overhead_fraction > 0.6
    assert mod.native_codec_savings_bytes(comp) == comp.scores_bytes + comp.input_ids_bytes


def test_rejects_nonsense_sizes() -> None:
    mod = _module()
    for kwargs in (
        {"n_tokens": -1, "n_vocab": 10, "n_batch": 8, "native_state_bytes": 0},
        {"n_tokens": 4, "n_vocab": 0, "n_batch": 8, "native_state_bytes": 0},
        {"n_tokens": 4, "n_vocab": 10, "n_batch": 0, "native_state_bytes": 0},
    ):
        try:
            mod.llama_state_blob_composition(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
