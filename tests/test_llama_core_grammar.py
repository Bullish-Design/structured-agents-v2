from __future__ import annotations

import numpy as np
import pytest

from structured_agents.llama_core.grammar import apply_packed_bitmask_inplace


def test_packed_mask_masks_padded_model_vocabulary_ids() -> None:
    logits = np.arange(65, dtype=np.float32)
    # Permit tokens 0 and 64 only; model ids 10..63 emulate padded logits.
    mask = np.array([1, 0, 1], dtype=np.int32)

    apply_packed_bitmask_inplace(logits, mask, 65)

    assert logits[0] == 0
    assert logits[64] == 64
    assert np.isneginf(logits[1])
    assert np.isneginf(logits[63])


def test_packed_mask_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="smaller"):
        apply_packed_bitmask_inplace(np.zeros(4), np.zeros(1, dtype=np.int32), 33)
