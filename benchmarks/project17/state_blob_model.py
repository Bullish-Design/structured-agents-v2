"""Analytic model of the llama-cpp-python ``LlamaState`` pickle blob.

This is GPU-free teaching/analysis code.  It explains *why* the persistent
whole-state prefix cache blob grows the way the CUDA artifacts recorded, by
decomposing it into the native llama.cpp state versus the ``LlamaState``
score/input-id buffers that :meth:`llama_cpp.Llama.save_state` copies.

The key teaching fact: for a hybrid recurrent model such as Ornith the native
state is dominated by a large, nearly token-count-independent recurrent
(GatedDeltaNet) cache, while ``LlamaState.scores`` — a ``(min(n_tokens,
n_batch), n_vocab)`` float32 array of *prefill logits that the restore
lifecycle never uses* — is what actually makes the blob grow with prefix
length.  A native-only codec removes that term.
"""

from __future__ import annotations

from dataclasses import dataclass

FLOAT32_BYTES = 4
INT32_BYTES = 4


@dataclass(frozen=True, slots=True)
class BlobComposition:
    """Predicted byte breakdown of one pickled ``LlamaState`` snapshot."""

    n_tokens: int
    scored_rows: int
    scores_bytes: int
    input_ids_bytes: int
    native_state_bytes: int

    @property
    def llama_state_overhead_bytes(self) -> int:
        """Bytes present only because of the Python ``LlamaState`` wrapper."""
        return self.scores_bytes + self.input_ids_bytes

    @property
    def total_bytes(self) -> int:
        """Total wrapper overhead plus native state (excludes pickle framing)."""
        return self.native_state_bytes + self.llama_state_overhead_bytes

    @property
    def overhead_fraction(self) -> float:
        return self.llama_state_overhead_bytes / self.total_bytes


def llama_state_blob_composition(
    *,
    n_tokens: int,
    n_vocab: int,
    n_batch: int,
    native_state_bytes: int,
) -> BlobComposition:
    """Predict the ``LlamaState`` pickle composition for one checkpoint.

    ``save_state`` copies ``self.scores[: n_tokens, :]``.  ``self.scores`` is
    allocated as ``(n_batch, n_vocab)`` when ``logits_all`` is false, so the row
    count saturates at ``n_batch``.  ``input_ids`` is one int32 per token.
    ``native_state_bytes`` is ``llama_state_get_size(ctx)`` for the checkpoint.
    """
    if n_tokens < 0 or n_vocab <= 0 or n_batch <= 0 or native_state_bytes < 0:
        raise ValueError("token/vocab/batch/native sizes must be sensible non-negatives")
    scored_rows = min(n_tokens, n_batch)
    return BlobComposition(
        n_tokens=n_tokens,
        scored_rows=scored_rows,
        scores_bytes=scored_rows * n_vocab * FLOAT32_BYTES,
        input_ids_bytes=n_tokens * INT32_BYTES,
        native_state_bytes=native_state_bytes,
    )


def native_codec_savings_bytes(composition: BlobComposition) -> int:
    """Bytes a native ``llama_state_get_data`` codec would stop persisting.

    A native codec restores the exact same context state (it is the same C
    entry point ``load_state`` already wraps) but never serializes the
    ``LlamaState`` score/input arrays, so it drops the entire wrapper overhead.
    """
    return composition.llama_state_overhead_bytes
