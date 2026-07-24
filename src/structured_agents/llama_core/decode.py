"""A small, explicit llama.cpp decode loop.

This module deliberately uses ``llama_decode`` and the sampler C API instead
of the high-level completion helper.  It is the teaching reference for later
grammar integration: a mask hook receives the zero-copy logits view before
sampler transforms run.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any, Protocol


class LogitsHook(Protocol):
    """Mutate a logits vector before sampler application."""

    def __call__(self, logits: Any) -> None: ...


class TokenHook(Protocol):
    """Observe exactly the token selected by the sampler."""

    def __call__(self, token: int) -> None: ...


def _llama_cpp() -> Any:
    """Import lazily so core-only installs do not import the native library."""
    import llama_cpp

    return llama_cpp


class OwnedLlamaDecoder:
    """Single-sequence owned decode loop backed by a ``llama_cpp.Llama``.

    ``Llama`` remains responsible only for model lifecycle and tokenization.
    Evaluation, zero-copy logits access, sampling, and sampler acceptance are
    owned here.  The sampler is accepted exactly once per output token.
    """

    def __init__(self, llm: Any, sampler: Any | None = None) -> None:
        self.llm = llm
        self._native = _llama_cpp()
        self._owns_sampler = sampler is None
        self.sampler = sampler if sampler is not None else self._new_greedy_sampler()
        self._batch = self._native.llama_batch_init(1, 0, 1)
        self._closed = False

    def _new_greedy_sampler(self) -> Any:
        chain = self._native.llama_sampler_chain_init(self._native.llama_sampler_chain_default_params())
        self._native.llama_sampler_chain_add(chain, self._native.llama_sampler_init_greedy())
        return chain

    def close(self) -> None:
        """Free the owned batch and, when applicable, sampler chain."""
        if self._closed:
            return
        self._native.llama_batch_free(self._batch)
        if self._owns_sampler:
            self._native.llama_sampler_free(self.sampler)
        self._closed = True

    def __enter__(self) -> OwnedLlamaDecoder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _decode_one(self, token: int, position: int) -> None:
        batch = self._batch
        batch.n_tokens = 1
        batch.token[0] = token
        batch.pos[0] = position
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1
        result = self._native.llama_decode(self.llm._ctx.ctx, batch)
        if result != 0:
            raise RuntimeError(f"llama_decode failed with code {result}")

    def _candidate_array(self) -> tuple[Any, Any]:
        """Copy logits into llama.cpp's candidate representation for sampling."""
        import numpy as np

        logits_pointer = self._native.llama_get_logits_ith(self.llm._ctx.ctx, 0)
        logits = np.ctypeslib.as_array(logits_pointer, shape=(self.llm.n_vocab(),))
        candidate_type = self._native.llama_token_data * self.llm.n_vocab()
        candidates = candidate_type()
        for index, logit in enumerate(logits):
            candidates[index].id = index
            candidates[index].logit = float(logit)
            candidates[index].p = 0.0
        array = self._native.llama_token_data_array(candidates, len(candidates), -1, False)
        return logits, array

    def generate_tokens(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
        logits_hook: LogitsHook | None = None,
        token_hook: TokenHook | None = None,
        stop_tokens: frozenset[int] = frozenset(),
    ) -> list[int]:
        """Generate greedily or with the supplied sampler, without completion APIs."""
        if self._closed:
            raise RuntimeError("OwnedLlamaDecoder is closed")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        # A new decoder is normally paired with a fresh context.  Reset is
        # necessary for hybrid models, and keeps lifecycle ownership explicit.
        self.llm.reset()
        for position, token in enumerate(prompt_tokens):
            self._decode_one(token, position)

        generated: list[int] = []
        next_position = len(prompt_tokens)
        for _ in range(max_tokens):
            logits, candidates = self._candidate_array()
            if logits_hook is not None:
                logits_hook(logits)
                # Candidate logits are copied after masking to make the order
                # visible and unambiguous to readers of the loop.
                for index, logit in enumerate(logits):
                    candidates.data[index].logit = float(logit)
            self._native.llama_sampler_apply(self.sampler, ctypes.byref(candidates))
            token = int(candidates.data[candidates.selected].id)
            # This is the sole llama sampler acceptance; the convenience
            # sampling function would accept internally.
            self._native.llama_sampler_accept(self.sampler, token)
            if token_hook is not None:
                token_hook(token)
            generated.append(token)
            if token in stop_tokens:
                break
            self._decode_one(token, next_position)
            next_position += 1
        return generated

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int,
        logits_hook: LogitsHook | None = None,
        token_hook: TokenHook | None = None,
    ) -> str:
        """Tokenize, run the owned loop, and detokenize its generated tokens."""
        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        tokens = self.generate_tokens(
            prompt_tokens,
            max_tokens=max_tokens,
            logits_hook=logits_hook,
            token_hook=token_hook,
            stop_tokens=frozenset({self.llm.token_eos()}),
        )
        return self.llm.detokenize(tokens, special=True).decode("utf-8", errors="replace")
