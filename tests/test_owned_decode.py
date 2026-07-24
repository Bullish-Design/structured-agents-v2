from __future__ import annotations

from structured_agents.llama_core.decode import OwnedLlamaDecoder


def test_owned_loop_contains_one_explicit_sampler_accept() -> None:
    """Gate 1 regression: the teaching loop must not delegate to sample()."""
    source = OwnedLlamaDecoder.generate_tokens.__doc__
    assert source is not None

    # The implementation's public contract is independently pinned here.  A
    # native smoke (when the model is available) tests the C API separately.
    import inspect

    implementation = inspect.getsource(OwnedLlamaDecoder.generate_tokens)
    assert implementation.count(".llama_sampler_accept(") == 1
    assert "llama_sampler_sample" not in implementation
