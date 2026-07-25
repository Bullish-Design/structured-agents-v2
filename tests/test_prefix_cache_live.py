"""Tests for the live per-seq KV state bridge (prefix_cache_live).

The unit tests use a token-list fake for the ctypes bridge, so they exercise the
capture -> publish -> lookup -> restore -> suffix-decode wiring, the n_seq_max
portability gate, and the fail-closed set_data reject with no GPU.  The final
test is a GPU-gated integration proving restored continuation == cold-prefill
continuation, token-exact greedy, and is skipped without a model/GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from structured_agents.llama_core.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from structured_agents.llama_core.prefix_cache import (
    CacheRejectionReason,
    PersistentPrefixCache,
    PrefixCacheKey,
)
from structured_agents.llama_core.prefix_cache_live import (
    N_SEQ_MAX_FACT,
    InMemoryPrefixCache,
    LiveRestoreResult,
    capture_prefix,
    restore_and_continue,
)


def _fingerprint(model_digest: str = "a" * 64) -> LlamaEngineFingerprint:
    artifact = ArtifactIdentity(path="/models/ornith.gguf", sha256=model_digest, size_bytes=1, mtime_ns=1, inode=1)
    return LlamaEngineFingerprint(
        model=artifact,
        tokenizer=artifact,
        llama_cpp_python_version="0.3.34",
        llama_cpp_commit="b10103",
        backend="cuda",
        n_ctx=2048,
    )


class FakeSeqBridge:
    """Duck-typed stand-in for LlamaSeqStateBridge with no ctypes/GPU.

    KV "state" is modeled as the exact token list decoded into a sequence slot.
    ``last_token`` is a deterministic function of the slot's full token list, so
    a restore that reconstructs prefix+suffix yields the same token as a cold
    prefill of prefix+suffix -- the fake analogue of the GPU correctness bar.
    """

    def __init__(self, n_vocab: int = 1009) -> None:
        self.n_vocab = n_vocab
        self.n_batch = 8
        self.seq_state: dict[int, list[int]] = {}
        self.reject_set_data = False
        self._last_seq = 0
        self.decode_calls: list[tuple[int, list[int], int]] = []

    def capture_seq_state(self, ctx: object, seq_id: int) -> bytes:
        return json.dumps(self.seq_state.get(seq_id, [])).encode()

    def restore_seq_state(self, ctx: object, blob: bytes, seq_id: int) -> int:
        if self.reject_set_data:
            return 0
        self.seq_state[seq_id] = list(json.loads(blob.decode()))
        self._last_seq = seq_id
        return len(blob)

    def decode_tokens(self, ctx: object, tokens: object, seq_id: int, start_pos: int) -> None:
        slot = self.seq_state.setdefault(seq_id, [])
        # The bridge must supply explicit positions immediately after the slot's
        # current tokens; a gap would mean the restored KV was not honored.
        assert start_pos == len(slot), (start_pos, len(slot))
        slot.extend(tokens)
        self._last_seq = seq_id
        self.decode_calls.append((seq_id, list(tokens), start_pos))

    def last_token(self, ctx: object) -> int:
        return sum(self.seq_state.get(self._last_seq, [])) % self.n_vocab


def _cold_baseline(bridge: FakeSeqBridge, tokens: tuple[int, ...], seq_id: int = 3) -> int:
    bridge.seq_state[seq_id] = []
    bridge.decode_tokens(None, tokens, seq_id, 0)
    return bridge.last_token(None)


def test_in_memory_cache_round_trips_and_records_n_seq_max() -> None:
    cache = InMemoryPrefixCache()
    bridge = FakeSeqBridge()
    fp = _fingerprint()
    prefix = (1, 2, 3, 4, 5)

    entry = capture_prefix(
        bridge, None, cache=cache, namespace="demo", fingerprint=fp, prefix_token_ids=prefix, n_seq_max=2, seq_id=0
    )

    assert dict(entry.runtime_facts)[N_SEQ_MAX_FACT] == "2"
    found = cache.lookup(entry.key)
    assert found.hit and found.state == json.dumps(list(prefix)).encode()


def test_capture_then_restore_matches_cold_continuation() -> None:
    cache = InMemoryPrefixCache()
    fp = _fingerprint()
    prefix = (11, 22, 33, 44)
    suffix = (55, 66)
    request = prefix + suffix

    cold = _cold_baseline(FakeSeqBridge(), request)

    capture_bridge = FakeSeqBridge()
    capture_bridge.seq_state[0] = []
    entry = capture_prefix(
        capture_bridge, None, cache=cache, namespace="demo", fingerprint=fp,
        prefix_token_ids=prefix, n_seq_max=2, seq_id=0,
    )

    restore_bridge = FakeSeqBridge()
    result = restore_and_continue(
        restore_bridge, None, cache=cache, key=entry.key, request_token_ids=request, n_seq_max=2, seq_id=1
    )

    assert result.restored and result.reason == "hit"
    assert result.next_token == cold
    assert result.suffix_token_count == len(suffix)
    # The suffix decode landed after the restored prefix at explicit positions.
    assert restore_bridge.decode_calls == [(1, list(suffix), len(prefix))]


def test_restore_rejects_n_seq_max_mismatch_before_touching_state() -> None:
    cache = InMemoryPrefixCache()
    fp = _fingerprint()
    prefix, suffix = (7, 8, 9), (10,)
    entry = capture_prefix(
        FakeSeqBridge(), None, cache=cache, namespace="demo", fingerprint=fp,
        prefix_token_ids=prefix, n_seq_max=2, seq_id=0,
    )

    restore_bridge = FakeSeqBridge()
    result = restore_and_continue(
        restore_bridge, None, cache=cache, key=entry.key, request_token_ids=prefix + suffix, n_seq_max=4, seq_id=0
    )

    assert result.rejected
    assert result.reason == str(CacheRejectionReason.N_SEQ_MAX_MISMATCH)
    assert restore_bridge.decode_calls == []  # never decoded on a portability reject


def test_restore_fails_closed_when_set_data_returns_zero() -> None:
    cache = InMemoryPrefixCache()
    fp = _fingerprint()
    prefix, suffix = (2, 4, 6), (8,)
    entry = capture_prefix(
        FakeSeqBridge(), None, cache=cache, namespace="demo", fingerprint=fp,
        prefix_token_ids=prefix, n_seq_max=2, seq_id=0,
    )

    restore_bridge = FakeSeqBridge()
    restore_bridge.reject_set_data = True
    result = restore_and_continue(
        restore_bridge, None, cache=cache, key=entry.key, request_token_ids=prefix + suffix, n_seq_max=2, seq_id=0
    )

    assert result.rejected
    assert result.reason == str(CacheRejectionReason.STATE_SET_DATA_REJECTED)
    assert result.set_data_return == 0


def test_restore_rejects_request_that_does_not_extend_prefix() -> None:
    cache = InMemoryPrefixCache()
    fp = _fingerprint()
    entry = capture_prefix(
        FakeSeqBridge(), None, cache=cache, namespace="demo", fingerprint=fp,
        prefix_token_ids=(1, 2, 3), n_seq_max=2, seq_id=0,
    )

    result = restore_and_continue(
        FakeSeqBridge(), None, cache=cache, key=entry.key, request_token_ids=(1, 2, 99, 4), n_seq_max=2, seq_id=0
    )

    assert result.rejected
    assert result.reason == str(CacheRejectionReason.REQUEST_DOES_NOT_EXTEND_PREFIX)


def test_restore_miss_is_data_not_exception() -> None:
    cache = InMemoryPrefixCache()
    missing = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(5, 6))
    result = restore_and_continue(
        FakeSeqBridge(), None, cache=cache, key=missing, request_token_ids=(5, 6, 7), n_seq_max=2, seq_id=0
    )
    assert isinstance(result, LiveRestoreResult)
    assert result.rejected and result.reason == "miss_not_found"


def test_persistent_cache_backs_the_bridge_round_trip(tmp_path: Path) -> None:
    cache = PersistentPrefixCache(tmp_path)
    fp = _fingerprint()
    prefix, suffix = (3, 1, 4, 1, 5), (9, 2)
    request = prefix + suffix
    cold = _cold_baseline(FakeSeqBridge(), request)

    entry = capture_prefix(
        FakeSeqBridge(), None, cache=cache, namespace="demo", fingerprint=fp,
        prefix_token_ids=prefix, n_seq_max=2, seq_id=0,
    )
    # A fresh cache object proves durability of the blob + n_seq_max fact.
    reopened = PersistentPrefixCache(tmp_path)
    result = restore_and_continue(
        FakeSeqBridge(), None, cache=reopened, key=entry.key, request_token_ids=request, n_seq_max=2, seq_id=1
    )
    assert result.restored and result.next_token == cold


class _FakeNativeRestore:
    """Minimal ``llama_cpp`` stand-in exposing only ``llama_state_seq_set_data``.

    Models the multi-slot restore path used by the context-pool router: the same
    blob is loaded into several seq slots of one context.  ``reject`` forces the
    ``0`` return (n_seq_max mismatch / incompatible blob) so the fail-closed
    branch is exercised without a GPU.
    """

    def __init__(self, reject: bool = False) -> None:
        self.reject = reject
        self.loaded: list[tuple[int, bytes]] = []

    def llama_state_seq_set_data(self, ctx: object, array: object, size: int, seq_id: int) -> int:
        if self.reject:
            return 0
        self.loaded.append((seq_id, bytes(array)))
        return size


def test_restore_blob_into_seq_loads_same_blob_into_multiple_slots() -> None:
    from structured_agents.llama_core.prefix_cache_live import LlamaSeqStateBridge

    native = _FakeNativeRestore()
    bridge = LlamaSeqStateBridge(n_batch=8, n_vocab=1009, native=native)
    blob = b"prefix-kv-blob"

    codes = [bridge.restore_blob_into_seq(None, blob, seq_id) for seq_id in range(3)]

    assert codes == [len(blob)] * 3
    assert [seq for seq, _ in native.loaded] == [0, 1, 2]
    assert all(loaded == blob for _, loaded in native.loaded)


def test_restore_blob_into_seq_fails_closed_on_zero() -> None:
    from structured_agents.llama_core.prefix_cache_live import (
        LlamaSeqStateBridge,
        SeqRestoreRejected,
    )

    bridge = LlamaSeqStateBridge(n_batch=8, n_vocab=1009, native=_FakeNativeRestore(reject=True))

    with pytest.raises(SeqRestoreRejected) as excinfo:
        bridge.restore_blob_into_seq(None, b"blob", seq_id=2)
    assert excinfo.value.seq_id == 2


# --------------------------------------------------------------------------- #
# GPU-gated integration test: the real correctness bar.
# --------------------------------------------------------------------------- #

_MODEL_PATH = Path(
    os.environ.get(
        "LLAMA_TEST_MODEL",
        "/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf",
    )
)


def _gpu_ready() -> bool:
    if not os.environ.get("LLAMA_CPP_LIB_PATH") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return False
    if not _MODEL_PATH.exists():
        return False
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gpu_ready(), reason="requires GPU env (LLAMA_CPP_LIB_PATH, CUDA, model)")
def test_gpu_restored_continuation_matches_cold_prefill(tmp_path: Path) -> None:
    from llama_cpp import Llama, llama_cpp

    from structured_agents.llama_core.prefix_cache_live import LlamaSeqStateBridge

    n_ctx, n_batch, n_seq_max, seq_id = 2048, 128, 2, 1
    llm = Llama(model_path=str(_MODEL_PATH), n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=-1, seed=17018, verbose=False)
    model = llm._model.model
    bridge = LlamaSeqStateBridge(n_batch=n_batch, n_vocab=llm._n_vocab)

    def make_ctx() -> object:
        params = llama_cpp.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_ubatch = n_batch
        params.n_seq_max = n_seq_max
        return llama_cpp.llama_new_context_with_model(model, params)

    tokens = list(llm.tokenize(b"Shared router prefix for KV state reuse. " * 40, add_bos=True, special=True))
    prefix = tuple(tokens[:200])
    request = tuple(tokens[:201])  # one uncached suffix token

    # Cold prefill baseline: decode prefix + suffix fresh, read greedy argmax.
    cold_ctx = make_ctx()
    bridge.decode_tokens(cold_ctx, list(request), seq_id, 0)
    cold_token = bridge.last_token(cold_ctx)
    llama_cpp.llama_free(cold_ctx)

    # Capture the prefix blob through the real cache (with disk round-trip).
    fp = _fingerprint()
    capture_ctx = make_ctx()
    cache = PersistentPrefixCache(tmp_path)
    entry = capture_prefix(
        bridge, capture_ctx, cache=cache, namespace="gpu", fingerprint=fp,
        prefix_token_ids=prefix, n_seq_max=n_seq_max, seq_id=seq_id,
    )
    llama_cpp.llama_free(capture_ctx)

    # Restore into a fresh context and decode only the suffix.
    restore_ctx = make_ctx()
    result = restore_and_continue(
        bridge, restore_ctx, cache=PersistentPrefixCache(tmp_path), key=entry.key,
        request_token_ids=request, n_seq_max=n_seq_max, seq_id=seq_id,
    )
    llama_cpp.llama_free(restore_ctx)
    llm.close()

    assert result.restored, result.reason
    assert result.set_data_return and result.set_data_return > 0
    assert result.next_token == cold_token  # token-exact greedy correctness bar
