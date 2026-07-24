from __future__ import annotations

from pathlib import Path

from structured_agents.llama_core.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from structured_agents.llama_core.prefix_cache import (
    PersistentPrefixCache,
    PrefixCacheEntry,
    PrefixCacheKey,
    plan_restore,
    restore_then_decode_suffix,
)


def _key(tokens: tuple[int, ...] = (1, 2)) -> PrefixCacheKey:
    artifact = ArtifactIdentity(path="/model", sha256="a" * 64, size_bytes=1, mtime_ns=1, inode=1)
    fp = LlamaEngineFingerprint(
        model=artifact,
        tokenizer=artifact,
        llama_cpp_python_version="0.3.34",
        backend="cuda",
        n_ctx=64,
    )
    return PrefixCacheKey.from_fingerprint(namespace="test", fingerprint=fp, prefix_token_ids=tokens)


def test_atomic_publish_survives_new_store_and_accounts_hit(tmp_path: Path) -> None:
    store = PersistentPrefixCache(tmp_path)
    entry = store.publish(_key(), b"state", llama_state_version="whole-state-v1")
    assert entry.state_blob_path is not None
    assert not list((tmp_path / "blobs").glob("*.tmp"))
    found = PersistentPrefixCache(tmp_path).lookup(_key())
    assert found.hit and found.state == b"state" and found.reason == "hit"


def test_corruption_and_key_mismatch_are_safe_misses(tmp_path: Path) -> None:
    store = PersistentPrefixCache(tmp_path)
    entry = store.publish(_key(), b"state")
    assert not store.lookup(_key((1, 3))).hit
    (tmp_path / str(entry.state_blob_path)).write_bytes(b"corrupt")
    result = store.lookup(_key())
    assert not result.hit
    assert "state_" in result.reason


def test_restore_contract_loads_then_decodes_suffix() -> None:
    entry = PrefixCacheEntry(_key(), 1, "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881")
    decision = plan_restore(entry, _key(), (1, 2, 3))
    events: list[object] = []
    assert decision.plan is not None
    restore_then_decode_suffix(
        decision.plan,
        b"x",
        load_state=lambda _: events.append("load"),
        decode_suffix=lambda x: events.append(x),
    )
    assert events == ["load", (3,)]
