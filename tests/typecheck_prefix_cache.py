from __future__ import annotations

from typing import assert_type

from structured_agents.llama_core.prefix_cache import (
    CacheCompatibility,
    PrefixCacheBlobStore,
    PrefixCacheEntry,
    PrefixCacheIndex,
    PrefixCacheKey,
    RestorePlanDecision,
    check_compatibility,
    check_state_integrity,
    plan_restore,
)


def check_types(
    entry: PrefixCacheEntry,
    key: PrefixCacheKey,
    blob_store: PrefixCacheBlobStore,
    index: PrefixCacheIndex,
) -> None:
    assert_type(index.get(key), PrefixCacheEntry | None)
    assert_type(blob_store.read_blob(entry), bytes | None)
    assert_type(check_compatibility(entry, key), CacheCompatibility)
    assert_type(check_state_integrity(entry, b"state"), CacheCompatibility)
    assert_type(plan_restore(entry, key, (1,)), RestorePlanDecision)
