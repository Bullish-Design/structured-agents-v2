"""Contracts for a persistent, exact-prefix llama.cpp state snapshot cache.

This module deliberately has no llama.cpp dependency and performs no I/O.  It
defines the compatibility boundary that a later filesystem implementation and
state-capture codec must obey.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .fingerprint import LlamaEngineFingerprint

CACHE_FORMAT_VERSION = 1


class CacheRejectionReason(StrEnum):
    """Why an entry must not be used for a restore."""

    NAMESPACE_MISMATCH = "namespace_mismatch"
    FORMAT_VERSION_MISMATCH = "format_version_mismatch"
    ENGINE_FINGERPRINT_MISMATCH = "engine_fingerprint_mismatch"
    PREFIX_TOKEN_IDS_MISMATCH = "prefix_token_ids_mismatch"
    CHECKPOINT_TOKEN_COUNT_MISMATCH = "checkpoint_token_count_mismatch"
    REQUEST_DOES_NOT_EXTEND_PREFIX = "request_does_not_extend_prefix"
    SUFFIX_DECODE_REQUIRED = "suffix_decode_required"
    STATE_SIZE_MISMATCH = "state_size_mismatch"
    STATE_CHECKSUM_MISMATCH = "state_checksum_mismatch"


@dataclass(frozen=True, slots=True)
class PrefixCacheKey:
    """Identity of one exact checkpoint, including its compatible engine."""

    namespace: str
    engine_fingerprint_key: str
    prefix_token_ids: tuple[int, ...]
    checkpoint_token_count: int
    format_version: int = CACHE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if not self.engine_fingerprint_key:
            raise ValueError("engine_fingerprint_key must be non-empty")
        if self.format_version <= 0:
            raise ValueError("format_version must be positive")
        if self.checkpoint_token_count != len(self.prefix_token_ids):
            raise ValueError("checkpoint_token_count must equal len(prefix_token_ids)")
        if any(token < 0 for token in self.prefix_token_ids):
            raise ValueError("prefix_token_ids must contain non-negative token IDs")

    @classmethod
    def from_fingerprint(
        cls,
        *,
        namespace: str,
        fingerprint: LlamaEngineFingerprint,
        prefix_token_ids: tuple[int, ...],
        format_version: int = CACHE_FORMAT_VERSION,
    ) -> PrefixCacheKey:
        """Construct a key from the complete, frozen engine fingerprint."""
        return cls(
            namespace=namespace,
            engine_fingerprint_key=fingerprint.cache_key(),
            prefix_token_ids=prefix_token_ids,
            checkpoint_token_count=len(prefix_token_ids),
            format_version=format_version,
        )

    @property
    def storage_key(self) -> str:
        """Injective, deterministic serialization for an index or blob name.

        The complete serialized fields remain in the key instead of relying on
        a digest alone, so distinct fingerprints or token sequences cannot map
        to the same value through this contract.
        """
        fields = (
            str(self.format_version),
            _encode_text(self.namespace),
            _encode_text(self.engine_fingerprint_key),
            str(self.checkpoint_token_count),
            ".".join(str(token) for token in self.prefix_token_ids),
        )
        return "prefix-cache:" + "|".join(fields)


@dataclass(frozen=True, slots=True)
class PrefixCacheEntry:
    """Indexed metadata for one persisted whole-state snapshot."""

    key: PrefixCacheKey
    state_size_bytes: int
    state_checksum_sha256: str

    def __post_init__(self) -> None:
        if self.state_size_bytes < 0:
            raise ValueError("state_size_bytes must be non-negative")
        if len(self.state_checksum_sha256) != 64:
            raise ValueError("state_checksum_sha256 must be a SHA-256 hex digest")
        try:
            int(self.state_checksum_sha256, 16)
        except ValueError as exc:
            raise ValueError("state_checksum_sha256 must be a SHA-256 hex digest") from exc

    @property
    def format_version(self) -> int:
        return self.key.format_version


@dataclass(frozen=True, slots=True)
class CacheCompatibility:
    """A non-throwing compatibility result suitable for a cache fallback."""

    accepted: bool
    reason: CacheRejectionReason | None = None
    detail: str | None = None

    @classmethod
    def reject(cls, reason: CacheRejectionReason, detail: str) -> CacheCompatibility:
        return cls(accepted=False, reason=reason, detail=detail)


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """The only safe restore data flow: restore then decode fresh suffix tokens."""

    entry: PrefixCacheEntry
    cached_prefix_token_ids: tuple[int, ...]
    uncached_suffix_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.uncached_suffix_token_ids:
            raise ValueError("restore plan requires at least one suffix token decode for fresh logits")


@dataclass(frozen=True, slots=True)
class RestorePlanDecision:
    """A restore plan or an explicit reason to fall back to normal prefill."""

    plan: RestorePlan | None = None
    rejection: CacheCompatibility | None = None

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.rejection is None):
            raise ValueError("restore decision must contain exactly one of plan or rejection")

    @property
    def can_restore(self) -> bool:
        return self.plan is not None


class PrefixCacheBlobStore(Protocol):
    """Blob boundary for a later atomic filesystem implementation."""

    def read_blob(self, entry: PrefixCacheEntry) -> bytes | None: ...

    def write_blob(self, entry: PrefixCacheEntry, state: bytes) -> None: ...


class PrefixCacheIndex(Protocol):
    """Checkpoint-boundary metadata lookup; deliberately not a radix tree."""

    def get(self, key: PrefixCacheKey) -> PrefixCacheEntry | None: ...

    def put(self, entry: PrefixCacheEntry) -> None: ...


def check_compatibility(entry: PrefixCacheEntry, requested_key: PrefixCacheKey) -> CacheCompatibility:
    """Require an exact namespace, format, engine, and prefix-token match."""
    cached = entry.key
    if cached.namespace != requested_key.namespace:
        return CacheCompatibility.reject(CacheRejectionReason.NAMESPACE_MISMATCH, "cache namespace differs")
    if cached.format_version != requested_key.format_version:
        return CacheCompatibility.reject(CacheRejectionReason.FORMAT_VERSION_MISMATCH, "cache format version differs")
    if cached.engine_fingerprint_key != requested_key.engine_fingerprint_key:
        return CacheCompatibility.reject(
            CacheRejectionReason.ENGINE_FINGERPRINT_MISMATCH, "frozen engine fingerprint differs"
        )
    if cached.checkpoint_token_count != requested_key.checkpoint_token_count:
        return CacheCompatibility.reject(
            CacheRejectionReason.CHECKPOINT_TOKEN_COUNT_MISMATCH, "checkpoint token count differs"
        )
    if cached.prefix_token_ids != requested_key.prefix_token_ids:
        return CacheCompatibility.reject(CacheRejectionReason.PREFIX_TOKEN_IDS_MISMATCH, "exact token IDs differ")
    return CacheCompatibility(accepted=True)


def check_state_integrity(entry: PrefixCacheEntry, state: bytes) -> CacheCompatibility:
    """Verify blob size and checksum before passing bytes to a restore codec."""
    if len(state) != entry.state_size_bytes:
        return CacheCompatibility.reject(CacheRejectionReason.STATE_SIZE_MISMATCH, "persisted state size differs")
    actual = hashlib.sha256(state).hexdigest()
    if actual != entry.state_checksum_sha256.lower():
        return CacheCompatibility.reject(
            CacheRejectionReason.STATE_CHECKSUM_MISMATCH, "persisted state checksum differs"
        )
    return CacheCompatibility(accepted=True)


def plan_restore(
    entry: PrefixCacheEntry, requested_key: PrefixCacheKey, request_token_ids: tuple[int, ...]
) -> RestorePlanDecision:
    """Make a safe restore plan, requiring a post-restore suffix decode.

    Saved llama.cpp state excludes the output logits buffer.  A zero-token
    suffix would therefore invite callers to consume stale logits.
    """
    compatible = check_compatibility(entry, requested_key)
    if not compatible.accepted:
        return RestorePlanDecision(rejection=compatible)
    prefix = entry.key.prefix_token_ids
    if request_token_ids[: len(prefix)] != prefix:
        return RestorePlanDecision(
            rejection=CacheCompatibility.reject(
                CacheRejectionReason.REQUEST_DOES_NOT_EXTEND_PREFIX, "request does not begin with cached token IDs"
            )
        )
    suffix = request_token_ids[len(prefix) :]
    if not suffix:
        return RestorePlanDecision(
            rejection=CacheCompatibility.reject(
                CacheRejectionReason.SUFFIX_DECODE_REQUIRED, "restore requires one or more uncached suffix tokens"
            )
        )
    return RestorePlanDecision(plan=RestorePlan(entry, prefix, suffix))


def _encode_text(value: str) -> str:
    """Length-prefix UTF-8 text so delimiter characters remain unambiguous."""
    encoded = value.encode("utf-8").hex()
    return f"{len(encoded)}:{encoded}"
