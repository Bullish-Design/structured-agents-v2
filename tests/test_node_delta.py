"""Tests for the codebase context-tree layer (node_delta / _live / _blend).

The unit tests reuse the token-list fake pattern from ``test_prefix_cache_live``:
KV "state" is the exact token list decoded into a seq slot and ``last_token`` is
a deterministic function of that list, so a reconstructed chain (or a blended
assembly) yields the same greedy token as a cold prefill of the same tokens --
the no-GPU analogue of the token-exact correctness bar.  A fake native provides
``llama_get_memory`` / ``llama_memory_seq_rm`` so the DFS rewind between siblings
is exercised without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from structured_agents.llama_core.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from structured_agents.llama_core.node_blend_live import (
    PullInSpec,
    blend_by_reanchor,
    blend_by_redecode,
)
from structured_agents.llama_core.node_delta import (
    ChainRejectionReason,
    NodeDelta,
    NodeDeltaIndex,
    NodeInclusion,
    plan_chain,
    resolve_chain,
)
from structured_agents.llama_core.node_delta_live import (
    TreeNodeSpec,
    capture_tree,
    reconstruct_chain,
)
from structured_agents.llama_core.prefix_cache import PrefixCacheEntry, PrefixCacheKey
from structured_agents.llama_core.prefix_cache_live import InMemoryPrefixCache


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


class _FakeMemory:
    """Stand-in for the llama.cpp memory handle; truncation targets one bridge."""

    def __init__(self, bridge: "FakeTreeBridge") -> None:
        self.bridge = bridge


class _FakeNative:
    """Minimal native exposing only the two memory calls ``_seq_rm`` uses."""

    def __init__(self, bridge: "FakeTreeBridge") -> None:
        self._memory = _FakeMemory(bridge)

    def llama_get_memory(self, ctx: object) -> _FakeMemory:
        return self._memory

    def llama_memory_seq_rm(self, memory: _FakeMemory, seq_id: int, p0: int, p1: int) -> None:
        slot = memory.bridge.seq_state.setdefault(seq_id, [])
        # p1 == -1 means "to the end"; the DFS only ever rewinds a whole tail.
        end = len(slot) if p1 == -1 else p1
        del slot[p0:end]


class FakeTreeBridge:
    """Token-list bridge (as in test_prefix_cache_live) plus a rewindable native."""

    def __init__(self, n_vocab: int = 1009) -> None:
        self.n_vocab = n_vocab
        self.n_batch = 8
        self.seq_state: dict[int, list[int]] = {}
        self.reject_set_data = False
        self._last_seq = 0
        self.decode_calls: list[tuple[int, list[int], int]] = []
        self._native = _FakeNative(self)

    def capture_seq_state(self, ctx: object, seq_id: int) -> bytes:
        import json

        return json.dumps(self.seq_state.get(seq_id, [])).encode()

    def restore_seq_state(self, ctx: object, blob: bytes, seq_id: int) -> int:
        import json

        if self.reject_set_data:
            return 0
        self.seq_state[seq_id] = list(json.loads(blob.decode()))
        self._last_seq = seq_id
        return len(blob)

    def decode_tokens(self, ctx: object, tokens: object, seq_id: int, start_pos: int) -> None:
        slot = self.seq_state.setdefault(seq_id, [])
        assert start_pos == len(slot), (start_pos, len(slot))
        slot.extend(tokens)
        self._last_seq = seq_id
        self.decode_calls.append((seq_id, list(tokens), start_pos))

    def last_token(self, ctx: object) -> int:
        return sum(self.seq_state.get(self._last_seq, [])) % self.n_vocab


def _cold_token(tokens: tuple[int, ...], n_vocab: int = 1009) -> int:
    return sum(tokens) % n_vocab


def _key(tokens: tuple[int, ...], fp: LlamaEngineFingerprint) -> PrefixCacheKey:
    return PrefixCacheKey.from_fingerprint(namespace="tree", fingerprint=fp, prefix_token_ids=tokens)


# --------------------------------------------------------------------------- #
# Pure tree policy: resolve_chain / plan_chain / invariants
# --------------------------------------------------------------------------- #


def _index_with_chain(tmp_path: Path, fp: LlamaEngineFingerprint) -> NodeDeltaIndex:
    idx = NodeDeltaIndex(tmp_path)
    idx.put(NodeDelta("root", None, _key((1, 2), fp), (1, 2), 0))
    idx.put(NodeDelta("file", "root", _key((1, 2, 3, 4), fp), (3, 4), 2, source_path="a.py"))
    idx.put(NodeDelta("sym", "file", _key((1, 2, 3, 4, 5), fp), (5,), 4))
    return idx


def test_resolve_chain_returns_root_to_target(tmp_path: Path) -> None:
    idx = _index_with_chain(tmp_path, _fingerprint())
    decision = resolve_chain(idx, "sym")
    assert decision.can_restore
    assert [n.node_id for n in decision.plan.chain] == ["root", "file", "sym"]
    assert decision.plan.cumulative_token_ids == (1, 2, 3, 4, 5)


def test_exclude_node_drops_its_tokens_from_chain(tmp_path: Path) -> None:
    idx = _index_with_chain(tmp_path, _fingerprint())
    # Re-mark "file" as EXCLUDE: it contributes no tokens but still parents "sym".
    idx.put(NodeDelta("file", "root", _key((1, 2, 3, 4), _fingerprint()), (3, 4), 2, inclusion=NodeInclusion.EXCLUDE))
    decision = resolve_chain(idx, "sym")
    assert [n.node_id for n in decision.plan.chain] == ["root", "sym"]


def test_broken_parent_link_is_data_not_exception(tmp_path: Path) -> None:
    idx = NodeDeltaIndex(tmp_path)
    idx.put(NodeDelta("orphan", "ghost", _key((7,), _fingerprint()), (7,), 0))
    decision = resolve_chain(idx, "orphan")
    assert not decision.can_restore
    assert decision.reason == ChainRejectionReason.BROKEN_PARENT_LINK


def test_missing_node_is_data(tmp_path: Path) -> None:
    idx = NodeDeltaIndex(tmp_path)
    decision = resolve_chain(idx, "nope")
    assert decision.reason == ChainRejectionReason.NODE_NOT_FOUND


def test_node_delta_rejects_inconsistent_span_and_position() -> None:
    fp = _fingerprint()
    with pytest.raises(ValueError):
        # base_position + len(span) != checkpoint count
        NodeDelta("x", None, _key((1, 2, 3), fp), (2, 3), base_position=0)
    with pytest.raises(ValueError):
        # key prefix does not end with span
        NodeDelta("x", None, _key((1, 2, 3), fp), (9,), base_position=2)


def test_plan_chain_builds_restore_with_prompt_as_suffix(tmp_path: Path) -> None:
    fp = _fingerprint()
    idx = _index_with_chain(tmp_path, fp)
    entry = PrefixCacheEntry(_key((1, 2, 3, 4, 5), fp), 10, "0" * 64)
    decision = plan_chain(idx, entry, "sym", (99,))
    assert decision.plan is not None
    assert decision.plan.cached_prefix_token_ids == (1, 2, 3, 4, 5)
    assert decision.plan.uncached_suffix_token_ids == (99,)


def test_plan_chain_requires_a_prompt(tmp_path: Path) -> None:
    fp = _fingerprint()
    idx = _index_with_chain(tmp_path, fp)
    entry = PrefixCacheEntry(_key((1, 2, 3, 4, 5), fp), 10, "0" * 64)
    with pytest.raises(ValueError):
        plan_chain(idx, entry, "sym", ())


# --------------------------------------------------------------------------- #
# Live capture_tree DFS + reconstruct_chain
# --------------------------------------------------------------------------- #


def _capture_demo_tree(bridge: FakeTreeBridge, cache: InMemoryPrefixCache, idx: NodeDeltaIndex, fp) -> None:
    nodes = [
        TreeNodeSpec("root", None, (1, 2)),
        TreeNodeSpec("fileA", "root", (3, 4)),
        TreeNodeSpec("symA", "fileA", (5,)),
        TreeNodeSpec("fileB", "root", (7, 8, 9)),  # sibling of fileA
        TreeNodeSpec("symB", "fileB", (6,)),
    ]
    capture_tree(
        bridge, None, cache=cache, index=idx, namespace="tree", fingerprint=fp,
        nodes=nodes, n_seq_max=2, seq_id=0,
    )


def test_capture_tree_decodes_each_node_once_and_rewinds_siblings(tmp_path: Path) -> None:
    fp = _fingerprint()
    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    _capture_demo_tree(bridge, cache, idx, fp)

    # Each node's own span decoded exactly once (5 spans, DFS order).
    assert [tokens for _, tokens, _ in bridge.decode_calls] == [[1, 2], [3, 4], [5], [7, 8, 9], [6]]
    # fileB's span was decoded at position 2 -- proving symA's subtree was rewound
    # back to root's boundary, not appended after it.
    assert bridge.decode_calls[3] == (0, [7, 8, 9], 2)
    # Every node has a durable blob under its cumulative-chain key.
    for cum in [(1, 2), (1, 2, 3, 4), (1, 2, 3, 4, 5), (1, 2, 7, 8, 9), (1, 2, 7, 8, 9, 6)]:
        assert cache.lookup(_key(cum, fp)).hit


def test_reconstruct_chain_matches_cold_prefill(tmp_path: Path) -> None:
    fp = _fingerprint()
    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    _capture_demo_tree(bridge, cache, idx, fp)

    prompt = (42,)
    # symB's cumulative chain is root+fileB+symB = (1,2,7,8,9,6); + prompt.
    cold = _cold_token((1, 2, 7, 8, 9, 6) + prompt)

    restore_bridge = FakeTreeBridge()
    result = reconstruct_chain(
        restore_bridge, None, cache=cache, index=idx, node_id="symB",
        prompt_token_ids=prompt, n_seq_max=2, seq_id=1,
    )
    assert result.restored, getattr(result, "reason", result)
    assert result.next_token == cold  # token-exact: no codebase prefill, prompt only
    # Only the prompt was decoded live; the whole chain came from the blob.
    assert restore_bridge.decode_calls == [(1, list(prompt), 6)]


def test_reconstruct_missing_node_is_chain_decision(tmp_path: Path) -> None:
    fp = _fingerprint()
    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    _capture_demo_tree(bridge, cache, idx, fp)
    result = reconstruct_chain(
        FakeTreeBridge(), None, cache=cache, index=idx, node_id="ghost",
        prompt_token_ids=(1,), n_seq_max=2, seq_id=0,
    )
    assert result.reason == ChainRejectionReason.NODE_NOT_FOUND


def test_summarize_node_gives_children_the_digest(tmp_path: Path) -> None:
    fp = _fingerprint()
    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    nodes = [
        TreeNodeSpec("root", None, (1, 2)),
        TreeNodeSpec("big", "root", (3, 4, 5, 6, 7, 8), NodeInclusion.SUMMARIZE, summary_token_ids=(90, 91)),
        TreeNodeSpec("sym", "big", (5,)),
    ]
    capture_tree(bridge, None, cache=cache, index=idx, namespace="tree", fingerprint=fp,
                 nodes=nodes, n_seq_max=2, seq_id=0)

    # The raw body (3..8) was never decoded; the summary (90,91) took its place.
    decoded = [tokens for _, tokens, _ in bridge.decode_calls]
    assert decoded == [[1, 2], [90, 91], [5]]
    # sym's cumulative chain therefore inherits the digest, not the body.
    assert cache.lookup(_key((1, 2, 90, 91, 5), fp)).hit
    assert idx.get("big").span_token_ids == (90, 91)


# --------------------------------------------------------------------------- #
# Cross-chain blend
# --------------------------------------------------------------------------- #


def test_blend_by_redecode_admits_within_budget_and_matches_cold(tmp_path: Path) -> None:
    fp = _fingerprint()
    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    _capture_demo_tree(bridge, cache, idx, fp)
    # Two pull-in defs on their own chains (indexed only; no blob needed -- blend
    # re-decodes their token IDs, it does not restore their KV).
    idx.put(NodeDelta("defA", None, _key((50, 51, 52), fp), (50, 51, 52), 0))
    idx.put(NodeDelta("defB", None, _key((60, 61, 62, 63, 64), fp), (60, 61, 62, 63, 64), 0))

    prompt = (999,)
    blend_bridge = FakeTreeBridge()
    result = blend_by_redecode(
        blend_bridge, None, cache=cache, index=idx, base_node_id="fileA",
        pull_ins=[PullInSpec("defA"), PullInSpec("defB"), PullInSpec("ghost")],
        prompt_token_ids=prompt, n_seq_max=2, token_budget=6, seq_id=1,
    )
    assert result.ok
    assert result.admitted == ("defA",)
    assert dict(result.dropped)["defB"] == "token_budget_exhausted"
    assert dict(result.dropped)["ghost"] == str(ChainRejectionReason.NODE_NOT_FOUND)
    # fileA chain = (1,2,3,4); + defA span (50,51,52) + prompt (999).
    assembled = (1, 2, 3, 4, 50, 51, 52) + prompt
    assert result.prompt_next_token == _cold_token(assembled)
    assert result.end_position == len(assembled)
    # The base chain was restored (not re-decoded); only pull-in + prompt decoded.
    assert blend_bridge.decode_calls == [(1, [50, 51, 52, 999], 4)]


def test_blend_base_tree_failure_is_chain_decision(tmp_path: Path) -> None:
    fp = _fingerprint()
    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    _capture_demo_tree(bridge, cache, idx, fp)
    result = blend_by_redecode(
        FakeTreeBridge(), None, cache=cache, index=idx, base_node_id="ghost",
        pull_ins=[], prompt_token_ids=(1,), n_seq_max=2, seq_id=0,
    )
    assert result.reason == ChainRejectionReason.NODE_NOT_FOUND


def test_blend_by_reanchor_is_stub_with_position_delta(tmp_path: Path) -> None:
    fp = _fingerprint()
    delta = NodeDelta("defA", None, _key((50, 51, 52), fp), (50, 51, 52), base_position=0)
    with pytest.raises(NotImplementedError) as excinfo:
        blend_by_reanchor(FakeTreeBridge(), None, pull_in=delta, new_base_position=4, heal_fraction=0.1)
    assert "position_delta=4" in str(excinfo.value)
