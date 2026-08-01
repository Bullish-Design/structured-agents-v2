"""Mixed-batch per-sequence LoRA routing — the P2 fork's ctypes surface (Pillar).

The private P2 fork of the pinned llama.cpp adds two C entry points that let a
single ``llama_decode`` carry a *mix* of adapters — different sequences in one
ubatch use different LoRA adapters (the vLLM/Punica capability):

* ``llama_set_seq_adapters(ctx, adapters, n)`` — register the ordered adapter pool
  once; the pool index becomes the routing id.
* ``llama_set_seq_adapter(ctx, seq_id, idx)`` — assign a sequence to a pool slot
  (``-1`` = no adapter / raw base model).

Provenance and evidence live in ``.scratch/projects/17-…`` (``20-P2-MIXED-BATCH-GO``,
``21-P2-THROUGHPUT``). This module lifts the hand-bound ctypes from
``benchmarks/project17/context_pool_router.py`` (``enable_seq_routing`` /
``run_seq_routed``) into the library behind a narrow, capability-guarded surface.

Fail-closed contract (repo standing rule): the routing symbols only exist on a
fork lib. On a stock lib the guard raises :class:`SeqRoutingUnavailable` at *bind*
time; callers (the router's ``auto`` backend) treat that as "capability absent" and
fall back to the context-pool path — a missing capability is never an inference
failure. Pydantic stays at the config boundary; this binding is plain ctypes.
"""

from __future__ import annotations

import ctypes
from typing import Any

# The two fork-only symbols. A lib is fork-capable iff it exports both.
SEQ_ROUTING_SYMBOLS: tuple[str, ...] = ("llama_set_seq_adapters", "llama_set_seq_adapter")

# Sentinel adapter index meaning "no adapter / raw base model" (matches the fork's
# seq_adapter_map default and llama_set_seq_adapter's -1 contract).
NO_ADAPTER: int = -1


class SeqRoutingUnavailable(RuntimeError):
    """Raised when the loaded libllama lacks the P2-fork routing symbols.

    Signals *missing capability*, not a fault: the ``auto`` router backend catches
    it and falls back to the context-pool path.
    """


def library_supports_seq_routing(lib: Any) -> bool:
    """True iff ``lib`` (a loaded ctypes CDLL) exports both fork routing symbols."""
    return all(hasattr(lib, name) for name in SEQ_ROUTING_SYMBOLS)


class SeqRoutingBinding:
    """Typed ctypes binding for the P2 fork's per-sequence LoRA routing calls.

    Construct from the ``llama_cpp.llama_cpp`` module (the high-level package's
    ctypes namespace, exposing ``_lib`` and the shared ``argtypes`` aliases). The
    constructor probes for the fork symbols and installs ``argtypes`` / ``restype``
    exactly as the benchmark reference bound them; if the symbols are absent it
    raises :class:`SeqRoutingUnavailable` and touches nothing.
    """

    def __init__(self, llama_cpp_module: Any) -> None:
        self._C = llama_cpp_module
        lib = getattr(llama_cpp_module, "_lib", None)
        if lib is None:
            raise SeqRoutingUnavailable("llama_cpp module exposes no loaded library (_lib is None)")
        if not library_supports_seq_routing(lib):
            raise SeqRoutingUnavailable(
                "loaded libllama has no "
                f"{'/'.join(SEQ_ROUTING_SYMBOLS)} — need the P2 fork build "
                "(set LLAMA_CPP_LIB_PATH to a p2fork lib dir)"
            )
        self._lib = lib

        # argtypes/restypes copied verbatim from context_pool_router.py:523-535.
        lib.llama_set_seq_adapters.argtypes = [
            self._C.llama_context_p_ctypes,
            ctypes.POINTER(self._C.llama_adapter_lora_p_ctypes),
            ctypes.c_size_t,
        ]
        lib.llama_set_seq_adapters.restype = ctypes.c_int32
        lib.llama_set_seq_adapter.argtypes = [
            self._C.llama_context_p_ctypes,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.llama_set_seq_adapter.restype = ctypes.c_int32

    def set_seq_adapters(self, ctx: Any, adapter_ptrs: list[Any]) -> None:
        """Register the ordered adapter pool on ``ctx`` (index = routing id)."""
        n = len(adapter_ptrs)
        arr = (self._C.llama_adapter_lora_p_ctypes * n)(*adapter_ptrs)
        if self._lib.llama_set_seq_adapters(ctx, arr, n) != 0:
            raise RuntimeError("llama_set_seq_adapters failed")

    def set_seq_adapter(self, ctx: Any, seq_id: int, adapter_idx: int) -> None:
        """Route sequence ``seq_id`` to pool slot ``adapter_idx`` (``-1`` = base)."""
        if self._lib.llama_set_seq_adapter(ctx, seq_id, adapter_idx) != 0:
            raise RuntimeError(f"llama_set_seq_adapter(seq={seq_id}, idx={adapter_idx}) failed")
