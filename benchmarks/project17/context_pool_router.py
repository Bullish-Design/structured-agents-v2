"""Context-pool multi-LoRA router (guide 14 §10 / Decision D2 path a).

The no-fork router: load the base model ONCE, then keep a pool of llama_contexts
that SHARE those base weights, each context pinned to a different LoRA adapter via
llama_set_adapters_lora. A request carries an adapter name; the router routes it to
that adapter's context, batches concurrent requests within the context (proven
~4x multi-seq decode), and multiplexes across contexts.

This is NOT mixed-batch multi-LoRA (that needs the §7 fork): within a single
llama_decode every sequence uses the same adapter. Different adapters live in
different contexts. That is exactly what the library supports today, with zero
C++ changes, reusing the own-batch decode + per-seq primitives proven in
run_seq_reuse.py / run_seq_batch_breakeven.py.

GPU-only usage (see memory `llama-cpp-gpu-driver-stub-fix`):
  LD_LIBRARY_PATH=/run/opengl-driver/lib:<zlib>:<stdcxx>:<build/lib>
  CUDA_VISIBLE_DEVICES=0  LLAMA_CPP_LIB_PATH=<build/lib>
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Dogfood the library's live prefix-cache bridge. The benchmark runners put
# ``src`` on PYTHONPATH; add it here too so this module imports standalone.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inferference.prefix_cache_live import (  # noqa: E402
    LlamaSeqStateBridge,
)

BASE = None  # sentinel adapter name for "no adapter" (raw base model)


@dataclass
class Request:
    rid: str
    prompt: str
    adapter: Optional[str]        # adapter name, or BASE (None) for raw base
    max_tokens: int = 32


@dataclass
class Generation:
    rid: str
    adapter: Optional[str]
    prompt_tokens: list[int]
    tokens: list[int] = field(default_factory=list)


@dataclass
class PrefixCache:
    """A KV blob for a shared prefix, captured under a specific adapter's context.

    The blob is adapter-specific: layer-3 K/V are perturbed by the pinned adapter,
    so a prefix cached under adapter X is only valid when restored into adapter X's
    context. Only portable into a context with the same n_seq_max (proven rule).
    """
    adapter: Optional[str]
    tokens: list[int]
    n: int
    blob: bytes


class ContextPoolRouter:
    """One shared base model, one pinned-adapter context per adapter (+ base)."""

    def __init__(
        self,
        model_path: str,
        adapters: dict[str, str],   # name -> LoRA gguf path
        *,
        n_ctx: int = 2048,
        n_batch: int = 256,
        n_seq_max: int = 8,
        scale: float = 1.0,
        seed: int = 17018,
        include_base: bool = True,
    ) -> None:
        from llama_cpp import Llama, llama_cpp

        self.C = llama_cpp
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_seq_max = n_seq_max
        # Load the base model exactly once. All contexts share llm._model.model.
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch,
                         n_gpu_layers=-1, seed=seed, verbose=False)
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        # Library bridge owns the correctness-critical per-seq KV discipline:
        # ctypes capture (llama_state_seq_get_data), own-batch suffix decode with
        # explicit positions, greedy last-token read, and the fail-closed
        # set_data==0 restore reject. The router keeps only its multi-seq batching.
        self._bridge = LlamaSeqStateBridge(n_batch=n_batch, n_vocab=self.n_vocab,
                                           native=self.C)

        # Load each adapter against the shared model.
        self.adapter_ptr: dict[str, Any] = {}
        for name, path in adapters.items():
            ptr = self.C.llama_adapter_lora_init(self.model, path.encode("utf-8"))
            if not ptr:
                raise RuntimeError(f"llama_adapter_lora_init failed for {name!r} ({path})")
            self.adapter_ptr[name] = ptr

        # One context per adapter, each pinned to its adapter. Optionally a base ctx.
        self.ctx: dict[Optional[str], Any] = {}
        names: list[Optional[str]] = ([BASE] if include_base else []) + list(adapters)
        for name in names:
            c = self._make_ctx(n_seq_max)
            if name is not BASE:
                self._pin_adapter(c, self.adapter_ptr[name], scale)
            self.ctx[name] = c

    # ---- low-level primitives (mirrors the proven runners) ----

    def _make_ctx(self, n_seq_max: int) -> Any:
        p = self.C.llama_context_default_params()
        p.n_ctx = self.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = n_seq_max
        ctx = self.C.llama_new_context_with_model(self.model, p)
        if not ctx:
            raise RuntimeError("llama_new_context_with_model returned NULL")
        return ctx

    def _pin_adapter(self, ctx: Any, ptr: Any, scale: float) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rc = self.C.llama_set_adapter_lora(ctx, ptr, scale)  # shim -> set_adapters_lora
        if rc != 0:
            raise RuntimeError(f"llama_set_adapters_lora rc={rc}")

    def _seq_rm(self, ctx: Any, seq_id: int) -> None:
        mem = self.C.llama_get_memory(ctx)
        self.C.llama_memory_seq_rm(mem, seq_id, -1, -1)

    def tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def _decode_prefill(self, ctx: Any, tokens: list[int], seq_id: int,
                        start_pos: int = 0) -> int:
        """Prefill one sequence at [start_pos, ...); return first sampled token.

        Own-batch decode with explicit positions (logits only on the last token)
        and the greedy last-token read are delegated to the library bridge — the
        same primitive that makes a restored prefix's suffix decode land on the
        cells right after the restored KV.
        """
        self._bridge.decode_tokens(ctx, tokens, seq_id, start_pos)
        return self._bridge.last_token(ctx)

    def _get_seq(self, ctx: Any, seq_id: int) -> bytes:
        return self._bridge.capture_seq_state(ctx, seq_id)

    def _batched_step(self, ctx: Any, cur: list[int], pos: list[int]) -> list[int]:
        """One token for each of len(cur) sequences (seq_id == row index)."""
        s = len(cur)
        batch = self.C.llama_batch_init(s, 0, 1)
        batch.n_tokens = s
        for i in range(s):
            batch.token[i] = cur[i]
            batch.pos[i] = pos[i]
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = i
            batch.logits[i] = 1
        rc = self.C.llama_decode(ctx, batch)
        if rc != 0:
            self.C.llama_batch_free(batch)
            raise RuntimeError(f"batched llama_decode rc={rc}")
        nxt = []
        for i in range(s):
            ptr = self.C.llama_get_logits_ith(ctx, i)
            nxt.append(int(np.argmax(np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)))))
        self.C.llama_batch_free(batch)
        return nxt

    # ---- generation ----

    def _generate_wave(self, ctx: Any, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
        """Greedily generate up to n_seq_max sequences concurrently in one context.

        Fixed-length greedy (no early EOS stop) so the batched decode keeps a
        stable seq_id==row invariant and matches a single-seq baseline exactly.
        """
        s = len(prompts)
        assert s <= self.n_seq_max, f"{s} > n_seq_max {self.n_seq_max}"
        for i in range(s):
            self._seq_rm(ctx, i)  # reset slot from any prior wave
        outputs: list[list[int]] = []
        cur: list[int] = []
        pos: list[int] = []
        for i in range(s):
            first = self._decode_prefill(ctx, prompts[i], i)
            outputs.append([first])
            cur.append(first)
            pos.append(len(prompts[i]))
        # Feed the just-sampled token at its own position, then advance. The first
        # sampled token sits at pos == len(prompt) (right after the prompt); pos is
        # advanced only AFTER the step so KV positions stay consecutive.
        for _ in range(max_tokens - 1):
            cur = self._batched_step(ctx, cur, pos)
            for i in range(s):
                outputs[i].append(cur[i])
                pos[i] += 1
        return outputs

    # ---- cached shared-prefix path ----

    def cache_prefix(self, adapter: Optional[str], prefix_tokens: list[int]) -> PrefixCache:
        """Prefill a shared prefix ONCE in the adapter's context and capture its
        per-sequence KV blob. The blob is adapter-specific (see PrefixCache)."""
        if adapter is not BASE and adapter not in self.adapter_ptr:
            raise KeyError(f"unknown adapter {adapter!r}")
        ctx = self.ctx[adapter]
        self._seq_rm(ctx, 0)
        self._decode_prefill(ctx, prefix_tokens, seq_id=0, start_pos=0)
        blob = self._get_seq(ctx, 0)
        self._seq_rm(ctx, 0)  # leave the context clean for later waves
        return PrefixCache(adapter, list(prefix_tokens), len(prefix_tokens), blob)

    def _generate_wave_cached(self, ctx: Any, cache: PrefixCache,
                              suffixes: list[list[int]], max_tokens: int) -> list[list[int]]:
        """Restore the shared prefix into each seq slot, then decode only the
        per-request suffix and generate. Skips re-prefilling the shared prefix."""
        s = len(suffixes)
        assert s <= self.n_seq_max, f"{s} > n_seq_max {self.n_seq_max}"
        for i in range(s):
            self._seq_rm(ctx, i)
            # Fail-closed restore (set_data==0 reject) via the library bridge.
            self._bridge.restore_blob_into_seq(ctx, cache.blob, i)
        outputs: list[list[int]] = []
        cur: list[int] = []
        pos: list[int] = []
        for i in range(s):
            # Suffix positions start right after the restored prefix.
            first = self._decode_prefill(ctx, suffixes[i], seq_id=i, start_pos=cache.n)
            outputs.append([first])
            cur.append(first)
            pos.append(cache.n + len(suffixes[i]))
        for _ in range(max_tokens - 1):
            cur = self._batched_step(ctx, cur, pos)
            for i in range(s):
                outputs[i].append(cur[i])
                pos[i] += 1
        return outputs

    def run_cached(self, adapter: Optional[str], cache: PrefixCache,
                   suffix_requests: list[Request]) -> list[Generation]:
        """Run a batch of requests that all share `cache` (same adapter+prefix).

        Each request.prompt is treated as the SUFFIX appended after the cached
        prefix. Processed in waves of n_seq_max within the adapter's context.
        """
        if cache.adapter is not adapter:
            raise ValueError("cache adapter does not match requested adapter")
        ctx = self.ctx[adapter]
        out: list[Generation] = []
        for w in range(0, len(suffix_requests), self.n_seq_max):
            wave = suffix_requests[w:w + self.n_seq_max]
            suffixes = [self.tokenize_suffix(r.prompt) for r in wave]
            gen = self._generate_wave_cached(ctx, cache, suffixes, max(r.max_tokens for r in wave))
            for slot, r in enumerate(wave):
                out.append(Generation(rid=r.rid, adapter=adapter,
                                      prompt_tokens=cache.tokens + suffixes[slot],
                                      tokens=gen[slot][:r.max_tokens]))
        return out

    def tokenize_suffix(self, text: str) -> list[int]:
        """Suffix tokens WITHOUT a leading BOS (the prefix already carried it)."""
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def run(self, requests: list[Request]) -> list[Generation]:
        """Route requests to their adapter's context, batching within each context.

        Requests keep their input order in the returned list. Within an adapter,
        requests are processed in waves of n_seq_max.
        """
        by_adapter: dict[Optional[str], list[int]] = {}
        for idx, r in enumerate(requests):
            if r.adapter is not BASE and r.adapter not in self.adapter_ptr:
                raise KeyError(f"unknown adapter {r.adapter!r}")
            by_adapter.setdefault(r.adapter, []).append(idx)

        results: dict[int, Generation] = {}
        for adapter, idxs in by_adapter.items():
            ctx = self.ctx[adapter]
            for w in range(0, len(idxs), self.n_seq_max):
                wave = idxs[w:w + self.n_seq_max]
                prompts = [self.tokenize(requests[j].prompt) for j in wave]
                maxs = [requests[j].max_tokens for j in wave]
                gen = self._generate_wave(ctx, prompts, max(maxs))
                for slot, j in enumerate(wave):
                    r = requests[j]
                    results[j] = Generation(
                        rid=r.rid, adapter=adapter,
                        prompt_tokens=prompts[slot],
                        tokens=gen[slot][:r.max_tokens],
                    )
        return [results[i] for i in range(len(requests))]

    def baseline(self, request: Request) -> Generation:
        """Ground truth: run one request ALONE on a fresh single-seq context pinned
        to its adapter. Independent of any batching/multiplexing."""
        ctx = self._make_ctx(1)
        try:
            if request.adapter is not BASE:
                self._pin_adapter(ctx, self.adapter_ptr[request.adapter], 1.0)
            prompt = self.tokenize(request.prompt)
            out = self._generate_wave(ctx, [prompt], request.max_tokens)[0]
            return Generation(rid=request.rid, adapter=request.adapter,
                              prompt_tokens=prompt, tokens=out[:request.max_tokens])
        finally:
            self.C.llama_free(ctx)

    # ---- grammar-constrained decoding (xgrammar) ----

    def enable_grammar(self, hf_tokenizer_dir: str) -> None:
        """Build the xgrammar TokenizerInfo + GrammarCompiler once (cached).

        vocab_size MUST be llama.cpp's n_vocab (not len(hf_tokenizer)) so padded
        logit ids stay masked (see 12-XGRAMMAR-API-FINDINGS.md).
        """
        import xgrammar as xgr
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_dir)
        self._xgr = xgr
        self._tok_info = xgr.TokenizerInfo.from_huggingface(hf_tok, vocab_size=self.n_vocab)
        self._grammar_compiler = xgr.GrammarCompiler(self._tok_info, cache_enabled=True)

    def compile_json_schema(self, schema: Any) -> Any:
        """Compile a JSON-schema dict (e.g. pydantic .model_json_schema()) to a grammar."""
        if not hasattr(self, "_grammar_compiler"):
            raise RuntimeError("call enable_grammar(hf_tokenizer_dir) first")
        import json as _json
        s = schema if isinstance(schema, str) else _json.dumps(schema)
        return self._grammar_compiler.compile_json_schema(s, strict_mode=True)

    def _apply_bitmask(self, logits: "np.ndarray", bitmask_row: "np.ndarray") -> None:
        """Set logits of disallowed tokens to -inf. bitmask bit=1 means allowed
        (little-endian within each int32 word); numpy hot path, torch-free."""
        bits = np.unpackbits(bitmask_row.view(np.uint8), bitorder="little")
        allowed = bits[: self.n_vocab].astype(bool)
        logits[: self.n_vocab][~allowed] = -np.inf

    def _constrained_sample(self, ctx: Any, row: int, matcher: Any) -> int:
        """Greedy argmax under the matcher's mask, then accept the token (fail-closed)."""
        bitmask = np.zeros(self._xgr.get_bitmask_shape(1, self.n_vocab), dtype=np.int32)
        if matcher.fill_next_token_bitmask(bitmask):
            ptr = self.C.llama_get_logits_ith(ctx, row)
            logits = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
            self._apply_bitmask(logits, bitmask[0])
            tok = int(np.argmax(logits))
        else:
            ptr = self.C.llama_get_logits_ith(ctx, row)
            tok = int(np.argmax(np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))))
        if not matcher.accept_token(tok):
            raise RuntimeError(f"grammar rejected accepted token {tok} (fail-closed)")
        return tok

    def _constrained_loop(self, ctx: Any, matchers: list[Any], outputs: list[list[int]],
                          cur: list[int], pos: list[int], done: list[bool],
                          max_tokens: int) -> list[list[int]]:
        """Shared batched constrained decode loop. Assumes each seq is already
        prefilled (prompt or restored-prefix+suffix) and its first token sampled
        into outputs/cur/pos/done. Runs up to max_tokens-1 further steps."""
        for _ in range(max_tokens - 1):
            if all(done):
                break
            si = len(cur)  # terminated slots ride along to keep seq_id == row
            batch = self.C.llama_batch_init(si, 0, 1)
            batch.n_tokens = si
            for i in range(si):
                batch.token[i] = cur[i]
                batch.pos[i] = pos[i]
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = i
                batch.logits[i] = 1
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"constrained batched decode rc={rc}")
            for i in range(si):
                pos[i] += 1
                if done[i]:
                    continue
                tok = self._constrained_sample(ctx, i, matchers[i])
                outputs[i].append(tok)
                cur[i] = tok
                if matchers[i].is_terminated():
                    done[i] = True
        return outputs

    def _generate_wave_constrained(self, ctx: Any, prompts: list[list[int]],
                                   grammar: Any, max_tokens: int) -> list[list[int]]:
        """Batched greedy generation with a per-sequence grammar matcher.

        Sequences terminate independently when the grammar completes; terminated
        slots stay in the batch (their tokens are frozen and discarded) to keep the
        seq_id==row invariant, until all terminate or max_tokens is hit.
        """
        s = len(prompts)
        assert s <= self.n_seq_max, f"{s} > n_seq_max {self.n_seq_max}"
        matchers = [self._xgr.GrammarMatcher(grammar) for _ in range(s)]
        for i in range(s):
            self._seq_rm(ctx, i)
        outputs: list[list[int]] = [[] for _ in range(s)]
        cur: list[int] = [0] * s
        pos: list[int] = [0] * s
        done: list[bool] = [False] * s
        for i in range(s):
            self._decode_prefill(ctx, prompts[i], seq_id=i, start_pos=0)  # decode; logits at -1
            tok = self._constrained_sample(ctx, -1, matchers[i])
            outputs[i].append(tok)
            cur[i] = tok
            pos[i] = len(prompts[i])
            done[i] = matchers[i].is_terminated()
        return self._constrained_loop(ctx, matchers, outputs, cur, pos, done, max_tokens)

    def _generate_wave_constrained_cached(self, ctx: Any, cache: PrefixCache,
                                          suffixes: list[list[int]], grammar: Any,
                                          max_tokens: int) -> list[list[int]]:
        """Cached shared-prefix restore + grammar-constrained generation.

        Restores the shared prefix KV into each seq slot, decodes only the suffix,
        then samples grammar-constrained. The full path-(a) MVP: shared router
        prompt cached once, per-request suffix, guaranteed-valid output.
        """
        s = len(suffixes)
        assert s <= self.n_seq_max, f"{s} > n_seq_max {self.n_seq_max}"
        matchers = [self._xgr.GrammarMatcher(grammar) for _ in range(s)]
        for i in range(s):
            self._seq_rm(ctx, i)
            # Fail-closed restore (set_data==0 reject) via the library bridge.
            self._bridge.restore_blob_into_seq(ctx, cache.blob, i)
        outputs: list[list[int]] = [[] for _ in range(s)]
        cur: list[int] = [0] * s
        pos: list[int] = [0] * s
        done: list[bool] = [False] * s
        for i in range(s):
            self._decode_prefill(ctx, suffixes[i], seq_id=i, start_pos=cache.n)
            tok = self._constrained_sample(ctx, -1, matchers[i])
            outputs[i].append(tok)
            cur[i] = tok
            pos[i] = cache.n + len(suffixes[i])
            done[i] = matchers[i].is_terminated()
        return self._constrained_loop(ctx, matchers, outputs, cur, pos, done, max_tokens)

    def run_constrained_cached(self, adapter: Optional[str], cache: PrefixCache,
                               suffix_requests: list[Request], grammar: Any) -> list[Generation]:
        """Full path-(a) MVP: shared cached prefix + per-request suffix + grammar.

        Each request.prompt is the SUFFIX appended after the cached prefix; every
        output is constrained to `grammar`. Waves of n_seq_max in the adapter ctx.
        """
        if not hasattr(self, "_xgr"):
            raise RuntimeError("call enable_grammar(...) first")
        if cache.adapter is not adapter:
            raise ValueError("cache adapter does not match requested adapter")
        ctx = self.ctx[adapter]
        out: list[Generation] = []
        for w in range(0, len(suffix_requests), self.n_seq_max):
            wave = suffix_requests[w:w + self.n_seq_max]
            suffixes = [self.tokenize_suffix(r.prompt) for r in wave]
            gen = self._generate_wave_constrained_cached(
                ctx, cache, suffixes, grammar, max(r.max_tokens for r in wave))
            for slot, r in enumerate(wave):
                out.append(Generation(rid=r.rid, adapter=adapter,
                                      prompt_tokens=cache.tokens + suffixes[slot],
                                      tokens=gen[slot][:r.max_tokens]))
        return out

    def run_constrained(self, requests: list[Request], grammar: Any) -> list[Generation]:
        """Route + batch like run(), but constrain each sequence to `grammar`."""
        if not hasattr(self, "_xgr"):
            raise RuntimeError("call enable_grammar(...) first")
        by_adapter: dict[Optional[str], list[int]] = {}
        for idx, r in enumerate(requests):
            if r.adapter is not BASE and r.adapter not in self.adapter_ptr:
                raise KeyError(f"unknown adapter {r.adapter!r}")
            by_adapter.setdefault(r.adapter, []).append(idx)
        results: dict[int, Generation] = {}
        for adapter, idxs in by_adapter.items():
            ctx = self.ctx[adapter]
            for w in range(0, len(idxs), self.n_seq_max):
                wave = idxs[w:w + self.n_seq_max]
                prompts = [self.tokenize(requests[j].prompt) for j in wave]
                gen = self._generate_wave_constrained(
                    ctx, prompts, grammar, max(requests[j].max_tokens for j in wave))
                for slot, j in enumerate(wave):
                    r = requests[j]
                    results[j] = Generation(rid=r.rid, adapter=adapter,
                                            prompt_tokens=prompts[slot],
                                            tokens=gen[slot][:r.max_tokens])
        return [results[i] for i in range(len(requests))]

    def baseline_from_tokens(self, adapter: Optional[str], tokens: list[int],
                             max_tokens: int, rid: str = "baseline") -> Generation:
        """Ground truth from an explicit token list (avoids re-tokenizing a joined
        prefix+suffix string, which could merge tokens differently)."""
        ctx = self._make_ctx(1)
        try:
            if adapter is not BASE:
                self._pin_adapter(ctx, self.adapter_ptr[adapter], 1.0)
            out = self._generate_wave(ctx, [tokens], max_tokens)[0]
            return Generation(rid=rid, adapter=adapter, prompt_tokens=tokens,
                              tokens=out[:max_tokens])
        finally:
            self.C.llama_free(ctx)

    # ---- P2 fork: true mixed-batch multi-LoRA (one context, per-seq adapters) ----

    def enable_seq_routing(self, pool: list[str]) -> None:
        """Bind the fork's llama_set_seq_adapters/llama_set_seq_adapter and create a
        single context holding the ordered adapter pool. Requires the P2 fork lib."""
        import ctypes
        C = self.C
        lib = C._lib
        if not hasattr(lib, "llama_set_seq_adapters"):
            raise RuntimeError("loaded libllama has no llama_set_seq_adapters — need the P2 fork build")
        lib.llama_set_seq_adapters.argtypes = [
            C.llama_context_p_ctypes, ctypes.POINTER(C.llama_adapter_lora_p_ctypes), ctypes.c_size_t]
        lib.llama_set_seq_adapters.restype = ctypes.c_int32
        lib.llama_set_seq_adapter.argtypes = [C.llama_context_p_ctypes, ctypes.c_int32, ctypes.c_int32]
        lib.llama_set_seq_adapter.restype = ctypes.c_int32
        self._lib = lib
        self._seq_pool = list(pool)
        self._seq_ctx = self._make_ctx(self.n_seq_max)
        arr = (C.llama_adapter_lora_p_ctypes * len(pool))(*[self.adapter_ptr[n] for n in pool])
        if lib.llama_set_seq_adapters(self._seq_ctx, arr, len(pool)) != 0:
            raise RuntimeError("llama_set_seq_adapters failed")

    def run_seq_routed(self, requests: list[Request]) -> list[Generation]:
        """True mixed-batch: one llama_decode step covers all sequences, each using
        its own adapter (from the registered pool). BASE (None) -> no adapter (-1)."""
        ctx = self._seq_ctx
        out: list[Generation] = []
        for w in range(0, len(requests), self.n_seq_max):
            wave = requests[w:w + self.n_seq_max]
            for i, r in enumerate(wave):
                idx = -1 if r.adapter is BASE else self._seq_pool.index(r.adapter)
                self._lib.llama_set_seq_adapter(ctx, i, idx)
            prompts = [self.tokenize(r.prompt) for r in wave]
            gen = self._generate_wave(ctx, prompts, max(r.max_tokens for r in wave))
            for i, r in enumerate(wave):
                out.append(Generation(rid=r.rid, adapter=r.adapter,
                                      prompt_tokens=prompts[i], tokens=gen[i][:r.max_tokens]))
        return out

    def detokenize(self, tokens: list[int]) -> str:
        return self.llm.detokenize(tokens).decode("utf-8", errors="replace")

    def close(self) -> None:
        for c in self.ctx.values():
            self.C.llama_free(c)
        self.ctx.clear()
        self.llm.close()
