"""Multi-LoRA agent-router surface — the project-17 flagship as a library API.

One base model loaded once; one pinned-adapter ``llama_context`` per adapter (plus
an optional base context). A request carries an adapter name; the router routes it
to that adapter's context, batches concurrent requests within a context (own-batch
multi-sequence decode), and multiplexes across contexts. Optional grammar-constrained
decoding reuses :mod:`structured_agents.llama_core.grammar`.

Design boundaries (repo standing rule): Pydantic validates at the edges
(:class:`RouterConfig`, :class:`RouteRequest`, :class:`RouteResult`); the decode hot
path passes plain token ids and numpy logit views. This mirrors
:mod:`structured_agents.llama_core.decode`, extended from one sequence to a routed,
batched wave.

The research provenance (GPU-validated, exact-match vs isolated baselines) lives in
``benchmarks/project17/``; this module is the teaching-quality library form.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .decode import FINISH_LENGTH, FINISH_STOP
from .grammar import JsonSchemaGrammar, apply_packed_bitmask_inplace
from .models import EngineConfig, GenerationResult

BASE: None = None  # sentinel adapter name meaning "no adapter / raw base model"


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterSpec(_BoundaryModel):
    """One LoRA adapter in the router's pool."""

    name: str
    gguf_path: str
    scale: float = Field(default=1.0, gt=0)


class RouterConfig(_BoundaryModel):
    """Everything needed to stand up a router over one base model."""

    engine: EngineConfig
    adapters: tuple[AdapterSpec, ...]
    n_seq_max: int = Field(default=8, gt=0)
    include_base: bool = True


class RouteRequest(_BoundaryModel):
    """A generation request tagged with the adapter it should route to."""

    prompt: str
    adapter: str | None = None  # None -> base context
    max_tokens: int = Field(default=64, gt=0)
    request_id: str | None = None


class RouteResult(GenerationResult):
    """A completed route: the generation plus which adapter served it."""

    adapter: str | None = None
    decision: dict[str, Any] | None = None  # parsed JSON when grammar-constrained


class MultiLoRARouter:
    """Shared base model, one pinned-adapter context per adapter, batched routing."""

    def __init__(self, config: RouterConfig) -> None:
        from llama_cpp import Llama, llama_cpp

        self.config = config
        self.C = llama_cpp
        eng = config.engine
        self.n_seq_max = config.n_seq_max
        self.n_batch = eng.n_batch
        self.llm = Llama(
            model_path=eng.model_path, n_ctx=eng.n_ctx, n_batch=eng.n_batch,
            n_gpu_layers=eng.n_gpu_layers, seed=eng.seed, verbose=False,
        )
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self._eos = int(self.llm.token_eos())

        self._adapter_ptr: dict[str, Any] = {}
        for spec in config.adapters:
            ptr = self.C.llama_adapter_lora_init(self.model, spec.gguf_path.encode("utf-8"))
            if not ptr:
                raise RuntimeError(f"failed to load adapter {spec.name!r} from {spec.gguf_path}")
            self._adapter_ptr[spec.name] = ptr
        self._scale = {s.name: s.scale for s in config.adapters}

        self._ctx: dict[str | None, Any] = {}
        names: list[str | None] = ([BASE] if config.include_base else []) + [s.name for s in config.adapters]
        for name in names:
            ctx = self._make_ctx()
            if name is not BASE:
                self._pin_adapter(ctx, self._adapter_ptr[name], self._scale[name])
            self._ctx[name] = ctx
        self._grammar: JsonSchemaGrammar | None = None

    # ---- setup ----

    def _make_ctx(self) -> Any:
        p = self.C.llama_context_default_params()
        p.n_ctx = self.config.engine.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = self.n_seq_max
        ctx = self.C.llama_new_context_with_model(self.model, p)
        if not ctx:
            raise RuntimeError("llama_new_context_with_model returned NULL (out of VRAM?)")
        return ctx

    def _pin_adapter(self, ctx: Any, ptr: Any, scale: float) -> None:
        arr = (self.C.llama_adapter_lora_p_ctypes * 1)(ptr)
        scales = (ctypes.c_float * 1)(scale)
        if self.C.llama_set_adapters_lora(ctx, arr, 1, scales) != 0:
            raise RuntimeError("llama_set_adapters_lora failed")

    def enable_grammar(self, tokenizer: Any, schema: dict[str, Any]) -> None:
        """Compile a JSON-schema grammar; constrained runs mask to it per sequence."""
        self._grammar = JsonSchemaGrammar.from_huggingface(tokenizer, schema, vocab_size=self.n_vocab)

    # ---- decode primitives (plain tokens / numpy on the hot path) ----

    def _seq_rm(self, ctx: Any, seq_id: int) -> None:
        self.C.llama_memory_seq_rm(self.C.llama_get_memory(ctx), seq_id, -1, -1)

    def _prefill(self, ctx: Any, tokens: list[int], seq_id: int) -> None:
        n = len(tokens)
        for off in range(0, n, self.n_batch):
            chunk = tokens[off:off + self.n_batch]
            m = len(chunk)
            batch = self.C.llama_batch_init(m, 0, 1)
            batch.n_tokens = m
            for i, t in enumerate(chunk):
                batch.token[i] = t
                batch.pos[i] = off + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 1 if off + m == n and i == m - 1 else 0
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"prefill llama_decode rc={rc}")

    def _sample_row(self, ctx: Any, row: int, matcher: Any | None) -> int:
        ptr = self.C.llama_get_logits_ith(ctx, row)
        logits = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        if matcher is not None:
            import xgrammar as xgr

            bitmask = np.zeros(xgr.get_bitmask_shape(1, self.n_vocab), dtype=np.int32)
            if matcher.fill_next_token_bitmask(bitmask):
                apply_packed_bitmask_inplace(logits, bitmask[0], self.n_vocab)
        tok = int(np.argmax(logits))
        if matcher is not None and matcher.accept_token(tok) is False:
            raise RuntimeError(f"grammar rejected token {tok} (fail-closed)")
        return tok

    def _advance(self, out: list[int], tok: int, matcher: Any | None) -> str | None:
        """Record a sampled token; return a finish_reason if the sequence is done.

        Grammar completion or a sampled EOS is a clean ``stop``; otherwise the
        sequence continues (``None``) until ``max_tokens`` yields a ``length`` cut.
        """
        if matcher is not None:
            out.append(tok)
            return FINISH_STOP if matcher.is_terminated() else None
        if tok == self._eos:
            return FINISH_STOP  # EOS is the boundary, not content — do not emit it
        out.append(tok)
        return None

    def _generate_wave(self, ctx: Any, prompts: list[list[int]], max_tokens: int,
                       matchers: list[Any] | None) -> list[tuple[list[int], str]]:
        """Greedy batched decode of up to n_seq_max sequences; seq_id == row.

        Returns ``(tokens, finish_reason)`` per sequence so truncation is
        observable. Finished slots ride along (frozen) to keep the seq_id==row
        invariant until all finish or ``max_tokens`` is hit.
        """
        s = len(prompts)
        for i in range(s):
            self._seq_rm(ctx, i)
        out: list[list[int]] = [[] for _ in range(s)]
        cur: list[int] = [0] * s
        pos: list[int] = [0] * s
        finish: list[str | None] = [None] * s
        for i in range(s):
            self._prefill(ctx, prompts[i], i)
            cur[i] = self._sample_row(ctx, -1, matchers[i] if matchers else None)
            pos[i] = len(prompts[i])
            finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
        for _ in range(max_tokens - 1):
            if all(f is not None for f in finish):
                break
            batch = self.C.llama_batch_init(s, 0, 1)
            batch.n_tokens = s
            for i in range(s):
                batch.token[i] = cur[i]
                batch.pos[i] = pos[i]
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = i
                batch.logits[i] = 1
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"batched llama_decode rc={rc}")
            for i in range(s):
                pos[i] += 1
                if finish[i] is not None:
                    continue
                cur[i] = self._sample_row(ctx, i, matchers[i] if matchers else None)
                finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
        return [(out[i], finish[i] or FINISH_LENGTH) for i in range(s)]

    # ---- public API ----

    def _tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def run(self, requests: list[RouteRequest], *, constrained: bool = False) -> list[RouteResult]:
        """Route + batch a mixed workload; optionally constrain to the grammar."""
        if constrained and self._grammar is None:
            raise RuntimeError("constrained=True requires enable_grammar(...) first")
        by_adapter: dict[str | None, list[int]] = {}
        for idx, r in enumerate(requests):
            if r.adapter is not BASE and r.adapter not in self._adapter_ptr:
                raise KeyError(f"unknown adapter {r.adapter!r}")
            by_adapter.setdefault(r.adapter, []).append(idx)

        results: dict[int, RouteResult] = {}
        for adapter, idxs in by_adapter.items():
            ctx = self._ctx[adapter]
            for w in range(0, len(idxs), self.n_seq_max):
                wave = idxs[w:w + self.n_seq_max]
                prompts = [self._tokenize(requests[j].prompt) for j in wave]
                matchers = [self._grammar.new_matcher() for _ in wave] if constrained else None
                gen = self._generate_wave(ctx, prompts, max(requests[j].max_tokens for j in wave), matchers)
                for slot, j in enumerate(wave):
                    r = requests[j]
                    toks, finish_reason = gen[slot]
                    toks = toks[:r.max_tokens]
                    text = self.llm.detokenize(toks).decode("utf-8", errors="replace")
                    # Only parse when the grammar actually completed; a length cut
                    # leaves the JSON truncated, and validated must reflect that.
                    decision = None
                    if constrained and finish_reason == FINISH_STOP:
                        try:
                            decision = json.loads(text)
                        except ValueError:
                            decision = None
                    results[j] = RouteResult(
                        text=text, token_ids=tuple(toks), prompt_token_count=len(prompts[slot]),
                        completion_token_count=len(toks),
                        finish_reason=finish_reason, request_id=r.request_id,
                        adapter=adapter, decision=decision,
                        validated=(decision is not None) if constrained else None,
                    )
        return [results[i] for i in range(len(requests))]

    def close(self) -> None:
        for ctx in self._ctx.values():
            self.C.llama_free(ctx)
        self._ctx.clear()
        self.llm.close()

    def __enter__(self) -> MultiLoRARouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
