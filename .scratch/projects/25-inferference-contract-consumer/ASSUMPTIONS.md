# ASSUMPTIONS — 25 inferference contract consumer

## The layering the owner asked for

`inferference` is the base layer this library builds on. It already decided
what KIND of base layer it is: **a serving substrate**, not an importable
decoder. Owner decision 2026-09-04 (`inferference/AGENTS.md:126-134`) —
"THIS LIBRARY OWNS ALL LLM SERVING. There is no other server."

So this library depends on inferference for **types, capabilities and a
client**. It does not depend on it for the native runtime.

## Why not import the decoder

Three blockers, each read from inferference's source:

1. `MultiLoRARouter.run` is synchronous and wave-shaped (`router.py:552`); a
   Pydantic AI `Model.request` is async and per-request.
2. `enable_grammar` compiles ONE grammar onto the router (`router.py:339`);
   this library needs one constraint per agent.
3. The router holds native handles and VRAM. DBOS recovery on a second worker
   would reload a multi-GiB GGUF onto a card the reconciler has already placed.

Durability is this library's point; serving is inferference's. Each keeps its
own.

## Constraints

- This package holds no card and never loads a model.
- The ABI-anchor pair must not be reachable by default resolution: PyPI
  `llama-cpp-python` is a STOCK build.
