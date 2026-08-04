"""Multi-LoRA agent-router demo — the project-17 flagship as a library call.

One base model, several LoRA adapters, one mixed batch of requests routed to their
adapters, optionally with grammar-guaranteed JSON decisions. Uses the typed library
surface in ``inferference.router``.

Run inside the pinned project environment (CUDA build on PATH via LLAMA_CPP_LIB_PATH;
real driver ahead of the CUDA stub on LD_LIBRARY_PATH). Example:

    python examples/multi_lora_router.py \
        --model .../Ornith-1.0-9B-UD-Q4_K_XL.gguf \
        --adapter probe-a=.../probe-a.gguf --adapter probe-b=.../probe-b.gguf \
        --constrained
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from inferference.models import EngineConfig
from inferference.router import (
    AdapterSpec,
    MultiLoRARouter,
    RouterConfig,
    RouteRequest,
)
from pydantic import BaseModel


class Route(BaseModel):
    """The grammar-constrained routing decision schema."""

    tool: Literal["search", "calculator", "calendar", "smart_home", "none"]
    confidence: Literal["low", "medium", "high"]


TASKS = [
    "book a meeting for tomorrow at 3pm",
    "what is 17 times 23",
    "turn off the kitchen lights",
    "find the latest news about mars",
]
INSTR = ("You are a tool router. Respond with ONLY a JSON object "
         '{"tool": ..., "confidence": ...}. User request: ')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--adapter", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable; e.g. probe-a=/path/probe-a.gguf")
    ap.add_argument("--tokenizer", default="deepreinforce-ai/Ornith-1.0-9B")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-seq-max", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--constrained", action="store_true", help="grammar-guarantee JSON output")
    args = ap.parse_args()

    specs = []
    for item in args.adapter:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"--adapter must be NAME=PATH, got {item!r}")
        specs.append(AdapterSpec(name=name, gguf_path=path))
    if not specs:
        raise SystemExit("provide at least one --adapter NAME=PATH")
    names = [s.name for s in specs]

    config = RouterConfig(
        engine=EngineConfig(model_path=str(args.model), n_ctx=args.n_ctx,
                            n_batch=256, n_gpu_layers=-1, backend="cuda"),
        adapters=tuple(specs), n_seq_max=args.n_seq_max, include_base=True,
    )

    # Mixed workload: cycle each task across the adapters.
    requests = [
        RouteRequest(request_id=f"t{i}", prompt=INSTR + TASKS[i % len(TASKS)],
                     adapter=names[i % len(names)], max_tokens=args.max_tokens)
        for i in range(len(TASKS))
    ]

    with MultiLoRARouter(config) as router:
        if args.constrained:
            from transformers import AutoTokenizer

            router.enable_grammar(AutoTokenizer.from_pretrained(args.tokenizer),
                                  Route.model_json_schema())
        results = router.run(requests, constrained=args.constrained)

    for r in results:
        line = {"request_id": r.request_id, "adapter": r.adapter,
                "finish_reason": r.finish_reason,
                "decision": r.decision if args.constrained else r.text.strip()[:80]}
        print(json.dumps(line, sort_keys=True))


if __name__ == "__main__":
    main()
