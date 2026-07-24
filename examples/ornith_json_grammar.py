"""Generate valid-by-construction JSON from Ornith with the owned xgrammar loop.

Run inside the pinned project environment with a local GGUF and the matching
Hugging Face tokenizer cached or available to Transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from structured_agents.llama_core.benchmark import BenchmarkTimer, write_benchmark_record
from structured_agents.llama_core.decode import OwnedLlamaDecoder
from structured_agents.llama_core.grammar import JsonSchemaGrammar


class CapitalAnswer(BaseModel):
    city: str
    country: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", default="deepreinforce-ai/Ornith-1.0-9B")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()
    if args.requests <= 0:
        raise ValueError("--requests must be positive")

    from llama_cpp import Llama
    from transformers import AutoTokenizer

    llm = Llama(model_path=str(args.model), n_ctx=512, n_batch=128, n_threads=8, logits_all=False, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    grammar = JsonSchemaGrammar.from_huggingface(
        tokenizer,
        CapitalAnswer.model_json_schema(),
        vocab_size=llm.n_vocab(),
    )
    prompt = "Return only a JSON object naming the capital city and country of France."
    prompt_tokens = len(llm.tokenize(prompt.encode(), add_bos=False))
    for request_index in range(args.requests):
        # The grammar is shared; its matcher is intentionally fresh per request.
        matcher = grammar.new_matcher()
        benchmark = BenchmarkTimer(
            "ornith-xgrammar-json",
            metadata={"request_index": request_index, "vocab_size": llm.n_vocab()},
        )
        with OwnedLlamaDecoder(llm) as decoder:
            generated = decoder.generate_text(
                prompt,
                max_tokens=args.max_tokens,
                logits_hook=grammar.logits_hook(matcher, benchmark=benchmark),
                token_hook=grammar.token_hook(matcher),
                benchmark=benchmark,
            )
        if generated.finish_reason != "stop":
            raise RuntimeError(
                f"request {request_index} stopped on {generated.finish_reason!r} after "
                f"{generated.completion_token_count} tokens; output is likely truncated: {generated.text!r}"
            )
        with benchmark.measure("validation"):
            answer = CapitalAnswer.model_validate_json(generated.text)
        artifact = write_benchmark_record(
            benchmark.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=generated.completion_token_count,
            ),
            args.artifacts,
        )
        print(json.dumps(answer.model_dump(), sort_keys=True))
        print(artifact)


if __name__ == "__main__":
    main()
