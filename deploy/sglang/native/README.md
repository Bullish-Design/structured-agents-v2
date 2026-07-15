# Isolated SGLang Gemma 4 GGUF spike

This is an intentionally disabled-by-default compatibility test for the exact
Unsloth Gemma 4 12B QAT `UD-Q4_K_XL` GGUF. It must never replace or restart
the GPU-1 vLLM endpoint. When enabled, it binds only `127.0.0.1:8002`, uses GPU
0, has one request slot, uses a dedicated cache, and forces Hugging Face and
Transformers offline.

`devenv.nix` and `pyproject.toml` pin a standalone Python 3.12/SGLang 0.5.14
environment. `run.sh` enters that environment; it does not reference the vLLM
directory or virtual environment. `serve.sh` requires pre-existing local target and tokenizer paths. It uses the
SGLang `--load-format gguf` and `--tokenizer-path` interface, disables CPU
offload, and logs the fully resolved invocation. MTP is an explicit safety
failure—not a fallback—until a version-specific configuration proves the Q8
GGUF assistant has nonzero proposal and acceptance telemetry.

## Validation sequence

Run `./test_serve.sh` first; it is a shell contract test only. Then run
`nix flake check --no-build` and a non-activating NixOS build before modifying
the host configuration. A runtime attempt requires stopping the GPU-0
llama.cpp service through systemd, capturing its pre-state, starting this
separate service, and restoring llama.cpp through systemd afterward. GPU 1
and `structured-agents-vllm.service` are out of scope.

After startup succeeds, `./verify.sh` checks health, model listing, and a basic
chat completion. Structured JSON/regex/grammar, LoRA, MTP, and performance are
unverified until their individual runtime artifacts are captured.

## Source evidence (2026-07-14)

- SGLang's current server-arguments reference documents `--load-format gguf`,
  `--model-path`, and `--tokenizer-path`.
- Its grammar backend accepts `xgrammar`, `outlines`, `llguidance`, or `none`;
  that does not establish compatibility with this model or the library's
  `structured_outputs` wire shape.
- SGLang's current speculative settings describe a draft model path and
  algorithms such as EAGLE, but do not prove a GGUF target + GGUF Gemma 4 MTP
  assistant works. This spike therefore does not synthesize a speculative
  configuration.

Consulted sources: `sgl-project/sglang` server-arguments and server-args
references, SGLang serving benchmark documentation, and Unsloth's GGUF serving
guidance. Exact source URLs and runtime outcomes belong in the timestamped
artifact directory for each attempt.
