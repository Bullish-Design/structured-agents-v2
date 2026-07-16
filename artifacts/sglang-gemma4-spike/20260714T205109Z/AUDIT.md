# SGLang Gemma 4 GGUF spike — audit and static validation

UTC started: 2026-07-14T20:51:09Z

## Isolation baseline

Read-only host inspection found the following active services before any SGLang
work. No service was stopped, restarted, or reconfigured during this audit.

| lane | systemd unit | GPU / port | observed command |
|---|---|---|---|
| production | `structured-agents-vllm.service` | GPU 1 / `127.0.0.1:8000` | `vllm serve unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL ... --max-model-len 16384 --quantization gguf --cpu-offload-gb 0 --structured-outputs-config.backend xgrammar` |
| comparison | `structured-agents-llama-cpp.service` | GPU 0 / `127.0.0.1:8001` | `llama-server ... --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --parallel 5` |

Host GPU state at inspection: two NVIDIA GeForce RTX 3060 cards, driver 595.84,
12,288 MiB each. GPU 0 had `llama-server` using 8,400 MiB; GPU 1 had
`VLLM::EngineCore` using 10,784 MiB. Python is 3.13.13. The sandbox cannot see
the host systemd or driver, so the baseline was captured with an approved,
read-only host query.

## Local immutable inputs

- Target: `/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165` (6.3 GiB).
- Tokenizer/config: `/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac` (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json` present).
- Q8 MTP assistant: 444 MiB, SHA-256 `13331068b6af643c3dc75e619373b674c1f75a1958e7c82e2020d96a17c63809`.

## Source evidence consulted

Current official SGLang source references (main branch, consulted
2026-07-14) document `--model-path`, `--tokenizer-path`, `--load-format gguf`,
`--context-length`, loopback host binding, and the GGUF load-format choice.
The server arguments also expose grammar backends (`xgrammar`, `outlines`,
`llguidance`, `none`) and speculative-draft settings. This source evidence does
**not** establish this dynamic-quant Gemma 4 GGUF, its multimodal details, the
Q8 GGUF MTP assistant, or the library's `structured_outputs` request shape.

The SGLang serving benchmark reference records TTFT, inter-token latency,
throughput, detailed request results, and SGLang speculative accept length when
available from `/server_info`; it will be used only after baseline startup.

## Static results (not runtime results)

- `bash deploy/sglang/native/test_serve.sh`: PASS. This only captures the
  launcher arguments with a fake Python executable and verifies the MTP safety
  gate; it does not import SGLang, access CUDA, or load weights.
- `nix flake check --no-build`: PASS after marking only the new files as
  intent-to-add, which makes them visible to Nix without staging their content.
- `nixos-rebuild build --flake /home/andrew/Documents/Projects/nix-meta#server`:
  PASS (non-activating build), result
  `/nix/store/kgfxyqwr58am0kmi9wdl02r2avs8lmlb-nixos-system-server-26.11.20260705.d407951`.

The SGLang environment is not installed on the host and no `sglang` executable
or import was found. Therefore no SGLang runtime claim has been made and no
GPU-0/llama.cpp handoff has occurred.

## Dedicated environment update

On 2026-07-14 the separate `deploy/sglang/native` devenv was created and
locked for Linux x86_64 with Python 3.12 and `sglang==0.5.14`. It has its own
`.venv`, `uv.lock`, `devenv.lock`, CUDA 13 toolchain, and `run.sh`; no path in
that definition references `deploy/vllm/native` or its virtual environment.

`uv sync --locked` completed in that isolated directory. A no-model, no-GPU
check printed `sglang 0.5.14`. Its installed launcher help includes
`--model-path`, `--tokenizer-path`, `--load-format gguf`, `--context-length`,
`--cpu-offload-gb`, and `--grammar-backend`. The launcher also notes that
`sglang serve` is the preferred spelling while the retained
`python -m sglang.launch_server` entrypoint remains supported.

The shell contract test and `nix flake check --no-build` passed after this
change. These remain static/environment checks only: no model was loaded, no
GPU runtime was started, and neither llama.cpp nor vLLM was modified.

## Current implementation

`deploy/sglang/native/` now supplies a disabled-by-default NixOS module,
launcher, API verifier, shell contract test, and README. The launcher requires
existing local model/tokenizer paths; forces an isolated cache and Hugging Face/
Transformers offline mode; binds only GPU 0 and `127.0.0.1:8002`; enforces one
slot, 16k context, and zero CPU weight offload; and refuses MTP outright until
a version-specific configuration is proven.
