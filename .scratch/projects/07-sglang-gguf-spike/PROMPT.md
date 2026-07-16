# SGLang GGUF / Gemma 4 test spike

Work in `/home/andrew/Documents/Projects/structured-agents-v2`. The NixOS host
configuration is in `/home/andrew/Documents/Projects/nix-meta`.

## Goal

Implement a carefully isolated SGLang compatibility and performance test spike for
the local quantized Gemma 4 deployment. Determine through runtime evidence, not
documentation alone, whether SGLang can serve the exact target GGUF on the RTX
3060 and whether it is a credible future replacement for the local vLLM or
llama.cpp serving lanes.

Do not make SGLang the default backend. Do not remove, replace, restart,
benchmark, or otherwise disturb the vLLM endpoint on GPU 1. Do not claim Gemma
4 GGUF, MTP, structured outputs, or performance works until it is runtime-tested.

## Current live topology

- GPU 0: llama.cpp, loopback `127.0.0.1:8001/v1`.
- GPU 1: vLLM, loopback `127.0.0.1:8000/v1`.
- vLLM target: Unsloth Gemma 4 12B QAT `UD-Q4_K_XL`.
- llama.cpp target: the same immutable GGUF.
- llama.cpp settings: 16k total context, q8 KV cache, Flash Attention, full
  target/drafter GPU offload, MTP enabled, `--parallel 5`.
- Current llama.cpp drafter:
  `/home/andrew/.cache/structured-agents/models/mtp-gemma-4-12B-it.gguf`.
- Cached Q8 MTP assistant:
  `/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf`.
- Target model path:
  `/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165`.

## Known evidence

- Active llama.cpp version: `9842 (6f4f53f)`.
- GPU 0 is an RTX 3060 with 12 GiB VRAM.
- Existing llama.cpp MTP is real: logs have shown nonzero draft and accepted-token counts.
- The Q8 assistant cache file is 465,127,040 bytes with SHA-256
  `13331068b6af643c3dc75e619373b674c1f75a1958e7c82e2020d96a17c63809`.
- SGLang officially exposes GGUF loading/quantization options, and Unsloth
  documents GGUF serving with `--tokenizer-path`. This does **not** prove support
  for this exact Gemma 4 QAT GGUF or GGUF MTP configuration.

## Repository context

- Existing native vLLM deployment: `deploy/vllm/native/`.
- Existing native llama.cpp deployment: `deploy/llama-cpp/native/`.
- NixOS modules are exported from `flake.nix`.
- The Python library has one OpenAI-compatible `Backend` abstraction in
  `src/structured_agents_v2/backend.py`.
- The library currently assumes vLLM-like capabilities by default: XGrammar
  structured outputs and LoRA.
- Decoder request shapes: `src/structured_agents_v2/decoder.py`.
- Dual-path records currently hard-code local identity as `kind="vllm"` in
  `src/structured_agents_v2/dual_path/record.py`.

## Operational constraints

1. Preserve all unrelated dirty changes in both repositories. Use `apply_patch`
   for edits. Never reset, delete, or overwrite unrelated work.
2. Do not touch vLLM files or the GPU-1 vLLM service. Record its initial process,
   arguments, and GPU assignment, and confirm it remains unaffected afterward.
3. Treat SGLang as a separate spike:
   - new `deploy/sglang/native/` structure;
   - a separate NixOS module/service name;
   - loopback-only, distinct port (suggest `8002`);
   - no Tailscale publication by default;
   - a dedicated persistent cache path;
   - no silent model redownloads;
   - no reuse or mutation of the vLLM virtual environment/cache.
4. Do not stop or restart the live llama.cpp service without saying so explicitly.
   GPU 0 likely cannot host it and an SGLang target simultaneously. If an isolated
   GPU-0 runtime test requires stopping llama.cpp, prepare all static/config/build
   validation first, use normal systemd/NixOS controls, preserve/restore the
   service, and never kill it directly. If interactive sudo is required, prepare
   everything possible then give exactly one clear command for the user to run.
5. Do not assume generic GGUF support covers this Unsloth dynamic quant, Gemma 4
   architecture/multimodal details, the Q8 GGUF MTP assistant, 16k context on
   12 GiB, or the library's `structured_outputs` request fields. Test each item
   or mark it unverified.

## Required workflow

### A. Audit and research

Inspect both repositories' status, active systemd units/process arguments, GPU
usage, CUDA/driver environment, current llama.cpp/vLLM versions, modules, and
launch scripts.

Read current official SGLang and Unsloth sources for GGUF loading,
`--tokenizer-path`, Gemma 4, Gemma 4 MTP/NEXTN, structured outputs (JSON Schema,
regex, EBNF/grammar), LoRA API behavior, and required CUDA/PyTorch/SGLang versions.
Record the exact versions/commits/docs consulted in artifacts.

### B. Build an isolated SGLang deployment spike

Create at least:

- `deploy/sglang/native/nixos-module.nix`
- `deploy/sglang/native/serve.sh`
- `deploy/sglang/native/verify.sh`
- `deploy/sglang/native/test_serve.sh`
- `deploy/sglang/native/README.md`

Optionally use a separate locked Python/devenv/uv environment, benchmark runner,
and capability probe. Make enable, repository/user/group, SGLang package/version,
target GGUF, tokenizer path/revision, model name, isolated GPU, port, context,
memory/KV settings, quantization/load format, optional drafter, MTP settings,
and API key configurable.

The launcher must bind only to loopback, require existing model/tokenizer paths,
never auto-download target weights, make MTP disabled by default until proven,
and log resolved arguments. Distinguish target, tokenizer, and drafter paths.

### C. Static validation

Before runtime work, run and label shell contract tests, Python/static tests as
applicable, `nix flake check --no-build`, and non-activating
`nixos-rebuild build --flake /home/andrew/Documents/Projects/nix-meta#server`.
Do not describe those as runtime validation.

### D. Runtime tests

If GPU-0 llama.cpp must be stopped, state that explicitly and use the reversible
normal-service path. Never disturb GPU 1/vLLM. Save all artifacts beneath:

`artifacts/sglang-gemma4-spike/<UTC timestamp>/`

Run, in order:

1. **Startup/load-only:** exact target GGUF, explicit HF tokenizer, GPU 0 only,
   no MTP, one slot, 16k context if it fits. Capture loader metadata, selected
   implementation, quantization recognition, and post-load VRAM. Fail clearly on
   CPU offload or an inability to fit.
2. **OpenAI compatibility:** health, `/v1/models`, normal chat completion, and
   streaming if available.
3. **Structured outputs:** JSON Schema with the library's `response_format`
   shape, regex and grammar/EBNF with the library's current `structured_outputs`
   shapes. Validate output, not merely HTTP status. If SGLang needs a different
   shape, document it and propose a narrow adapter rather than broadly changing
   the library.
4. **LoRA:** inspect/document the real API; do not download/train an adapter just
   for this. Call it documented-but-unverified unless an existing local adapter
   can be used without a download.
5. **MTP:** attempt only after baseline loading. Determine whether SGLang accepts
   the GGUF target with the cached Q8 GGUF assistant. Use version-appropriate
   settings; capture an exact error if unsupported; never substitute unrelated
   HF weights. MTP is working only if telemetry shows nonzero proposals and
   nonzero accepted tokens/accept length.
6. **Controlled benchmark:** one slot, same target/context/KV/offload/sampling,
   nine fixed 192-token requests, `temperature=1.0`, `top_p=0.95`, `top_k=64`,
   no hidden thinking. Compare llama.cpp no-MTP, llama.cpp Unsloth MTP, SGLang
   GGUF baseline, and SGLang GGUF+MTP only when it actually starts. Save raw API
   responses, logs, GPU captures, CSV, and Markdown. Report output tok/s,
   latency, TTFT when available, draft/generated/accepted tokens, acceptance
   rate, error rate, and VRAM.

### E. Integration proposal only

Do not broadly refactor the library. Add a concise artifact/design note covering
generic OpenAI backend suitability, vLLM/llama.cpp/SGLang capability matrix,
possible `BackendCaps` additions, generalized runtime identity, wire-contract
compatibility, endpoint-routing policy, and a staged migration plan only when
runtime evidence supports it.

## Success criteria

- SGLang target-GGUF startup is runtime-proven on GPU 0 with no CPU weight offload.
- Exact VRAM and 16k-context viability are recorded.
- Basic OpenAI API compatibility is proven.
- Structured JSON/regex/grammar support is proven or explicitly incompatible.
- MTP is called working only with nonzero acceptance telemetry.
- vLLM/GPU 1 remains unchanged.
- No performance claims lack controlled measurement artifacts.
- Final reporting clearly separates source evidence, contract/static tests,
  Nix-build tests, runtime startup/API/constraint evidence, benchmarks, and
  unverified items.

Start with inspection and a brief progress update. Do not request a NixOS rebuild
before static validation is complete.
