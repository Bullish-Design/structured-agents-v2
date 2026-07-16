# Next build: investigate Gemma 4 GGUF expert-implementation construction blocker

Work in `/home/andrew/Documents/Projects/structured-agents-v2`.

## Mission

Continue the isolated GPU-0 SGLang effort to serve the exact locally cached
Unsloth Gemma 4 12B QAT `UD-Q4_K_XL` GGUF. Do not substitute native
safetensors weights. The current goal remains base GGUF startup followed by
one valid non-streaming chat completion. MTP, structured output, and 16k
viability remain out of scope until that proof exists.

The attention-construction blocker has been resolved locally: the latest
attempt passed SGLang's Gemma 4 Triton attention selection, initialized Torch
distributed, and reached `Load weight begin`. It then failed before the GGUF
loader read any tensor because current Transformers validates an unsupported
expert implementation while constructing SGLang's `Gemma4ForCausalLM`.

## Read in full before acting

1. `.scratch/CRITICAL_RULES.md`
2. `.scratch/REPO_RULES.md`
3. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md`
4. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/METADATA_REPORT.md`
5. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/MINIMAL_REPRODUCTION.md`
6. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/NEXT_BUILD_PROMPT.md`
7. `deploy/sglang/native/CONTAINER_TESTING.md`
8. `deploy/sglang/native/{README.md,run.sh,serve.sh,sitecustomize.py,gemma4_gguf_compat.py,test_gemma4_gguf_compat.py,pyproject.toml,devenv.nix}`
9. `artifacts/sglang-gemma4-spike/20260715T015057Z/{host-topology-before.txt,sglang-gguf-launch.txt,host-topology-final-observation.txt}`

Use the `$build-run-investigation-loop` skill for the full evidence and
research workflow. Do not use subagents.

## Exact target and fixed inputs

```text
MODEL_PATH=/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165
TOKENIZER_PATH=/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac
```

The target GGUF is immutable and extensionless. `MODEL_LOAD_FORMAT=gguf`
declares its format. Do not modify the blob, tokenizer/config cache, or MTP
assets.

## Proven state

- Environment: SGLang `0.5.14`, Python `3.12.13`, Torch `2.11.0+cu130`,
  Transformers `5.14.0.dev0` at
  `ab1771c9e42891d893189978a8009426d70b4688`.
- Hardware: GPU 0 and GPU 1 are RTX 3060 12 GiB. GPU 1 runs the live vLLM
  service on `127.0.0.1:8000`; it was active at 10,784 MiB before and after
  the latest GPU-0 attempt. GPU 0 was free before launch and returned to idle
  after SGLang cleaned up.
- The exact text-only GGUF has 48 layers and a semantic 5:1 repeating
  sliding/full attention pattern. Sliding KV heads are 8; full KV heads are 1;
  full layers share K=V; PLE width is 0.
- `gemma4_gguf_compat.py` correctly reconstructs those config fields and
  aliases the gguf-py tensor map from `gemma4` to `gemma4_text`.
- `prepare_sglang_gemma4_construction` sets only Gemma-4's Transformers
  attention implementation to `eager` immediately before SGLang constructs
  its own model. SGLang still selects and uses its Triton/RadixAttention
  runtime path. The prior SDPA error is gone.

## Current runtime evidence and blocker

Latest command properties: exact target, adapter unset,
`MODEL_LOAD_FORMAT=gguf`, `CUDA_VISIBLE_DEVICES=0`, loopback port 8002,
`CONTEXT_LENGTH=16384`, one slot, no CPU offload, MTP disabled, offline cache.

Observed sequence:

```text
Use triton as default attention backend for Gemma4
Init torch distributed ends. elapsed=0.43 s, mem usage=0.02 GB
Load weight begin. avail mem=11.48 GB
ValueError: Gemma4ForCausalLM does not support setting experts implementation.
```

Trace classification: `transformers.modeling_utils` reaches
`_grouped_mm_can_dispatch` from `PreTrainedModel.__init__` during SGLang's
`Gemma4ForCausalLM` construction. It happens before GGUF tensors are loaded,
so this is not evidence about tensor mapping, dynamic Q4 kernels, VRAM, API,
MTP, or output correctness.

Transformers' relevant behavior is:

```python
applicable_experts = "grouped_mm" if requested_experts is None else requested_experts
...
if applicable_experts == "grouped_mm":
    try:
        self._grouped_mm_can_dispatch()
    except (ValueError, ImportError):
        if requested_experts == "grouped_mm":
            raise
        applicable_experts = "eager"
```

The observed raise therefore means the resolved configuration requests
`grouped_mm` explicitly. Determine where that request is set and compare
current SGLang main and its matching Transformers context before changing any
field. The target is dense; do not infer that a generic MoE/expert backend
should be forced globally.

## Operational constraints

- Read `deploy/sglang/native/CONTAINER_TESTING.md` and follow it exactly.
- Container-only commands cannot inspect host GPU/Nix state reliably. Use
  `require_escalated` for host snapshots, the Nix/devenv regression, and every
  GPU-0 launch. Do not bypass the daemon socket or run the venv Python outside
  `devenv`.
- GPU 1/vLLM is out of scope: never stop, restart, benchmark, or alter it.
- Never stop `paseo.service`. Only terminate a specifically identified stale
  GPU-0 SGLang child after recording it and confirming it is the experiment.
- Bind only `127.0.0.1:8002`; use GPU 0 only; keep downloads disabled.
- Create a fresh UTC directory under `artifacts/sglang-gemma4-spike/` for each
  attempt. Capture before/during/after topology, full command/log, and all
  test output.
- Use `apply_patch` for repository edits and preserve unrelated dirty work.
- Do not test MTP or call API support successful until base GGUF startup and
  the mandated HTTP evidence pass.

## Required loop

1. Run container-safe static checks from `CONTAINER_TESTING.md`.
2. Use approved host execution to record topology and run the exact metadata
   regression. Do not launch if the regression fails.
3. Research the new blocker before patching: inspect current local
   Transformers/SGLang sources, compare SGLang main and its Gemma 4 cookbook
   dependency context, and search primary upstream sources for the exception
   and related Gemma 4 construction issues. Create a dated research report in
   this project with direct links, version/commit context, hypotheses, and the
   smallest safe fix.
4. Add a focused regression test first. Implement only a narrow, reversible
   configuration/constructor change justified by the evidence; preserve
   SGLang's runtime attention and avoid global Transformers behavior changes.
5. Re-run static checks, host metadata regression, topology snapshots, and the
   exact GPU-0 command. Save full logs in a fresh artifact directory.
6. If a new failure appears, classify it precisely and return to research. Do
   not patch past an unexplained failure.
7. If the server starts, run `/health`, `/v1/models`, one non-streaming chat,
   and one streaming chat; save raw responses and verify meaningful output.

## Deliverables

- A dated research report for the expert-implementation blocker with primary
  source links and version-qualified conclusions.
- A focused regression test and minimal reversible fix only if justified.
- Fresh artifacts for every host-side test attempt.
- Updated `ANALYSIS.md` and `MINIMAL_REPRODUCTION.md`, clearly separating the
  resolved SDPA blocker, current expert blocker, tensor-loader status, API,
  MTP, and 16k capacity.
