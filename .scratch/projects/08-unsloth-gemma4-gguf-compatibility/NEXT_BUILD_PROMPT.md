# Next build: advance exact Unsloth Gemma 4 GGUF through SGLang

Work in `/home/andrew/Documents/Projects/structured-agents-v2`.

## Mission

Continue the isolated GPU-0 effort to serve the exact locally cached Unsloth
Gemma 4 12B QAT `UD-Q4_K_XL` GGUF in SGLang. Do not substitute the native
safetensors checkpoint. The immediate goal is still base GGUF startup and one
valid chat completion; MTP follows only after that is runtime-proven.

The previous session made genuine progress: the exact GGUF now reaches
SGLang's `Load weight begin`. The current blocker is model construction:

```text
ValueError: Gemma4ForCausalLM does not support an attention implementation
through torch.nn.functional.scaled_dot_product_attention yet.
...
load your model with attn_implementation="eager"
```

This happens before weights are loaded. Resolve it narrowly, preserve SGLang's
own Gemma 4 Triton attention path, and then expose the next real tensor-loader
or runtime result.

## Read in full before acting

1. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md`
2. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/METADATA_REPORT.md`
3. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/MINIMAL_REPRODUCTION.md`
4. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/NEXT_BUILD_PROMPT.md`
5. `deploy/sglang/native/{README.md,run.sh,serve.sh,sitecustomize.py,gemma4_gguf_compat.py,test_gemma4_gguf_compat.py,pyproject.toml,devenv.nix}`
6. The latest raw log:
   `artifacts/sglang-gemma4-spike/20260714T234306Z/patched-gguf-loader-alias-launch.txt`

Also inspect the earlier raw evidence in that artifact directory, particularly:

- `clean-stock-gguf-launch.txt` — clean adapter-free initial failure;
- `gguf-metadata.json` and `gguf-dump-raw.txt` — immutable target metadata;
- `gemma4-gguf-compat-regression.txt` — compatibility regression test;
- `patched-gguf-launch-supervised.txt` — config patch reached `Load weight begin`;
- `host-topology-before.txt`, `native-control-after-termination.txt`, and
  `status-latest.txt` — GPU topology/non-interference evidence.

## Exact target and local assets

Base GGUF (immutable; do not alter):

```text
/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165
```

Cached config/tokenizer:

```text
/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac
```

MTP assets (out of scope until base API success):

```text
/home/andrew/.cache/structured-agents/models/mtp-gemma-4-12B-it.gguf
/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf
```

## Safety and operational constraints

- GPU 1 has live vLLM at `127.0.0.1:8000`; never stop, restart, benchmark, or
  otherwise modify it. Take before/after host process and GPU snapshots for
  every GPU-0 attempt.
- GPU 0 is currently free after the previous failed SGLang experiment. Do not
  restore llama.cpp. Bind SGLang only to `127.0.0.1:8002`.
- The shared `paseo.service` hosts Codex and other work. Never stop it. If a
  stale GPU-0 SGLang child exists, terminate only that identified child process
  after recording its PID/arguments and confirming it is the experimental
  server.
- Use only `deploy/sglang/native/`; do not mutate/reuse vLLM's environment or
  cache. Keep downloads disabled.
- Use `apply_patch` for repository edits. Preserve unrelated dirty work.
- Save new evidence under a fresh UTC directory below
  `artifacts/sglang-gemma4-spike/`.
- Do not call load, API support, MTP, or 16k viability successful without the
  corresponding runtime evidence.

## Current environment and confirmed facts

- SGLang `0.5.14`; Python `3.12.13`; Torch `2.11.0+cu130`.
- Transformers `5.14.0.dev0`, pinned to
  `ab1771c9e42891d893189978a8009426d70b4688` in `uv.lock`.
- Two RTX 3060 12 GiB GPUs; GPU 0 has around 11.48 GiB free when idle.
- The target is a text-only GGUF: `general.architecture=gemma4`, 48 blocks,
  667 tensors (338 F32, 329 Q4_0), no vision/projector tensors.
- Its mixed attention metadata is semantically meaningful:
  - 16 attention heads;
  - `attention.head_count_kv` repeats `[8, 8, 8, 8, 8, 1]` eight times;
  - the matching sliding/full pattern is five sliding layers then one full;
  - full layers have K=V (no V projection); sliding/local KV heads are 8 and
    full/global KV heads are 1;
  - `hidden_size_per_layer_input=0`; do not accept the HF default 256.
- SGLang native Google Gemma 4 QAT safetensors previously served HTTP 200 on
  GPU 0. That proves the general GPU/CUDA/SGLang Gemma 4 path, not GGUF.

## Local compatibility patch already present

`deploy/sglang/native/gemma4_gguf_compat.py`, installed by
`sitecustomize.py`, is temporary and has an explicit removal condition. It:

1. converts the GGUF's per-layer metadata into valid `Gemma4TextConfig` fields
   (`num_key_value_heads=8`, `num_global_key_value_heads=1`, `layer_types`,
   `attention_k_eq_v=True`, PLE width 0);
2. applies SGLang's required native Gemma 4 local/global-to-SWA inversion to a
   text-only config; and
3. aliases gguf-py's `gemma4` tensor map to Transformers/SGLang's
   `gemma4_text` model type.

The regression test reads the exact target metadata without loading tensors:

```bash
cd deploy/sglang/native
export NIXPKGS_ALLOW_UNFREE=1
MODEL_PATH=/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165 \
  devenv shell --impure -- uv run --locked --no-sync python test_gemma4_gguf_compat.py
```

Do not replace this with a hardcoded list or `max()`. If modifying it, add or
update a regression assertion based on target metadata and state why it is
semantically correct.

## The current blocker: attention implementation selection

The patch's last launch got to:

```text
Init torch distributed ends. elapsed=0.42 s, mem usage=0.02 GB
Load weight begin. avail mem=11.48 GB
```

It then failed in `transformers.modeling_utils` during construction of
SGLang's `Gemma4ForCausalLM`: Transformers selected SDPA and rejects it for
Gemma 4. This is not an indication that the GGUF tensors are bad.

The log had already selected SGLang Triton attention:

```text
Use triton as default attention backend for Gemma4
```

Investigate how current SGLang main/the cookbook prevents Transformers'
`PreTrainedModel` initialization from choosing SDPA for its Gemma 4 model
class. The narrow fix may set the HF config's internal attention implementation
to eager solely during SGLang model construction, while retaining SGLang's
Triton `RadixAttention` runtime. Do not globally degrade native Transformers
attention, force an unrelated backend, or claim output correctness until chat
completion succeeds.

Look first at:

```text
deploy/sglang/native/.venv/lib/python3.12/site-packages/sglang/srt/models/gemma4_causal.py
deploy/sglang/native/.venv/lib/python3.12/site-packages/sglang/srt/model_loader/loader.py
deploy/sglang/native/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py
```

Compare with SGLang main before patching. If main has a small upstream fix,
faithfully port only that behavior into the local reversible compatibility
layer, test it, and record the upstream commit/removal condition.

## Official references

- SGLang Gemma 4 cookbook:
  https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Google/Gemma4.mdx
  - confirms hybrid sliding/full attention, PLE, native SGLang-main support,
    Triton attention, and a newer matching Transformers commit
    `1423d22f7a3b62e8c70ad67b58ec25cd9b675897`;
  - its QAT path is `qat-q4_0-unquantized` bf16 and its MTP examples use HF
    assistant models; it does **not** document this GGUF or GGUF MTP.
- Transformers Gemma 4 config:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py
- Transformers GGUF loader:
  https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_gguf_pytorch_utils.py
- SGLang server arguments:
  https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- ggml GGUF specification:
  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

Use primary sources and local source inspection. Search current upstream only
after preserving this exact version context.

## Required workflow

1. Record current host topology, current SGLang/Transformers/Torch versions,
   `sitecustomize` state, and git status. Confirm GPU 1/vLLM remains unchanged.
2. Run the metadata/config regression test above. Save output in a new artifact
   directory.
3. Inspect SGLang main plus the cookbook's matching dependency context for the
   SDPA/eager construction solution. Make the smallest reversible local patch;
   add a regression test that proves the intended config/constructor behavior
   without loading the target weights if practical.
4. Run the exact target with: adapter unset, `MODEL_LOAD_FORMAT=gguf`, GPU 0,
   loopback port 8002, no CPU offload, MTP disabled, one slot, and offline
   caches. Capture resolved command, full trace, GPU/process snapshots.
5. If it reaches a new failure, classify it precisely (tensor name mapping,
   quantization type/kernel, missing parameter, VRAM/CUDA, API). Do not patch
   past it without understanding it.
6. If it starts, run `/health`, `/v1/models`, one non-streaming chat request,
   and one streaming request; save raw responses and validate meaningful output.
   Then document actual VRAM/KV capacity. Do not test MTP until these pass.

## Deliverables

- Update `ANALYSIS.md` with dated, version-qualified evidence and a clear
  outcome table.
- Update `METADATA_REPORT.md` only if new metadata/mapping evidence changes it.
- Add/refresh an upstream-ready minimal reproduction for the current blocker.
- Keep all raw outputs in a fresh `artifacts/sglang-gemma4-spike/<UTC>/`.
- Final report must distinguish: adapter-free stock reproduction, local-patch
  progress, current blocker, and unverified capabilities (API, MTP, 16k,
  structured output).
