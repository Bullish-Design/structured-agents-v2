# Unsloth Gemma 4 GGUF compatibility investigation

**Status:** active feasibility investigation  
**Created:** 2026-07-14  
**Goal:** make the locally cached Unsloth Gemma 4 12B QAT `UD-Q4_K_XL` GGUF
serve through SGLang on GPU 0, then determine whether its GGUF MTP assistant
can be used. This is independent of the already-working native SGLang Gemma 4
QAT path.

## Executive finding

SGLang's `--load-format gguf` option is real, but the exact target has **not
yet** loaded through SGLang. The first startup attempt, under the earlier
Transformers dependency state, fails before any weights are loaded because its
GGUF reader lacks `gemma4` architecture support. The subsequently pinned
Transformers source checkout does contain a `gemma4` GGUF mapping; a clean
stock-GGUF retry in that *current* environment remains the immediate next
experiment. An intentionally narrow local-config adapter was also tried: it
preserves the GGUF as the weight path and supplies the already-cached upstream
Gemma configuration only for config resolution. That adapter gets past the
first failure but fails during Gemma configuration validation: the GGUF
metadata has a *per-layer list* of KV-head counts while the `Gemma4Config` path
being used requires one integer.

This is not a VRAM failure, a tokenizer-path failure, or evidence that native
Gemma 4 support is broken. The same isolated SGLang environment successfully
loaded native Google Gemma 4 QAT safetensors and served an OpenAI-compatible
request on the same GPU. The outstanding issue is the combination of:

1. whether the current Transformers Gemma 4 GGUF converter can turn this
   metadata into a valid `gemma4_text` configuration; and
2. SGLang's GGUF loader/model-weight mapping for this dynamic-quantized,
   multimodal Gemma 4 architecture.

The Q8 GGUF MTP assistant has **not** been attempted in SGLang because the base
GGUF has not reached weight loading. No MTP compatibility claim is justified.

## Scope and preserved baseline

Target base model (immutable local blob):

```text
/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165
```

Relevant companion assets:

- Cached upstream Gemma config/tokenizer directory:
  `/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac`
- llama.cpp's working MTP draft model:
  `/home/andrew/.cache/structured-agents/models/mtp-gemma-4-12B-it.gguf`
- Q8 assistant cache (verified earlier):
  `/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf`
  (SHA-256 `13331068b6af643c3dc75e619373b674c1f75a1958e7c82e2020d96a17c63809`).

The existing vLLM service stays active on GPU 1 and was neither restarted nor
modified. llama.cpp on GPU 0 was stopped by the user to allow the isolated
runtime experiments. The current SGLang native-QAT experiment is bound to
`127.0.0.1:8002`; it must not be confused with the target GGUF attempt.

## Environment tested

- GPU: NVIDIA RTX 3060, 12 GiB VRAM (GPU 0)
- SGLang: `0.5.14`, dedicated `deploy/sglang/native/.venv`
- Python: 3.12
- Transformers: upstream `main`, locked at commit
  `ab1771c9e42891d893189978a8009426d70b4688` (`5.14.0.dev0` reported)
- Model launch settings: one request, GPU 0, loopback `127.0.0.1:8002`,
  `--context-length 16384`, `--mem-fraction-static 0.80`, no CPU offload,
  XGrammar selected, CUDA graphs disabled.

The Transformers source checkout was necessary because the released package
could not parse the newer `gemma4_unified` configuration. SGLang 0.5.14 also
tries to re-register some model config names already present in that newer
Transformers build; the local adapter makes those registrations idempotent.
Neither change fixes GGUF weight compatibility.

## Runtime evidence: exact target GGUF

### Attempt 0 — current-stack read-only parser and metadata inspection (2026-07-14)

The current isolated environment was verified as SGLang `0.5.14`, Python
`3.12.13`, Torch `2.11.0+cu130`, and Transformers `5.14.0.dev0`; the
`SGLANG_GGUF_CONFIG_PATH` diagnostic adapter was unset. A local `GGUFReader`
read of the immutable target found `general.architecture=gemma4`, 48 blocks,
and an alternating per-layer `gemma4.attention.head_count_kv` list: five `8`
values (sliding attention) then `1` (full attention), repeated eight times.

This pin now recognizes `gemma4` and produces a `gemma4_text` config, so the
old unsupported-architecture failure has been superseded. But its mapping
still directly assigns the list to `Gemma4Config.num_key_value_heads`, which
is a strict scalar `int`. The emitted current-stack config is saved as
`transformers-current-config-from-gguf.json`; its list is invalid for model
config instantiation, matching the prior strict-validation failure. The
converter has no Gemma-4-specific equivalent of its LFM2 per-layer conversion.

This conclusively locates the next blocker before weight loading: **the
Transformers Gemma 4 GGUF config converter does not reconstruct the native
mixed-attention fields from `attention.head_count_kv`**. The minimal credible
fix must preserve the per-layer semantics (scalar sliding/global KV counts plus
layer types, or a supported equivalent); taking `max()` is incorrect.

Evidence and a mapping comparison:
[metadata report](METADATA_REPORT.md) and
`artifacts/sglang-gemma4-spike/20260714T234306Z/`.

At the initial snapshot GPU 0 was occupied by an already-running native QAT
SGLang control server on port 8002 (about 9.9 GiB), while GPU 1's vLLM stayed
active (about 10.8 GiB). The user subsequently authorized replacement of that
GPU-0 control process; the resulting clean runtime attempt is recorded below.

### Attempt 0b — decisive clean current-stack SGLang startup (2026-07-15)

After a before/after snapshot, only the GPU-0 native-QAT SGLang process was
terminated; the shared Paseo daemon was not stopped. GPU 0 returned to 1 MiB
used, and GPU 1 remained at 10.8 GiB with
`structured-agents-vllm.service` active. The exact target was then launched
through `deploy/sglang/native/run.sh` with these non-negotiable properties:

- `SGLANG_GGUF_CONFIG_PATH` explicitly unset;
- `MODEL_LOAD_FORMAT=gguf`, `CUDA_VISIBLE_DEVICES=0`, loopback port 8002;
- the exact immutable blob and cached tokenizer/config path;
- `ENABLE_MTP=0`, `CPU_OFFLOAD_GB=0`, one request, and 16k requested context;
- offline Hugging Face/Transformers settings and a fresh artifact-local cache.

The clean launch failed with exit code 1 before any model weight allocation or
GPU residency:

```text
huggingface_hub.errors.StrictDataclassFieldValidationError:
Validation error for field 'num_key_value_heads':
  TypeError: Field 'num_key_value_heads' expected int, got list
```

This is the required adapter-free current-stack reproduction. It confirms the
failure classification as **config conversion**, not tokenizer, tensor-loader,
CUDA, VRAM, or API failure. Full command, traceback, and GPU/process evidence:
`artifacts/sglang-gemma4-spike/20260714T234306Z/clean-stock-gguf-launch.txt`,
`native-control-{before,after}-termination.txt`, and
`host-topology-clean-launch-live.txt`.

### Attempt 1 — stock SGLang GGUF path

Invocation (with explicit local tokenizer/config assets):

```text
python3 -m sglang.launch_server \
  --model-path <target blob> --tokenizer-path <cached Gemma config> \
  --load-format gguf --host 127.0.0.1 --port 8002 \
  --served-model-name base --context-length 16384 \
  --mem-fraction-static 0.80 --max-running-requests 1 \
  --cpu-offload-gb 0 --grammar-backend xgrammar --disable-cuda-graph
```

Observed failure, before model load / GPU residency:

```text
File transformers/modeling_gguf_pytorch_utils.py, line 648, in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
ValueError: GGUF model with architecture gemma4 is not supported yet.
```

Evidence: [launch and traceback](../../../artifacts/sglang-gemma4-spike/20260714T213101Z/sglang-log-tail-after-load.txt).

Interpretation: SGLang's config path invokes Transformers' GGUF conversion to
derive a Hugging Face configuration. The dependency state used for this
attempt had no usable `gemma4` converter, so the command did not reach
SGLang's Gemma 4 CUDA model implementation.

Later source inspection changes the next action: the pinned current
Transformers checkout *does* include a `gemma4` `GGUF_CONFIG_MAPPING` entry,
includes `gemma4` in the supported-architecture set, maps a text-only GGUF to
`gemma4_text`, and maps `gemma4_text` back to the GGUF `gemma4` tensor names.
The original unsupported trace is therefore evidence for the earlier resolved
dependency state, not proof that the current pin will fail identically. We
must make one clean, adapter-free startup attempt before concluding the
upstream converter is incomplete.

### Attempt 2 — local config adapter to isolate the next blocker

The opt-in adapter at
[sitecustomize.py](../../../deploy/sglang/native/sitecustomize.py) replaces only
SGLang's config lookup when `SGLANG_GGUF_CONFIG_PATH` is explicitly set. The
GGUF blob remains `--model-path`; only config discovery is redirected to the
cached upstream `config.json`. This was a diagnostic experiment, not a claimed
deployment solution.

Observed failure:

```text
TypeError: Field 'num_key_value_heads' expected int, got list
value: [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, ...]
```

It is wrapped by `huggingface_hub.errors.StrictDataclassFieldValidationError`.
Evidence: [adapter launch and traceback](../../../artifacts/sglang-gemma4-spike/20260714T213101Z/sglang-adapter-log-tail.txt).

Interpretation: Gemma 4 has heterogeneous layers (the alternating values are
consistent with its local/global-attention design), whereas the configuration
class reached by the adapter accepts a scalar KV-head count. Therefore simply
injecting an HF config is insufficient: the GGUF metadata-to-SGLang config
translation must preserve Gemma 4's per-layer structure, and the downstream
model/loader must be able to consume it.

## Control result: native Gemma 4 works

With the exact same SGLang environment and GPU, the locally cached Google
`gemma-4-12B-it-qat-w4a16-ct` safetensors checkpoint loaded as
`Gemma4UnifiedForConditionalGeneration` with `quant=compressed-tensors`:

```text
Load weight end ... avail mem=2.74 GB, mem usage=8.74 GB.
Use sliding window memory pool. full_layer_tokens=1710, swa_layer_tokens=1368
```

It served `/health`, `/v1/models`, and a first `/v1/chat/completions` request
with HTTP 200. Evidence: [native load log](../../../artifacts/sglang-gemma4-spike/20260714T213101Z/sglang-native-qat-retry-stdout-stderr.log),
[model response](../../../artifacts/sglang-gemma4-spike/20260714T213101Z/native-qat-models-update.json), and
[GPU/API capture](../../../artifacts/sglang-gemma4-spike/20260714T213101Z/gpu-native-qat-api.csv).

This proves the following independently:

- GPU, CUDA, SGLang installation, tokenizer, and basic Gemma 4 SGLang model
  implementation are viable.
- 16k is accepted as configured but **not usable capacity** on this GPU with
  this model: SGLang allocated only 1,710 full-layer tokens. This is a separate
  memory constraint to address after GGUF startup.
- The exact Unsloth GGUF is the compatibility problem; replacing it with native
  weights is useful for feasibility but does not satisfy the GGUF goal.

## Why generic GGUF support is not enough

SGLang's documented `gguf` load format means that selected GGUF architectures
and their tensor naming/metadata mappings are supported. It does not imply:

- a Transformers GGUF converter for every `general.architecture` value;
- support for Gemma 4's alternating/per-layer KV-head metadata;
- mapping of Unsloth's dynamic QAT `UD-Q4_K_XL` tensor types into SGLang's CUDA
  kernels;
- multimodal Gemma 4 projector/vision weights in a GGUF representation; or
- GGUF base plus GGUF MTP assistant support in SGLang's speculative decoder.

The target works through llama.cpp, which implements the GGUF format and its
Gemma 4/MTP conventions directly. vLLM also serves it using an explicit
`--quantization gguf` configuration and `--hf-config-path`; that is evidence
for those two engines, not automatic evidence for SGLang.

## Upstream references consulted

Primary sources, checked 2026-07-14:

- [SGLang server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md): documents `gguf` as a load format,
  but not a compatibility promise for Gemma 4 GGUF.
- [SGLang releases](https://github.com/sgl-project/sglang/releases): describe
  Gemma 4 and later Gemma 4 MTP work. The release notes are oriented to native
  SGLang model support and do not state that Unsloth Gemma 4 GGUF or a GGUF MTP
  assistant is supported.
- [SGLang issue #22370](https://github.com/sgl-project/sglang/issues/22370):
  a Gemma 4 loading report using an AutoRound/native-style checkpoint. It
  illustrates active Gemma 4 work but not GGUF coverage.
- [SGLang issue #22277](https://github.com/sgl-project/sglang/issues/22277):
  Gemma4 KV-cache behavior issue. It independently confirms that Gemma 4's
  architecture/KV behavior needs model-specific handling even on native paths.
- [SGLang source tree](https://github.com/sgl-project/sglang): the codebase
  exposes a GGUF loader and a separate Gemma 4 model implementation; the local
  call stack demonstrated that config conversion is the first blocker here.
- [Transformers source tree](https://github.com/huggingface/transformers):
  source of `modeling_gguf_pytorch_utils.py`, which emitted the exact
  unsupported-`gemma4` error above.
- [Unsloth Gemma 4 GGUF repository](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF): source of the target artifact/format family. It should be
  consulted for the exact GGUF metadata and quantization provenance before a
  converter patch is proposed.
- [SGLang Gemma 4 cookbook](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Google/Gemma4.mdx), reviewed 2026-07-15:
  confirms Gemma 4's hybrid sliding/full attention, PLE, and native SGLang-main
  support, and pins SGLang main with Transformers commit
  `1423d22f7a3b62e8c70ad67b58ec25cd9b675897`. Its QAT instructions describe
  `qat-q4_0-unquantized` bf16 checkpoints and matching HF assistant models;
  it does not document GGUF loading, Unsloth dynamic Q4, or a GGUF MTP draft.
  Thus it supports the semantic requirements of the local patch but is not
  evidence that this exact GGUF is supported.

Searches also included SGLang GitHub issues and Reddit. GitHub yielded useful
architecture-specific reports; Reddit did not produce a reproducible SGLang
Gemma 4 GGUF solution and is not relied on for a technical conclusion.

## Next investigation sequence

### Construction-time SDPA investigation (2026-07-15)

The patched GGUF startup reached `Load weight begin` and then failed while
SGLang constructed its own `Gemma4ForCausalLM`, before a GGUF tensor was read.
The exception originates in Transformers' `PreTrainedModel.__init__`: the
resolved `Gemma4TextConfig` retains an explicit SDPA selection, while that HF
base class does not advertise SDPA support for Gemma 4.

This is distinct from SGLang runtime attention. The captured server arguments
already selected `attention_backend='triton'`, and SGLang's Gemma 4 module uses
`RadixAttention` in its forward path. Current SGLang documentation likewise
states that Gemma 4 automatically selects Triton attention. Therefore the
local compatibility layer now changes only a `gemma4_text` config's internal
Transformers construction selection to `eager`, immediately before
`sglang.srt.model_loader.loader._initialize_model` calls the SGLang model
constructor. It does not change `--attention-backend`, SGLang's attention
classes, or any non-Gemma config.

The source-level regression asserts both the Gemma-only eager rewrite and the
non-Gemma no-op. The full metadata/config regression and GPU-0 retry remain
pending a host Nix/GPU session: the present sandbox cannot communicate with
the NVIDIA driver or Nix daemon. The patch must be removed when the pinned
Transformers/SGLang combination no longer carries the incompatible explicit
SDPA selection (or when upstream makes this construction-time workaround
unnecessary); it is not evidence of tensor loading or API success.

### Attempt 4 — eager attention construction patch (2026-07-15)

The exact target was launched on host GPU 0 with `SGLANG_GGUF_CONFIG_PATH`
unset, `MODEL_LOAD_FORMAT=gguf`, loopback port 8002, one slot, no CPU offload,
and MTP disabled. The metadata/config regression passed first. A before
snapshot recorded GPU 0 at 4 MiB and GPU 1's unchanged vLLM engine at
10,784 MiB; the after snapshot recorded GPU 0 at 1 MiB and the same GPU 1
engine, with `structured-agents-vllm.service` still active.

The eager construction patch removed the earlier SDPA exception. The server
selected SGLang Triton attention, initialized Torch distributed, and reached:

```text
Load weight begin. avail mem=11.48 GB
```

It then failed during the same SGLang `Gemma4ForCausalLM` constructor, but at
the next Transformers base-class validation:

```text
ValueError: Gemma4ForCausalLM does not support setting experts implementation.
```

The trace is in `transformers.modeling_utils._grouped_mm_can_dispatch`, before
the GGUF loader reads a tensor. Current Transformers defaults an unspecified
expert implementation to `grouped_mm`; its explicit-request path raises for
the SGLang model class. This is a second construction-compatibility issue, not
a tensor-name, quantization, CUDA, VRAM, API, or output-correctness result.
Do not patch it until the resolved configuration's explicit expert setting and
SGLang main's intended handling are compared. Raw launch and topology evidence:
`artifacts/sglang-gemma4-spike/20260715T015057Z/`.

1. **Patched current-stack retry (first).** Start the exact GGUF with the
   compatibility layer enabled, then capture and classify the next loader
   result.
   current Transformers checkout, no `SGLANG_GGUF_CONFIG_PATH` and no adapter.
   Capture the complete log. This isolates the new upstream GGUF converter
   from the diagnostic adapter.
2. **Read the target GGUF metadata without loading weights.** Compare its
   architecture, tensor names/types, and per-layer fields against the current
   converter expectations. Save the comparison in this project.
3. **Inspect current SGLang `main` and current Transformers source** for a
   post-0.5.14 Gemma 4 GGUF converter/loader change. If upstream support exists,
   test it in a *new isolated environment* before changing the existing spike.
4. **If no upstream support exists, create a minimal reproduction** containing
   the metadata (not model weights), the exact command, environment versions,
   and both stack traces. Use it to search existing issues and, if genuinely
   novel, prepare an upstream issue rather than patching config fields blindly.
5. **Only after base GGUF reaches `Load weight begin/end`**, test a short chat
   completion. Then determine whether SGLang has a documented way to give it
   the exact Q8 GGUF MTP assistant. Require proposal and acceptance telemetry;
   no inference from command acceptance alone.
6. **After successful base API and MTP tests**, revisit memory. A 12 GiB 3060
   gave native QAT only 1,710 usable full-layer tokens at 80% static memory.
   GGUF may differ, but 16k must be measured, not assumed.

## Guardrails

- Do not change or restart vLLM/GPU 1.
- Do not present the native safetensors result as GGUF success.
- Do not mutate the target blob, tokenizer cache, or MTP cache.
- Keep all new runtime attempts and raw logs in a fresh timestamped artifact
  directory under `artifacts/sglang-gemma4-spike/`.
- Prefer an upstream-supported release or a narrowly scoped upstream patch over
  permanent monkeypatches of model configuration semantics.
