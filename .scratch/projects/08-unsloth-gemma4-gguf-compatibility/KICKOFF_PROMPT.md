# Focused session: make the local Unsloth Gemma 4 GGUF work in SGLang

Work in `/home/andrew/Documents/Projects/structured-agents-v2`.

## Mission

Determine, with runtime evidence, how to serve the exact locally cached
Unsloth Gemma 4 12B QAT `UD-Q4_K_XL` GGUF through SGLang on GPU 0. Do not
substitute the native safetensors checkpoint as the solution. If the current
SGLang/Transformers stack cannot support it, produce a precise, minimal,
upstream-actionable diagnosis and identify the smallest credible fix or
version change.

The immediate goal is **base GGUF startup and one successful chat completion**.
MTP is strictly a follow-on task. Do not spend time benchmarking, broad library
integration, LoRA, or structured outputs until the exact GGUF starts.

## Read this first

Read all of:

1. `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md`
2. `.scratch/projects/07-sglang-gguf-spike/PROMPT.md`
3. `deploy/sglang/native/{README.md,serve.sh,run.sh,sitecustomize.py,pyproject.toml,devenv.nix}`
4. The relevant logs in
   `artifacts/sglang-gemma4-spike/20260714T213101Z/`:
   - `sglang-log-tail-after-load.txt` — first exact-GGUF failure
   - `sglang-adapter-log-tail.txt` — diagnostic adapter failure
   - `sglang-native-qat-retry-stdout-stderr.log` — native control success
   - `processes-before.txt`, `processes-after-sglang-failure.txt`, and GPU
     snapshots — topology/non-interference evidence

The analysis document is the current source of truth. Correct it if new
evidence supersedes it; do not repeat an earlier conclusion without preserving
the version context that produced it.

## Exact target and local assets

Base target GGUF (immutable blob; do not alter it):

```text
/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165
```

Cached upstream Gemma config/tokenizer directory:

```text
/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac
```

Existing MTP-related GGUF files (not for the initial base-model test):

```text
/home/andrew/.cache/structured-agents/models/mtp-gemma-4-12B-it.gguf
/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf
```

The assistant cache SHA-256 previously verified as:

```text
13331068b6af643c3dc75e619373b674c1f75a1958e7c82e2020d96a17c63809
```

## Non-negotiable safety and scope

- GPU 1 hosts live vLLM at `127.0.0.1:8000`. Do not modify, restart, stop,
  benchmark, or otherwise disturb it. Capture a before/after process and GPU
  snapshot for every GPU-0 experiment.
- GPU 0 is available for SGLang because the user stopped llama.cpp. Do not
  restore llama.cpp unless asked. Do not touch its model or MTP cache files.
- Bind experimental SGLang only to `127.0.0.1:8002`.
- Use the isolated environment under `deploy/sglang/native/`; never reuse or
  mutate the vLLM virtual environment or caches.
- Keep model downloads disabled unless the user explicitly approves a download.
  Existing local files and source inspection are in scope.
- Use `apply_patch` for repository edits; preserve unrelated dirty changes.
- Save all new runtime evidence under a fresh UTC directory beneath
  `artifacts/sglang-gemma4-spike/`.
- Do not call a configuration accepted, a model loaded, MTP working, or 16k
  viable until the relevant runtime evidence exists.

## What is already known

### Baseline topology

- Both GPUs are NVIDIA RTX 3060 12 GiB.
- vLLM on GPU 1 currently serves the same Unsloth GGUF successfully, using
  `--quantization gguf` and an explicit `--hf-config-path`.
- llama.cpp previously served the same GGUF on GPU 0 with MTP. That proves
  the artifact is usable in llama.cpp; it does not prove SGLang support.

### Isolated SGLang environment

- SGLang is pinned at `0.5.14`, Python 3.12.
- `pyproject.toml` overrides Transformers with a pinned source checkout:
  commit `ab1771c9e42891d893189978a8009426d70b4688`, reporting
  `5.14.0.dev0`.
- The source checkout was selected because released Transformers could not
  parse Gemma 4's `gemma4_unified` configuration.
- `sitecustomize.py` has two separate purposes:
  1. tolerate duplicate AutoConfig registration caused by mixing SGLang 0.5.14
     with the newer Transformers source; and
  2. **only when `SGLANG_GGUF_CONFIG_PATH` is set**, redirect SGLang config
     lookup to a local config file. That second behavior is a diagnostic
     adapter and must be disabled for the first clean retry.

### Runtime evidence and the crucial version distinction

1. The original direct GGUF attempt failed before weights loaded with:

   ```text
   ValueError: GGUF model with architecture gemma4 is not supported yet.
   ```

   That trace came from the earlier Transformers dependency state.

2. A diagnostic local-config adapter bypassed that parser but then failed with:

   ```text
   Field 'num_key_value_heads' expected int, got list
   [8, 8, 8, 8, 8, 1, ...]
   ```

   Do **not** infer that a stock current parser will necessarily fail this way:
   the adapter intentionally skipped its conversion logic.

3. Inspection of the current pinned Transformers source shows a `gemma4`
   entry in `GGUF_CONFIG_MAPPING`, Gemma-4-to-`gemma4_text` config handling,
   and GGUF tensor-name mapping for `gemma4_text`. This means the old failure
   is not sufficient to reject the current stack.

4. In the same SGLang environment, local native Google Gemma 4 QAT safetensors
   loaded and served HTTP 200. It consumed about 8.74 GiB for weights and
   yielded only 1,710 full-layer KV tokens at the configured 80% static memory
   fraction. This proves GPU/CUDA/SGLang native Gemma 4 viability, but it is
   not GGUF success and does not prove 16k capacity.

## Primary workflow

### 1. Record state and validate the isolated setup

Before runtime work, record:

- vLLM service/process/arguments and GPU assignment;
- GPU memory and active compute processes;
- SGLang, Transformers, torch, CUDA, driver, and gguf-py versions;
- the exact `sitecustomize.py` behavior actually loaded by the environment.

Run non-destructive checks first. If entering `devenv shell` needs host Nix
daemon access, use the normal approval mechanism rather than trying to bypass
it. A previous attempt was blocked only by sandbox approval timeout, not by a
technical environment failure.

### 2. First decisive experiment: stock current GGUF parser

Run the exact GGUF through the **current pinned environment** with:

- `MODEL_LOAD_FORMAT=gguf`
- the exact blob as `MODEL_PATH`
- cached tokenizer/config directory as `TOKENIZER_PATH`
- `SGLANG_GGUF_CONFIG_PATH` unset
- `ENABLE_MTP=0`
- `CUDA_VISIBLE_DEVICES=0`
- loopback port 8002, one request, no CPU offload

Use `deploy/sglang/native/run.sh` / `serve.sh` where possible. Do not add a
config override, model rewrite, or MTP flag to this first retry. Capture the
full resolved command, environment versions, logs, process tree, health probe,
and GPU snapshots.

Classify the result exactly:

- config conversion failure;
- model implementation selection failure;
- GGUF tensor-name/type/quantization loader failure;
- CUDA/kernel/VRAM failure;
- successful weight load but API failure; or
- successful base API.

### 3. Read-only GGUF metadata and mapping analysis

Regardless of the result, inspect the GGUF metadata without loading tensors.
Capture at minimum:

- `general.architecture`, name, file/quantization metadata;
- `gemma4.*` config fields, especially `attention.head_count_kv`, key length,
  sliding-window, block count, and context length;
- tensor-name prefixes, tensor quantization types, and any multimodal/projector
  tensors;
- how those fields map through current Transformers
  `GGUF_CONFIG_MAPPING`, `GGUF_SUPPORTED_ARCHITECTURES`, and
  `get_gguf_hf_weights_map`;
- whether the installed `gguf` package recognizes the `gemma4` architecture.

Do not rely on a grep alone: use the local GGUF reader and save a concise,
reproducible metadata report. Compare metadata expectations against the cached
native Gemma config.

### 4. Research only after reproducing on the current stack

Search official primary sources first:

- SGLang source, releases, docs, and GitHub issues;
- Hugging Face Transformers source/issues/PRs involving Gemma 4 GGUF;
- ggml-org/llama.cpp GGUF specification, Gemma 4 conversion code, and issues;
- Unsloth’s model page/repository for the exact target's format and dynamic
  quantization details.

Useful starting references:

- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- https://github.com/sgl-project/sglang/releases
- https://github.com/sgl-project/sglang/issues/22370
- https://github.com/sgl-project/sglang/issues/22277
- https://github.com/huggingface/transformers
- https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF

GitHub issue links are evidence of ongoing support work, not proof of
compatibility. Reddit may be used for leads but cannot be the basis for a
technical conclusion.

### 5. Choose the smallest next fix based on evidence

Preferred order:

1. an upstream release/commit that supports the exact conversion and loader;
2. a narrowly scoped local compatibility patch that faithfully ports the
   upstream conversion/model-loader behavior and is reversible;
3. an upstream issue/reproducer if the capability is genuinely absent.

Do not permanently paper over a per-layer KV-head list by taking `max()`,
hardcoding a scalar, or editing target metadata. Gemma 4 mixed attention is
semantically meaningful; an apparently successful startup could produce
incorrect output.

If a local patch is warranted, add a minimal regression test that parses the
target metadata or a sanitized metadata fixture, plus a clear removal condition
(for example, a specific upstream version/PR).

### 6. Only after base GGUF success

Run `/health`, `/v1/models`, a basic non-streaming chat completion, and a
streaming completion. Save raw responses and validate meaningful output.

Only then investigate speculative decoding/MTP. First find a documented SGLang
interface compatible with a GGUF target plus a GGUF draft/MTP assistant. MTP is
working only if logs/telemetry show nonzero proposals **and** accepted tokens.
Do not use a different HF draft checkpoint as a substitute for the requested
Q8 GGUF assistant.

## Deliverables

1. Update `ANALYSIS.md` with a dated evidence table distinguishing each
   environment/version and its exact outcome.
2. Add `METADATA_REPORT.md` in this project with the read-only metadata and
   mapping comparison.
3. Store raw command output and runtime captures under a new timestamped
   `artifacts/sglang-gemma4-spike/<UTC>/` directory.
4. If unsuccessful, provide an upstream-ready minimal reproduction: exact
   package versions/commits, non-sensitive metadata excerpt, command, complete
   traceback, expected behavior, and the suspected component.
5. If successful, document the supported launch command, actual VRAM/KV
   capacity, basic API proof, and remaining MTP/structured-output gaps.

## Definition of done for this session

At minimum, do one clean adapter-free current-stack attempt and conclusively
locate the next blocker using the saved evidence. Preferably make the exact
target load and return a valid chat completion. Never report GGUF or MTP
support based solely on native safetensors success or source-code presence.
