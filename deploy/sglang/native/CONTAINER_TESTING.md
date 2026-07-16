# Running the isolated SGLang tests from a Codex container

This runbook is for agents operating in the repository container while the
actual NVIDIA GPUs, systemd services, and Nix daemon live on the host.

The target is an experimental SGLang server on GPU 0 only. GPU 1 runs the live
vLLM service and is out of scope.

## Before you run anything

Read these documents first:

- `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/ANALYSIS.md`
- `.scratch/projects/08-unsloth-gemma4-gguf-compatibility/NEXT_BUILD_PROMPT.md`
- `deploy/sglang/native/{README.md,serve.sh,run.sh,sitecustomize.py,gemma4_gguf_compat.py}`

Important boundaries:

- Container-only shell commands cannot reliably see host NVIDIA devices or
  host processes. `nvidia-smi` may report that it cannot communicate with the
  driver; do not interpret that as a GPU failure.
- GPU/process inspection and any actual SGLang launch need host execution
  approval (`require_escalated`). Request it through the normal approval flow;
  do not attempt to bypass the sandbox.
- Never stop `structured-agents-vllm.service`, touch its files/environment, or
  use GPU 1.
- Never stop `paseo.service`: it hosts Codex and other work. If GPU 0 has a
  stale SGLang server, identify and terminate only that exact child process,
  and only when the user has authorized a replacement experiment.
- Keep downloads disabled. The launch scripts enforce offline Hugging Face and
  Transformers mode.

## Paths and fixed runtime inputs

```bash
export REPO=/home/andrew/Documents/Projects/structured-agents-v2
export NATIVE="$REPO/deploy/sglang/native"
export MODEL_PATH=/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165
export TOKENIZER_PATH=/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac
```

Never modify the blob or the tokenizer/config cache. The exact file is
extensionless; `MODEL_LOAD_FORMAT=gguf` is what declares its format.

## 1. Container-safe static checks

These checks do not use host CUDA, do not start a server, and should be run
before requesting GPU access.

```bash
cd "$REPO"
git status --short
git diff --check
./deploy/sglang/native/test_serve.sh
python -m py_compile \
  deploy/sglang/native/sitecustomize.py \
  deploy/sglang/native/gemma4_gguf_compat.py \
  deploy/sglang/native/test_gemma4_gguf_compat.py
```

`test_serve.sh` is a shell contract test only. Its printed `sglang.launch_server`
command is expected; it uses a temporary fake Python executable and does not
import SGLang or load a model.

## 2. Required host snapshot

Run this only with host approval. Create a fresh UTC artifact directory first.

```bash
cd "$REPO"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/sglang-gemma4-spike/$STAMP"
mkdir -p "$OUT"

{
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,ppid,user,unit,args | rg -i '[v]llm|[l]lama-server|[s]glang'
  systemctl is-active structured-agents-vllm.service
  systemctl is-active structured-agents-llama-cpp.service || true
} > "$OUT/host-topology-before.txt"
```

Acceptance conditions before a GPU-0 test:

- GPU 1 has vLLM, and `structured-agents-vllm.service` is `active`.
- GPU 0 is free, or the user explicitly authorized replacement of the
  identified GPU-0 experimental SGLang process.
- No service-wide operation against `paseo.service` is needed or permitted.

## 3. Entering the isolated Nix/uv environment

Host execution needs the unfree CUDA setting before `devenv` evaluates:

```bash
cd "$NATIVE"
export NIXPKGS_ALLOW_UNFREE=1
devenv shell --impure -- uv run --locked --no-sync python -c '
import sglang, torch, transformers
print("sglang", sglang.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
'
```

Do not run the venv Python directly outside `devenv`: it may fail to find Nix
runtime libraries such as `libstdc++.so.6`, producing misleading NumPy errors.

## 4. Read-only Gemma 4 GGUF regression test

This is the required check after changing the local compatibility layer. It
memory-maps metadata and creates a config; it does not load GGUF tensors or a
CUDA model.

```bash
cd "$NATIVE"
export NIXPKGS_ALLOW_UNFREE=1
MODEL_PATH="$MODEL_PATH" \
  devenv shell --impure -- uv run --locked --no-sync \
  python test_gemma4_gguf_compat.py
```

The test verifies the exact target’s essential semantics:

- local/sliding KV heads: 8;
- global/full KV heads: 1;
- the 48-layer 5:1 sliding/full pattern;
- no per-layer embeddings (`hidden_size_per_layer_input=0`);
- K=V for full-attention layers; and
- the SGLang GGUF tensor-map alias for `gemma4_text`.

Capture stdout/stderr in the fresh artifact directory. A failure means do not
launch GPU startup yet.

## 5. Exact GPU-0 startup attempt

This is an authorized host action. It uses GPU 0, loopback port 8002, one slot,
no CPU offload, no downloads, and no MTP. The config adapter must remain unset.

```bash
cd "$REPO"
mkdir -p "$OUT/hf-cache"

env -u SGLANG_GGUF_CONFIG_PATH \
  MODEL_PATH="$MODEL_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  MODEL_LOAD_FORMAT=gguf \
  CUDA_VISIBLE_DEVICES=0 \
  PORT=8002 \
  CONTEXT_LENGTH=16384 \
  MAX_RUNNING_REQUESTS=1 \
  CPU_OFFLOAD_GB=0 \
  ENABLE_MTP=0 \
  SGLANG_CACHE_DIR="$OUT/hf-cache" \
  bash deploy/sglang/native/run.sh \
  2>&1 | tee "$OUT/sglang-gguf-launch.txt"
```

Do not use `SGLANG_GGUF_CONFIG_PATH` for this test. It activates a diagnostic
adapter and invalidates an adapter-free compatibility claim.

If the process stays alive, collect a second host snapshot while it starts:

```bash
{
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,ppid,user,unit,args | rg -i '[v]llm|[s]glang'
  systemctl is-active structured-agents-vllm.service
} > "$OUT/host-topology-during-launch.txt"
```

How to classify the result:

| Last evidence | Classification |
| --- | --- |
| error before `Load weight begin` | config/model-selection failure |
| `Load weight begin`, then tensor/key/type error | GGUF tensor-map or quantization loader failure |
| CUDA OOM/kernel error | CUDA/VRAM failure |
| `Load weight end`, server exits | post-load server/API failure |
| health endpoint available | proceed to API proof |

Never silently patch values such as KV-head count to `max()`; Gemma 4’s
per-layer attention is semantically required.

## 6. API proof after a successful load

Only after logs show the server is listening on `127.0.0.1:8002`:

```bash
curl -fsS http://127.0.0.1:8002/health | tee "$OUT/health.json"
curl -fsS http://127.0.0.1:8002/v1/models | tee "$OUT/models.json"
curl -fsS http://127.0.0.1:8002/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"base","messages":[{"role":"user","content":"Reply with exactly: GGUF works."}],"max_tokens":32,"temperature":0}' \
  | tee "$OUT/chat.json"
```

Then run a streaming request and save raw SSE output. Verify meaningful model
text, not only HTTP 200. Do not test MTP, structured output, LoRA, or a
benchmark before baseline chat completion works.

## 7. Shutdown and after snapshot

When the experiment is finished, terminate only the identified SGLang launch
process and its children. Do not stop `paseo.service`; do not restart llama.cpp
unless the user asks. Then collect a final host snapshot using the same command
as Section 2 and save it as `host-topology-after.txt`.

The final report must name the exact command, package versions/commits, latest
failure stage or API proof, GPU memory state, and whether GPU 1 vLLM remained
active.
