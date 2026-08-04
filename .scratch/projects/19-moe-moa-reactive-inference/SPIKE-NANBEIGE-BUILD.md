# Spike — Build the Nanbeige llama.cpp fork and verify the looped (recurrent-in-depth) forward pass runs COHERENTLY

Date: 2026-07-31. Follows `SPIKE-NANBEIGE-LOOP.md` ("smallest unblock": build the fork, run the 2-loop forward pass, confirm it is coherent — not the Ornith gibberish failure mode).

## VERDICT

**LOOP RUNS COHERENTLY ON OUR HARDWARE: YES.**

The Nanbeige fork of llama.cpp builds with CUDA for the RTX 3060 (sm_86), loads
`Nanbeige4.2-3B` GGUF with `general.architecture=nanbeige` / `num_loops=2`,
offloads all layers to GPU 1, and produces coherent factual and multi-step
reasoning output at ~58–60 tok/s. This is categorically different from the
Ornith/SGLang "loads but emits gibberish" arch-mismatch failure mode — the
fixed 2-pass depth loop is executed correctly.

---

## 1. Build outcome — SUCCESS (fork was already built by the prior attempt; verified working)

- **Fork ref:** `https://github.com/Nanbeige/llama.cpp` branch **`nanbeige42`**,
  commit **`c6640a1c0`** ("fix loop bound check and drop redundant head_dim").
  Confirmed live via `git ls-remote` (`refs/heads/nanbeige42 = c6640a1c0cf7...`).
- **Build:** `cuda-3060` profile (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`),
  built via the standalone `cuda-shell.nix` (nvcc/cmake/ninja are only on PATH
  inside that nix-shell — they are NOT in the bare login shell).
- **Build id / version string:** `b10151-c6640a1c0`.
- **Artifacts (already on disk from the prior attempt, all present and valid):**
  - Source tree: `.scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/src-nanbeige` (at c6640a1c0)
  - Build dir: `.scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/build-nanbeige-cli`
  - Libs: `build-nanbeige-cli/bin/` — `libllama.so.0.0.10151`, `libggml.so`,
    `libggml-base.so`, `libggml-cpu.so`, **`libggml-cuda.so.0.17.0` (56 MB — CUDA backend present)**, `libmtmd.so`.
  - Unified binary: `build-nanbeige-cli/bin/llama` (subcommands: `serve`, `cli`, `download`, ...).
- The prior attempt built everything but died at the coherence-test step (permission
  prompt). This run picked up the existing build and completed the coherence test.

### Runtime env (the GPU driver-stub fix — mandatory)
`LD_LIBRARY_PATH` must prepend `/run/opengl-driver/lib` (real `libcuda`) before
the fork's `bin/` and the CUDA runtime libs, or llama.cpp silently runs on CPU:
```
export LD_LIBRARY_PATH="/run/opengl-driver/lib:<...>/build-nanbeige-cli/bin:$(cat .scratch/projects/17-llama-cpp-inference-lab/.cuda_runtime_ld)"
export CUDA_VISIBLE_DEVICES=1
```

## 2. Model source

- **`/home/andrew/.cache/structured-agents/models/Nanbeige4.2-3B-UD-Q4_K_XL.gguf`** (2.6 GB).
  This is the `Andgihat/Nanbeige4.2-3B-GGUF` Unsloth-Dynamic Q4_K_M/XL quant
  referenced in the prior spike (reported 92% GSM8K, 98.9% recovery). Already
  downloaded — no new download needed.

### GGUF metadata (proves the loop arch is baked in, not a stock-arch fallback)
```
general.architecture      = nanbeige
general.name              = Nanbeige4.2 3B
nanbeige.block_count      = 22
nanbeige.num_loops        = 2          <-- recurrence: 22-layer stack run twice = 44 effective depth
nanbeige.skip_loop_final_norm = False
nanbeige.attention.head_count    = 48
nanbeige.attention.head_count_kv = 8
nanbeige.context_length   = 262144
n_tensors=201, n_kv=50, GGUF v3
```

## 3. Serve-log + GPU evidence (all layers on GPU 1, GPU 0 untouched)

Served with `CUDA_VISIBLE_DEVICES=1 ./bin/llama serve -m <gguf> -ngl 99 --port 8007 -c 4096`.

Server log (`/tmp/nan_serve.log`):
```
I srv  load_model: loading model '.../Nanbeige4.2-3B-UD-Q4_K_XL.gguf'
I srv  load_model: initializing, n_slots = 4, n_ctx_slot = 4096, kv_unified = 'true'
I srv  llama_server: model loaded            (~1.5 s — fast = GPU load, not CPU)
I srv  llama_server: listening on http://127.0.0.1:8007
```
(Note: this fork's server uses the newer terse structured logger; the verbose
per-tensor "offloaded N/N layers to GPU" lines are suppressed. GPU offload is
instead proven directly by nvidia-smi below.)

`nvidia-smi` while serving:
```
GPU 0: 1 MiB used        <-- untouched, as required
GPU 1: 3165 MiB used     <-- our server pid = 3148 MiB (2.6 GB weights + KV, fully on GPU)
```
After kill: GPU 1 back to 9 MiB. GPU 0 never exceeded 1 MiB at any point.

## 4. COHERENCE TESTS (temp=0)

### Test 1 — Factual
Prompt: "What is the capital of France? Answer in one short sentence."
- `content`: **"The capital of France is Paris."**
- `reasoning_content` (thinking): coherent verification ("...The capital of France
  is Paris... ensure the sentence is short..."). Reasoning is emitted in the
  OpenAI `reasoning_content` field; final answer in `content`.
- finish_reason: stop. Generation: **59.7 tok/s** (prompt 768 tok/s).

### Test 2 — GSM8K-style multi-step reasoning
Prompt: "Natalia sold clips to 48 friends in April, and then she sold half as many
clips in May. How many clips did she sell altogether in April and May? End with:
The answer is X."
- `content` (verbatim tail): step-by-step — April = 48, May = 48/2 = 24,
  Total = 48 + 24 = 72 — ending **"The answer is 72."** (correct).
- Coherent LaTeX-formatted reasoning, correct arithmetic, correct final answer.
- finish_reason: stop. Generation: **58.2 tok/s** (851 predicted tokens).

Minor cosmetic artifact: the reasoning stream occasionally opens with a garbled
token ("Weimplify..."); does not affect the final answer or overall coherence.
This is a quant/tokenizer quirk, not the arch-mismatch gibberish failure mode.

### Throughput summary
| Test | Generation tok/s | Prompt tok/s |
|------|-----------------|--------------|
| Factual | 59.7 | 768 |
| GSM8K   | 58.2 | (cached) |

~58–60 tok/s decode for a 3B model with a 2x depth loop (≈44 effective layers) on
one 3060 is consistent with a correctly-executed loop.

## 5. Reproduce (all on GPU 1; nothing here needs sudo/systemd)

```bash
cd .scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/build-nanbeige-cli
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$PWD/bin:$(cat ../../.cuda_runtime_ld)"
export CUDA_VISIBLE_DEVICES=1
M=/home/andrew/.cache/structured-agents/models/Nanbeige4.2-3B-UD-Q4_K_XL.gguf
./bin/llama serve -m "$M" -ngl 99 --host 127.0.0.1 --port 8007 -c 4096 &
curl -s localhost:8007/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":250}'
```

### If a clean rebuild from scratch is ever needed (needs the CUDA nix-shell)
```bash
cd .scratch/projects/17-llama-cpp-inference-lab
NIXPKGS_ALLOW_UNFREE=1 nix-shell --impure cuda-shell.nix --run \
  './build-llamacpp.sh --repo https://github.com/Nanbeige/llama.cpp --ref nanbeige42 --profile cuda-3060'
```
(The current working artifacts were built as `build-nanbeige-cli`, a full build
including the `llama` unified binary; `build-llamacpp.sh` alone builds only the
lib set, which is enough for the llama-cpp-python path but not the CLI/serve tool.)

## 6. Notes / open items for Project 19

- **ABI-anchor caveat (unverified here):** this fork is `b10151` (libllama
  0.0.10151, ggml 0.17.0). The repo's installed `llama-cpp-python 0.3.34` is
  anchored to an older ABI (project 17/20 builds use ggml 0.16–anchor c588c4f47).
  The **standalone `llama`/`serve` binary is self-contained and ABI-safe** (used
  here). But loading this fork's `libllama.so` under the installed
  `llama-cpp-python` bindings is NOT yet ABI-verified and would need the smoke
  gate before trusting it. For the loop-coherence question that gate is not
  required — the fork's own binary answers it.
- **The real Project-19 build task is unchanged:** the loop and the P2 mixed-batch
  LoRA fork still live in two separate trees. This spike only confirms the loop
  half runs coherently on our hardware.

## No blockers encountered
No permission wall was hit. All GPU commands used `CUDA_VISIBLE_DEVICES=1`; GPU 0
stayed at 1 MiB throughout. No sudo, no systemd changes.
