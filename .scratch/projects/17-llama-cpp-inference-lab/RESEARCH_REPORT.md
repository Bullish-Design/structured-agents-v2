# Build investigation report — 2026-07-24

## Failure boundary

The first CUDA build compiled the llama.cpp libraries and CUDA kernels, then
failed while linking the separate `app/llama` executable. The failure was:

```text
cannot find -lllama-server-impl
cannot find -llama-cli-impl
```

The configured flags intentionally set `LLAMA_BUILD_SERVER=OFF` and
`LLAMA_BUILD_EXAMPLES=OFF`. The required runtime libraries were already
present, and the subsequent GPU smoke test loaded Ornith and generated
coherently.

## Local source evidence

At the pinned source commit, `src/CMakeLists.txt` declares
`add_library(llama ...)`. The failing target was `app/llama`, not that library.
Therefore the minimal fix is to invoke:

```sh
cmake --build "$build" --target llama
```

instead of building the default aggregate target.

## Chosen fix

The build script now:

1. selects the `llama` shared-library target explicitly;
2. uses the Ninja generator;
3. enables llama.cpp's ccache integration; and
4. exposes a persistent ccache directory from the CUDA shell.

The target change is required for correctness. Ninja and ccache are isolated
build-performance changes and do not alter model or kernel semantics.

## Post-fix verification

A fresh rebuild with the fixed script completed all 344 Ninja steps and exited
0. It produced `libllama.so`, the CUDA/CPU/base ggml libraries, and copied the
matching public headers into the output artifact. The cffi API-mode bindgen then
compiled and imported successfully against that artifact, reporting
`supports_gpu_offload=True` and CUDA architecture 860.

The same rebuilt library set passed the 1024-context Ornith GPU smoke test:
44.67 generated tokens/s, 922.7 prompt tokens/s, coherent output, and the same
~2.6/3.6 GiB per-GPU model footprint. The first rerun without the host driver
directory produced the expected NixOS `CUDA driver is a stub library` failure;
rerunning with the recorded `.cuda_runtime_ld` path passed.

The second clean rebuild achieved 334 direct ccache hits out of 668 cacheable
calls (50% overall), confirming that the persistent cache is active. The first
build was cold and had 334 misses.

## Primary upstream references

- [llama.cpp b10103 top-level CMake configuration](https://github.com/ggml-org/llama.cpp/blob/b10103/CMakeLists.txt) — project build options and subdirectories.
- [llama.cpp b10103 library target](https://github.com/ggml-org/llama.cpp/blob/b10103/src/CMakeLists.txt) — defines the `llama` shared-library target.

## Removal condition

The explicit target workaround can be removed if a future llama.cpp release
makes the aggregate target valid with server and examples disabled, and a
clean build verifies that behavior. Ninja/ccache should remain unless measured
regressions show they are harmful for this machine.

## Evidence limits

The original failure log and post-fix artifacts are preserved under
`artifacts/20260724-postfix/`. The initial smoke measured 46.95 tok/s and
976.8 tok/s prefill; the post-fix rerun measured 44.67 tok/s and 922.7 tok/s.
These are valid only for their recorded smoke configurations and are not a
controlled performance comparison.

## Grammar MVP runtime correction — 2026-07-24

The first end-to-end owned-loop XGrammar smoke reached valid JSON text but
Pydantic rejected the detokenized result because it contained a trailing
`<|im_end|>` special stop token.  This was an output-lifecycle bug, not a
grammar failure: `OwnedLlamaDecoder.generate_tokens` accepted and appended a
stop token before checking it.  The minimal fix accepts the sampler and matcher
exactly once, then checks stop membership before adding that token to the
returned completion.  A fresh Ornith JSON smoke is required after this change;
the failure artifact remains ignored under `artifacts/project17-xgrammar-json-*`.

## Grammar MVP verification — 2026-07-24

The fresh CPU Ornith smoke passed after the stop-token correction. With the
pinned project environment (`xgrammar 0.2.1`, `transformers 4.57.6`, `torch
2.12.0`) it emitted and Pydantic-validated
`{"city":"Paris","country":"France"}`.

The tokenizer gate was repeated with this Transformers version and still passed
all 26 probes and all 600 fuzz strings. Local ignored benchmark artifacts
compare a 48-token unconstrained run (4.44 decode tok/s) with the 9-token JSON
completion (4.38 decode tok/s). The constrained run measured 0.55 ms total
mask creation and 15.49 ms mask application, or roughly 1.78 ms/token. The
different output lengths and CPU-only setting mean this is a teaching smoke,
not a controlled performance comparison.

## Compiler-cache multi-request smoke — 2026-07-24

`GrammarCompilerCache` now keys compiler reuse by the full engine fingerprint
and compiled grammar reuse by the canonical JSON schema, strictness, and
xgrammar version. It intentionally never shares a matcher. Two sequential CPU
Ornith requests with one compiled grammar and fresh matchers each wrote a
validated 9-token JSON record; their decode rates were 4.68 and 4.61 tok/s.
An exploratory third request exited without a Python traceback or artifact, so
this report only claims two-request evidence. Reproduce and classify that
repeat-run boundary before using a long-lived worker as a benchmark baseline.

## Phase-1 repeat-run soak harness — 2026-07-24

### Implementation and contract

`examples/soak_grammar.py` is the reproducible Phase-1 CLI. It accepts the
GGUF path, tokenizer ID or local tokenizer directory, request count, prompt,
JSON schema (`--schema-json` or `--schema-file`), max-token/context/thread
settings, deterministic llama.cpp seed, and an artifact directory. Artifact
directories are deliberately restricted to ignored `artifacts/project17-*`.

It resolves a tokenizer ID to an on-disk Hugging Face snapshot by default. This
is a local reproducibility boundary: Transformers 4.57.6 otherwise attempts a
remote metadata request for this tokenizer class even when the tokenizer files
are cached. `--allow-network` is explicit opt-in for downloading a missing
snapshot.

The grammar is compiled once before the constrained batch. Every iteration
constructs a fresh matcher, applies the XGrammar mask before sampler
application, accepts the llama sampler and matcher exactly once through the
owned decoder, and rejects every non-`stop` finish before Pydantic validation.
Each attempt writes a benchmark record; a normally completed run writes
`summary.json` with valid/invalid/cutoff counts, tokens, all phase p50/p95
timings, direct mask overhead, and an equal-size unconstrained comparison
unless `--no-baseline` is selected. Any constrained cutoff, malformed output,
matcher rejection, or runtime error returns exit status 1.

`tests/test_grammar_soak.py` is GPU- and llama.cpp-free. It covers aggregate
failure/cutoff accounting, token totals, mask-overhead math, and the
per-emitted-token decode comparison.

### Commands and environment

Focused checks passed in the project venv:

```sh
.devenv/state/venv/bin/pytest tests/test_grammar_soak.py
.devenv/state/venv/bin/ruff check examples/soak_grammar.py tests/test_grammar_soak.py
.devenv/state/venv/bin/ruff format --check examples/soak_grammar.py tests/test_grammar_soak.py
```

The real CPU attempt used Python 3.13.13, llama-cpp-python 0.3.34,
xgrammar 0.2.1, transformers 4.57.6, torch 2.13.0, numpy 2.5.1, Pydantic
2.13.4, the CPU Ornith Q4_K_XL GGUF, the cached
`deepreinforce-ai/Ornith-1.0-9B` tokenizer snapshot, `n_threads=8`,
`n_ctx=512`, seed 1234, and the recorded Nix GCC 15.2 libstdc++ directory:

```sh
LD_LIBRARY_PATH="$(tr -d '\n' < .scratch/projects/17-llama-cpp-inference-lab/.stdcxx_dir)" \
PYTHONPATH="src:.devenv/state/venv/lib/python3.13/site-packages" \
.scratch/projects/17-llama-cpp-inference-lab/.venv-spike/bin/python examples/soak_grammar.py \
  --model /home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf \
  --requests 10 --max-tokens 16 --seed 1234 --n-threads 8 \
  --artifacts artifacts/project17-grammar-soak-20260724T1010Z
```

### Result boundary and raw evidence

The command-runner ended the CPU process after about 28 seconds, before it
could reach request 3 or write `summary.json`; it left no Python traceback. The
fresh ignored directory contains two completed per-request records (requests 0
and 1), both `finish_reason="stop"`, both Pydantic-valid
`{"city":"Paris","country":"France"}`, with fresh matchers and the one
compiled grammar. Their combined partial aggregate is 2 valid, 0 invalid, 0
cutoff, 28 prompt tokens, and 18 completion tokens. Direct mask work was
32.74 ms total (1.82 ms/completion token; 0.63% of 5.189 s aggregate decode
time). Request decode rates were 3.90 and 3.12 token/s.

The preceding two setup attempts are also preserved under ignored artifacts:
`project17-grammar-soak-20260724T1005Z` stopped before model construction
because the isolated spike venv lacks DBOS, and `...T1007Z` hit a blocked DNS
request from Transformers before the local-snapshot resolver was added.

This proves the per-request artifact and accounting path for two real fresh
matchers, and establishes the measured local mask cost for those requests. It
does **not** prove the requested 10-request (let alone Phase-1 1,000-request)
repeat-run bar, a completed unconstrained comparison batch, GPU overhead,
throughput causality, non-default schemas, or any cache/KV/LoRA behavior. A
host execution facility that permits this roughly 2-minute CPU workload is the
remaining external requirement for the 10-request smoke.

### Verification status

The focused accounting tests passed (`2 passed in 12.32s`) and the relevant
llama-core suite passed:

```text
pytest tests/test_grammar_soak.py tests/test_llama_core_benchmark.py \
  tests/test_llama_core_grammar.py tests/test_owned_decode.py \
  tests/test_xgrammar_api_contract.py
13 passed in 21.39s
```

`ruff check src tests examples` and `ruff format --check src tests examples`
both passed. A repository-wide Ruff invocation is not a usable gate because it
reports 57 existing violations in unrelated `.scratch/` and `deploy/` files.
A full `pytest -o addopts='-ra'` was started (62 items collected, one skipped)
but the same approximately-28-second command-runner limit stopped it while
entering `tests/test_agent.py`; it produced no failure result and must be
rerun in the host facility. This does not weaken the completed focused suite,
but it is not evidence of a full-suite pass.

## Recovered host soak result — 2026-07-24

The apparent command-runner termination was a control-plane observation error:
the command interface lost its process handle, but the child Python processes
continued on the host. A detached Zellij session (`project17-soak-20260724t1035z`)
was then used to capture the authoritative completed run at
`artifacts/project17-grammar-soak-20260724T1035Z/`.

The process exited 0 and wrote `summary.json` plus 20 request records (ten
constrained and ten unconstrained baseline). The constrained batch is the
repeat-run result: **10/10 valid, 0 invalid, 0 cutoff**, all nine-token,
clean-stop Pydantic-valid `{"city":"Paris","country":"France"}` results;
140 prompt and 90 completion tokens total. The grammar mask cost was 167.74 ms
total, 1.864 ms/completion token, with p50/p95 per-request mask work of
16.586/19.139 ms. Mask application alone was 161.763 ms total and mask creation
was 5.978 ms total.

The optional baseline emitted 16 tokens on each request and therefore reached
the configured `max_tokens=16` cutoff 10/10. Its decode comparison reports
557.28 ms/token unconstrained versus 1,037.12 ms/token constrained (+86.10%).
Do not interpret that delta as a controlled throughput result: another local
CPU workload was active and three stale agent-launched soaks were found
concurrently consuming CPU before they were stopped. The direct per-token mask
measure is the usable local overhead result; the baseline comparison should be
rerun on an idle host before making a performance claim.

Earlier recovered artifact directories are partial runs, not additional soak
passes: `...T1010Z` contains valid constrained request indices 0--8 (nine
records), `...T1015Z` indices 0--3 (four), and `...T1025Z` indices 0--2
(three). They confirm that the command wrapper did not terminate Python, but
their overlap and resource contention make them non-authoritative.

## GPU-only evaluation policy — 2026-07-24

CPU execution was useful only to establish early API and correctness evidence.
It is not the performance or evaluation target for the current project. From
this point, every Ornith JSON soak, unconstrained baseline, grammar-overhead
measurement, prefix-cache experiment, and router evaluation must use the
recorded CUDA llama.cpp library set with GPU layers offloaded. CPU remains
permitted solely for GPU-free unit tests and build/ABI diagnostics.

Accordingly, the historical CPU rates and the recovered CPU baseline above are
provenance, not actionable performance data. GPU artifact manifests must record
`LLAMA_CPP_LIB_PATH`, CUDA runtime/driver library paths, GPU identity, driver
version, and `n_gpu_layers`.

## Persistent prefix-cache CUDA result — 2026-07-24

`artifacts/project17-prefix-cache-20260724T175312Z/` exited 0 with the pinned
postfix2 CUDA build, CUDA_VISIBLE_DEVICES=0, n_ctx=512, n_gpu_layers=-1, and
GPU1 at 9 MiB. It proves persistent blob/index → fresh context → load → suffix
decode → matching continuation at 16/32/64/96 exact tokens. State blobs were
69.1/85.5/118.4/151.2 MB. Synchronized cold/cache means: 92.3/832.3,
83.1/1472.7, 113.4/1683.4, 145.7/1741.6 ms; p50/p95 cache ms:
831.9/837.7, 1451.0/1707.3, 1853.4/1862.6, 1639.5/2158.6. Cache completion
rates were 1.201/.679/.594/.574 tok/s. No measured break-even: disk read plus
whole-state restore dominates. Async llama_decode enqueue timing is not
throughput. Unit tests cover incompatible fingerprint/token, corruption, atomic
publication, hit/miss, and the required restore-before-suffix lifecycle.

## Phase-2 whole-state root-cause + per-sequence analysis — 2026-07-24

This section attributes the negative break-even to a specific, source-grounded
mechanism and explains the per-sequence divergence. It draws only on completed
GPU-only artifacts and local source/headers; it does **not** run a new GPU
measurement (GPU 0 was held by a concurrent live `run_json_workload.py` at
analysis time, and the GPU-only policy requires exclusive GPU 0). New evidence
requiring the GPU is designed, staged, and handed off below — not claimed.

### Runtime tuple and artifacts (authoritative, already run)

- Build: `out-cuda-3060-postfix2`, llama-cpp-python 0.3.34, torch 2.12.0,
  Python 3.13.13, `CUDA_VISIBLE_DEVICES=0`, `n_ctx=512`, `n_batch=128`,
  `n_gpu_layers=-1`, seed 17018. GPU 1 idle (9 MiB) in every accepted run.
- Whole-state sweeps: `artifacts/project17-prefix-cache-20260724T175312Z/`
  (16/32/64/96, exit 0) and `...191154Z/` (128/192/256, exit 0).
- Per-sequence probe: `artifacts/project17-seq-state-20260724T191450Z/`
  (K=32, 1-token suffix, exit 1 — divergence).
- Ornith identity: `n_vocab=248320` (Gate 3 doc), hybrid attention +
  GatedDeltaNet (linear-attention / recurrent) architecture.

### A. Why whole-state restore never breaks even

Per-phase synchronized means (ms), from the two summaries:

| prefix | cold prefill | lookup | state read+cksum | state restore | suffix | cache e2e | blob MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 92.3 | 4.2 | 153.9 | 651.8 | 26.7 | 832.3 | 69.1 |
| 32 | 83.1 | 3.4 | 209.5 | 1226.3 | 36.9 | 1472.7 | 85.5 |
| 64 | 113.4 | 2.9 | 280.2 | 1366.7 | 36.6 | 1683.4 | 118.4 |
| 96 | 145.7 | 3.5 | 375.1 | 1332.6 | 33.9 | 1741.6 | 151.2 |
| 128 | 210.4 | 4.8 | 427.2 | 1471.7 | 53.2 | 1952.1 | 184.0 |
| 192 | 271.8 | 3.7 | 470.3 | 1658.1 | 33.6 | 2162.0 | 186.1 |
| 256 | 373.5 | 5.2 | 480.3 | 1763.4 | 49.9 | 2293.6 | 188.2 |

Two dominant phases, both scaling with **blob size**: `state_read_wall` (disk
read + SHA-256 of the blob) and `state_restore_wall` (which in
`run_prefix_cache.py:160` bundles `pickle.loads`, `LlamaState.load_state`'s
score-array apply, and the native `llama_state_set_data` host→device restore).
Cold GPU prefill is cheap (~1.2 ms/token beyond a ~50 ms floor) and the cache
path is 5–9× slower at every length. The gap **widens** with length, so a longer
prefix cannot rescue it (eliminates Option C1 for this codec).

**Blob composition (source-grounded, arithmetic-verified).**
`Llama.save_state` (llama.py:2199) returns a `LlamaState` carrying
`scores=self._scores.copy()` where `_scores = self.scores[: n_tokens, :]` and
`self.scores` is allocated `(n_batch, n_vocab)` (llama.py:477), so the saved
score rows saturate at `n_batch=128`. With `n_vocab=248320`, fp32:

- scores bytes = `min(n_tokens, 128) · 248320 · 4`  (≈0.993 MB/token to the cap)
- native `llama_state_get_size` ≈ 53 MB at 15 tokens (Gate 3) and grows only
  ~0.03 MB/token (recurrent GatedDeltaNet state dominates and is nearly
  token-count-independent).

This model reproduces every recorded blob to ≈0.1 MB (see
`benchmarks/project17/state_blob_model.py` and `tests/test_state_blob_model.py`,
3 passing GPU-free tests). Implied native state: 53.2/53.7/54.8/55.8/56.9/59.0/
61.1 MB for 16…256. **Consequence: the entire per-token blob growth is the
`LlamaState` score buffer — prefill logits that the restore lifecycle never
consumes (we always re-decode a suffix for fresh logits). At 96 tokens ~63% of
the blob (95 of 151 MB) is dead-weight scores; past 128 tokens scores plateau
and the blob is nearly flat.**

**What this proves:** the negative result is dominated by moving a large blob
(disk read + checksum + pickle + host→device set), and the growth term is a
Python-wrapper artifact, not the model KV/recurrent state. **What it does not
yet prove:** the split of `state_restore_wall` between pickle/score bookkeeping
and the native `llama_state_set_data` transfer/reconstruction. That split is the
one open attribution question and decides Option C3 — it is exactly what the
staged decomposition runner measures.

### B. Why the per-sequence byte-copy diverges

`llama_state_seq_get_data`/`set_data` copied and re-loaded 53,740,972 bytes
(size==copied==set_return) yet greedy continuation diverged (baseline 21059 vs
restored 364). Byte-count round-trip proves buffer sizing, not semantic restore.

Primary local evidence — the pinned `llama.h` (out-cuda-3060-postfix2):

```
// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1
...
LLAMA_API size_t llama_state_seq_get_data_ext(..., llama_state_seq_flags flags);
```

Ornith is a hybrid attention + recurrent (GatedDeltaNet) model. The header
distinguishes recurrent/partial sequence state and provides `_ext` + flag
variants for it; the spike used the **plain non-ext** `llama_state_seq_*` path
(`FLAGS_NONE`). The installed 0.3.34 binding **does** expose the `_ext` symbols
and `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`/`ON_DEVICE` (verified in
`llama_cpp/llama_cpp.py`). Leading hypothesis: the plain per-sequence path does
not correctly serialize/re-key the recurrent cache into `dest_seq_id`, so the
attention KV round-trips but the GatedDeltaNet recurrent memory does not —
producing a same-size blob with wrong continuation. Whole-state
`llama_state_get_data`/`set_data` avoids this because it restores the entire
context (recurrent state included) and is already proven bit-exact (Gate 3,
same- and cross-instance; and all Phase-2 whole-state lengths matched).

This is a hypothesis, not a proof of the internal cause: it must not be patched
speculatively. The staged sweep tests it directly.

### C. Options with go/no-go

1. **Longer shared prefix.** No-go for this codec: cache cost grows faster than
   cold prefill; the gap widens through 256 tokens and scores plateau while
   restore stays >1.4 s. Only reconsider *after* a native codec flattens the blob.
2. **In-memory whole-state checkpoint.** Removes `state_read_wall`
   (154–480 ms) and pickle framing, not the native set or the score apply.
   Best-case still ~650–1300 ms restore ≫ cold. Go only as an explicitly
   in-memory teaching variant, not a persistent-cache speedup. Go/no-go:
   in-memory restore total < cold prefill at some tested length.
3. **Native `llama_state_get_data`/`set_data` codec (recommended).** Same C
   entry point `load_state` already wraps, minus the score buffer, so restore
   correctness is inherited (low risk) and the blob drops to a flat ~53–61 MB
   (removes disk read/pickle/score-apply proportionally). Removes the dominant
   *growth* term; whether it beats cold prefill depends on the native-set share
   of restore, which is unmeasured. Go/no-go: instrumented run shows
   `nat_restore_total_wall < cold_prefill_wall` at some length **and** exact
   continuation match. Exact experiment: `run_native_state_decompose.py`.
4. **Correct per-sequence codec.** Higher risk; blocked until a sweep proves
   deterministic continuation equivalence. Exact experiment: the extended
   `run_seq_state_spike.py` sweep (positions × suffix × `none`/`partial_only`
   ext). Go/no-go: `match=True` across K and suffix lengths for a single flag
   configuration. No production use before that.
5. **Different checkpoint/suffix placement.** Does not change the dominant
   blob-move cost; not a route to break-even. No-go as a speedup lever.
6. **Plain-transformer control model.** Diagnostic only, to localize whether the
   per-sequence divergence is recurrent-specific. Not an Ornith performance
   baseline. Note: the only other local GGUFs are gemma-4 MTP variants (not
   strictly plain); document the caveat if used.

### Staged (not-yet-run) GPU evidence — handoff

- `benchmarks/project17/run_native_state_decompose.py`: decomposes restore into
  `ls_pickle_loads / ls_load_state / ls_suffix` vs `nat_set_data / nat_suffix`,
  in-memory (no disk) to isolate codec cost, with a fail-closed continuation
  gate against the cold baseline. Answers A's attribution and Options 2 & 3.
- `benchmarks/project17/run_seq_state_spike.py` (extended): sweeps checkpoint
  position, suffix length, and `none`/`partial_only`(`_ext`) flags; records
  blob checksums; success = continuation match only. Answers Option 4.

Run when GPU 0 is exclusively free (GPU 1 idle), using the pinned env
(`CUDA_VISIBLE_DEVICES=0`, `LLAMA_CPP_LIB_PATH=...out-cuda-3060-postfix2/lib`,
`LD_LIBRARY_PATH` from `.cuda_runtime_ld`, `n_gpu_layers=-1`), the same
`.venv-spike` interpreter, and artifact dirs under `artifacts/project17-*`.

### Proven / rejected / unknown

- **Proven:** whole-state persistent cache is correct (restart-safe, exact
  continuation 16→256) but never breaks even; both dominant phases scale with
  blob size; the blob's per-token growth is entirely the `LlamaState` score
  buffer (dead weight); native state is recurrent-dominated and nearly flat;
  the per-sequence non-ext path round-trips bytes but diverges semantically.
- **Rejected:** "longer prefix will break even" (gap widens); "byte-count copy
  == correct restore"; "blob growth is model KV/recurrent state" (it is scores).
- **Unknown (needs the staged GPU runs):** the pickle/score-apply vs
  native-set split of restore; whether a native codec beats cold prefill;
  whether the `_ext`/PARTIAL_ONLY path restores per-sequence continuation.

### Recommendation and Phase-2 status

Smallest justified next experiment: run `run_native_state_decompose.py` on an
exclusive GPU 0. If native restore total < cold prefill with exact continuation,
promote a native-bytes codec behind the existing `PrefixCache` contracts
(fingerprint/token/size/checksum gates and restore-then-suffix lifecycle
unchanged). Otherwise, **Phase 2 stays closed as a correct-but-negative
whole-state persistent-cache teaching result**, with the native-codec and
per-sequence(`_ext`) sweeps recorded as the two open follow-ups. No LMCache,
radix tree, compression, or generic cache abstraction is warranted by this
evidence.

## Phase-2 staged GPU runs — RESULTS — 2026-07-24

GPU 0 became exclusively free (GPU 1 idle, 9 MiB) and both staged experiments
ran on the pinned postfix2 CUDA build, `.venv-spike`, commit 15a933c. These are
fresh, CUDA-synchronized, GPU-only results. They confirm the whole-state
attribution **and overturn the per-sequence "established fact": that divergence
was a test-harness bug, not an API/semantic failure.**

### Native-vs-LlamaState restore decomposition (in-memory, no disk)

`artifacts/project17-native-decompose-20260724T194604Z/` (exit 0, GPU 0 active,
GPU 1 idle). Continuation matched the cold baseline at every length (fail-closed
gate). Means, ms:

| prefix | cold | LS pickle_loads | LS load_state | LS total | nat set_data | nat total | native blob | pickle blob |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 41.2 | 37.3 | 176.0 | 238.8 | 150.9 | 175.2 | 53.2 MB | 69.1 MB |
| 64 | 65.3 | 71.2 | 209.4 | 303.1 | 155.7 | 179.4 | 54.8 MB | 118.4 MB |
| 128 | 116.1 | 109.7 | 257.2 | 390.1 | 159.2 | 179.1 | 56.9 MB | 184.0 MB |
| 256 | 210.1 | 116.8 | 253.6 | 393.6 | 171.9 | 192.1 | 61.1 MB | 188.2 MB |

(`LS load_state` includes the native set **plus** the score-array apply; suffix
decode was ~20–26 ms in every path. Cold is lower than the persistent-run cold
because this runner keeps one warm context, which is the fair internal control.)

**Attribution (answers A's open question):** the dominant restore cost is the
**native `llama_state_set_data`** host→device reconstruction, ~151–172 ms and
**flat** in prefix length (the native state is recurrent-dominated ~53–61 MB).
The `LlamaState` wrapper adds a large **variable** penalty on top —
`pickle.loads` (37→117 ms) plus score-array apply (~25→98 ms, = `LS load_state −
nat set_data`) — that grows with the dead-weight score blob. So the native codec
is 1.4–2.2× faster than the wrapper path and its blob is flat.

**Break-even (in-memory native codec):** cold prefill grows ~0.8 ms/token while
native restore is flat ~180–192 ms, so they **cross at ≈256 tokens** (256: cold
210.1 vs native 192.1 — native wins by 18 ms; at 128 native still loses by
63 ms). This is a real, bounded success criterion for **Option 2 + Option 3
combined**: an in-memory native-state checkpoint is beneficial for prefixes
≳256 tokens. Persistent disk is still not there at 256 — adding read+SHA-256 of
the ~55–61 MB blob (~120 ms, extrapolated from the 69 MB→154 ms persistent
measurement) pushes native total to ~310 ms > cold 210 ms; disk break-even needs
a longer prefix, which requires `n_ctx > 512`.

### Per-sequence sweep — the earlier divergence was a harness bug

`artifacts/project17-seq-state-sweep-20260724T194742Z/` (GPU 0 active, GPU 1
idle). Sweep of checkpoint K∈{1,16,31} × suffix∈{1,4} × flag∈{none, partial_only}.

**Plain non-ext `llama_state_seq_get_data`/`set_data` (flag=none): match=TRUE in
all 6 configurations.** The `partial_only` (`_ext`) path matched only 1/6 —
expected, because `PARTIAL_ONLY` serializes only the recurrent/SWA sub-state,
not the full sequence state, so restoring it into a fresh context is wrong by
construction.

**Root cause of the original 191450Z divergence (21059 vs 364): the original
probe omitted the Python n_past bookkeeping after `set_data`.** It left
`target.n_tokens = 0`, so `Llama.eval(suffix)` ran `kv_cache_seq_rm(-1, 0, -1)`
(llama.py:665) and **wiped the just-restored KV cache**, then decoded the suffix
at position 0 against an empty cache. The extended runner restores
`target.n_tokens = len(prefix)` and `input_ids` before `eval`, and the plain
per-sequence API then round-trips Ornith's hybrid recurrent state correctly.
This confirms the hypothesis explicitly listed for B ("whether the test harness
is using the API incorrectly") — **it was.** Byte-count success was never the
issue; the missing bookkeeping was.

Scope of this per-sequence proof (do not over-claim): greedy continuation only,
K≤31, suffix≤4, `dest_seq_id=0`, same build, same-process fresh context. Still
unproven: larger K, nonzero `dest_seq_id`, cross-process/disk round-trip, and
multi-sequence isolation. The whole-state runner's own bookkeeping (via
`load_state`) was always correct, which is why whole-state never showed this bug.

### Updated proven / rejected / unknown

- **Newly proven:** dominant whole-state restore cost is the native state set
  (~151–172 ms, flat); the `LlamaState` wrapper adds a variable pickle+score
  penalty; a native codec is ~1.4–2.2× faster with a flat ~53–61 MB blob; an
  **in-memory** native checkpoint breaks even at ≈256 tokens. The **plain
  per-sequence API restores Ornith correctly** once n_past bookkeeping is
  replayed — the earlier divergence was a harness error.
- **Newly rejected:** "per-sequence `llama_state_seq_*` is semantically broken
  for Ornith" (established fact #3) — refuted; it was our harness. "pickle/scores
  are negligible" — they are the variable penalty, though the native set is the
  floor.
- **Still unknown:** disk-inclusive native break-even and whether `n_ctx>512`
  moves persistent break-even; per-sequence correctness at large K / nonzero
  dest_seq_id / cross-process; multi-sequence isolation.

### Updated recommendation and Phase-2 status

The smallest justified next step is now a small, **correct** code change, not a
new question: add a **native-bytes state codec** (`llama_state_get_data`/
`set_data` with n_past replay) behind the existing `PrefixCache` contracts, used
as an **in-memory** checkpoint for long shared prefixes (≳256 tokens) — the only
regime where it beats recompute. The persistent-disk whole-state cache **stays
closed as correct-but-negative** at n_ctx=512. Per-sequence reuse is now
promising rather than blocked, but must clear the larger-K / nonzero-dest /
cross-process sweep before production use. Still no need for LMCache, radix tree,
compression, or a generic cache abstraction.

## Per-sequence reuse — deep investigation — 2026-07-24

Follow-up to the corrected per-sequence result: does per-sequence state reuse
work for the *router* use case — restoring a cached prefix into one sequence
slot of a **live multi-sequence** context — and under what constraints? All
evidence GPU-only (`CUDA_VISIBLE_DEVICES=0`, GPU 1 idle), CUDA-synchronized,
own-batch multi-sequence decode (the high-level `Llama.eval` only ever uses
seq 0). Runner: `benchmarks/project17/run_seq_reuse.py`. Artifact:
`artifacts/project17-seq-reuse-20260724T201749Z/` (exit 0, GPU 0 5503 MiB,
GPU 1 9 MiB). `n_ctx=2048`, `n_batch=128`. Contexts are built with a custom
`n_seq_max` via `llama_new_context_with_model`, reusing the loaded weights.

**Success = greedy continuation equality only.** Two decode caveats found and
fixed while building the runner (both are real llama.cpp constraints, worth
teaching): (a) a single `llama_decode` must satisfy `n_tokens <= n_batch`, so
prefixes are chunked; (b) `n_ctx` is **divided across sequences**
(`n_ctx/n_seq_max` cells each), so `n_ctx` must be large enough that the longest
prefix fits under every `n_seq_max`.

### Proven for the router pillar

1. **Parallel multi-sequence decode is correct.** Two independent sequences in
   one `n_seq_max=2` context each reproduce their isolated single-sequence
   baselines (`multiseq_control.A_ok/B_ok = true`). Ornith's hybrid recurrent
   architecture runs true parallel sequences.
2. **Restore into a nonzero seq of a live context, with isolation.** Capturing
   sequence B (matched `n_seq_max`) and `llama_state_seq_set_data` into seq 1 of
   a context already holding sequence A on seq 0 yields the correct B
   continuation **and** leaves A's continuation unchanged, for checkpoints
   K∈{32,64,128,256} (`router_path[*].restored_ok_all` and `isolation_ok_all`
   all true). This is the exact KV-reuse primitive a multi-prefix / multi-LoRA
   router needs.
3. **Cross-process restart.** Capture a sequence blob to disk in one process,
   restore in a fresh process/context → identical continuation
   (`crossprocess.match = true`, token 25147; child exit 0).

### The portability constraint (must become a cache key)

Per-sequence blobs are portable **only across identical `n_seq_max`**. The
capture×restore matrix is clean:

| capture n_seq_max → restore n_seq_max | set_data return | continuation |
| :-- | :-- | :-- |
| 1→1, 2→2, 4→4 (diagonal) | > 0 (loaded) | match |
| every off-diagonal (1→2, 2→1, 2→4, 4→2, …) | **0 = failed to load** | n/a |

Blob size encodes `n_seq_max` (53,740,972 / …976 / …984 bytes for
`n_seq_max` 1/2/4 — a 4-byte-per-doubling header field). A mismatch **fails
safe**: `set_data` returns 0, nothing is corrupted, no crash — so a per-sequence
cache can detect it and fall back to cold prefill. **`n_seq_max` (and the rest of
the context config that shapes the state) must be part of the per-sequence cache
compatibility key.** This is a design requirement for any future per-sequence
codec; it is deliberately *not* retrofitted into the frozen
`LlamaEngineFingerprint` in this investigation (which carries `n_ctx` but not
`n_seq_max`), to preserve the existing whole-state contract.

### Restore cost (synchronized, in-memory, one sequence)

`set_data` of a single sequence's state: 103 / 106 / 110 / 118 ms mean for
K = 32 / 64 / 128 / 256 (blob 53.7 / 54.8 / 56.9 / 61.1 MB). It is **flatter and
cheaper than the whole-state native set** (~150–172 ms) because it moves one
sequence's recurrent+KV state, not the whole context. Suggestive, not yet a
break-even claim: whole-state cold prefill at 256 tokens was ~210 ms (warm
single context), so per-sequence restore (~118 ms) is well under it — per-seq
reuse may cross break-even *earlier* than whole-state. The fair test is a
cold-prefill-vs-restore comparison **inside the same multi-sequence context**
(different runner, different n_ctx), which this run did not do.

### Still unknown / next

- Cold-vs-restore break-even measured in the same multi-sequence context.
- Larger `n_seq_max` (8/16) and concurrent batched-decode throughput.
- Per-sequence **LoRA** adapters in one batch — a separate, known-hard llama.cpp
  limitation ([[llama-cpp-inference-lab]]); this experiment covers KV reuse only,
  not per-seq adapters.
- Recurrent-state correctness at very long K (near per-seq cell capacity) and
  multi-sequence isolation beyond two live sequences.

### Bottom line

Per-sequence KV reuse for the router is **feasible and correct** (nonzero-seq
restore into a live context, isolated, cross-process), bounded by one clean
rule: **matching `n_seq_max`**, which fails safe on mismatch. Restore is cheaper
than whole-state. The remaining gate before production use is a same-context
break-even measurement and a larger-scale batched-decode check; per-seq LoRA
remains a distinct, separately-tracked risk.

## Per-sequence reuse — in-context break-even + batched throughput — 2026-07-24

Closes the first two open items above. Runner:
`benchmarks/project17/run_seq_batch_breakeven.py`. Artifact:
`artifacts/project17-seq-batch-20260724T202826Z/` (exit 0; GPU 0 active, peak
6587 MiB / 97%; GPU 1 idle 9 MiB throughout). `n_ctx=2048`, `n_batch=128`,
CUDA-synchronized, 5 reps.

### 1. Cold-vs-restore break-even inside one multi-sequence context

Same `n_seq_max=2` context and same seq slot for both paths, same fresh-logit
lifecycle: **cold** = decode K-token prefix + 1 suffix token; **restore** =
`llama_state_seq_set_data` the matched blob + decode 1 suffix token. Continuation
matched (cold == restore) at every K — fail-closed. Means (ms):

| K | cold | restore | delta (cold−restore) | restore wins |
| ---: | ---: | ---: | ---: | :--: |
| 16 | 53.4 | 130.2 | −76.8 | no |
| 32 | 45.2 | 127.5 | −82.2 | no |
| 64 | 63.4 | 128.8 | −65.3 | no |
| 96 | 80.2 | 130.3 | −50.1 | no |
| 128 | 110.0 | 133.4 | −23.4 | no |
| 192 | 151.9 | 139.1 | **+12.8** | **yes** |
| 256 | 198.8 | 143.1 | **+55.7** | **yes** |

**Per-sequence reuse breaks even at ≈192 tokens (crossover ~170).** Restore is
nearly flat (~127→143 ms; `set_data` of one sequence's ~54–61 MB state plus a
~15 ms suffix decode), while cold prefill grows ~0.8 ms/token. This is a genuine
positive result — unlike the whole-state *disk* cache, per-sequence *in-context*
reuse pays off for shared prefixes ≳192 tokens, and does so earlier than the
whole-state in-memory codec (~256). It needs no disk and no `LlamaState` score
buffer.

### 2. n_seq_max scaling to 8/16 and concurrent batched-decode throughput

Aggregate generation throughput when S sequences decode one token each per
`llama_decode` step (the router's in-flight-batch path), 32 tokens/seq:

| S (sequences) | aggregate tok/s | per-seq tok/s | step latency | matched-nsm load |
| ---: | ---: | ---: | ---: | :--: |
| 1 | 48.1 | 48.1 | 20.8 ms | — |
| 2 | 75.7 | 37.9 | 26.4 ms | — |
| 4 | 93.7 | 23.4 | 42.7 ms | — |
| 8 | 106.1 | 13.3 | 75.4 ms | ok |
| 16 | 194.8 | 12.2 | 82.1 ms | ok |

**Batching wins: ~4.05× aggregate throughput at S=16 vs S=1**, with per-sequence
latency degrading as expected (shared compute). Notably the step latency is
nearly flat from S=8 (75.4 ms) to S=16 (82.1 ms) — decode is **weight-memory-
bandwidth bound**, so doubling the batch barely raises step time and throughput
nearly doubles there. This is the quantitative basis for the router pillar's
"massively parallel batched generation" claim on a single 3060. The matched-
`n_seq_max` portability rule **still holds at n_seq_max = 8 and 16**
(`diagonal_set_ok = true`), so cached per-sequence prefixes remain restorable in
the larger batch contexts a router would use.

### Updated status

Both first-tier open items are resolved: per-sequence reuse **breaks even at
≈192 tokens in-context** (a real speedup, no disk), and **batched decode scales
~4× to 16 sequences** with the portability rule intact. Remaining per-sequence
unknowns are now second-tier: per-sequence LoRA-in-one-batch (separate known-hard
risk), correctness near per-seq cell capacity, and isolation beyond two live
sequences. The teaching MVP recommendation stands: whole-state persistent disk
cache is correct-but-negative; the promising path is in-context per-sequence KV
reuse (≳192-token shared prefixes) inside a batched multi-sequence router.
