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
