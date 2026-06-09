{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # --- Inference backend (OpenAI-compatible) -------------------------------
  # The remote box currently runs llama.cpp serving a Qwen3.5-9B GGUF quant.
  # The library targets a vLLM (XGrammar + per-agent LoRA) backend that will
  # eventually replace this server with a drop-in Docker container exposing the
  # same OpenAI-compatible base URL.
  env.LLM_BASE_URL = "http://remora-server:8000/v1";
  env.LLM_API_KEY = "sk-no-key-required";   # llama.cpp ignores; vLLM will enforce
  env.LLM_MODEL = "Qwen3.5-9B-UD-Q6_K_XL.gguf";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.uv
    pkgs.curl       # probing the inference server
    pkgs.jq         # inspecting JSON responses
    ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages = {
      python = {
          enable = true;
          version = "3.13";
          venv.enable = true;
          uv.enable = true;
        };
    };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # Sync the project venv (pydantic-ai etc.) from pyproject/uv.lock.
  scripts.spike-sync.exec = ''
    uv sync
  '';

  # Probe whatever OpenAI-compatible server $LLM_BASE_URL points at.
  scripts.llm-probe.exec = ''
    set -e
    echo "Probing $LLM_BASE_URL ..."
    curl -sS "$LLM_BASE_URL/models" | jq '.data[] | {id, owned_by: (.owned_by // "?")}'
  '';

  # Run the request-path spike against $LLM_BASE_URL.
  scripts.spike-run.exec = ''
    uv run python .scratch/projects/01-xgrammar-concept/spike/run_spike.py "$@"
  '';

  enterShell = ''
    hello
    git --version
    echo "LLM backend: $LLM_BASE_URL  (model: $LLM_MODEL)"
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
