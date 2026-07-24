{ pkgs, ... }:

{
  packages = [
    pkgs.git
    pkgs.uv
    pkgs.zellij
  ];

  scripts.project17-pytest-zellij.exec = ''
    set -euo pipefail
    session="project17-pytest-$(date -u +%Y%m%dT%H%M%SZ)"
    artifact_dir="$PWD/artifacts/$session"
    mkdir -p "$artifact_dir"
    zellij attach "$session" --create-background
    ZELLIJ_SESSION_NAME="$session" zellij run --name pytest --close-on-exit --cwd "$PWD" -- \
      ${pkgs.bash}/bin/bash -lc "pytest -o addopts='-ra' > '$artifact_dir/stdout-stderr.log' 2>&1; status=\$?; printf '%s\\n' \"\$status\" > '$artifact_dir/exit-status.txt'; exit \$status"
    echo "pytest is running in: zellij attach $session"
    echo "artifacts: $artifact_dir"
  '';

  scripts.project17-json-workload-zellij.exec = ''
    set -euo pipefail
    : "''${PROJECT17_MODEL:?set PROJECT17_MODEL to the Ornith GGUF path}"
    project17_corpus="''${PROJECT17_CORPUS:-benchmarks/project17/json_workload_100.jsonl}"
    project17_requests="''${PROJECT17_REQUESTS:-100}"
    project17_mode="''${PROJECT17_MODE:-constrained}"
    project17_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    project17_session="project17-json-$project17_stamp"
    project17_artifacts="$PWD/artifacts/project17-json-$project17_mode-$project17_stamp"
    project17_lib="$PWD/.scratch/projects/17-llama-cpp-inference-lab/.llamacpp-builds/out-cuda-3060-postfix2/lib"
    project17_cuda_ld="$(tr -d '\n' < "$PWD/.scratch/projects/17-llama-cpp-inference-lab/.cuda_runtime_ld")"
    mkdir -p "$project17_artifacts"
    {
      printf 'session=%s\n' "$project17_session"
      printf 'started_at_utc=%s\n' "$project17_stamp"
      printf 'mode=%s\ncorpus=%s\nrequests=%s\nmodel=%s\n' "$project17_mode" "$project17_corpus" "$project17_requests" "$PROJECT17_MODEL"
      printf 'llama_cpp_lib_path=%s\n' "$project17_lib"
      printf 'cuda_visible_devices=0\nn_gpu_layers=-1\n'
    } > "$project17_artifacts/run-config.txt"
    git status --short > "$project17_artifacts/git-status-before.txt"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv,noheader > "$project17_artifacts/gpu-before.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$project17_artifacts/gpu-processes-before.csv" || true
    ps -eo pid,ppid,stat,args > "$project17_artifacts/processes-before.txt"
    zellij attach "$project17_session" --create-background
    ZELLIJ_SESSION_NAME="$project17_session" zellij run --name project17-json --close-on-exit --cwd "$PWD" -- \
      ${pkgs.bash}/bin/bash -lc '
        set -euo pipefail
        out="$1"; mode="$2"; corpus="$3"; requests="$4"; model="$5"; lib="$6"; cuda_ld="$7"; root="$8"
        cd "$root"
        baseline=""
        same=""
        case "$mode" in
          constrained) ;;
          baseline) baseline="--baseline-only" ;;
          same-grammar-repeated) same="--same-grammar-repeated" ;;
          *) printf "unknown PROJECT17_MODE: %s\\n" "$mode" >&2; exit 2 ;;
        esac
        export CUDA_VISIBLE_DEVICES=0
        export LLAMA_CPP_LIB_PATH="$lib"
        export LD_LIBRARY_PATH="$cuda_ld''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export PYTHONPATH="src:.devenv/state/venv/lib/python3.13/site-packages''${PYTHONPATH:+:$PYTHONPATH}"
        printf "CUDA_VISIBLE_DEVICES=%s\\nLLAMA_CPP_LIB_PATH=%s\\nLD_LIBRARY_PATH=%s\\nPYTHONPATH=%s\\n" "$CUDA_VISIBLE_DEVICES" "$LLAMA_CPP_LIB_PATH" "$LD_LIBRARY_PATH" "$PYTHONPATH" > "$out/runtime-environment.txt"
        printf "%q " .scratch/projects/17-llama-cpp-inference-lab/.venv-spike/bin/python benchmarks/project17/run_json_workload.py --model "$model" --corpus "$corpus" --requests "$requests" --artifacts "$out" $baseline $same > "$out/command.txt"
        printf "\\n" >> "$out/command.txt"
        ( while :; do date -u +%FT%TZ; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 15; done ) > "$out/gpu-during.csv" 2>&1 &
        monitor=$!
        trap "kill $monitor 2>/dev/null || true" EXIT
        set +e
        .scratch/projects/17-llama-cpp-inference-lab/.venv-spike/bin/python benchmarks/project17/run_json_workload.py --model "$model" --corpus "$corpus" --requests "$requests" --artifacts "$out" $baseline $same > "$out/stdout-stderr.log" 2>&1
        rc=$?
        set -e
        printf "%s\\n" "$rc" > "$out/exit-status.txt"
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv,noheader > "$out/gpu-after.csv" || true
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$out/gpu-processes-after.csv" || true
        ps -eo pid,ppid,stat,args > "$out/processes-after.txt"
        exit "$rc"
      ' -- "$project17_artifacts" "$project17_mode" "$project17_corpus" "$project17_requests" "$PROJECT17_MODEL" "$project17_lib" "$project17_cuda_ld" "$PWD"
    printf 'Project 17 benchmark running in Zellij: zellij attach %s\n' "$project17_session"
    printf 'Artifacts: %s\n' "$project17_artifacts"
  '';

  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    uv.enable = true;
  };

  enterShell = ''
    git --version
  '';
}
