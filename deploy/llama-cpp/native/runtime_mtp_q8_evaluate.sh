#!/usr/bin/env bash
# Root-run, reversible GPU-0 MTP comparison.  It deliberately refuses to run
# beside the production service and restores the configured fallback on exit.
set -euo pipefail

repo=/home/andrew/Documents/Projects/structured-agents-v2
service=structured-agents-llama-cpp.service
user=andrew
group=users
server_bin=${LLAMA_SERVER_BIN:-/nix/store/yrv3h66zp3yxbc3c3sfglyiqa0s4is96-llama-cpp-9842/bin/llama-server}
target=/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165
unsloth=/home/andrew/.cache/structured-agents/models/mtp-gemma-4-12B-it.gguf
q8=/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifacts="$repo/artifacts/mtp-q8-evaluation/$stamp"

[[ $(id -u) -eq 0 ]] || { echo "run this script via sudo" >&2; exit 2; }
[[ -x $server_bin && -f $target && -f $unsloth && -f $q8 ]] || { echo "missing server or model file" >&2; exit 2; }

mkdir -p "$artifacts"
chown "$user:$group" "$artifacts"
systemctl is-active --quiet "$service" || { echo "$service must be active before its reversible benchmark stop" >&2; exit 2; }

restore() {
  systemctl start "$service" || true
}
trap restore EXIT

# This is the sole intentional service interruption.  vLLM is never named or touched.
systemctl stop "$service"

run_variant() {
  local variant=$1 draft=${2:-}
  local dir="$artifacts/$variant" log="$artifacts/$variant/server.log"
  mkdir -p "$dir"
  chown "$user:$group" "$dir"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader >"$dir/gpu-before.csv"
  local -a cmd=("$server_bin" --model "$target" --host 127.0.0.1 --port 8001 --alias base
    --ctx-size 16384 --n-gpu-layers 999 --cache-type-k q8_0 --cache-type-v q8_0
    --flash-attn on --parallel 1 --threads 8 --threads-batch 8)
  if [[ -n $draft ]]; then
    cmd+=(--spec-type draft-mtp --spec-draft-model "$draft" --spec-draft-n-max 4 --n-gpu-layers-draft 999)
  fi
  printf '%q ' "${cmd[@]}" >"$dir/command.txt"; printf '\n' >>"$dir/command.txt"
  systemd-run --quiet --collect --unit "structured-agents-llama-cpp-benchmark-$variant" \
    --uid="$user" --gid="$group" --working-directory="$repo/deploy/llama-cpp/native" \
    --property="StandardOutput=append:$log" --property="StandardError=append:$log" \
    /usr/bin/env CUDA_VISIBLE_DEVICES=0 "${cmd[@]}"
  local tries=0
  until /etc/profiles/per-user/andrew/bin/curl --fail --silent http://127.0.0.1:8001/v1/models >"$dir/models.json"; do
    (( ++tries >= 120 )) && { echo "$variant server did not become ready" >&2; return 1; }
    sleep 1
  done
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader >"$dir/gpu-loaded.csv"
  runuser -u "$user" -- /usr/bin/env PATH=/etc/profiles/per-user/andrew/bin:/run/current-system/sw/bin \
    python3 "$repo/deploy/llama-cpp/native/benchmark_mtp.py" --base-url http://127.0.0.1:8001/v1 \
    --variant "$variant" --artifact-dir "$dir" >"$dir/client.json"
  systemctl stop "structured-agents-llama-cpp-benchmark-$variant.service"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader >"$dir/gpu-after.csv"
  chown -R "$user:$group" "$dir"
}

run_variant no-mtp
run_variant unsloth "$unsloth"
run_variant q8 "$q8"

runuser -u "$user" -- python3 - "$artifacts" <<'PY'
import csv, json, re, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    requests = list(csv.DictReader((d / "requests.csv").open()))
    log = (d / "server.log").read_text(errors="replace")
    accepted = generated = 0
    for a, g in re.findall(r"\\(\\s*(\\d+) accepted /\\s*(\\d+) generated\\)", log):
        accepted += int(a); generated += int(g)
    rows.append({"variant": d.name, "mean_output_tok_s": round(sum(float(r["output_tok_s"]) for r in requests)/len(requests), 3), "mean_latency_s": round(sum(float(r["latency_seconds"]) for r in requests)/len(requests), 3), "draft_tokens": generated, "accepted_tokens": accepted, "acceptance_rate": round(accepted/generated, 4) if generated else 0})
with (root / "comparison.json").open("w") as f: json.dump(rows, f, indent=2); f.write("\n")
with (root / "comparison.csv").open("w", newline="") as f: w=csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
with (root / "comparison.md").open("w") as f:
    f.write("| variant | mean output tok/s | mean latency s | draft tokens | accepted tokens | acceptance rate |\n|---|---:|---:|---:|---:|---:|\n")
    for r in rows: f.write("| {variant} | {mean_output_tok_s} | {mean_latency_s} | {draft_tokens} | {accepted_tokens} | {acceptance_rate} |\n".format(**r))
PY

echo "artifacts: $artifacts"
