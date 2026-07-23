#!/usr/bin/env bash
# Basic runtime API verification; run only after a successful isolated startup.
set -euo pipefail

base_url="${LLM_BASE_URL:-http://127.0.0.1:8003/v1}"
base_url="${base_url%/}"
root_url="${base_url%/v1}"
model="${LLM_MODEL:-base}"
headers=(-H 'Content-Type: application/json')
[[ -n "${LLM_API_KEY:-}" ]] && headers+=(-H "Authorization: Bearer $LLM_API_KEY")

curl -fsS "$root_url/health" >/dev/null
models="$(curl -fsS "${headers[@]}" "$base_url/models")"
grep -q "\"$model\"" <<<"$models"
response="$(curl -fsS "${headers[@]}" -X POST "$base_url/chat/completions" \
  --data "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ready\"}],\"temperature\":0,\"max_tokens\":8}")"
grep -q '"choices"' <<<"$response"
echo "OK — SGLang is serving $model at $base_url"
