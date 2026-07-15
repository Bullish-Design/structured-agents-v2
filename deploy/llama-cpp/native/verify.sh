#!/usr/bin/env bash
# Smoke test the OpenAI-compatible llama.cpp endpoint.  Kept separate from the
# vLLM verifier because llama.cpp does not implement vLLM's XGrammar extension.
set -euo pipefail

base_url="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
base_url="${base_url%/}"
model="${LLM_MODEL:-base}"
api_key="${LLM_API_KEY:-${API_KEY:-}}"
root_url="${base_url%/v1}"
headers=(-H 'Content-Type: application/json')
[[ -n "$api_key" ]] && headers+=(-H "Authorization: Bearer $api_key")

curl -fsS "$root_url/health" >/dev/null
models="$(curl -fsS "${headers[@]}" "$base_url/models")"
grep -q "\"$model\"" <<<"$models"
curl -fsS "${headers[@]}" -X POST "$base_url/chat/completions" \
  --data "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ready\"}],\"temperature\":0,\"max_tokens\":8}" \
  >/dev/null

echo "OK — llama.cpp is serving $model at $base_url"
