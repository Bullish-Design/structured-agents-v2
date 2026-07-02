#!/usr/bin/env bash
# Verify a running structured-agents vLLM endpoint, end to end.
#
# Checks, in order (each prints PASS/FAIL; the script exits non-zero if any hard check fails):
#   1. reachable     — GET /health
#   2. models        — GET /v1/models lists the served model (and any LoRA adapters)
#   3. json_schema   — a response_format=json_schema completion returns schema-valid JSON
#   4. xgrammar      — a structured_outputs.regex completion returns text matching the regex
#                      (this is the part the llama.cpp box CAN'T do — proves real XGrammar)
#   5. lora          — (optional) a request with model=<adapter> returns 200
#
# Config comes from the environment (defaults mirror deploy/vllm/.env):
#   LLM_BASE_URL   default http://localhost:8000/v1   (e.g. http://tower:8000/v1 over Tailscale)
#   LLM_API_KEY    default $VLLM_API_KEY              (sent as Bearer when non-empty)
#   LLM_MODEL      default base                       (SERVED_MODEL_NAME)
#   LORA_NAME      optional                           (a name from LORA_MODULES; enables check 5)
#
# Usage:
#   deploy/vllm/verify.sh                       # curl-only smoke (no project deps)
#   LLM_BASE_URL=http://tower:8000/v1 deploy/vllm/verify.sh
#   deploy/vllm/verify.sh --pytest              # also run the library's live test markers
#
# Requires: bash, curl. Optional: python3 (stricter JSON validation), the repo + devenv (--pytest).
set -uo pipefail

BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
API_KEY="${LLM_API_KEY:-${VLLM_API_KEY:-}}"
MODEL="${LLM_MODEL:-base}"
LORA_NAME="${LORA_NAME:-}"
ROOT_URL="${BASE_URL%/v1}"
RUN_PYTEST=0
[[ "${1:-}" == "--pytest" ]] && RUN_PYTEST=1

pass=0 fail=0
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }
ok() {
  green "PASS  $*"
  pass=$((pass + 1));
}
bad() {
  red "FAIL  $*"
  fail=$((fail + 1));
}
hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

auth=(-H "Content-Type: application/json")
[[ -n "$API_KEY" ]] && auth+=(-H "Authorization: Bearer $API_KEY")

# curl_json METHOD URL [DATA] -> sets $BODY/$CODE
curl_json() {
  local method="$1" url="$2" data="${3:-}" tmp
  tmp="$(mktemp)"
  # curl -w always prints a status (000 on a connection failure); don't add a fallback.
  if [[ -n "$data" ]]; then
    CODE="$(curl -s --max-time 30 -o "$tmp" -w '%{http_code}' -X "$method" "${auth[@]}" -d "$data" "$url")"
  else
    CODE="$(curl -s --max-time 10 -o "$tmp" -w '%{http_code}' -X "$method" "${auth[@]}" "$url")"
  fi
  BODY="$(cat "$tmp")"
  rm -f "$tmp"
}

# content of choices[0].message.content from $BODY (python if available, else grep)
extract_content() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' <<<"$BODY" 2>/dev/null
  else
    grep -oE '"content":"[^"]*"' <<<"$BODY" | head -1 | sed 's/^"content":"//; s/"$//'
  fi
}

is_valid_json() { command -v python3 >/dev/null 2>&1 && python3 -c 'import sys,json; json.load(sys.stdin)' >/dev/null 2>&1 <<<"$1"; }

echo "Target: $BASE_URL   model: $MODEL   auth: $([[ -n "$API_KEY" ]] && echo yes || echo no)"

# 1. reachable -------------------------------------------------------------------------
hdr "1. reachable (/health)"
curl_json GET "$ROOT_URL/health"
if [[ "$CODE" == "200" ]]; then ok "/health -> 200"; else
  bad "/health -> $CODE (is the container up? model still loading? wrong URL?)"
  red "Endpoint unreachable — skipping remaining checks."
  echo; echo "Summary: $pass passed, $fail failed"; exit 1
fi

# 2. models ----------------------------------------------------------------------------
hdr "2. served models (/v1/models)"
curl_json GET "$BASE_URL/models"
if [[ "$CODE" == "200" ]] && grep -q "\"$MODEL\"" <<<"$BODY"; then
  ok "model '$MODEL' is served"
else
  bad "model '$MODEL' not found in /v1/models (code $CODE)"
fi
if [[ -n "$LORA_NAME" ]]; then
  grep -q "\"$LORA_NAME\"" <<<"$BODY" && ok "LoRA adapter '$LORA_NAME' is served" || bad "LoRA adapter '$LORA_NAME' not listed"
fi

# 3. json_schema -----------------------------------------------------------------------
hdr "3. json_schema structured output"
read -r -d '' JS_REQ <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"Return a command to create notes.txt"}],
 "response_format":{"type":"json_schema","json_schema":{"name":"Command","strict":true,
   "schema":{"type":"object","properties":{"action":{"type":"string"},"target":{"type":"string"}},
     "required":["action","target"],"additionalProperties":false}}}}
EOF
curl_json POST "$BASE_URL/chat/completions" "$JS_REQ"
content="$(extract_content)"
if [[ "$CODE" == "200" ]] && is_valid_json "$content" && grep -q '"action"' <<<"$content"; then
  ok "json_schema -> valid JSON: $content"
else
  bad "json_schema (code $CODE): ${content:-<no content>}"
fi

# 4. xgrammar (regex) ------------------------------------------------------------------
hdr "4. XGrammar bare-string (regex) — vLLM-only"
REGEX='git (status|diff|add|commit) [a-zA-Z0-9._/ -]*'
# max_tokens is REQUIRED here: the regex ends in an unbounded `*`, so without a cap the
# model generates to max_model_len and the request exceeds curl's timeout (code 000) on
# slow GPUs. XGrammar still guarantees every emitted token matches the regex, so a capped
# (possibly truncated) output is a valid full match for the anchored grep below.
read -r -d '' RX_REQ <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"Show the working tree status with git."}],
 "structured_outputs":{"regex":"$REGEX"},"max_tokens":64}
EOF
curl_json POST "$BASE_URL/chat/completions" "$RX_REQ"
content="$(extract_content)"
if [[ "$CODE" == "200" ]] && grep -qE "^$REGEX\$" <<<"$content"; then
  ok "regex-constrained output matches: '$content'"
elif [[ "$CODE" != "200" ]]; then
  bad "xgrammar request failed (code $CODE) — is STRUCTURED_OUTPUTS_BACKEND=xgrammar on a vLLM 0.10+ tag?"
else
  bad "output did not match regex: '$content'"
fi

# 5. lora (optional) -------------------------------------------------------------------
if [[ -n "$LORA_NAME" ]]; then
  hdr "5. LoRA adapter round-trip (model=$LORA_NAME)"
  read -r -d '' LR_REQ <<EOF
{"model":"$LORA_NAME","messages":[{"role":"user","content":"hello"}]}
EOF
  curl_json POST "$BASE_URL/chat/completions" "$LR_REQ"
  [[ "$CODE" == "200" ]] && ok "adapter '$LORA_NAME' served a completion" || bad "adapter '$LORA_NAME' request -> $CODE"
fi

# optional: the library's own live assertions ------------------------------------------
if [[ "$RUN_PYTEST" == "1" ]]; then
  hdr "library live tests (pytest -m live)"
  if command -v devenv >/dev/null 2>&1; then
    ( cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" &&
      SAV_LIVE=1 SAV_LIVE_XGRAMMAR=1 LLM_BASE_URL="$BASE_URL" LLM_API_KEY="$API_KEY" LLM_MODEL="$MODEL" \
        devenv shell -- uv run --extra dev pytest -q -m live ) &&
      ok "live pytest markers passed" || bad "live pytest markers failed"
  else
    red "SKIP  devenv not found — run from the repo on the dev box for --pytest"
  fi
fi

# summary ------------------------------------------------------------------------------
hdr "summary"
echo "$pass passed, $fail failed"
[[ "$fail" -eq 0 ]] && { green "OK — endpoint verified."; exit 0; } || { red "verification incomplete"; exit 1; }
