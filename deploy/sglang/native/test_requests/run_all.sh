#!/usr/bin/env bash
# Replay every saved test request against a running server and save responses.
# Usage: ./run_all.sh [base_url] [output_dir]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_url="${1:-http://127.0.0.1:8002/v1}"
base_url="${base_url%/}"
out_dir="${2:-$here/responses}"
mkdir -p "$out_dir"

curl -fsS "${base_url%/v1}/health" >/dev/null || {
  echo "server at $base_url is not healthy" >&2
  exit 1
}

status=0
for req in "$here"/[0-9][0-9]_*.json; do
  name="$(basename "$req" .json)"
  echo "=== $name ==="
  if curl -fsS -H 'Content-Type: application/json' \
      -X POST "$base_url/chat/completions" \
      --data @"$req" \
      -o "$out_dir/$name.response.json"; then
    python3 -c "
import json, sys
d = json.load(open('$out_dir/$name.response.json'))
msg = d['choices'][0]['message']
print('finish_reason:', d['choices'][0].get('finish_reason'))
print('content:', (msg.get('content') or '')[:500])
if msg.get('tool_calls'):
    print('tool_calls:', json.dumps(msg['tool_calls'], indent=2))
"
  else
    echo "REQUEST FAILED: $name" >&2
    status=1
  fi
  echo
done

exit "$status"
