#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

need() {
  local url="$1"
  shift
  echo "→ $url"
  curl -sfS "$@" "$url"
  echo
}

need "http://127.0.0.1:8001/health"
need "http://127.0.0.1:8002/health"
need "http://127.0.0.1:3000/health"

echo "→ http://127.0.0.1/health (expect redirect to https)"
redirect="$(curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' http://127.0.0.1/health)"
echo "$redirect"
echo "$redirect" | grep -q '^301 ' || {
  echo "expected HTTP 301 from port 80"
  exit 1
}

need "https://127.0.0.1/health" -k

REF="$ROOT/models/voices/default/ref.wav"
if [[ -f "$REF" ]]; then
  echo "→ POST /v1/audio/transcriptions ($REF)"
  curl -sfS "http://127.0.0.1:8001/v1/audio/transcriptions" \
    -F "file=@${REF}" \
    -F "language=ru"
  echo
else
  echo "skip STT file test: $REF is missing"
fi

echo "ok"
