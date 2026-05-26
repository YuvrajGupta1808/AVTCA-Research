#!/usr/bin/env bash
# Codex login for remote SFSU workspace (device-auth disabled).
# Run ON srva after forwarding port 1455 from your laptop (see header).

set -euo pipefail

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
PORT=1455

echo "==> Stopping stale Codex processes..."
pkill -f 'codex app-server' 2>/dev/null || true
sleep 1

echo "==> Clearing revoked credentials..."
codex logout 2>/dev/null || true
rm -f "$CODEX_HOME/auth.json"

if ! ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo ""
  echo "ERROR: Nothing is listening on localhost:${PORT}."
  echo "Forward port ${PORT} before running this script:"
  echo ""
  echo "  Option A — Cursor/VS Code: Ports panel → Forward port ${PORT}"
  echo "  Option B — Laptop terminal:"
  echo "    ssh -L ${PORT}:localhost:${PORT} 922933190@srva"
  echo ""
  exit 1
fi

echo "==> Port ${PORT} is forwarded. Starting browser login..."
codex login

echo ""
echo "==> Verifying API access..."
if codex exec "reply with exactly: ok" -a never 2>&1 | head -5 | grep -qi ok; then
  echo "SUCCESS: Codex auth is working."
  codex login status
  exit 0
fi

echo "WARNING: Login finished but API test did not return 'ok'."
echo "  - Close Codex on your laptop (only one active session)."
echo "  - Do NOT use: codex login --device-auth (disabled for SFSU)."
echo "  - Re-run this script after: codex logout (on laptop and srva)."
codex login status
exit 1
