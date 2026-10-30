#!/usr/bin/env bash
set -euo pipefail
: "${PIUU_CODEX_ROOT:?Absolute PIUU_SERVER checkout path required}"
: "${PIUU_CODEX_PYTHON:?Absolute Python venv interpreter required}"
[[ $PIUU_CODEX_ROOT = /* && $PIUU_CODEX_PYTHON = /* ]] || exit 64
if [[ -z ${PIUU_CODEX_BINARY:-} ]]; then
    PIUU_CODEX_BINARY=$(command -v codex) || exit 69
    export PIUU_CODEX_BINARY
fi
[[ $PIUU_CODEX_BINARY = /* && -x $PIUU_CODEX_BINARY ]] || exit 69
export PYTHONPATH="$PIUU_CODEX_ROOT/src"
# User systemd managers may be unable to apply PrivateDevices. Bubblewrap
# provides a private synthetic /dev without requiring a root service. Keep
# Codex's own restricted filesystem/network sandbox inside this outer boundary.
bwrap_binary=${PIUU_CODEX_BWRAP:-/usr/bin/bwrap}
[[ $bwrap_binary = /* && -x $bwrap_binary ]] || exit 69
exec "$bwrap_binary" --die-with-parent --bind / / --dev /dev \
    --unshare-pid --proc /proc -- "$PIUU_CODEX_PYTHON" -m bridge.codex.gateway
