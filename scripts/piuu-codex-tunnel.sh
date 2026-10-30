#!/usr/bin/env bash
set -euo pipefail
: "${PIUU_CODEX_SSH_HOST:?Use a reviewed SSH hostname or alias}"
: "${PIUU_CODEX_SSH_USER:?Set the dedicated tunnel user}"
: "${PIUU_CODEX_SSH_KEY:?Set the absolute private key path}"
local_port=${PIUU_CODEX_PORT:-4500}
remote_port=${PIUU_CODEX_REMOTE_PORT:-4500}
ssh_port=${PIUU_CODEX_SSH_PORT:-22}
for port in "$local_port" "$remote_port" "$ssh_port"; do
    [[ $port =~ ^[0-9]+$ ]] && ((10#$port > 0 && 10#$port < 65536)) || exit 64
done
[[ $PIUU_CODEX_SSH_HOST =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 64
[[ $PIUU_CODEX_SSH_USER =~ ^[a-zA-Z0-9_][a-zA-Z0-9_-]*$ ]] || exit 64
[[ $PIUU_CODEX_SSH_KEY = /* && -f $PIUU_CODEX_SSH_KEY ]] || exit 64
printf '%s\n' 'CODEX_TUNNEL_CONNECTING'
# SSH reports forwarding errors/disconnections to journald; systemd owns retries.
# Log success only after authentication/forwarding using SSH's LocalCommand.
exec /usr/bin/ssh -NT -p "$ssh_port" -l "$PIUU_CODEX_SSH_USER" \
    -i "$PIUU_CODEX_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o ConnectTimeout=10 -o ConnectionAttempts=1 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=yes \
    -o LogLevel=ERROR -o PermitLocalCommand=yes \
    -o 'LocalCommand=echo CODEX_TUNNEL_CONNECTED' \
    -R "127.0.0.1:${remote_port}:127.0.0.1:${local_port}" -- "$PIUU_CODEX_SSH_HOST"
