#!/usr/bin/env bash
# Start / stop / status for the local Claude CLI proxy (claude_cli_proxy.py) used as an
# OpenAI-compatible reflection endpoint by the offline learners (--reflection-client vllm,
# HOSTED_VLLM_API_BASE=http://127.0.0.1:8000/v1).
#
#   bash offline_learning/scripts/claude_proxy_ctl.sh start|stop|restart|status
#
# Kept as a script (not an inline shell command) because `pgrep -f` on the proxy's name
# would otherwise match the invoking shell's own command line and kill it.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT/logs/2026-08-24/claude_proxy"
PORT="${CLAUDE_PROXY_PORT:-8000}"
PATTERN="scripts/claude_cli_proxy[.]py"
export CLAUDE_PROXY_MAX_CONCURRENCY="${CLAUDE_PROXY_MAX_CONCURRENCY:-8}"
export CLAUDE_PROXY_TIMEOUT_SECONDS="${CLAUDE_PROXY_TIMEOUT_SECONDS:-900}"

pids() { pgrep -f "$PATTERN" | grep -v "^$$\$" || true; }

stop() {
  local p; p="$(pids)"
  if [ -n "$p" ]; then kill $p 2>/dev/null; sleep 1; echo "stopped: $p"; else echo "not running"; fi
}

start() {
  if [ -n "$(pids)" ]; then echo "already running: $(pids)"; return 0; fi
  mkdir -p "$LOG_DIR"
  cd "$ROOT" || exit 1
  nohup uv run python offline_learning/scripts/claude_cli_proxy.py --port "$PORT" \
    >> "$LOG_DIR/proxy.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 40); do
    curl -s "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
    sleep 1
  done
  echo "started pid=$pid concurrency=$CLAUDE_PROXY_MAX_CONCURRENCY timeout=$CLAUDE_PROXY_TIMEOUT_SECONDS"
  curl -s "http://127.0.0.1:$PORT/healthz"; echo
}

status() {
  echo "pids: $(pids)"
  curl -s "http://127.0.0.1:$PORT/healthz" || echo "healthz: no response"
  echo
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 2 ;;
esac
