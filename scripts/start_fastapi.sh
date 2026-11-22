#!/usr/bin/env bash
set -euo pipefail

# Simple starter script for FastAPI using uvicorn
# Usage:
#   ./scripts/start_fastapi.sh
#   APP_MODULE=main:app RELOAD=true PORT=8080 ./scripts/start_fastapi.sh

ENV_FILE=".env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
APP_MODULE=${APP_MODULE:-main:app}
RELOAD=${RELOAD:-false}

UVICORN_CMD=(uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}")
if [ "${RELOAD}" = "true" ] || [ "${RELOAD}" = "1" ]; then
  UVICORN_CMD+=(--reload)
fi

echo "Starting FastAPI: ${APP_MODULE} on ${HOST}:${PORT} (reload=${RELOAD})"
exec "${UVICORN_CMD[@]}"
