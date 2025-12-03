#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-/app/models/sentiment}
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export UVICORN_WORKERS=${UVICORN_WORKERS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

echo "Starting SmartReview API"
echo "MODEL_DIR=${MODEL_DIR}, HOST=${API_HOST}, PORT=${API_PORT}, WORKERS=${UVICORN_WORKERS}"

# start uvicorn
exec uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --workers "$UVICORN_WORKERS" --log-level info
