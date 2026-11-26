#!/usr/bin/env bash
# Single-container start-up (Hugging Face Space or `docker run`): API on 127.0.0.1:8000 (internal only),
# Streamlit dashboard on $PORT (7860 by default) - the only publicly exposed port.
set -euo pipefail
cd "$(dirname "$0")/.."

export DR_STATE_DIR="${DR_STATE_DIR:-$PWD/state}"        # on a Space with persistent storage set DR_STATE_DIR=/data
export DR_MODELS_DIR="$DR_STATE_DIR/models"
export DR_DB_PATH="$DR_STATE_DIR/database/dr_mlops.sqlite3"
export DR_UPLOAD_DIR="$DR_STATE_DIR/uploads"
export DR_DATASET_NPZ="${DR_DATASET_NPZ:-$PWD/data/retinamnist_224.npz}"
export DR_API_URL="http://127.0.0.1:8000"

python -m scripts.prepare_runtime

uvicorn src.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

echo "Waiting for the API ..."
for i in $(seq 1 120); do
  if python -m scripts.healthcheck; then echo "API is up."; break; fi
  if ! kill -0 $API_PID 2>/dev/null; then echo "API process died during start-up"; exit 1; fi
  sleep 2
done

exec streamlit run app/dashboard.py \
  --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true \
  --server.enableXsrfProtection false --server.enableCORS false --browser.gatherUsageStats false
