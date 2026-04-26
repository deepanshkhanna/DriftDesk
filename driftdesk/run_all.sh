#!/bin/bash
set -e
APP_DIR="/app"
LOG="/tmp/training.log"
touch "$LOG" && chmod 666 "$LOG" || true

# HF/cache dirs need to be writable by the runtime user.
export HOME=/tmp
export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf
export HF_HUB_CACHE=/tmp/hf
export XDG_CACHE_HOME=/tmp/cache
export HF_HUB_ENABLE_HF_TRANSFER=0
# DATA_DIR must point to the app directory so eval_harness.py and server/ are found
export DATA_DIR=/app
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" || true

echo "[run_all] Starting minimal log-server on port 7860..."
cd "$APP_DIR"
uvicorn log_server:app --host 0.0.0.0 --port 7860 &
SERVER_PID=$!
echo "[run_all] Server PID: $SERVER_PID"

# Brief wait so the port is bound before we start the heavy training process
for i in $(seq 1 20); do
    if curl -sf http://localhost:7860/healthz > /dev/null 2>&1; then
        echo "[run_all] Log-server ready after ${i}s."
        break
    fi
    sleep 1
done

echo "[run_all] Launching train.py -> $LOG"
nohup python train.py > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "[run_all] Training PID: $TRAIN_PID"

# Keep container alive
wait $SERVER_PID
