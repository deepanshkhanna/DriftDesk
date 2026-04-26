#!/bin/bash
JUPYTER_TOKEN="${JUPYTER_TOKEN:=huggingface}"
APP_DIR="/home/user/app"
LOG="$APP_DIR/training.log"

# Launch training in background before JupyterLab starts
if [ -f "$APP_DIR/train.py" ]; then
    echo "[start_server] Launching train.py in background -> $LOG"
    cd "$APP_DIR"
    nohup python train.py > "$LOG" 2>&1 &
    echo $! > "$APP_DIR/training.pid"
    echo "[start_server] Training PID: $(cat $APP_DIR/training.pid)"
else
    echo "[start_server] train.py not found — skipping auto-training."
fi

jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

exec jupyter-lab \
    --ip 0.0.0.0 \
    --port 7860 \
    --no-browser \
    --allow-root \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --ServerApp.tornado_settings="{'headers': {'Content-Security-Policy': 'frame-ancestors *'}}" \
    --ServerApp.cookie_options="{'SameSite': 'None', 'Secure': True}" \
    --ServerApp.disable_check_xsrf=True \
    --LabApp.news_url=None \
    --LabApp.check_for_updates_class="jupyterlab.NeverCheckForUpdate" \
    --notebook-dir="$APP_DIR"
