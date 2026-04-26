"""Minimal training-log server for the HF training Space.

The Space MUST bind port 7860 to stay alive. The training script (`train.py`)
talks to an EXTERNAL DriftDesk env (DRIFTDESK_ENV_URL), so we don't need the
heavy openenv-core stack here. This keeps the build lean.
"""
import os

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()

LOG_CANDIDATES = ["/app/training.log", "/tmp/training.log", "/home/user/app/training.log"]


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def root():
    return PlainTextResponse(
        "DriftDesk training Space — see /training-log for live logs.\n"
    )


@app.get("/training-log", response_class=PlainTextResponse)
def training_log(tail: int = 200):
    for log_path in LOG_CANDIDATES:
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", errors="replace") as f:
                    lines = f.readlines()
                return PlainTextResponse(
                    f"# log_path={log_path} bytes={os.path.getsize(log_path)}\n"
                    + "".join(lines[-tail:])
                )
            except Exception as e:
                return PlainTextResponse(f"# error reading {log_path}: {e}")

    out = ["training.log not found in: " + ", ".join(LOG_CANDIDATES), ""]
    for d in ["/app", "/tmp"]:
        out.append(f"=== ls {d} ===")
        try:
            out.extend(sorted(os.listdir(d))[:80])
        except Exception as e:
            out.append(f"(error: {e})")
        out.append("")
    out.append("=== env ===")
    for k in (
        "HF_TOKEN", "DRIFTDESK_ENV_URL", "GRPO_STEPS", "MAX_EPISODE_STEPS",
        "GRPO_TEMPERATURE", "GRPO_TOP_P", "GRPO_NUM_GENERATIONS", "DATA_DIR",
    ):
        v = os.environ.get(k)
        if v is not None and k == "HF_TOKEN":
            v = f"set(len={len(v)})"
        out.append(f"{k}={v}")
    return PlainTextResponse("\n".join(out))
