"""
FastAPI application for the DriftDesk Environment.

Endpoints (OpenEnv standard):
  POST /reset  — start a new episode
  POST /step   — execute an action
  GET  /state  — inspect current environment state
  GET  /schema — OpenAPI schema for action/observation
  WS   /ws     — WebSocket for persistent sessions
  GET  /healthz — health check
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.http_server import create_app
from models import DriftDeskAction, DriftDeskObservation
from server.driftdesk_environment import DriftDeskEnvironment

app = create_app(
    DriftDeskEnvironment,
    DriftDeskAction,
    DriftDeskObservation,
    env_name="driftdesk",
    max_concurrent_envs=4,
)

from fastapi.responses import PlainTextResponse

@app.get("/training-log")
def training_log(tail: int = 200):
    # Probe both common container layouts for the log file.
    candidates = ["/app/training.log", "/home/user/app/training.log"]
    for log_path in candidates:
        if os.path.exists(log_path):
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()
            return PlainTextResponse(
                f"# log_path={log_path} bytes={os.path.getsize(log_path)}\n"
                + "".join(lines[-tail:])
            )
    # Fallback: surface filesystem state so we can diagnose silent failures.
    out = ["training.log not found in candidates: " + ", ".join(candidates), ""]
    for d in ["/app", "/home/user/app", "/tmp"]:
        out.append(f"=== ls {d} ===")
        try:
            out.extend(sorted(os.listdir(d))[:50])
        except Exception as e:
            out.append(f"(error: {e})")
        out.append("")
    out.append("=== env ===")
    for k in ("HF_TOKEN", "DRIFTDESK_ENV_URL", "GRPO_STEPS", "MAX_EPISODE_STEPS",
             "GRPO_TEMPERATURE", "GRPO_TOP_P", "GRPO_NUM_GENERATIONS"):
        v = os.environ.get(k)
        if v is not None and k == "HF_TOKEN":
            v = f"set(len={len(v)})"
        out.append(f"{k}={v}")
    return PlainTextResponse("\n".join(out))


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)  # calls main()
