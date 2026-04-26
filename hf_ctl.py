#!/usr/bin/env python3
"""
hf_ctl.py — HF Agent Control Script for DriftDesk training
Usage:  python3 hf_ctl.py <command> [args]

Commands:
  status          — Space stage, hardware, last adapter commit
  logs            — Tail live Space logs
  push-notebook   — Upload driftdesk_grpo_hf_space.ipynb + eval_harness.py to Space
  checkpoints     — List all checkpoint commits in the adapter repo
  download-ckpt   — Download latest adapter checkpoint locally
  pause           — Pause the Space (save credits)
  resume          — Resume the Space (restart hardware)
  restart         — Restart the Space runtime
  watch [secs]    — Poll for new checkpoints every N seconds (default 60)
  set-sleep [secs] — Set Space inactivity sleep timer (default 7200s / 2h)
"""

import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, whoami

SPACE_ID   = "HelloOjasMutreja/driftdesk-training"
REPO_ID    = "HelloOjasMutreja/driftdesk-grpo-adapter"
NOTEBOOK   = Path(__file__).parent / "driftdesk" / "driftdesk_grpo_hf_space.ipynb"
HARNESS    = Path(__file__).parent / "driftdesk" / "eval_harness.py"
CKPT_DIR   = Path(__file__).parent / "driftdesk" / "driftdesk_grpo_output"

api = HfApi()


def cmd_status():
    rt = api.get_space_runtime(SPACE_ID)
    commits = list(api.list_repo_commits(REPO_ID, repo_type="model"))
    last = commits[0] if commits else None
    print(f"Space    : {SPACE_ID}")
    print(f"Stage    : {rt.stage}")
    print(f"Hardware : {rt.hardware}")
    print(f"Sleep in : {rt.sleep_time}s of inactivity")
    print(f"URL      : https://{SPACE_ID.replace('/', '-').lower()}.hf.space")
    print()
    print(f"Adapter  : {REPO_ID}")
    print(f"Commits  : {len(commits)}")
    if last:
        print(f"Latest   : [{last.commit_id[:7]}]  {last.created_at}  {last.title}")


def cmd_logs():
    import time
    print(f"Streaming logs for {SPACE_ID} (Ctrl+C to stop)...\n")
    try:
        for entry in api.get_space_runtime(SPACE_ID).__class__.__mro__:
            pass
    except Exception:
        pass
    # Use requests to stream logs
    import requests
    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    url = f"https://huggingface.co/api/spaces/{SPACE_ID}/logs"
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120) as r:
        for line in r.iter_lines():
            if line:
                print(line.decode("utf-8", errors="replace"))


def cmd_push_notebook():
    files = []
    if NOTEBOOK.exists():
        files.append((str(NOTEBOOK), "driftdesk_grpo_hf_space.ipynb"))
    else:
        print(f"WARN: notebook not found at {NOTEBOOK}")
    if HARNESS.exists():
        files.append((str(HARNESS), "eval_harness.py"))
    else:
        print(f"WARN: eval_harness.py not found at {HARNESS}")

    for local_path, path_in_repo in files:
        url = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"agent push: {path_in_repo}",
        )
        print(f"Pushed {path_in_repo} → {url}")


def cmd_checkpoints():
    commits = list(api.list_repo_commits(REPO_ID, repo_type="model"))
    print(f"Adapter repo: {REPO_ID}  ({len(commits)} commits)\n")
    for c in commits:
        print(f"  {c.commit_id[:7]}  {c.created_at}  {c.title}")


def cmd_download_ckpt():
    from huggingface_hub import snapshot_download
    dest = CKPT_DIR / "hub_latest"
    print(f"Downloading latest adapter from {REPO_ID} → {dest}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(dest),
        ignore_patterns=["*.pt", "rng_state*"],
    )
    print(f"Downloaded to {path}")


def cmd_pause():
    api.pause_space(SPACE_ID)
    print(f"Space {SPACE_ID} paused (hardware deallocated, no charges).")


def cmd_resume():
    rt = api.restart_space(SPACE_ID)
    print(f"Space {SPACE_ID} resuming → stage: {rt.stage}")


def cmd_restart():
    rt = api.restart_space(SPACE_ID)
    print(f"Space {SPACE_ID} restarting → stage: {rt.stage}")


def cmd_set_sleep():
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 7200
    api.set_space_sleep_time(SPACE_ID, sleep_time=secs)
    rt = api.get_space_runtime(SPACE_ID)
    print(f"Sleep timer set to {rt.sleep_time}s ({rt.sleep_time//60} min).")


def cmd_watch():
    import time
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print(f"Watching {REPO_ID} for new checkpoints (every {interval}s, Ctrl+C to stop)...\n")
    seen = set()
    while True:
        try:
            commits = list(api.list_repo_commits(REPO_ID, repo_type="model"))
            for c in commits:
                if c.commit_id not in seen:
                    seen.add(c.commit_id)
                    ts = c.created_at.strftime("%H:%M:%S")
                    print(f"[{ts}]  {c.commit_id[:7]}  {c.title}")
            rt = api.get_space_runtime(SPACE_ID)
            print(f"        Space: {rt.stage}  (checked at {time.strftime('%H:%M:%S')})", end="\r")
        except Exception as e:
            print(f"[warn] {e}")
        time.sleep(interval)


COMMANDS = {
    "status":          cmd_status,
    "logs":            cmd_logs,
    "push-notebook":   cmd_push_notebook,
    "checkpoints":     cmd_checkpoints,
    "download-ckpt":   cmd_download_ckpt,
    "pause":           cmd_pause,
    "resume":          cmd_resume,
    "restart":         cmd_restart,
    "watch":           cmd_watch,
    "set-sleep":       cmd_set_sleep,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]]()
