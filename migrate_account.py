#!/usr/bin/env python3
"""
migrate_account.py — Migrate DriftDesk training to a new HF account.

Usage:
    python3 migrate_account.py <new_username> [hf_token]

If hf_token is omitted, reads from ~/.cache/huggingface/token
"""

import sys
import os
import time
from pathlib import Path
from huggingface_hub import HfApi, whoami

# ── Config ────────────────────────────────────────────────────────────────────
DRIFTDESK_DIR = Path(__file__).parent / "driftdesk"
HF_CTL        = Path(__file__).parent / "hf_ctl.py"

# Files to upload to the Space repo (path_on_disk → path_in_repo)
SPACE_FILES = [
    # Top-level
    (DRIFTDESK_DIR / "Dockerfile",         "Dockerfile"),
    (DRIFTDESK_DIR / "requirements.txt",   "requirements.txt"),
    (DRIFTDESK_DIR / "openenv.yaml",       "openenv.yaml"),
    (DRIFTDESK_DIR / "on_startup.sh",      "on_startup.sh"),
    (DRIFTDESK_DIR / "start_server.sh",    "start_server.sh"),
    (DRIFTDESK_DIR / "train.py",           "train.py"),
    (DRIFTDESK_DIR / "models.py",          "models.py"),
    (DRIFTDESK_DIR / "schemas.py",         "schemas.py"),
    (DRIFTDESK_DIR / "client.py",          "client.py"),
    (DRIFTDESK_DIR / "dummy_env.py",       "dummy_env.py"),
    (DRIFTDESK_DIR / "eval_harness.py",    "eval_harness.py"),
    (DRIFTDESK_DIR / "__init__.py",        "__init__.py"),
    # Server package
    (DRIFTDESK_DIR / "server" / "__init__.py",                "server/__init__.py"),
    (DRIFTDESK_DIR / "server" / "app.py",                     "server/app.py"),
    (DRIFTDESK_DIR / "server" / "drift_controller.py",        "server/drift_controller.py"),
    (DRIFTDESK_DIR / "server" / "driftdesk_environment.py",   "server/driftdesk_environment.py"),
    (DRIFTDESK_DIR / "server" / "policy_injector.py",         "server/policy_injector.py"),
    (DRIFTDESK_DIR / "server" / "reward_engine.py",           "server/reward_engine.py"),
    # Task modules
    (DRIFTDESK_DIR / "server" / "task_modules" / "__init__.py",  "server/task_modules/__init__.py"),
    (DRIFTDESK_DIR / "server" / "task_modules" / "airline.py",   "server/task_modules/airline.py"),
    (DRIFTDESK_DIR / "server" / "task_modules" / "bank.py",      "server/task_modules/bank.py"),
    (DRIFTDESK_DIR / "server" / "task_modules" / "base.py",      "server/task_modules/base.py"),
    (DRIFTDESK_DIR / "server" / "task_modules" / "insurance.py", "server/task_modules/insurance.py"),
]

# Space environment variables
SPACE_VARS = {
    "CURRICULUM_STAGE": "0",
    "GRPO_STEPS":       "150",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}


def get_token():
    if len(sys.argv) >= 3:
        return sys.argv[2].strip()
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    raise RuntimeError(
        "No token found. Either run 'huggingface-cli login' or pass token as 2nd argument."
    )


def update_hf_ctl(new_user: str):
    text = HF_CTL.read_text()
    old_space = next(l for l in text.splitlines() if 'SPACE_ID' in l and '=' in l)
    old_repo  = next(l for l in text.splitlines() if 'REPO_ID'  in l and '=' in l)
    new_space_line = f'SPACE_ID   = "{new_user}/driftdesk-training"'
    new_repo_line  = f'REPO_ID    = "{new_user}/driftdesk-grpo-adapter"'
    text = text.replace(old_space, new_space_line).replace(old_repo, new_repo_line)
    HF_CTL.write_text(text)
    print(f"  hf_ctl.py updated → SPACE_ID={new_user}/driftdesk-training")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    new_user = sys.argv[1].strip()
    token    = get_token()
    api      = HfApi(token=token)

    # Verify token
    me = whoami(token=token)
    print(f"\nLogged in as: {me['name']}")
    if me['name'].lower() != new_user.lower():
        print(f"WARNING: logged-in user '{me['name']}' ≠ requested '{new_user}'")
        print("Continuing anyway — make sure you have write access.\n")

    new_space_id = f"{new_user}/driftdesk-training"
    new_repo_id  = f"{new_user}/driftdesk-grpo-adapter"

    # ── 1. Create Space ───────────────────────────────────────────────────────
    print(f"\n[1/5] Creating Space: {new_space_id}  (A100-large, Docker)")
    try:
        api.create_repo(
            repo_id=new_space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print("      Space repo created (or already exists).")
    except Exception as e:
        print(f"      Space creation: {e}")

    # ── 2. Request A100-large hardware ────────────────────────────────────────
    print("\n[2/5] Requesting A100-large hardware...")
    try:
        api.request_space_hardware(
            repo_id=new_space_id,
            hardware="a100-large",
        )
        print("      Hardware request submitted: a100-large")
    except Exception as e:
        print(f"      Hardware request failed (may need to do manually): {e}")
        print("      → Falling back to A10G for now (free tier).")
        try:
            api.request_space_hardware(repo_id=new_space_id, hardware="a10g-large")
        except Exception as e2:
            print(f"      A10G also failed: {e2}")

    # ── 3. Set env vars ───────────────────────────────────────────────────────
    print("\n[3/5] Setting Space environment variables...")
    for key, val in SPACE_VARS.items():
        api.add_space_variable(new_space_id, key, val)
        print(f"      {key} = {val}")

    # Disable sleep
    try:
        api.set_space_sleep_time(new_space_id, sleep_time=-1)
        print("      Sleep disabled (-1)")
    except Exception as e:
        print(f"      Sleep disable failed: {e}")

    # Adapter repo env var — set after repo creation below
    adapter_env_pending = True

    # ── 4. Create adapter repo ────────────────────────────────────────────────
    print(f"\n[4/5] Creating adapter repo: {new_repo_id}")
    try:
        api.create_repo(
            repo_id=new_repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        print("      Adapter repo created.")
    except Exception as e:
        print(f"      Adapter repo creation: {e}")

    # Now set the adapter repo env var on the Space
    api.add_space_variable(new_space_id, "ADAPTER_REPO_ID", new_repo_id)
    print(f"      ADAPTER_REPO_ID = {new_repo_id}")

    # ── 5. Upload Space files ─────────────────────────────────────────────────
    print(f"\n[5/5] Uploading {len(SPACE_FILES)} files to Space...")
    for local_path, repo_path in SPACE_FILES:
        if not local_path.exists():
            print(f"      SKIP (not found): {local_path}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=new_space_id,
                repo_type="space",
                commit_message=f"migrate: {repo_path}",
            )
            print(f"      ✓ {repo_path}")
        except Exception as e:
            print(f"      ✗ {repo_path}: {e}")

    # ── 6. Update hf_ctl.py ───────────────────────────────────────────────────
    print("\n[6/6] Updating hf_ctl.py with new account IDs...")
    update_hf_ctl(new_user)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Migration complete!")
    print(f"  Space   : https://huggingface.co/spaces/{new_space_id}")
    print(f"  Adapter : https://huggingface.co/         {new_repo_id}")
    print(f"  URL     : https://{new_space_id.replace('/', '-').lower()}.hf.space")
    print("\nSpace is building — check status with:")
    print(f"  python3 hf_ctl.py status")
    print("Training will auto-start once the Space is RUNNING.")
    print("="*60)


if __name__ == "__main__":
    main()
