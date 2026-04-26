#!/bin/bash
# on_startup.sh — runs as root at IMAGE BUILD TIME (Dockerfile RUN instruction).
# Only does directory setup. Training is launched at runtime by start_server.sh.

mkdir -p /home/user/app/driftdesk_grpo_output
mkdir -p /home/user/app/driftdesk_sft_warmup
chown -R user:user /home/user/app/driftdesk_grpo_output /home/user/app/driftdesk_sft_warmup 2>/dev/null || true
echo "[build] on_startup.sh complete — training will start at runtime via start_server.sh"
