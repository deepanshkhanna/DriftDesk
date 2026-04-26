---
title: DriftDesk Demo
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.6.0
python_version: 3.11
app_file: app.py
pinned: false
hardware: t4-small
---

# DriftDesk Demo

Public demo of a Qwen2.5-3B + LoRA agent trained with **GRPO** to recover from
silent schema drift in a multi-task executive-assistant environment.

- Adapter: [HelloOjasMutreja/driftdesk-grpo-adapter](https://huggingface.co/HelloOjasMutreja/driftdesk-grpo-adapter)
- Training Space: [HelloOjasMutreja/driftdesk-training](https://huggingface.co/spaces/HelloOjasMutreja/driftdesk-training)
- Env server: `lokiontheloose-driftdesk.hf.space`

Pick a seed → run a full 10-step episode → see the trajectory and reward
decomposition (task completion, drift recovery, efficiency, priority score).
