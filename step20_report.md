# DriftDesk GRPO — Step 20 Evaluation Report

**Date:** April 25, 2026  
**Model:** Qwen/Qwen2.5-3B-Instruct + QLoRA (r=16, α=32)  
**Training:** GRPO, checkpoint-20 (current batch, resumed from batch-0 checkpoint-20)  
**Eval set:** 50 deterministic episodes, seeds 1000–1049, curriculum stage 1 (cued drift)  
**Environment:** DriftDesk local server (`http://localhost:8000`)  

---

## Summary

| Metric | Step-20 GRPO | Rule-Based Baseline | Δ |
|--------|:-----------:|:-------------------:|:--:|
| **Mean reward** | 0.138 | 0.747 | **-0.609** |
| **Task completion** | 0.380 | 1.000 | **-0.620** |
| **Drift recovery** | 0.155 | 0.333 | **-0.178** |
| **Efficiency** | 0.151 | 1.000 | **-0.849** |
| **Loop penalty** | -0.241 | 0.000 | **-0.241** |
| **Policy grounding** | **1.000** | 1.000 | 0.000 |
| **Priority score** | 0.580 | 1.000 | -0.420 |

The model is significantly behind the rule-based baseline at step 20. This is expected at this stage of GRPO training.

---

## Detailed Metrics

### Reward

| Stat | Value |
|------|-------|
| Mean | 0.138 |
| Min | -0.160 |
| Max | 0.807 |
| Std dev | 0.269 |

#### Reward Distribution (50 episodes)

| Bucket | Count | % |
|--------|------:|--:|
| < 0 (failure) | 16 | 32% |
| 0.0 – 0.2 | 15 | 30% |
| 0.2 – 0.5 | 13 | 26% |
| 0.5 – 0.75 | 5 | 10% |
| ≥ 0.75 (near-baseline) | 1 | 2% |

62% of episodes scored below 0.2, and 32% returned a negative reward — primarily due to the loop penalty. Only 1 episode (~2%) reached baseline-level performance.

---

### Task Completion

| tc score | Episodes | Notes |
|----------|------:|-------|
| 0.00 (failed all tasks) | 18 | 36% |
| 0.33 (1/3 tasks) | 13 | 26% |
| 0.67 (2/3 tasks) | 13 | 26% |
| 1.00 (all tasks) | 6 | 12% |

The model completed **all tasks in only 6/50 episodes (12%)** vs the baseline's perfect 100%. It shows partial task completion ability — completing at least 1 task in 64% of episodes.

---

### Drift Recovery

| Stat | Value |
|------|-------|
| Mean drift recovery | 0.155 |
| Episodes with drift events | 50 / 50 (all) |
| Episodes with successful recovery | 14 / 50 (28%) |
| Mean drift events per episode | 1.50 |
| Mean successful recoveries per episode | 0.30 |

All episodes had at least one schema drift event (curriculum stage 1). The model recovered successfully in **14 of 50 episodes** but generally failed to adapt its actions after receiving drift error feedback.

---

### Loop Penalty

| Stat | Value |
|------|-------|
| Mean loop penalty | -0.241 |
| Min (worst) | -0.300 |
| Max (no penalty) | 0.000 |
| Std dev | 0.094 |

The loop penalty is the **primary performance bottleneck**. A mean of -0.241 out of a possible 0.0 indicates the model frequently repeats the same action instead of adapting — the core behavior GRPO is designed to train away. This is expected at 20 steps and should improve significantly with more training.

---

### Efficiency & Steps

| Stat | Value |
|------|-------|
| Mean efficiency | 0.151 |
| Mean steps per episode | 9.24 |
| Min steps | 3 |
| Max steps | 10 (episode limit) |

The model is hitting the 10-step episode limit in many episodes, which drives down efficiency. The rule-based agent typically solves tasks in 3–4 steps.

---

### Policy Grounding

**1.000 / 1.000** — Perfect score across all 50 episodes.

The model consistently outputs **valid JSON actions** matching the expected `{"module": ..., "payload": {...}}` format. This is a strong signal: the model has internalized the output format from SFT warm-up. The problem is not format compliance but action correctness (wrong field values, not adapting to drift errors).

---

## Notable Episodes

| Episode | Seed | Reward | tc | dr | Notes |
|---------|------|--------|----|----|-------|
| 28 | 1027 | **0.807** | 1.00 | 0.50 | Best episode — full task + drift recovery |
| 40 | 1039 | 0.740 | 1.00 | 0.50 | Full task, recovered from drift |
| 2 | 1001 | 0.596 | 1.00 | 0.25 | Full task completion |
| 11 | 1010 | 0.597 | 1.00 | 0.00 | Full task, no drift recovery |
| 5 | 1004 | -0.160 | 0.00 | 0.00 | Worst group — loop, no completion |

---

## Training Context at Step 20

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct |
| LoRA rank | 16 |
| Trainable params | 29.9M / 3.1B (0.96%) |
| Training steps completed | 20 |
| Target steps (this run) | 50 |
| GRPO batch size | 4 |
| Rollouts per step | 4 |
| Learning rate at step 20 | ~1.7e-6 (decaying) |
| Training loss (last logged) | ~0.00003 |

---

## Diagnosis & Expected Trajectory

### Root causes of low performance

1. **Loop penalty (-0.241 mean):** The model repeats the same failing action rather than modifying its payload after a DRIFT or VALIDATION error. GRPO will penalize this directly — expect significant improvement as training progresses.

2. **Low efficiency (0.151):** Most episodes hit the 10-step limit. The model is "trying" (outputting actions) but not converging on correct payloads quickly.

3. **Drift recovery gap (0.155 vs 0.333):** The model rarely uses `missing_fields` from the error response to update its payload schema. This is a reasoning failure that GRPO should address by rewarding the 14 successful recovery episodes more strongly.

### What's working

- **Perfect policy grounding (1.0):** Valid JSON output in every episode — SFT warm-up succeeded.
- **Partial task completion (64% complete ≥1 task):** The model knows the task structure.
- **Best episodes near baseline (0.807):** Proves the model is capable when reasoning correctly.

### Expected improvement by step 50

Based on GRPO dynamics and the fact that the reward signal is non-zero (model is receiving gradients), expect:
- Loop penalty to decrease toward 0.0 (model learns to change actions after errors)
- Task completion to increase from 38% toward 60–70%
- Mean reward to move from 0.138 toward 0.4–0.5

---

## Next Steps

1. **Run step-50 eval** after current HF training completes (~18:55 UTC)
2. **Compare step-20 vs step-50** — quantify GRPO improvement per 30 steps
3. If loop penalty persists at step 50, consider increasing `beta` (KL penalty) or adding an explicit loop-avoidance reward component
4. At step 100+, re-evaluate drift recovery — this requires multi-turn reasoning which takes longer to learn

---

*Generated by GitHub Copilot from `step20_eval_results.csv` (50 episodes, exit code 1 due to `None` in `no_spurious_rewrite` field — all episode data was written successfully before the summary crash).*
