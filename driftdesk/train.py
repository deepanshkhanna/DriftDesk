#!/usr/bin/env python3
"""
train.py — DriftDesk GRPO training script (auto-run on HF Space startup)
Converted from driftdesk_grpo_hf_space.ipynb for headless execution.
"""
import os, sys, gc, csv, glob, json, torch
from pathlib import Path

# NOTE: expandable_segments is disabled — causes CUDA allocator asserts on Blackwell/RTX50xx

# ── HF login ─────────────────────────────────────────────────────────────────
from huggingface_hub import login, whoami, create_repo

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    import time as _login_time
    for _attempt in range(5):
        try:
            login(token=hf_token, add_to_git_credential=False)
            me = whoami()
            print(f"Logged in as: {me['name']}")
            break
        except Exception as _e:
            if _attempt < 4:
                print(f"HF login attempt {_attempt+1} failed ({_e}), retrying in 30s...")
                _login_time.sleep(30)
            else:
                print(f"WARNING: HF login failed after 5 attempts: {_e}")
else:
    print("WARNING: HF_TOKEN not set — checkpoints will NOT be pushed to Hub.")

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Auto-start env server if running as HF Job (no external server available) ────
import subprocess as _subp, time as _time, threading
_ENV_AUTOSTART = os.environ.get("AUTOSTART_ENV_SERVER", "0") == "1"
if _ENV_AUTOSTART:
    print('[env] Auto-starting DriftDesk env server on :8000 ...')
    _env_proc = _subp.Popen(
        [sys.executable, '-m', 'uvicorn', 'server.app:app', '--host', '0.0.0.0', '--port', '8000'],
        cwd=os.environ.get('DATA_DIR', '/app'),
        stdout=_subp.DEVNULL, stderr=_subp.STDOUT
    )
    # Wait up to 30s for server to be healthy
    import urllib.request as _ur
    for _i in range(30):
        try:
            _ur.urlopen('http://localhost:8000/health', timeout=2)
            print(f'[env] Server ready after {_i+1}s'); break
        except Exception:
            _time.sleep(1)
    else:
        print('[env] WARNING: server health check timed out')
    os.environ['DRIFTDESK_ENV_URL'] = 'http://localhost:8000'

import transformers, trl, peft, accelerate, bitsandbytes
print(f"transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")

# ── Configuration ─────────────────────────────────────────────────────────────
from huggingface_hub import whoami, create_repo

ENV_URL = os.environ.get(
    "DRIFTDESK_ENV_URL",
    "https://lokiontheloose-driftdesk.hf.space",
)
WS_URL = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

# Auto-detect container layout — the LokiOnTheLoose Space uses /home/user/app
# but the HelloOjasMutreja Space (Dockerfile WORKDIR=/app) uses /app. Pick
# whichever exists and is writable.
def _pick_data_dir() -> str:
    override = os.environ.get("DATA_DIR")
    if override:
        return override
    for cand in ("/app", "/home/user/app"):
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return os.getcwd()
DATA_DIR             = _pick_data_dir()
print(f"DATA_DIR resolved to: {DATA_DIR}", flush=True)
ADAPTER_SAVE_PATH    = f"{DATA_DIR}/driftdesk_adapter"
RESULTS_CSV          = f"{DATA_DIR}/grpo_training_results.csv"
BASELINE_CSV         = f"{DATA_DIR}/baseline_eval_results.csv"

MODEL_NAME           = "Qwen/Qwen2.5-3B-Instruct"
# A100 80GB has the headroom for r=32 (was r=16 on A10). Bigger LoRA = more
# capacity to learn drift adaptation. Override via env if instability appears.
LORA_R               = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA           = int(os.environ.get("LORA_ALPHA", str(LORA_R * 2)))
MAX_SEQ_LEN          = 2048
# A100 default: 8 generations × 8 batch = 64 rollouts/step (8x local throughput)
# Local RTX 5070 default: 4 × 2 = 8 rollouts/step
GRPO_NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "8"))
GRPO_LEARNING_RATE   = float(os.environ.get("LR", "5e-6"))
GRPO_TEMPERATURE     = float(os.environ.get("GRPO_TEMPERATURE", "1.2"))
GRPO_TOP_P           = float(os.environ.get("GRPO_TOP_P", "1.0"))
GRPO_STEPS           = int(os.environ.get("GRPO_STEPS", "500"))
GRPO_BATCH_SIZE      = int(os.environ.get("GRPO_BATCH_SIZE", "4"))
CURRICULUM_STAGE     = int(os.environ.get("CURRICULUM_STAGE", "1"))  # 0=no-drift, 1=cued-drift
# USE_QUANTIZATION=0 on A100 (80GB): skip 4-bit, use full bf16 for better throughput
USE_QUANTIZATION     = os.environ.get("USE_QUANTIZATION", "1") == "1"

HF_REPO_ID = None
if hf_token:
    try:
        username = whoami()["name"]
        HF_REPO_ID = f"{username}/driftdesk-grpo-adapter"
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
        print(f"Hub repo ready: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"Hub repo setup failed: {e}")

print(f"ENV_URL  : {ENV_URL}")
print(f"Model    : {MODEL_NAME}")
print(f"Rollouts : {GRPO_NUM_GENERATIONS * GRPO_BATCH_SIZE} per step")
print(f"Hub repo : {HF_REPO_ID or 'disabled'}")

# ── Load model (QLoRA on consumer GPUs; full bf16 on A100) ───────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if USE_QUANTIZATION:
    print('Loading model with 4-bit QLoRA (consumer GPU mode)...')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
else:
    print('Loading model in full bf16 (A100/large GPU mode)...')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# ── DriftDesk client + rollout helpers ───────────────────────────────────────
import websocket
from typing import List

WS_URL = ENV_URL.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws'


class DriftDeskSession:
    def __init__(self, timeout: float = 30.0):
        self._ws = websocket.create_connection(WS_URL, timeout=timeout)

    def _rr(self, msg: dict) -> dict:
        self._ws.send(json.dumps(msg))
        raw = self._ws.recv()
        if not raw:
            raise ConnectionError("Empty WebSocket response from environment server")
        return json.loads(raw).get('data', {})

    def reset(self, seed=None, curriculum_stage=None) -> dict:
        data = {}
        if seed is not None: data['seed'] = seed
        if curriculum_stage is not None: data['curriculum_stage'] = curriculum_stage
        return self._rr({'type': 'reset', 'data': data})

    def step(self, module: str, payload: dict) -> dict:
        return self._rr({'type': 'step', 'data': {'module': module, 'payload': payload}})

    def close(self):
        try:
            self._ws.send(json.dumps({'type': 'close'}))
            self._ws.close()
        except Exception:
            pass


def obs_to_messages(result: dict) -> list:
    obs = result.get('observation', result)
    policy = obs.get('policy_doc', '')
    tasks = obs.get('tasks', [])
    last_result = obs.get('last_result', {})
    step = obs.get('step_count', 0)
    pending = [t for t in tasks if not t.get('completed')]
    pending_str = '\n'.join(
        f"  [{t['priority']}] {t['module']}: {t['description']}" for t in pending
    )

    # Build schema hints so the model knows required payload fields.
    # Without this the model calls the right module but sends wrong fields → tc=0 always.
    SCHEMA_HINTS = {
        'airline_rebook':   '{"module": "airline_rebook", "payload": {"flight_id": "...", "passenger_name": "...", "new_date": "YYYY-MM-DD"}}',
        'bank_dispute':     '{"module": "bank_dispute", "payload": {"account_id": "...", "amount": 0.0, "merchant": "...", "description": "..."}}',
        'insurance_claim':  '{"module": "insurance_claim", "payload": {"claimant_id": "...", "incident_date": "YYYY-MM-DD", "amount": 0.0, "description": "..."}}',
    }
    pending_modules = [t['module'] for t in pending if t.get('module')]
    hints = [SCHEMA_HINTS[m] for m in pending_modules if m in SCHEMA_HINTS]
    schema_section = ''
    if hints:
        schema_section = '\nPayload schemas for pending tasks:\n' + '\n'.join(f'  {h}' for h in hints) + '\n'

    system = (
        'You are DriftDesk, an executive-assistant agent. You complete tasks by '
        'calling APIs. Reply with EXACTLY ONE JSON object on a single line, '
        'no prose, no markdown, no commentary. Schema:\n'
        '{"module": "<module_name>", "payload": {<fields>}}\n'
        'On a DRIFT error, copy missing_fields from the error into payload and retry. '
        'On a TRANSIENT_ERROR (http_status 500), retry with the SAME payload — do not change fields.'
    )
    user = (
        f'Step {step}. Active policy:\n{policy[:600]}\n\n'
        f'Pending tasks (lower priority number = more urgent):\n{pending_str}\n'
        f'{schema_section}\n'
        f'Last result:\n{json.dumps(last_result)[:400]}\n\n'
        'Extract field values from the task descriptions and respond with ONLY the JSON action.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def obs_to_prompt(result: dict) -> str:
    return tokenizer.apply_chat_template(
        obs_to_messages(result),
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_action(text: str):
    text = text.strip().split('<|im_end|>')[0].strip()
    start, end = text.find('{'), text.rfind('}') + 1
    if start == -1 or end == 0:
        return None, False
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None, False
    if not isinstance(obj, dict) or 'module' not in obj or 'payload' not in obj:
        return obj, True
    return obj, True


_IM_END_ID = tokenizer.convert_tokens_to_ids('<|im_end|>')
_EOS_IDS = [i for i in [tokenizer.eos_token_id, _IM_END_ID] if i is not None and i >= 0]


def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=MAX_SEQ_LEN - max_new_tokens).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=_EOS_IDS,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


print('WebSocket client + rollout helpers ready.')

# ── Oracle baseline eval ──────────────────────────────────────────────────────
import subprocess
if not os.path.exists(BASELINE_CSV):
    print("Running oracle baseline eval...")
    result = subprocess.run(
        [sys.executable, "eval_harness.py",
         "--env-url", ENV_URL,
         "--agent", "rule_based",
         "--out-csv", BASELINE_CSV,
         "--curriculum-stage", str(CURRICULUM_STAGE)],
        cwd=DATA_DIR,
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-800:])
else:
    print(f"Baseline CSV already exists: {BASELINE_CSV}")

# ── GRPO reward function ──────────────────────────────────────────────────────
training_log = []
# P0 (iter 2): align with env MAX_STEPS=10 so `done=True` can fire and
# `compute_episode_reward` actually returns a non-None scalar. With 5 here,
# 3 tasks + drift events never finish → reward stays at FORMAT_BONUS forever.
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", "10"))

MAX_WS_RETRIES = 2

def grpo_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    import time
    rewards = []
    # TRL passes dataset columns with their exact column name ('seed', not 'seeds')
    seed_list = kwargs.get('seed', [None] * len(completions))
    if not isinstance(seed_list, (list, tuple)):
        seed_list = [seed_list] * len(completions)
    FORMAT_BONUS = 0.05

    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        reward = 0.0
        for attempt in range(MAX_WS_RETRIES + 1):
            sess = None
            try:
                sess = DriftDeskSession(timeout=45.0)
                # Use a per-rollout randomised seed so that within a GRPO group
                # (N completions, same prompt/base-seed) each completion faces a
                # distinct environment state.  Without this, all rollouts converge
                # to identical episodes → reward_std = 0 → zero gradient.
                import random as _rnd
                rollout_seed = _rnd.randint(0, 2**31 - 1)
                result = sess.reset(seed=rollout_seed, curriculum_stage=CURRICULUM_STAGE)

                action, is_valid_json = parse_action(completion)
                shape_ok = isinstance(action, dict) and 'module' in action and 'payload' in action
                partial = (FORMAT_BONUS * (0.5 + 0.5 * float(shape_ok))) if is_valid_json else 0.0

                if not shape_ok:
                    reward = partial
                    break

                result = sess.step(action['module'], action.get('payload', {}))

                for _ in range(MAX_EPISODE_STEPS - 1):
                    if result.get('done'):
                        break
                    next_prompt = obs_to_prompt(result)
                    gen = generate_action(model, tokenizer, next_prompt, max_new_tokens=128)
                    next_action, _ = parse_action(gen)
                    if not (isinstance(next_action, dict)
                            and 'module' in next_action and 'payload' in next_action):
                        break
                    result = sess.step(next_action['module'], next_action.get('payload', {}))

                episode_reward = float(result.get('reward') or 0.0)

                # P0 (iter 4) F4-client: reshape reward client-side from
                # observation['reward_components']. The server reward is
                # dominated by policy_grounding=1.0 floor + loop_penalty,
                # which collapses every successful episode to ~0.05 — no
                # gradient. Re-weight to amplify the components that vary
                # with policy quality (tc, dr) and zero out constants.
                obs = result.get('observation', result) if isinstance(result, dict) else {}
                comps = (obs.get('reward_components') or {}) if isinstance(obs, dict) else {}
                tc = float(comps.get('task_completion', 0.0) or 0.0)
                dr = float(comps.get('drift_recovery', 0.0) or 0.0)
                eff = float(comps.get('efficiency', 0.0) or 0.0)
                ps = float(comps.get('priority_score', 0.0) or 0.0)
                # NOTE: pg and loop_penalty intentionally excluded — pg is
                # always 1.0 (no signal), loop_penalty is variance-suppressing
                # negative on every rollout.
                shaped = 1.5 * tc + 1.2 * dr + 0.3 * eff + 0.3 * ps
                # Use shaped reward when episode actually completed; otherwise
                # fall back to raw episode_reward (handles done=False edge case).
                use_shaped = bool(result.get('done')) and (tc + dr + eff + ps) > 0.0
                eff_episode_reward = shaped if use_shaped else episode_reward

                # Module-match bonus: reward the model for calling a module
                # that is actually in the current task list.  This creates
                # reward variance even when tc=0 (no full completion), giving
                # GRPO a gradient signal before the model learns to finish tasks.
                tasks_list = (obs.get('tasks', []) or []) if isinstance(obs, dict) else []
                task_modules = set(t.get('module', '') for t in tasks_list if t.get('module'))
                avail_mods = set((obs.get('available_modules', []) or []) if isinstance(obs, dict) else [])
                action_module = action.get('module', '') if isinstance(action, dict) else ''
                module_bonus = 0.04 if (action_module and action_module in task_modules) \
                    else (0.01 if (action_module and action_module in avail_mods) else 0.0)

                reward = eff_episode_reward + partial + module_bonus
                training_log.append({'reward': reward,
                                     'episode_reward': episode_reward,
                                     'shaped': shaped if use_shaped else None,
                                     'tc': tc, 'dr': dr, 'eff': eff, 'ps': ps,
                                     'partial': partial,
                                     'module_bonus': module_bonus,
                                     'done': result.get('done')})
                break
            except Exception as e:
                if attempt < MAX_WS_RETRIES:
                    print(f'  [reward_fn] Retry {attempt + 1} for completion {i}: {e}')
                    time.sleep(2.0 * (attempt + 1))
                else:
                    print(f'  [reward_fn] Error for completion {i}: {e}')
            finally:
                if sess:
                    sess.close()
        rewards.append(reward)

    # P0 (iter 2/4) diagnostic: surface reward distribution + component means.
    nz = sum(1 for r in rewards if r > 0.06)  # > FORMAT_BONUS+epsilon
    # Pull last `len(rewards)` entries from training_log for this batch
    recent = [t for t in training_log[-len(rewards):] if isinstance(t, dict)]
    tc_mean = sum(t.get('tc', 0.0) for t in recent) / max(len(recent), 1)
    dr_mean = sum(t.get('dr', 0.0) for t in recent) / max(len(recent), 1)
    mb_mean = sum(t.get('module_bonus', 0.0) for t in recent) / max(len(recent), 1)
    print(f'  [reward_fn] batch n={len(rewards)} '
          f'mean={sum(rewards)/max(len(rewards),1):.4f} '
          f'min={min(rewards):.3f} max={max(rewards):.3f} '
          f'scored={nz}/{len(rewards)} '
          f'tc={tc_mean:.3f} dr={dr_mean:.3f} mb={mb_mean:.3f}', flush=True)

    return rewards


print('GRPO reward function ready.')

# ── SFT warm-up ───────────────────────────────────────────────────────────────
SFT_SENTINEL = f"{DATA_DIR}/driftdesk_sft_warmup/.done"
# Set SKIP_SFT=1 to bypass SFT warmup (e.g. when going straight to GRPO)
_skip_sft = os.environ.get("SKIP_SFT", "0") == "1"
if _skip_sft and not os.path.exists(SFT_SENTINEL):
    os.makedirs(f'{DATA_DIR}/driftdesk_sft_warmup', exist_ok=True)
    open(SFT_SENTINEL, 'w').close()
    print('SKIP_SFT=1 — skipping SFT warmup.')
if not os.path.exists(SFT_SENTINEL):
    print("Running SFT warm-up (cold-start fix)...")
    import random as _rnd
    from torch.optim import AdamW

    sys.path.insert(0, DATA_DIR)
    from eval_harness import RuleBasedAgent

    records = []
    for s in range(3000, 3200):
        try:
            sess = DriftDeskSession()
            result = sess.reset(seed=s, curriculum_stage=CURRICULUM_STAGE)
            agent = RuleBasedAgent()
            agent.reset(result.get('observation', result).get('tasks', []))
            for _ in range(10):
                action = agent.act(result)
                if action is None:
                    break
                prompt_text = obs_to_prompt(result)
                target = json.dumps(action) + '<|im_end|>'
                records.append(prompt_text + target)
                result = sess.step(action['module'], action.get('payload', {}))
                if result.get('done'):
                    break
            sess.close()
        except Exception as e:
            print(f'  seed {s}: {e}')

    print(f'Collected {len(records)} SFT (prompt, action) pairs')
    gc.collect()
    torch.cuda.empty_cache()

    # Manual SFT loop — avoids Trainer/collator bugs with bnb on RTX 50xx
    model.train()
    sft_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    SFT_EPOCHS = 3
    SFT_BATCH  = 2
    MAX_SFT_LEN = 512

    _rnd.shuffle(records)
    total_loss = 0.0
    n_steps = 0
    for epoch in range(SFT_EPOCHS):
        _rnd.shuffle(records)
        for b_start in range(0, len(records), SFT_BATCH):
            batch_texts = records[b_start:b_start + SFT_BATCH]
            enc = tokenizer(batch_texts, return_tensors='pt', padding=True,
                            truncation=True, max_length=MAX_SFT_LEN).to(model.device)
            labels = enc['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            if (labels != -100).sum() == 0:
                continue  # skip all-pad batch
            try:
                out = model(**enc, labels=labels)
                loss = out.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                sft_optimizer.step()
                sft_optimizer.zero_grad()
                total_loss += loss.item()
                n_steps += 1
                if n_steps % 50 == 0:
                    print(f'  [sft] epoch={epoch+1} step={n_steps} loss={total_loss/n_steps:.4f}', flush=True)
            except RuntimeError as e:
                print(f'  [sft] skipped batch: {e}')
                sft_optimizer.zero_grad()
                torch.cuda.empty_cache()

    del sft_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f'SFT warm-up complete. Steps={n_steps} avg_loss={total_loss/max(n_steps,1):.4f}')

    os.makedirs(f'{DATA_DIR}/driftdesk_sft_warmup', exist_ok=True)
    open(SFT_SENTINEL, 'w').close()
else:
    print('SFT warm-up already done (sentinel found).')

# ── Build training dataset ────────────────────────────────────────────────────
from datasets import Dataset

def build_grpo_dataset(n_prompts: int = 200, seed_offset: int = 0) -> Dataset:
    records = []
    for i in range(n_prompts):
        try:
            sess = DriftDeskSession()
            result = sess.reset(seed=seed_offset + i, curriculum_stage=CURRICULUM_STAGE)
            prompt = obs_to_prompt(result)
            records.append({"prompt": prompt, "seed": seed_offset + i})
            sess.close()
        except Exception as e:
            print(f"  [dataset] seed {seed_offset + i} failed: {e}")
    print(f"Built dataset: {len(records)} prompts")
    return Dataset.from_list(records)

train_dataset = build_grpo_dataset(n_prompts=200, seed_offset=2000)  # F-SFT-Boost overnight: 200 prompts
print(train_dataset)

# ── Callbacks ─────────────────────────────────────────────────────────────────
from transformers import TrainerCallback

EVAL_EVERY = 50

class DeterministicEvalCallback(TrainerCallback):
    def __init__(self, csv_path=f"{DATA_DIR}/grpo_eval_during_training.csv",
                 n_eval=5, seed_start=1000):
        self.csv_path = csv_path
        self.n_eval = n_eval
        self.seed_start = seed_start
        self._written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % EVAL_EVERY != 0:
            return
        rewards, drs, tcs = [], [], []
        for s in range(self.seed_start, self.seed_start + self.n_eval):
            try:
                sess = DriftDeskSession()
                result = sess.reset(seed=s, curriculum_stage=CURRICULUM_STAGE)
                for _ in range(MAX_EPISODE_STEPS):
                    if result.get('done'): break
                    prompt = obs_to_prompt(result)
                    text = generate_action(model, tokenizer, prompt, max_new_tokens=128)
                    action, _ = parse_action(text)
                    if not (isinstance(action, dict) and 'module' in action and 'payload' in action):
                        break
                    result = sess.step(action['module'], action.get('payload', {}))
                ep_r = float(result.get('reward') or 0.0)
                obs = result.get('observation', result)
                comps = (obs.get('reward_components') or {})
                rewards.append(ep_r)
                drs.append(comps.get('drift_recovery', 0.0) or 0.0)
                tcs.append(comps.get('task_completion', 0.0) or 0.0)
                sess.close()
            except Exception as e:
                print(f'  [eval cb] seed {s}: {e}')
        if not rewards: return
        row = {'step': state.global_step,
               'eval_reward_mean': sum(rewards)/len(rewards),
               'eval_drift_recovery_mean': sum(drs)/len(drs) if drs else 0.0,
               'eval_task_completion_mean': sum(tcs)/len(tcs) if tcs else 0.0,
               'n': len(rewards)}
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not self._written:
                w.writeheader()
                self._written = True
            w.writerow(row)
        print(f'  [eval cb step {state.global_step}] reward={row["eval_reward_mean"]:.3f}')


HUB_PUSH_EVERY = 5  # DEMO: push every 5 steps so results appear quickly

class HubPushCallback(TrainerCallback):
    def __init__(self, repo_id: str, output_dir: str, push_every: int = HUB_PUSH_EVERY):
        self.repo_id    = repo_id
        self.output_dir = output_dir
        self.push_every = push_every

    def on_save(self, args, state, control, **kwargs):
        if not self.repo_id:
            return
        if state.global_step % self.push_every != 0:
            return
        from huggingface_hub import upload_folder
        tmp = self.output_dir + "/hub_push_tmp"
        try:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            upload_folder(
                repo_id=self.repo_id,
                folder_path=tmp,
                repo_type="model",
                commit_message=f"checkpoint step {state.global_step}",
                ignore_patterns=["optimizer.pt", "rng_state*", "scheduler.pt"],
            )
            print(f"  [hub push step {state.global_step}] -> {self.repo_id}")
        except Exception as e:
            print(f"  [hub push] WARN: {e}")


hub_push_cb = (HubPushCallback(
    repo_id=HF_REPO_ID,
    output_dir=f"{DATA_DIR}/driftdesk_grpo_output",
) if HF_REPO_ID else None)

# ── Early-abort watchdog ──────────────────────────────────────────────────────
TC_ZERO_ABORT_AFTER = int(os.environ.get("TC_ZERO_ABORT_AFTER", "150"))

class EarlyAbortCallback(TrainerCallback):
    """Kill training if tc=0 for TC_ZERO_ABORT_AFTER consecutive steps."""
    def __init__(self):
        self._tc_zero_streak = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not training_log:
            return
        # Collect tc values from last GRPO_BATCH_SIZE * GRPO_NUM_GENERATIONS entries
        window = [t for t in training_log[-GRPO_BATCH_SIZE * GRPO_NUM_GENERATIONS:]
                  if isinstance(t, dict)]
        if not window:
            return
        tc_mean = sum(t.get('tc', 0.0) for t in window) / len(window)
        if tc_mean == 0.0:
            self._tc_zero_streak += 1
            print(f'  [watchdog] tc=0 streak={self._tc_zero_streak}/{TC_ZERO_ABORT_AFTER}', flush=True)
            if self._tc_zero_streak >= TC_ZERO_ABORT_AFTER:
                print(f'\n{"="*60}', flush=True)
                print(f'FATAL: tc=0 for {TC_ZERO_ABORT_AFTER} consecutive steps.', flush=True)
                print(f'Model is not completing tasks — stopping to avoid wasted compute.', flush=True)
                print(f'{"="*60}\n', flush=True)
                control.should_training_stop = True
        else:
            self._tc_zero_streak = 0
            print(f'  [watchdog] tc={tc_mean:.3f} — learning signal detected ✓', flush=True)

# ── GRPO training ─────────────────────────────────────────────────────────────
from trl import GRPOConfig, GRPOTrainer

OUTPUT_DIR = f"{DATA_DIR}/driftdesk_grpo_output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

checkpoints = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
    key=lambda p: int(p.rsplit("-", 1)[-1])
)
resume_checkpoint = checkpoints[-1] if checkpoints else None
if resume_checkpoint:
    resume_step = int(resume_checkpoint.rsplit("-", 1)[-1])
    print(f"Resuming from checkpoint: {resume_checkpoint} (step {resume_step})")
    if resume_step >= GRPO_STEPS:
        raise RuntimeError(
            f"GRPO_STEPS={GRPO_STEPS} <= resume_step={resume_step}. "
            f"Trainer would do zero gradient steps. "
            f"Set env var GRPO_STEPS to a larger value (e.g. {resume_step + 30})."
        )
    print(f"Will train {GRPO_STEPS - resume_step} new steps -> target step {GRPO_STEPS}")
else:
    print(f"No checkpoint found — starting from scratch. Target: {GRPO_STEPS} steps.")

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    max_steps=GRPO_STEPS,
    per_device_train_batch_size=GRPO_BATCH_SIZE,
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "2")),
    gradient_checkpointing=not USE_QUANTIZATION,  # not needed when full bf16 on A100
    bf16=not USE_QUANTIZATION,        # enable native bf16 training on A100
    learning_rate=GRPO_LEARNING_RATE,
    num_generations=GRPO_NUM_GENERATIONS,
    max_completion_length=int(os.environ.get("MAX_COMPLETION_LEN", "128")),
    temperature=GRPO_TEMPERATURE,
    top_p=GRPO_TOP_P,
    beta=0.04,
    logging_steps=1,
    save_steps=int(os.environ.get("SAVE_STEPS", "5")),
    save_total_limit=None,
    seed=42,
    report_to="none",
    dataloader_num_workers=0,
)

_callbacks = [DeterministicEvalCallback(), EarlyAbortCallback()]
if hub_push_cb:
    _callbacks.append(hub_push_cb)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=grpo_reward_fn,
    args=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    callbacks=_callbacks,
)

_csv_path = RESULTS_CSV
_csv_written = os.path.exists(_csv_path) and os.path.getsize(_csv_path) > 0
_orig_log = trainer.log

def _log_to_csv(logs, *args, **kwargs):
    global _csv_written
    _orig_log(logs, *args, **kwargs)
    with open(_csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=logs.keys())
        if not _csv_written:
            w.writeheader()
            _csv_written = True
        w.writerow(logs)

trainer.log = _log_to_csv

print(f"Starting GRPO training -> {GRPO_STEPS} steps")
trainer.train(resume_from_checkpoint=resume_checkpoint)
print(f"Training complete at step {trainer.state.global_step}")

# ── Save final adapter ────────────────────────────────────────────────────────
Path(ADAPTER_SAVE_PATH).mkdir(parents=True, exist_ok=True)
model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
print(f"Adapter saved to {ADAPTER_SAVE_PATH}")

if HF_REPO_ID:
    from huggingface_hub import upload_folder
    upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=ADAPTER_SAVE_PATH,
        repo_type="model",
        commit_message=f"final adapter — step {trainer.state.global_step}",
    )
    print(f"Final adapter pushed to https://huggingface.co/{HF_REPO_ID}")

print("Done.")
