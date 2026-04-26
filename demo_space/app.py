"""
DriftDesk Demo Space — interactive storytelling UI.
Each episode plays out as a narrative: briefing → agent reasoning → actions → outcome.
"""
from __future__ import annotations
import json, os, html as _hl
from typing import Generator
import gradio as gr
import torch

BASE_MODEL   = os.environ.get("BASE_MODEL",   "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "HelloOjasMutreja/driftdesk-grpo-adapter")
ENV_URL      = os.environ.get("DRIFTDESK_ENV_URL", "https://lokiontheloose-driftdesk.hf.space")
WS_URL       = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
MAX_STEPS    = int(os.environ.get("MAX_EPISODE_STEPS", "10"))

# ── Model state ───────────────────────────────────────────────────────────────
_model = _tokenizer = None

MODULE_META = {
    "airline_rebook":  {"icon": "✈️", "label": "Flight Rebooking",  "mod_cls": "mod-air"},
    "bank_dispute":    {"icon": "🏦", "label": "Bank Dispute",       "mod_cls": "mod-bank"},
    "insurance_claim": {"icon": "🛡️", "label": "Insurance Claim",   "mod_cls": "mod-ins"},
}
SCHEMA_HINTS = {
    "airline_rebook":   {"flight_id": "...", "passenger_name": "...", "new_date": "YYYY-MM-DD"},
    "bank_dispute":     {"account_id": "...", "amount": 0.0, "merchant": "...", "description": "..."},
    "insurance_claim":  {"claimant_id": "...", "incident_date": "YYYY-MM-DD", "amount": 0.0, "description": "..."},
}

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    try:
        d = snapshot_download(repo_id=ADAPTER_REPO, repo_type="model")
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            base = PeftModel.from_pretrained(base, d)
            print("[demo] GRPO adapter loaded")
        else:
            print("[demo] No adapter weights yet — base model only")
    except Exception as ex:
        print(f"[demo] Adapter unavailable ({ex}); base model only")
    base.eval()
    _model, _tokenizer = base, tok
    return base, tok

def _ws_call(ws, msg: dict) -> dict:
    ws.send(json.dumps(msg))
    raw = ws.recv()
    return json.loads(raw).get("data", {}) if raw else {}

def obs_to_prompt(result: dict) -> str:
    obs   = result.get("observation", result)
    tasks = obs.get("tasks", []) or []
    policy = (obs.get("policy_doc", "") or "")[:400]
    pending = [t for t in tasks if not t.get("completed")]
    pending_str = "\n".join(
        f"  [{t['priority']}] {t['module']}: {t['description']}" for t in pending
    ) or "  (none)"
    last  = obs.get("last_result", {}) or {}
    step  = obs.get("step_count", 0)
    hints = []
    for t in pending:
        m = t.get("module")
        if m in SCHEMA_HINTS:
            hints.append(f'  {{"module": "{m}", "payload": {json.dumps(SCHEMA_HINTS[m])}}}')
    schema_section = "\nPayload schemas:\n" + "\n".join(hints) + "\n" if hints else ""
    return (
        f"<|im_start|>system\n{policy}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Step {step} | Pending tasks:\n{pending_str}\n"
        f"Last result: {json.dumps(last)}\n"
        f"{schema_section}"
        f"Extract field values from the task descriptions and respond with ONLY the JSON action.\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

@torch.inference_mode()
def _generate(model, tok, prompt: str, max_new: int = 128) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new, temperature=0.7, do_sample=True,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)

# ── HTML rendering ─────────────────────────────────────────────────────────────
def e(s: object) -> str:
    return _hl.escape(str(s))

STYLE = """<style>
.dd{font-family:'SF Mono','Fira Code',Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;border-radius:12px;overflow:hidden;border:1px solid #21262d}
.dd-top{background:#161b22;border-bottom:1px solid #30363d;padding:10px 16px;display:flex;align-items:center;gap:12px;font-size:12px;flex-wrap:wrap}
.dd-logo{font-weight:700;color:#58a6ff;letter-spacing:2.5px;font-size:13px}
.dd-badge{padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;white-space:nowrap}
.b-run{background:#0d2d1a;color:#3fb950;border:1px solid #238636}
.b-idle{background:#1c2128;color:#8b949e;border:1px solid #484f58}
.b-done{background:#0d2d1a;color:#3fb950;border:1px solid #238636}
.b-err{background:#3d1a1a;color:#f85149;border:1px solid #da3633}
.b-load{background:#0d1f3a;color:#58a6ff;border:1px solid #1f6feb}
.dd-sep{color:#484f58}
.dd-body{padding:14px;display:flex;flex-direction:column;gap:14px;max-height:540px;overflow-y:auto;scroll-behavior:smooth}
.dd-body::-webkit-scrollbar{width:5px}
.dd-body::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
.dd-sh{color:#8b949e;font-size:10px;letter-spacing:2.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;margin-bottom:8px}
.dd-sh::after{content:'';flex:1;height:1px;background:#21262d}
.dd-tasks{display:flex;flex-direction:column;gap:7px}
.dd-task{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 12px;display:flex;gap:10px;position:relative;transition:opacity .4s,border-color .4s}
.dd-task-done{opacity:.5;border-color:#238636}
.dd-task-done::after{content:'✅';position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:16px}
.mod-air{border-left:3px solid #58a6ff!important}
.mod-bank{border-left:3px solid #d29922!important}
.mod-ins{border-left:3px solid #3fb950!important}
.dd-tname{font-weight:600;font-size:12px;margin-bottom:3px;display:flex;align-items:center;gap:6px}
.dd-tdesc{font-size:11px;color:#8b949e;line-height:1.5}
.dd-pri{font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px;flex-shrink:0}
.pri-H{background:#3d1a1a;color:#f85149}
.pri-M{background:#2d2a16;color:#d29922}
.pri-L{background:#0d2128;color:#58a6ff}
.dd-feed{display:flex;flex-direction:column;gap:8px}
.dd-step{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.dd-step-new{animation:slin .35s ease both}
@keyframes slin{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.dd-shdr{padding:8px 12px;display:flex;align-items:center;gap:8px;background:#1c2128;border-bottom:1px solid #21262d;font-size:11px}
.dd-snum{color:#8b949e;font-weight:700;letter-spacing:1.5px;font-size:10px}
.dd-smod{font-weight:600;font-size:11px}
.smod-air{color:#58a6ff}.smod-bank{color:#d29922}.smod-ins{color:#3fb950}
.dd-srew{margin-left:auto;font-weight:700;font-size:12px}
.dd-sbody{padding:10px 12px;font-size:11px}
.dd-payload{background:#0d1117;border:1px solid #21262d;border-radius:5px;padding:7px 10px;margin:6px 0;color:#a5d6ff;overflow-x:auto}
.dd-payload pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:11px}
.dd-res{margin-top:6px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.res-ok{color:#3fb950;font-weight:700}.res-fail{color:#f85149;font-weight:700}
.dd-stat{color:#8b949e;font-size:10px}
.dd-thnk{padding:10px 12px;color:#8b949e;font-size:11px;display:flex;align-items:center;gap:6px}
.dd-dots span{animation:blk 1.4s infinite;display:inline-block}
.dd-dots span:nth-child(2){animation-delay:.22s}
.dd-dots span:nth-child(3){animation-delay:.44s}
@keyframes blk{0%,80%,100%{opacity:.15}40%{opacity:1}}
.dd-load{padding:14px;display:flex;align-items:center;gap:10px;color:#58a6ff;font-size:12px;background:#161b22;border:1px solid #1f6feb;border-radius:8px}
.dd-spin{width:14px;height:14px;border:2px solid #21262d;border-top-color:#58a6ff;border-radius:50%;animation:spn .7s linear infinite;flex-shrink:0}
@keyframes spn{to{transform:rotate(360deg)}}
.dd-score{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 14px}
.dd-scr-lbl{font-size:10px;color:#8b949e;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.dd-scr-row{display:flex;align-items:center;gap:10px}
.dd-scr-bar{flex:1;height:7px;background:#21262d;border-radius:4px;overflow:hidden}
.dd-scr-fill{height:100%;background:linear-gradient(90deg,#1a7f37,#3fb950);border-radius:4px;transition:width .6s ease}
.dd-scr-val{font-weight:700;font-size:15px;color:#3fb950;min-width:60px;text-align:right}
.dd-sum{background:#0d2d1a;border:1px solid #238636;border-radius:8px;padding:18px;text-align:center;animation:slin .4s ease both}
.dd-sum h3{color:#3fb950;margin:0 0 4px;font-size:17px;letter-spacing:1px}
.dd-sum-sub{color:#8b949e;font-size:12px;margin-bottom:14px}
.dd-sum-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.dd-sstat{background:#0d1117;border-radius:6px;padding:10px}
.dd-sstat .val{font-size:22px;font-weight:700;color:#3fb950}
.dd-sstat .key{font-size:10px;color:#8b949e;margin-top:2px;letter-spacing:1px;text-transform:uppercase}
.dd-err-box{background:#2d1616;border:1px solid #da3633;border-radius:8px;padding:12px;color:#f85149;font-size:12px}
.dd-welcome{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:28px;text-align:center;color:#8b949e}
.dd-welcome .big{font-size:42px;margin-bottom:10px}
.dd-welcome .title{color:#c9d1d9;font-size:15px;font-weight:600;margin-bottom:8px}
.dd-welcome .sub{font-size:12px;line-height:1.7;max-width:380px;margin:0 auto}
.dd-welcome kbd{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:1px 6px;font-size:11px;color:#c9d1d9}
.dd-narr{color:#8b949e;font-size:11px;font-style:italic;margin-bottom:6px}
</style>"""

# ── HTML helpers ───────────────────────────────────────────────────────────────

def _badge(status: str) -> str:
    labels = {"run": "● RUNNING", "idle": "○ IDLE", "done": "✓ COMPLETE",
              "err": "✗ ERROR", "load": "⟳ LOADING"}
    return f'<span class="dd-badge b-{status}">{labels.get(status, status)}</span>'

def _pri_badge(p: str) -> str:
    k = p[0] if p else "L"
    return f'<span class="dd-pri pri-{k}">{e(p)}</span>'

def _render_task(t: dict, done: bool = False, is_new: bool = False) -> str:
    m    = t.get("module", "")
    meta = MODULE_META.get(m, {"icon": "📋", "label": m, "mod_cls": ""})
    pri  = t.get("priority", "?")
    anim = " dd-step-new" if is_new else ""
    done_cls = " dd-task-done" if done else ""
    return (
        f'<div class="dd-task {meta["mod_cls"]}{done_cls}{anim}">'
        f'<span style="font-size:20px;flex-shrink:0">{meta["icon"]}</span>'
        f'<div style="flex:1;min-width:0">'
        f'<div class="dd-tname">{_pri_badge(pri)}{e(meta["label"])}</div>'
        f'<div class="dd-tdesc">{e(t.get("description", ""))}</div>'
        f'</div></div>'
    )

def _render_step(ev: dict, is_new: bool = False) -> str:
    step = ev.get("step", "?")
    anim = " dd-step-new" if is_new else ""

    if ev.get("t") == "thinking":
        return (
            f'<div class="dd-step{anim}">'
            f'<div class="dd-shdr"><span class="dd-snum">STEP {step}</span>'
            f'<span style="color:#484f58;margin-left:4px;font-size:10px">— reasoning</span></div>'
            f'<div class="dd-thnk">🤖 Agent evaluating task queue'
            f'<span class="dd-dots"><span>.</span><span>.</span><span>.</span></span>'
            f'</div></div>'
        )

    m       = ev.get("module", "")
    meta    = MODULE_META.get(m, {"icon": "📋", "label": m, "mod_cls": ""})
    payload = ev.get("payload", {})
    reward  = ev.get("reward", 0.0)
    tc      = ev.get("tc", 0.0)
    dr      = ev.get("dr", 0.0)
    success = ev.get("success", False)
    invalid = ev.get("invalid", False)

    smod_map = {"mod-air": "smod-air", "mod-bank": "smod-bank", "mod-ins": "smod-ins"}
    smod_cls = smod_map.get(meta["mod_cls"], "")
    rew_color = "#3fb950" if reward > 0.5 else ("#d29922" if reward > 0 else "#f85149")
    p_str    = json.dumps(payload, indent=2) if (payload and not invalid) else "{}"

    if invalid:
        body = (
            f'<div class="dd-sbody">'
            f'<div class="res-fail">⚠ Model returned malformed JSON — skipping step</div>'
            f'<div class="dd-payload"><pre>{e(ev.get("raw", ""))[:120]}</pre></div>'
            f'</div>'
        )
    else:
        body = (
            f'<div class="dd-sbody">'
            f'<div style="color:#8b949e;margin-bottom:3px">'
            f'Calling <code style="color:#a5d6ff">{e(m)}</code> with payload:</div>'
            f'<div class="dd-payload"><pre>{e(p_str)}</pre></div>'
            f'<div class="dd-res">'
            f'<span class="{"res-ok" if success else "res-fail"}">{"✅ SUCCESS" if success else "❌ FAILED"}</span>'
            f'<span class="dd-stat">task_completion={tc:.2f}</span>'
            f'<span class="dd-stat">drift_recovery={dr:.2f}</span>'
            f'</div></div>'
        )

    return (
        f'<div class="dd-step{anim}">'
        f'<div class="dd-shdr">'
        f'<span class="dd-snum">STEP {step}</span>'
        f'<span class="dd-smod {smod_cls}">{meta["icon"]} {e(meta["label"])}</span>'
        f'<span class="dd-srew" style="color:{rew_color}">+{reward:.3f}</span>'
        f'</div>{body}</div>'
    )

def render_events(evs: list) -> str:
    status       = "idle"
    briefing_ev  = None
    step_evs: list = []
    total_reward = 0.0
    summary_ev   = None
    error_ev     = None
    loading      = False
    done_modules: set = set()

    for ev in evs:
        t = ev.get("t")
        if t == "connect":
            status = "load"
        elif t == "briefing":
            briefing_ev = ev
            status = "run"
        elif t == "loading":
            loading = True
        elif t == "model_ready":
            loading = False
        elif t == "thinking":
            status = "run"
            if step_evs and step_evs[-1].get("t") == "thinking":
                step_evs[-1] = ev
            else:
                step_evs.append(ev)
        elif t == "action":
            total_reward += ev.get("reward", 0.0)
            if ev.get("success") and ev.get("module"):
                done_modules.add(ev["module"])
            if step_evs and step_evs[-1].get("t") == "thinking":
                step_evs[-1] = ev
            else:
                step_evs.append(ev)
        elif t == "summary":
            summary_ev = ev
            status = "done"
        elif t == "error":
            error_ev = ev
            status = "err"

    # Status bar
    meta_str = ""
    if briefing_ev:
        stg = briefing_ev.get("stage", 0)
        meta_str = (
            f'<span class="dd-sep">│</span>'
            f'<span>Seed&nbsp;<strong>{briefing_ev.get("seed","?")}</strong></span>'
            f'<span class="dd-sep">│</span>'
            f'<span>Stage&nbsp;{stg}&nbsp;—&nbsp;{"Policy Drift" if stg else "Baseline"}</span>'
        )

    top_bar = (
        f'<div class="dd-top">'
        f'<span class="dd-logo">🤖&nbsp;DRIFTDESK</span>'
        f'{_badge(status)}'
        f'{meta_str}'
        f'</div>'
    )

    # Welcome screen
    if not evs:
        body = (
            '<div class="dd-welcome">'
            '<div class="big">🤖</div>'
            '<div class="title">DriftDesk Agent — Ready</div>'
            '<div class="sub">'
            'Choose a seed &amp; curriculum stage, then press '
            '<kbd>▶ Run Episode</kbd> to watch the AI agent tackle '
            'enterprise escalations in real time.'
            '<br><br>'
            '<strong style="color:#c9d1d9">Stage 0</strong> — normal policy'
            '&nbsp;&nbsp;·&nbsp;&nbsp;'
            '<strong style="color:#d29922">Stage 1</strong> — policy shifts mid-episode'
            '</div></div>'
        )
        return f'{STYLE}<div class="dd">{top_bar}<div class="dd-body">{body}</div></div>'

    sections = []

    # Task manifest
    if briefing_ev:
        tasks = briefing_ev.get("tasks", []) or []
        if tasks:
            n_done = len(done_modules)
            cards  = "".join(
                _render_task(t, done=t.get("module") in done_modules)
                for t in tasks
            )
            sections.append(
                f'<div>'
                f'<div class="dd-sh">📋 Mission Briefing — {len(tasks)} tasks · {n_done} resolved</div>'
                f'<div class="dd-narr">Incoming enterprise escalations requiring immediate attention.</div>'
                f'<div class="dd-tasks">{cards}</div>'
                f'</div>'
            )

    # Loading indicator
    if loading:
        sections.append(
            '<div class="dd-load"><div class="dd-spin"></div>'
            '<span>Loading model weights… first run takes ~30 s on this hardware</span></div>'
        )

    # Live feed
    if step_evs:
        feed_items = [
            _render_step(ev, is_new=(i == len(step_evs) - 1))
            for i, ev in enumerate(step_evs)
        ]
        sections.append(
            f'<div>'
            f'<div class="dd-sh">⚡ Live Agent Feed</div>'
            f'<div class="dd-narr">The agent processes each task, calls the API module, and earns a reward signal.</div>'
            f'<div class="dd-feed">{"".join(feed_items)}</div>'
            f'</div>'
        )

    # Reward bar
    if step_evs and not loading:
        pct = min(100, int(total_reward / 3.0 * 100))
        sections.append(
            f'<div class="dd-score">'
            f'<div class="dd-scr-lbl">Cumulative Reward</div>'
            f'<div class="dd-scr-row">'
            f'<div class="dd-scr-bar"><div class="dd-scr-fill" style="width:{pct}%"></div></div>'
            f'<div class="dd-scr-val">{total_reward:.3f}</div>'
            f'</div></div>'
        )

    # Summary
    if summary_ev:
        n_steps = summary_ev.get("steps", len(step_evs))
        n_comp  = summary_ev.get("completed", len(done_modules))
        n_total = summary_ev.get("total", len(briefing_ev.get("tasks", [])) if briefing_ev else "?")
        sections.append(
            f'<div class="dd-sum">'
            f'<h3>🏆 Episode Complete</h3>'
            f'<div class="dd-sum-sub">The agent has finished processing the task queue.</div>'
            f'<div class="dd-sum-grid">'
            f'<div class="dd-sstat"><div class="val">{total_reward:.3f}</div><div class="key">Total Reward</div></div>'
            f'<div class="dd-sstat"><div class="val">{n_comp}/{n_total}</div><div class="key">Tasks Done</div></div>'
            f'<div class="dd-sstat"><div class="val">{n_steps}</div><div class="key">Steps Taken</div></div>'
            f'</div></div>'
        )

    # Error
    if error_ev:
        sections.append(
            f'<div class="dd-err-box">⚠️ {e(error_ev.get("msg", "Unknown error"))}</div>'
        )

    scroll_js = (
        '<script>setTimeout(()=>{'
        'var b=document.querySelector(".dd-body");'
        'if(b)b.scrollTop=b.scrollHeight;},80);</script>'
    )
    return f'{STYLE}<div class="dd">{top_bar}<div class="dd-body">{"".join(sections)}</div></div>{scroll_js}'


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(seed: int, curriculum_stage: int) -> Generator[str, None, None]:
    import websocket
    evs: list = []

    evs.append({"t": "connect", "url": ENV_URL})
    yield render_events(evs)

    try:
        ws = websocket.create_connection(WS_URL, timeout=20)
    except Exception as ex:
        evs.append({"t": "error", "msg": f"Cannot reach env server: {ex}"})
        yield render_events(evs)
        return

    try:
        result = _ws_call(ws, {"type": "reset",
                               "data": {"seed": seed, "curriculum_stage": curriculum_stage}})
        obs   = result.get("observation", result)
        tasks = obs.get("tasks", []) or []
        evs.append({"t": "briefing", "seed": seed, "stage": curriculum_stage, "tasks": tasks})
        yield render_events(evs)

        evs.append({"t": "loading"})
        yield render_events(evs)
        model, tok = _load_model()
        evs[-1] = {"t": "model_ready"}
        yield render_events(evs)

        total_reward = 0.0
        done_mods: set = set()

        for step_idx in range(MAX_STEPS):
            evs.append({"t": "thinking", "step": step_idx + 1})
            yield render_events(evs)

            prompt = obs_to_prompt(result)
            raw    = _generate(model, tok, prompt)

            try:
                action = json.loads(raw.split("<|im_end|>")[0].strip())
            except Exception:
                action = None

            if action is None or not isinstance(action, dict) or "module" not in action:
                evs[-1] = {"t": "action", "step": step_idx + 1,
                           "module": "?", "payload": {}, "reward": 0.0,
                           "tc": 0.0, "dr": 0.0, "success": False,
                           "invalid": True, "raw": raw[:120]}
                yield render_events(evs)
                break

            step_result = _ws_call(ws, {"type": "step", "data": {
                "module": action["module"],
                "payload": action.get("payload", {})}})
            r      = step_result.get("reward", 0.0) or 0.0
            total_reward += r
            obs2   = step_result.get("observation", step_result) or {}
            comps  = obs2.get("reward_components", {}) or {}
            tc     = comps.get("task_completion", 0.0) or 0.0
            dr     = comps.get("drift_recovery", 0.0) or 0.0
            success = (r > 0) or (tc > 0)
            if success:
                done_mods.add(action["module"])

            evs[-1] = {"t": "action", "step": step_idx + 1,
                       "module": action["module"],
                       "payload": action.get("payload", {}),
                       "reward": r, "tc": tc, "dr": dr, "success": success}
            result = step_result
            obs    = obs2
            yield render_events(evs)

            if step_result.get("done"):
                break

        n_action = sum(1 for ev in evs if ev.get("t") == "action")
        evs.append({"t": "summary",
                    "total_reward": total_reward,
                    "steps": n_action,
                    "completed": len(done_mods),
                    "total": len(tasks)})
        yield render_events(evs)

    finally:
        try:
            _ws_call(ws, {"type": "close"})
            ws.close()
        except Exception:
            pass


# ── Gradio UI ──────────────────────────────────────────────────────────────────

BLOCKS_CSS = """
.gradio-container { max-width: 980px !important; margin: auto; }
#ep-out { background: transparent !important; border: none !important; padding: 0 !important; }
#ep-out > div { background: transparent !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="DriftDesk — AI Agent Demo",
    css=BLOCKS_CSS,
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.HTML("""
<div style="text-align:center;padding:28px 0 12px;font-family:'SF Mono',Consolas,monospace">
  <div style="font-size:28px;font-weight:700;color:#58a6ff;letter-spacing:4px;margin-bottom:6px">
    DRIFTDESK
  </div>
  <div style="color:#8b949e;font-size:13px;margin-bottom:6px">
    GRPO-trained AI agent &nbsp;·&nbsp; Enterprise task execution &nbsp;·&nbsp; Policy drift adaptation
  </div>
  <div style="font-size:11px;color:#484f58">
    <code style="color:#a5d6ff">Qwen2.5-3B-Instruct</code> + LoRA adapter &nbsp;·&nbsp;
    <a href="https://huggingface.co/HelloOjasMutreja/driftdesk-grpo-adapter"
       style="color:#58a6ff;text-decoration:none">Adapter weights ↗</a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/spaces/HelloOjasMutreja/driftdesk-training/tree/main"
       style="color:#58a6ff;text-decoration:none">Training code ↗</a>
  </div>
</div>
""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=220):
            gr.HTML('<div style="font-family:monospace;font-size:10px;color:#8b949e;'
                    'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">'
                    '⚙ Configuration</div>')
            seed_in  = gr.Number(label="Episode Seed", value=42, precision=0,
                                 info="Controls which tasks are generated")
            stage_in = gr.Radio(
                choices=[("Stage 0 — Baseline", 0), ("Stage 1 — Policy Drift", 1)],
                value=0, label="Curriculum Stage",
                info="Stage 1 injects a policy change mid-episode"
            )
            run_btn = gr.Button("▶  Run Episode", variant="primary", size="lg")
            gr.HTML("""
<div style="margin-top:16px;font-family:monospace;font-size:11px;color:#484f58;
            border-top:1px solid #21262d;padding-top:14px;line-height:1.75">
  <div style="color:#8b949e;letter-spacing:1px;margin-bottom:4px">HOW IT WORKS</div>
  <div>The agent receives a queue of enterprise escalations.
  Each step it selects a task, calls the correct API module with
  extracted payload fields, and earns a reward signal.</div>
  <div style="margin-top:10px;color:#8b949e;letter-spacing:1px">TRAINING</div>
  <div>500-step GRPO · A100 80 GB<br>batch=8 · 64 rollouts/step<br>LoRA r=32, α=64</div>
  <div style="margin-top:10px;color:#8b949e;letter-spacing:1px">API MODULES</div>
  <div>✈️ airline_rebook</div>
  <div>🏦 bank_dispute</div>
  <div>🛡️ insurance_claim</div>
</div>
""")

        with gr.Column(scale=2):
            episode_out = gr.HTML(value=render_events([]), elem_id="ep-out")

    def _run(seed, stage):
        for html_chunk in run_episode(int(seed), int(stage)):
            yield html_chunk

    run_btn.click(_run, inputs=[seed_in, stage_in], outputs=[episode_out])

if __name__ == "__main__":
    demo.launch()
