# Post-Training Update Plan

Implement these changes **after** training completes and the adapter is saved.
Do NOT touch server code or the notebook while training is running.

---

## Priority 1 — Schema Validation (correctness, affects re-runs)

**What:** Enforce declared types from `FieldSpec.type` and centralize validation in a single helper.

**Where:** `driftdesk/server/drift_controller.py`, `driftdesk/server/task_modules/base.py`

**Steps:**
1. Add a `validate_payload(payload: dict, schema: dict) -> ValidationResult` helper in `base.py` (or a new `driftdesk/server/schema_validator.py`).
2. For each field in the schema, coerce and check against `FieldSpec.type` (e.g., `"string"`, `"int"`, `"bool"`). Raise a typed `ValidationError` on mismatch.
3. Replace all per-module ad-hoc checks in `airline.py`, `bank.py`, `insurance.py` with calls to this helper.
4. Add a unit test per module: wrong type → `VALIDATION_ERROR`, correct type → passes through.

---

## Priority 2 — Test Suite

**What:** Unit + golden + property tests covering the three main components.

**Where:** New `driftdesk/tests/` directory.

**Steps:**

### 2a. `SchemaDriftController` unit tests (`test_drift_controller.py`)
- `reset()` clears state and returns v1 schema.
- `maybe_drift(step)` returns a `DriftEvent` at the scheduled step, `None` otherwise.
- `get_transient_error(step, module)` returns the configured error or `None`.
- Test that drift events are idempotent (calling twice at the same step doesn't double-apply).

### 2b. Golden tests for task modules (`test_task_modules.py`)
- For each of `airline`, `bank`, `insurance`:
  - v1 schema: valid payload → `SUCCESS`.
  - v2 schema (after one drift): old payload → `DRIFT_ERROR` with correct `missing_fields`.
  - v99 schema (fully drifted): payload for v1 → full `DRIFT_ERROR`.
- Use deterministic seeds so golden outputs are stable.

### 2c. Reward engine property tests (`test_reward_engine.py`)
- No inflation when no transient errors are present: `reward ≤ 1.0` always.
- Loop penalty caps: repeated identical actions after N retries yield a penalty, not infinite reward.
- Transient retry: same payload on a transient error step scores at least as well as a new payload.
- Format bonus: valid JSON action always yields `partial > 0`, invalid JSON always yields `partial == 0`.

**Tooling:** `pytest` + `pytest-cov`. Add to `pyproject.toml` under `[project.optional-dependencies] dev`.

---

## Priority 3 — Remove `sys.path` Hacks / Fix Imports

**What:** Replace `sys.path.insert(0, ...)` with proper absolute package imports.

**Where:** `driftdesk_grpo_training.ipynb` (cell 4c / SFT warm-up), `eval_harness.py`.

**Steps:**
1. Verify `pyproject.toml` has `packages = ["driftdesk"]` (or equivalent `find:` config) and that `driftdesk/server/` is under the package root.
2. Replace:
   ```python
   sys.path.insert(0, os.path.dirname(os.path.abspath('eval_harness.py')))
   from eval_harness import RuleBasedAgent
   ```
   with:
   ```python
   from driftdesk.eval_harness import RuleBasedAgent
   ```
3. Do the same for any other `sys.path` hacks in the server modules.
4. Run `pip install -e .` in the project root and confirm all imports resolve without `sys.path` manipulation.

---

## Priority 4 — Deployment Hardening (only if Space stays public)

**What:** Rate limiting, request size caps, timeouts, optional API key, structured logging.

**Where:** `driftdesk/server/app.py`

**Steps:**
1. Add `slowapi` (or `starlette-ratelimit`) middleware — e.g., 60 WS connections/min per IP.
2. Set a max WebSocket message size (e.g., 64 KB) to reject oversized payloads.
3. Add a per-episode timeout (e.g., 120 s) that closes the WS and returns a `TIMEOUT` error.
4. Optional API key: read `DRIFTDESK_API_KEY` from env; if set, require it as a query param or header on connect.
5. Replace `print()`-based logging with `structlog` or Python `logging` with JSON formatter — include `episode_id`, `step`, `module`, `reward` in each log line.

**Skip if:** the Space is private or only used for hackathon evaluation.

---

## Priority 5 — Index Drift/Transient Schedules (skip for now)

**Verdict:** Not worth doing. At 150 steps × 3 modules the linear scan is microseconds.
Revisit only if scaling to 10,000+ steps or adding dozens of modules.

---

## Execution Order

```
1. Training finishes + adapter saved
2. Implement Priority 1 (schema validation) → re-run eval_harness to confirm no reward change
3. Implement Priority 2 (tests) → all green before merging
4. Implement Priority 3 (imports) → pip install -e . + smoke test
5. Implement Priority 4 (hardening) → only if deploying publicly
6. Skip Priority 5
```
