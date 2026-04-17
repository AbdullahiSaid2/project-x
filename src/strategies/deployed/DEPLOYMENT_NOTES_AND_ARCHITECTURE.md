# Top Bottom Ticking Deployment Notes

## 1. Should `state.py` be separate for each model?

Yes. Each deployed model should have its **own** `state.py` and its own state file on disk.

### Why
`state.py` is not strategy logic. It is the model's **runtime memory**. It stores operational things such as:

- seen `signal_id`s
- orders sent count
- last cycle summary
- last force-flat event
- last runtime error

If two different strategies share the same `state.py` / state file, they can interfere with each other.

### Risks of shared state across models
- one model can mark a signal as already seen for another model
- force-flat / cycle info can overwrite each other
- restart recovery becomes confusing
- debugging becomes much harder

### Best practice
Each deployed model should have its own:

- `app.py`
- `config.py`
- `execution.py`
- `state.py`
- `live_model.py`

The only things that should be shared are high-level infrastructure patterns such as:

- being launched from one command via `run_models.py`
- common prop-firm timing rules
- Databento live data workflow
- PickMyTrade / Tradovate order execution workflow

---

## 2. What each file does

### `app.py`
The orchestration loop.

It:
- runs every cycle
- checks session / force-flat timing
- asks `live_model.py` for signals
- sends valid signals to `execution.py`
- updates `state.py`

It should not contain the trading strategy logic itself.

### `live_model.py`
The market data + live signal generation layer.

It:
- fetches live data from Databento
- optionally does Databento historical warmup
- runs the strategy logic against the fresh candles
- returns normalized trade signals

It should not fall back to TradingView in live mode if you want Databento-only operation.

### `execution.py`
The order execution layer.

It:
- receives a normalized signal
- logs it
- sends it to PickMyTrade / Tradovate
- handles force-flat actions

It should not contain the strategy logic.

### `state.py`
The runtime persistence layer.

It:
- loads and saves runtime state from disk
- remembers seen signals
- remembers the last cycle
- remembers force-flat / error info

It should be separate per model.

### `config.py`
The deployment configuration layer.

It:
- loads environment variables
- stores model-specific runtime settings
- stores execution settings
- stores session / force-flat timing settings

It should also be separate per model.

---

## 3. Strategy logic separation

### `ict_fractal`
Should use only its own signal logic.

### `top_bottom_ticking`
Should use only its own signal logic from:
- `src/strategies/manual/ict_top_bottom_ticking.py`

These two strategies should **not** share trade/setup logic.

They may share:
- run orchestration pattern
- prop-firm timing concepts
- Databento live data workflow
- PickMyTrade execution workflow

---

## 4. Why MGC needed more rows

The live model requires a minimum amount of recent completed candles before it can safely evaluate structure.

In the current setup, the minimum was effectively:

- **11 rows for 1-minute candles**

That is typically:
- a **10-bar lookback**
- plus the current completed bar

With only 4 rows, the strategy does not have enough recent context for reliable structure / confirmation logic.

### Better fix
Use:
- Databento live rows
- plus Databento historical warmup

### Avoid
- TradingView fallback in live mode

---

## 5. Clear startup summaries in `run_models.py`

A clearer startup summary is useful because it makes it obvious:

- which model is starting
- which deployment folder it is using
- which `app.py` is being launched
- which mode / prop profile is active
- whether live orders are enabled
- whether the model is running independently

A good per-model startup summary should print:

- model name
- deployment directory
- app path
- mode
- prop profile
- live orders on/off
- whether it is expected to use its own files

Example:

```text
[top_bottom_ticking] startup summary
  deploy_dir: /.../src/strategies/deployed/top_bottom_ticking
  app: /.../src/strategies/deployed/top_bottom_ticking/app.py
  mode: live
  prop_profile: apex_pa_50k
  live_orders: off
  runtime_stack: app.py + config.py + execution.py + state.py + live_model.py
```

---

## 6. Recommended deployed structure

For each deployed model:

```text
src/strategies/deployed/<model_name>/
  app.py
  config.py
  execution.py
  state.py
  live_model.py
  logs/
  state/
```

This keeps models operationally separate while still allowing one shared launcher.
