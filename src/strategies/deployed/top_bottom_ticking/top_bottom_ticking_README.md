# Top Bottom Ticking

## What this model is

`top_bottom_ticking` is a deployed futures trading model built from the strategy logic in:

- `src/strategies/manual/top_bottom_ticking_shared.py`

and wrapped for forward testing / live execution under:

- `src/strategies/deployed/top_bottom_ticking/`

It is designed to trade **micro futures** and related index/commodity symbols using a **top/bottom sweep and reaction** idea. The deployed wrapper lets you run the same strategy logic in a live-style loop with:

- session controls
- force-flat logic
- Databento / TradingView data
- execution bridge reuse
- prop-profile compatible deployment flow

## Symbols traded

By default the model is configured for:

- `MNQ`
- `MES`
- `MYM`
- `MGC`
- `MCL`

These are controlled by:

```env
TOP_BOTTOM_TICKING_SYMBOLS=MNQ,MES,MYM,MGC,MCL
```

## Strategy variants

The deployed model supports these variants by default:

- `type2_baseline`
- `type1_sniper`

Configured by:

```env
TOP_BOTTOM_TICKING_VARIANTS=type2_baseline,type1_sniper
```

## How it trades

At a high level, the model looks for **top / bottom ticking behavior** after a price sweep and then attempts to enter in the expected reversal direction.

Core characteristics:

- monitors configured symbols on their configured timeframe
- runs the same strategy classes used in your manual backtests
- detects eligible entry setups from the latest bars
- converts qualifying entries into live-style actions
- passes those actions into the deployed execution flow

### Broad workflow

1. Load market data for each configured symbol.
2. Run the selected strategy variant on recent bars.
3. Extract fresh trade entries from the latest bar window.
4. Validate entry structure:
   - side
   - entry
   - stop
   - target
   - quantity
5. Apply session control:
   - market open / closed handling
   - force-flat window handling
6. Send allowed actions to execution, or simulate them if live orders are disabled.

## Data sources

The model can use:

- **Databento**
- **TradingView bars**
- fallback fetcher path

Current active source is controlled by:

```env
DATA_SOURCE=databento
```

Top-bottom-ticking specific controls include:

```env
TOP_BOTTOM_TICKING_USE_TRADINGVIEW_BARS=1
TOP_BOTTOM_TICKING_TV_FALLBACK_TO_FETCHER=1
TOP_BOTTOM_TICKING_TV_MIN_BARS=200
TOP_BOTTOM_TICKING_TRADINGVIEW_SYMBOL_MAP=MNQ:MNQ1!,MES:MES1!,MYM:MYM1!,MGC:MGC1!,MCL:MCL1!
```

Databento-related controls include:

```env
TOP_BOTTOM_TICKING_DATABENTO_SCHEMA=ohlcv-1m
TOP_BOTTOM_TICKING_DATABENTO_WARMUP_HOURS=72
TOP_BOTTOM_TICKING_DATABENTO_HISTORICAL_LAG_MINUTES=30
TOP_BOTTOM_TICKING_DATABENTO_SYMBOL_MAP=MNQ:MNQ.c.0,MES:MES.c.0,MYM:MYM.c.0,MGC:MGC.c.0,MCL:MCL.c.0
```

## Session behavior

The deployed app supports:

- regular loop while market is open
- slower idle heartbeat while market is closed
- optional force-flat handling near end of session
- Globex reopen handling

Key settings:

```env
TOP_BOTTOM_TICKING_FORCE_FLAT_ENABLED=1
TOP_BOTTOM_TICKING_FORCE_FLAT_HOUR_ET=16
TOP_BOTTOM_TICKING_FORCE_FLAT_MINUTE_ET=50
TOP_BOTTOM_TICKING_GLOBEX_REOPEN_HOUR_ET=18
TOP_BOTTOM_TICKING_GLOBEX_REOPEN_MINUTE_ET=0
TOP_BOTTOM_TICKING_LOOP_SECONDS=20
TOP_BOTTOM_TICKING_CLOSED_LOOP_SECONDS=60
```

## Execution behavior

The deployed `top_bottom_ticking` model reuses the shared execution bridge style already used in your deployed stack.

That means it can run in:

- dry / no-live-order mode
- live path with execution disabled
- live path with real orders enabled

Real sending is controlled by the runner flags and environment:

```env
DEPLOY_MODE=live
PROP_PROFILE=apex_pa_50k
LIVE_ORDERS=0
```

## Where the actual strategy logic lives

The core strategy logic is **not** inside the deployed wrapper.

It lives in:

```text
src/strategies/manual/top_bottom_ticking_shared.py
```

The deployed folder is the runtime wrapper around that strategy.

### In practice

- **manual file** = real trading logic
- **deployed/live_model.py** = data + signal extraction wrapper
- **deployed/app.py** = loop runner
- **deployed/execution.py** = execution bridge
- **deployed/config.py** = runtime config
- **deployed/state.py / monitoring.py** = persistence + logs

## Files in deployed folder

Expected deployed structure:

```text
src/strategies/deployed/top_bottom_ticking/
├── __init__.py
├── app.py
├── config.py
├── execution.py
├── live_model.py
├── monitoring.py
└── state.py
```

And the registry should include:

```python
"top_bottom_ticking": {
    "app_path": DEPLOYED_ROOT / "top_bottom_ticking" / "app.py",
    "description": "Top/bottom ticking deployed loop",
    "supports_prop_guard": True,
},
```

## How to run it

### 1. Dry-style / safe live-path check

Run the deployed loop without sending real orders:

```bash
python src/strategies/deployed/run_models.py --models top_bottom_ticking --prop-profile apex_pa_50k --mode live --no-live-orders
```

This is the recommended first step.

### 2. Run alongside other deployed models

Example with multiple models:

```bash
python src/strategies/deployed/run_models.py --models ict_fractal,top_bottom_ticking --prop-profile apex_pa_50k --mode live --no-live-orders
```

### 3. Live orders enabled

Only do this after validating data, signals, and execution flow:

```bash
python src/strategies/deployed/run_models.py --models top_bottom_ticking --prop-profile apex_pa_50k --mode live
```

This assumes:

- execution bridge is working
- account routing is correct
- prop profile is correct
- `LIVE_ORDERS=1`

## Recommended .env settings

Example starting point:

```env
TOP_BOTTOM_TICKING_EXECUTION_MODE=live
TOP_BOTTOM_TICKING_SYMBOLS=MNQ,MES,MYM,MGC,MCL
TOP_BOTTOM_TICKING_VARIANTS=type2_baseline,type1_sniper

TOP_BOTTOM_TICKING_LOOP_SECONDS=20
TOP_BOTTOM_TICKING_CLOSED_LOOP_SECONDS=60
TOP_BOTTOM_TICKING_HEARTBEAT_EVERY_CLOSED_CYCLES=5
TOP_BOTTOM_TICKING_HEARTBEAT_EVERY_OPEN_CYCLES=1

TOP_BOTTOM_TICKING_DEFAULT_QTY=1
TOP_BOTTOM_TICKING_SIGNAL_LOOKBACK_BARS=1

TOP_BOTTOM_TICKING_FORCE_FLAT_ENABLED=1
TOP_BOTTOM_TICKING_FORCE_FLAT_HOUR_ET=16
TOP_BOTTOM_TICKING_FORCE_FLAT_MINUTE_ET=50
TOP_BOTTOM_TICKING_GLOBEX_REOPEN_HOUR_ET=18
TOP_BOTTOM_TICKING_GLOBEX_REOPEN_MINUTE_ET=0

TOP_BOTTOM_TICKING_USE_TRADINGVIEW_BARS=1
TOP_BOTTOM_TICKING_TV_FALLBACK_TO_FETCHER=1
TOP_BOTTOM_TICKING_TV_MIN_BARS=200
TOP_BOTTOM_TICKING_TRADINGVIEW_SYMBOL_MAP=MNQ:MNQ1!,MES:MES1!,MYM:MYM1!,MGC:MGC1!,MCL:MCL1!

TOP_BOTTOM_TICKING_DATABENTO_SCHEMA=ohlcv-1m
TOP_BOTTOM_TICKING_DATABENTO_WARMUP_HOURS=72
TOP_BOTTOM_TICKING_DATABENTO_HISTORICAL_LAG_MINUTES=30
TOP_BOTTOM_TICKING_DATABENTO_SYMBOL_MAP=MNQ:MNQ.c.0,MES:MES.c.0,MYM:MYM.c.0,MGC:MGC.c.0,MCL:MCL.c.0
```

## Recommended rollout path

1. Run with `--no-live-orders`
2. Confirm:
   - signals are appearing
   - session behavior is correct
   - no import/runtime errors
   - data source is correct
3. Move to **1 micro only**
4. Confirm end-to-end flow:
   - model
   - execution bridge
   - broker / account routing
5. Scale only after stable forward performance

## Summary

`top_bottom_ticking` is the deployed live/forward-test wrapper for your top/bottom sweep-reaction strategy. It uses the same underlying logic as your manual backtest file, but adds:

- live data integration
- deployment loop
- session controls
- execution routing
- prop-profile compatibility

Use the deployed wrapper to forward test and live run the model, while keeping core strategy development in the manual strategy file.
