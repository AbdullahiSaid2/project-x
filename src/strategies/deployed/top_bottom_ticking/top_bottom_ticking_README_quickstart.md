# Top Bottom Ticking Quickstart

## Where it lives

Core strategy logic:
- `src/strategies/manual/top_bottom_ticking_shared.py`

Deployed runtime wrapper:
- `src/strategies/deployed/top_bottom_ticking/`

## Required deployed files

Make sure these exist:

```text
src/strategies/deployed/top_bottom_ticking/
├── __init__.py
├── app.py
├── config.py
├── execution.py
├── live_model.py
├── monitoring.py
├── state.py
└── README.md
```

And add this to your deployed model registry:

```python
"top_bottom_ticking": {
    "app_path": DEPLOYED_ROOT / "top_bottom_ticking" / "app.py",
    "description": "Top/bottom ticking deployed loop",
    "supports_prop_guard": True,
},
```

## Recommended `.env` settings

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

DATA_SOURCE=databento
TOP_BOTTOM_TICKING_DATABENTO_SCHEMA=ohlcv-1m
TOP_BOTTOM_TICKING_DATABENTO_WARMUP_HOURS=72
TOP_BOTTOM_TICKING_DATABENTO_HISTORICAL_LAG_MINUTES=30
TOP_BOTTOM_TICKING_DATABENTO_SYMBOL_MAP=MNQ:MNQ.c.0,MES:MES.c.0,MYM:MYM.c.0,MGC:MGC.c.0,MCL:MCL.c.0
```

## How to run

## 1) Safe forward-test run

Run without real order sending:

```bash
python src/strategies/deployed/run_models.py --models top_bottom_ticking --prop-profile apex_pa_50k --mode live --no-live-orders
```

## 2) Run with another model too

```bash
python src/strategies/deployed/run_models.py --models ict_fractal,top_bottom_ticking --prop-profile apex_pa_50k --mode live --no-live-orders
```

## 3) Real live orders

Only after confirming the full flow:

```bash
python src/strategies/deployed/run_models.py --models top_bottom_ticking --prop-profile apex_pa_50k --mode live
```

This requires:
- correct execution bridge
- correct prop profile
- correct account routing
- `LIVE_ORDERS=1`

## What to look for in logs

Healthy start:

```text
[top_bottom_ticking] starting deployed loop in mode=live ...
```

Healthy closed-market idle:

```text
"heartbeat": "idle_closed_market"
```

Healthy open-market cycle:
- `signals_seen`
- `orders_sent`
- `session_reason`
- `data_source_mode`

## Deployment sequence

1. Start with `--no-live-orders`
2. Verify:
   - no import/runtime errors
   - correct symbols
   - correct data source
   - correct session behavior
3. Move to 1 micro only
4. Verify end-to-end:
   - signal generation
   - execution routing
   - broker/account mapping
5. Scale only after stable forward testing

## Notes

- Strategy logic stays in `src/strategies/manual/top_bottom_ticking_shared.py`
- Deployed files are wrappers for live/forward operation
- If there is a traceback on first run, patch the specific deployed file rather than changing the strategy logic first


## How to run with hearbeat

- cd /Users/Abdullahi/trading-project/trading_system
source venv/bin/activate
python -u src/strategies/deployed/top_bottom_ticking/app.py