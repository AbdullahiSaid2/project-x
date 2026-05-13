
# Exact Strategy Adapter Patch

This package replaces the placeholder adapters with adapters that port the actual model logic from:

```text
src/strategies/manual/ict_top_bottom_ticking.py
src/strategies/manual/ict_multi_setup_v452.py
```

## Install

From repo root:

```bash
cd /Users/Abdullahi/trading-project/trading_system
unzip /path/to/algotec_exact_strategy_adapters_package.zip -d /tmp/algotec_exact_adapters
cp -R /tmp/algotec_exact_adapters/src/* src/
```

This overwrites:

```text
src/strategies/adapters/top_bottom_ticking_event_adapter.py
src/strategies/adapters/ict_fractal_event_adapter.py
```

## Important

These adapters use your real model feature builders and state-machine logic, but the new event engine still controls:

```text
fills
no same-bar exits
position sizing
commissions
prop guardrails
news blackout
session force-flat
standard event_trade_log output
```

So results will not be identical to old Backtesting.py because execution math is intentionally stricter.

## Run top_bottom_ticking smoke test

```bash
PYTHONPATH=. python -m src.backtesting.event_engine.run_event_backtest \
  --strategy top_bottom_ticking \
  --symbols MNQ MES MYM MGC \
  --prop-profile apex_50k_pa \
  --days-back 365 \
  --timeframe 1m \
  --no-tail \
  --commission-per-contract-side 2.0 \
  --min-planned-target-dollars 500 \
  --news-events src/strategies/manual/researched_prop_trend/news_events.csv \
  --output-prefix top_bottom_ticking_exact
```

## Run ict_fractal smoke test

```bash
PYTHONPATH=. python -m src.backtesting.event_engine.run_event_backtest \
  --strategy ict_fractal \
  --symbols MNQ MES MYM MGC \
  --prop-profile apex_50k_pa \
  --days-back 365 \
  --timeframe 1m \
  --no-tail \
  --commission-per-contract-side 2.0 \
  --min-planned-target-dollars 500 \
  --news-events src/strategies/manual/researched_prop_trend/news_events.csv \
  --output-prefix ict_fractal_exact
```

## Run payout lifecycle after smoke test

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator \
  --trade-log src/backtesting/event_engine/outputs/ict_fractal_exact_event_trade_log.csv \
  --lifecycle-profile apex_50k_eod_lifecycle_safe
```

or:

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator \
  --trade-log src/backtesting/event_engine/outputs/top_bottom_ticking_exact_event_trade_log.csv \
  --lifecycle-profile apex_50k_eod_lifecycle_safe
```
