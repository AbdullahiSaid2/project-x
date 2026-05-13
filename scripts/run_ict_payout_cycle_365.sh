#!/usr/bin/env bash
set -euo pipefail
cd /Users/Abdullahi/trading-project/trading_system
NEWS="src/strategies/manual/researched_prop_trend/news_events.csv"
PYTHONPATH=. python -m src.backtesting.event_engine.run_ict_payout_cycle_backtest \
  --symbols MNQ MES MYM MGC \
  --prop-profile apex_50k_pa \
  --days-back 365 \
  --timeframe 1m \
  --no-tail \
  --commission-per-contract-side 2.0 \
  --target-r 5.0 \
  --risk-per-trade 150 \
  --daily-profit-target 650 \
  --daily-soft-loss-stop 350 \
  --max-trades-per-day 6 \
  --pause-after-consecutive-losses 2 \
  --min-ict-payout-score 4 \
  --fvg-expiry-bars 20 \
  --min-planned-target-dollars 500 \
  --news-events "$NEWS" \
  --output-prefix ict_payout_cycle
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator \
  --trade-log src/backtesting/event_engine/outputs/ict_payout_cycle_event_trade_log.csv \
  --lifecycle-profile apex_50k_eod_lifecycle_ict_payout_cycle
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.payout_interval_report
