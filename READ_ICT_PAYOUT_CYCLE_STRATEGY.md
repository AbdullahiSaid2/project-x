# ICT Payout Cycle Model v1

This package adds a codified ICT strategy designed specifically for prop-firm payout behaviour.

## Logic

```text
HTF Bias -> Liquidity Sweep -> Reclaim / Displacement -> FVG -> Pullback to FVG CE -> Structural Stop -> PA-friendly Target
```

## Install

```bash
cd /Users/Abdullahi/trading-project/trading_system
unzip /path/to/ict_payout_cycle_strategy_package.zip -d /tmp/ict_payout_cycle_strategy
cp -R /tmp/ict_payout_cycle_strategy/src/* src/
mkdir -p scripts
cp -R /tmp/ict_payout_cycle_strategy/scripts/* scripts/
chmod +x scripts/run_ict_payout_cycle_365.sh
```

## Merge lifecycle profile

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.install_ict_payout_cycle_profile
```

## Run 365-day smoke test

```bash
PYTHONPATH=. python -m src.backtesting.event_engine.run_ict_payout_cycle_backtest   --symbols MNQ MES MYM MGC   --prop-profile apex_50k_pa   --days-back 365   --timeframe 1m   --no-tail   --commission-per-contract-side 2.0   --target-r 5.0   --risk-per-trade 150   --daily-profit-target 650   --daily-soft-loss-stop 350   --max-trades-per-day 6   --pause-after-consecutive-losses 2   --min-ict-payout-score 4   --fvg-expiry-bars 20   --min-planned-target-dollars 500   --news-events src/strategies/manual/researched_prop_trend/news_events.csv   --output-prefix ict_payout_cycle
```

## Run payout lifecycle

```bash
PYTHONPATH=. python -m src.strategies.manual.researched_prop_trend.prop_lifecycle_payout_simulator   --trade-log src/backtesting/event_engine/outputs/ict_payout_cycle_event_trade_log.csv   --lifecycle-profile apex_50k_eod_lifecycle_ict_payout_cycle
```

## One-command smoke test

```bash
scripts/run_ict_payout_cycle_365.sh
```

## Full 5-year test

Change `--days-back 365` to `--days-back 1825` once the smoke test works.
