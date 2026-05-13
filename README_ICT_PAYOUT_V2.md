# ICT Payout Cycle v2 Risk Patch

Fixes the v1 issue where tiny stops produced hundreds of micro contracts and huge commissions.

Install:

```bash
cd /Users/Abdullahi/trading-project/trading_system
unzip /path/to/ict_payout_cycle_v2_risk_patch.zip -d /tmp/ict_payout_cycle_v2
cp -R /tmp/ict_payout_cycle_v2/src/* src/
```

Run:

```bash
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
  --min-ict-payout-score 3 \
  --fvg-expiry-bars 30 \
  --min-planned-target-dollars 250 \
  --max-contracts MNQ:6,MES:6,MYM:8,MGC:4 \
  --relaxed-mode \
  --news-events src/strategies/manual/researched_prop_trend/news_events.csv \
  --output-prefix ict_payout_cycle
```
