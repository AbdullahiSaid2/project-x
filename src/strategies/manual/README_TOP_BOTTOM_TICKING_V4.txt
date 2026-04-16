ICT TOP AND BOTTOM TICKING V4

What this version adds:
- Type 2 Active on 5m to restore trade frequency
- Type 1 Sniper on 30s
- Type 1 Sniper on 5s

Data assumptions:
- 5m uses your normal fetcher path
- 30s and 5s require a real local 1-second parquet in:
  src/data/databento_cache/
  e.g. NQ_1s.parquet or MNQ_1s.parquet

Run from repo root:
  cd /Users/Abdullahi/trading-project/trading_system
  source venv/bin/activate
  PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared

Outputs:
- top_bottom_ticking_variant_summary.csv
- top_bottom_ticking_all_trades.csv
- top_bottom_ticking_monthly_summary.csv
- top_bottom_ticking_daily_summary.csv
- top_bottom_ticking_losing_clusters.csv

Suggested placement:
- src/strategies/manual/ict_top_bottom_ticking.py
- src/strategies/manual/top_bottom_ticking_shared.py

Important:
- 5s/30s are only valid if sourced from real 1-second data
- this is still a research model, not a final exact discretionary clone
