ICT TOP AND BOTTOM TICKING - V3 REFINEMENT PASS

What this adds:
1. Internal sweep mandatory test
2. Tightened session-window test
3. Losing-cluster analysis report
4. Type 1 sniper retained

Files:
- ict_top_bottom_ticking.py
- top_bottom_ticking_shared.py

Place here:
- src/strategies/manual/ict_top_bottom_ticking.py
- src/strategies/manual/top_bottom_ticking_shared.py

Run from repo root:
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared

Variants run by default on MNQ:
- type2_baseline
- type2_internal_required
- type2_internal_session
- type1_sniper

Outputs:
- top_bottom_ticking_variant_summary.csv
- top_bottom_ticking_all_trades.csv
- top_bottom_ticking_monthly_summary.csv
- top_bottom_ticking_daily_summary.csv
- top_bottom_ticking_losing_clusters.csv

Interpretation:
- type2_baseline = original broader version
- type2_internal_required = external + internal sweep required
- type2_internal_session = internal sweep required + NY AM focused session window
- type1_sniper = stricter sniper approximation

Notes:
- This remains an inferred codification from screenshots.
- It is intended as a disciplined research pass, not a claim of exact discretionary parity.
