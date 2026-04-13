V470 patch pack

What this pack does
- Fixes the V469 problem where the runner delegated to V468 and wrote the wrong filenames.
- Enforces a stricter V470 policy layer:
  - A-tier only
  - London + NYPM only
  - fixed 10 MNQ
  - minimum planned profit target $500 per trade
  - minimum planned RR 5.0
- Writes V470-named exports directly so the analyze/export scripts stop failing on missing files.

What this pack does NOT do
- It does not guarantee $100k realized PnL by itself. That depends on the underlying signal quality in the base engine.
- It assumes your repo already has a working ICT base engine from one of the prior versions.

Recommended placement
Copy these files into:
src/strategies/manual/

Run order
python src/strategies/manual/tmp_test_ict_multi_setup_v470.py
python src/strategies/manual/analyze_v470_results.py
python src/strategies/manual/monthly_pnl_export_v470.py
python src/strategies/manual/apex_50k_monthly_payout_export_v470.py
python src/strategies/manual/daily_pnl_export_v470.py
