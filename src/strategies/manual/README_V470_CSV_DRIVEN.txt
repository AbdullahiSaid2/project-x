V470 CSV-DRIVEN BUILD

Place these files in:
trading_system/src/strategies/manual/

Expected source input in trading_system root:
- v467_trade_log.csv

How V470 works:
1) Reads root-level v467_trade_log.csv
2) Filters to:
   - setup_tier == A
   - setup_type in {LONDON_CONTINUATION, NYPM_CONTINUATION}
   - planned_rr >= 5
   - target >= $500 at fixed 10 MNQ
3) Recomputes fixed-10-MNQ realized dollars from realized_points
4) Writes root-level V470 CSV outputs

Run order:
python src/strategies/manual/tmp_test_ict_multi_setup_v470.py
python src/strategies/manual/analyze_v470_results.py
python src/strategies/manual/monthly_pnl_export_v470.py
python src/strategies/manual/apex_50k_monthly_payout_export_v470.py
python src/strategies/manual/daily_pnl_export_v470.py
