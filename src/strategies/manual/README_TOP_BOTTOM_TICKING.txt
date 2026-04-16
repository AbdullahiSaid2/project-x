ICT TOP AND BOTTOM TICKING - FIRST PASS BACKTEST MODEL

What this is
- a first codified version of the model inferred from the screenshots
- built to be backtested in the same backtesting.py style as the V473 work
- focused on Type-2 execution first because it is cleaner to code faithfully

Files
- ict_top_bottom_ticking.py
- top_bottom_ticking_shared.py

Suggested placement
- place ict_top_bottom_ticking.py in: project-x/src/strategies/manual/
- place top_bottom_ticking_shared.py in: project-x/src/strategies/manual/

How to run
1. cd to your repo root
2. activate your venv
3. run:
   python src/strategies/manual/top_bottom_ticking_shared.py

What it currently codifies
- context anchor: 15m information derived from the 5m stream
- execution timeframe: 5m
- external sweep mandatory
- internal sweep optional filter
- two-sided reversal model:
  - short after external buyside sweep
  - long after external sellside sweep
- rejection-block proxy built from the sweep candle / mitigation zone
- CE-based entry
- optional CoS confirmation before entry
- fixed stop beyond the refined zone
- 5 micro contracts
- partials: 2 / 2 / 1
  - first partial at 1.0R
  - second partial at 2.25R
  - final runner at 4.25R
- breakeven move after first partial
- no DCA / one position at a time
- force-flat cutoff before the CME maintenance window

Important assumptions made from the screenshots
Because the screenshots are visual discretionary examples, this version makes explicit assumptions:
1. external sweep is mandatory
2. internal sweep is optional but available as a stricter filter
3. Type-2 is implemented first
4. 15m context + 5m execution is the first backtestable version
5. target multiples 1.0R / 2.25R / 4.25R are used as the first coded proxy for the liquidity/stdv target framework shown in the charts
6. rejection block is approximated from the sweep candle zone, not a fully discretionary manual marking process

Outputs
- top_bottom_ticking_trade_log.csv
- top_bottom_ticking_monthly_summary.csv
- top_bottom_ticking_daily_summary.csv
- top_bottom_ticking_debug_counts.csv

What to compare against V473
For fair comparison against ict_fractal / V473, compare:
- number of trades
- win rate
- avg R / avg dollars
- max drawdown
- MAE / MFE if you add those later
- time in trade
- setup frequency by session

Recommended next iteration after first results
1. add a stricter internal-sweep-required variant
2. add a 1m execution refinement version
3. add explicit 15m rejection-block detection instead of sweep-candle proxy
4. add target selection by opposing liquidity pool rather than fixed R only
5. add separate Type-1 sniper version
