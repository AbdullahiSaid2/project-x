# Top Bottom Ticking + AXL Overlay Improvements

This package contains full replacement files for the real `top_bottom_ticking` model files you uploaded:

- `src/strategies/manual/ict_top_bottom_ticking.py`
- `src/strategies/manual/top_bottom_ticking_shared.py`

It does **not** use or modify the separate `ICT_MULTI_SETUP` model.

## What changed

### 1. AXL overlay added to `ict_top_bottom_ticking.py`

The strategy now supports:

- HTF PD-array anchor detection using 15m/60m FVG/IFVG proxy levels.
- MTF POI detection using 5m FVG/IFVG proxy levels.
- Separate external liquidity vs internal liquidity tagging.
- Close-based ChOCh/COS tagging.
- Volume regime tagging.
- AXL setup scoring and grading:
  - `A`: strong setup, external + internal liquidity and high score.
  - `B`: acceptable but lower probability.
  - `C`: low quality / reject in filter mode.
- Safer/deeper ODE-style entries for B-grade setups.
- Full AXL trade-log metadata.

### 2. Force-flat session bug fixed

The old `_after_force_flat_cutoff()` blocked entries after 16:50 ET for the rest of the day.

The new logic blocks only:

```text
16:50 ET <= time < 18:00 ET
```

At 18:00 ET and later, trading can resume for the new Globex session.

### 3. CLI controls added to `top_bottom_ticking_shared.py`

New flags:

```bash
--axl-mode off|log_only|filter
--axl-min-b-score 5
--axl-min-a-score 8
--axl-require-internal
--axl-require-htf-anchor
--axl-require-mtf-poi
--axl-disable-safe-entry-for-b
--axl-liquidity-close-ticks 20
```

### 4. Enhanced reporting added

The shared runner now writes:

- Trade log
- Debug counts
- Daily summary
- Monthly summary
- Variant summary
- AXL grade summary

## Recommended rollout

### Step 1: Install

From your repo root:

```bash
cd /Users/Abdullahi/trading-project/trading_system
cp src/strategies/manual/ict_top_bottom_ticking.py src/strategies/manual/ict_top_bottom_ticking.py.bak
cp src/strategies/manual/top_bottom_ticking_shared.py src/strategies/manual/top_bottom_ticking_shared.py.bak
```

Then copy the two replacement files from this package into the matching paths.

### Step 2: Run baseline unchanged

This keeps current behaviour:

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail
```

### Step 3: Run AXL log-only mode

This does not block trades. It tags trades so you can check whether A/B setups outperform C setups.

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode log_only
```

Check:

```text
top_bottom_ticking_axl_grade_summary_apex_50k_eval_365d_notail_axllog_only.csv
top_bottom_ticking_trade_log_apex_50k_eval_365d_notail_axllog_only.csv
```

### Step 4: Run AXL filter mode

This blocks C-grade setups.

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode filter
```

### Step 5: Run stricter AXL mode

This requires internal liquidity, HTF anchor, and MTF POI.

```bash
PYTHONPATH=. python -m src.strategies.manual.top_bottom_ticking_shared \
  --prop-profile apex_50k_eval \
  --days-back 365 \
  --no-tail \
  --axl-mode filter \
  --axl-require-internal \
  --axl-require-htf-anchor \
  --axl-require-mtf-poi
```

## Important

Run `log_only` first. Do not go straight into strict filter mode until the AXL grade summary confirms that A/B setups are materially better than C setups.

