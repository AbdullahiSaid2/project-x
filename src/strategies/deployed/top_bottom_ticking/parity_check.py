"""
top_bottom_ticking/parity_check.py

PARITY CHECK OVERVIEW
=====================

Purpose
-------
This script is used to compare two ways of running the SAME strategy logic on the
SAME market data window:

1) FULL-PASS RUN
2) LIVE-STYLE ROLLING REPLAY

The goal is to verify that the strategy behaves consistently when run in a normal
backtest-style batch pass versus a live-style incremental replay.

Why this matters
----------------
A strategy can appear correct in a standard backtest, but behave differently in live
execution if any of the following happen:

- state is not preserved correctly across bars
- pending setups reset incorrectly
- bar preparation differs between backtest and live
- lookahead assumptions exist in batch processing
- live entry logic depends on incremental bar arrival in a way that differs from
  full dataframe execution

This parity script helps check whether the strategy logic itself is consistent.

----------------------------------------------------------------------
FULL-PASS RUN
----------------------------------------------------------------------

Definition
----------
The full-pass run is the normal backtest-style evaluation.

The strategy is given the FULL selected dataframe at once, for example:
- all 5-minute bars
- from START timestamp
- to END timestamp

The model then processes the entire bar sequence in one complete run.

Conceptually
------------
This is equivalent to saying:

    "Here is the full chart for this time window.
     Run the strategy over all of it and tell me:
     - what trades were logged
     - what pending setup remains
     - what final state the strategy ended in"

What it tells us
----------------
The full-pass summary shows the final backtest-style outcome over that window:
- number of bars
- number of trade log rows
- latest bar timestamp
- final pending setup state
- stop/target state
- active direction state
- stop bound configuration values

Important note
--------------
Even though the whole dataframe is supplied at once, a properly written strategy
should still behave as if it is evaluating progressively bar by bar internally.

So this mode is NOT intended to "cheat".
It is simply the normal batch/backtest execution style.

----------------------------------------------------------------------
LIVE-STYLE ROLLING REPLAY
----------------------------------------------------------------------

Definition
----------
The live-style rolling replay simulates live trading behavior more closely.

Instead of giving the strategy the whole selected window in one shot, the script:

1. warms up using an initial minimum number of bars
2. then adds ONE newly closed bar at a time
3. reruns the strategy on only the bars available up to that moment

Conceptually
------------
This is equivalent to saying:

    "Pretend we are live.
     At each newly closed candle, re-evaluate using only the information that
     would have existed at that time."

So the model experiences the data progressively:
- first 40 bars
- then 41 bars
- then 42 bars
- then 43 bars
- etc.

What it tells us
----------------
The rolling replay helps verify:
- whether incremental execution matches backtest-style execution
- whether state persists correctly across re-runs
- whether pending setups are preserved correctly
- whether entries occur only when they should
- whether any hidden lookahead/state bug exists

Rolling replay output includes:
- replay_steps
- emitted_signals_total
- final pending state
- final stop/target state
- trade log count at the end of replay

----------------------------------------------------------------------
WHAT "PARITY" MEANS HERE
----------------------------------------------------------------------

Parity means that for the same symbol, timeframe, and bar window, we want the
strategy's final behavior to match between:

- full-pass run
- live-style rolling replay

In a healthy result, both sides should usually agree on things like:
- final pending_direction
- final pending_entry_ce
- final stop / targets
- final setup type
- trade log row count
- whether trades were emitted or not

If the two modes diverge, it may indicate:
- a state reset bug
- a lookahead bug
- a live-vs-batch data preparation mismatch
- a resampling / cleaning mismatch
- execution path differences in incremental vs batch mode

----------------------------------------------------------------------
IMPORTANT LIMITATION
----------------------------------------------------------------------

This script does NOT place real broker orders.

It validates STRATEGY DECISION PARITY, not EXECUTION PARITY.

So this script answers:

    "Does the strategy logic behave the same in batch mode and live-style replay?"

It does NOT answer:

    "Did the broker receive and fill the order exactly the same way live would?"

Broker execution, latency, rejected orders, slippage, and platform integration are
outside the scope of this parity script.

----------------------------------------------------------------------
DATA SOURCE / REPLAY INTENT
----------------------------------------------------------------------

This parity script is intended to use replay bars for a recent intraday window,
typically within the last 24-48 hours, so that we can compare:
- current live-model behavior
- strategy behavior on recent market structure
without relying on older frozen local cache unless explicitly desired.

Typical ideal use:
- choose a recent window where live logs showed a pending setup, rejection, or no trade
- replay that exact recent window
- compare full-pass and rolling-replay summaries
- inspect whether no-trade behavior is genuine or caused by a mismatch

Important cache safety rule
---------------------------
This script must NEVER write into:

    src/data/databento_cache

That folder is your main historical/backtest cache.

Parity replay data is instead written into:

    src/data/replay_cache

That makes parity replay a separate scratch cache and prevents accidental overwrite
of long-history backtest files like:

    MES_1m.parquet
    MNQ_1m.parquet
    MYM_1m.parquet
    MGC_1m.parquet
    MCL_1m.parquet

----------------------------------------------------------------------
HOW TO INTERPRET RESULTS
----------------------------------------------------------------------

Case 1: Full-pass and rolling replay both show no trades
--------------------------------------------------------
This usually means the strategy genuinely found no completed entries in that window.

Case 2: Both show same pending/rejection state
----------------------------------------------
This suggests the strategy is behaving consistently and no live-vs-backtest mismatch
is present for that window.

Case 3: Full-pass shows a trade but rolling replay does not
-----------------------------------------------------------
This is a strong signal that something differs between batch mode and live-style
incremental processing.

Case 4: Rolling replay shows different final pending state
----------------------------------------------------------
This suggests state handling or setup carry-forward logic may differ between modes.

----------------------------------------------------------------------
HIGH-LEVEL WORKFLOW
----------------------------------------------------------------------

1. Parse CLI arguments:
   - symbol
   - timeframe
   - start
   - end
   - output directory

2. Check parity scratch cache in:
   - src/data/replay_cache

3. If no cached replay file exists, fetch replay bars for the requested recent
   intraday window from Databento.

4. Save replay 1-minute bars ONLY into:
   - src/data/replay_cache

5. Clean / normalize replay bars so they match strategy expectations.

6. Resample 1-minute replay bars into the requested target timeframe (for example 5m).

7. Run FULL-PASS strategy evaluation on the entire selected timeframe dataframe.

8. Run LIVE-STYLE ROLLING REPLAY by progressively expanding the dataframe and
   re-evaluating one closed bar at a time.

9. Build summaries for:
   - full_pass
   - rolling_replay

10. Compute a DIFF block so mismatches are easy to inspect.

11. Save:
   - parity report JSON
   - replay bars CSV
   - optional logs/output artifacts

----------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------

Single symbol example
---------------------
python src/strategies/deployed/top_bottom_ticking/parity_check.py \
  --symbol MYM \
  --timeframe 5m \
  --start "2026-04-22T18:30:00+00:00" \
  --end "2026-04-20T23:15:00+00:00" \
  --output-dir /Users/Abdullahi/trading-project/trading_system/src/strategies/deployed/top_bottom_ticking/logs

What this does
--------------
- fetches replay bars for MYM over the selected recent window
- writes replay scratch data only into src/data/replay_cache
- builds timeframe bars
- runs full-pass
- runs live-style rolling replay
- compares the two
- saves JSON report and bars CSV

Typical output files
--------------------
- parity_<symbol>_<timeframe>_<start>_<end>.json
- parity_<symbol>_<timeframe>_<start>_<end>_bars.csv

----------------------------------------------------------------------
SUMMARY
----------------------------------------------------------------------

Use this script when you want to answer:

    "Is the strategy logic behaving the same way in backtest-style processing
     and live-style bar-by-bar replay for this recent market window?"

If yes:
- your no-trade / pending / rejection behavior is likely genuine for that window

If no:
- investigate state handling, resampling, filtering, and incremental execution logic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import live_model as lm

try:
    import databento as db
except ImportError as exc:
    raise SystemExit(
        "databento package is not installed in this venv. "
        "Run: pip install databento"
    ) from exc


DATASET = "GLBX.MDP3"
SCHEMA_1M = "ohlcv-1m"
STYPE_IN = "parent"
DEFAULT_MAX_WAIT_SECONDS = 20

# Databento OHLCV prices are fixed-precision integers for many schemas.
# Divide by 1e9 to get the real decimal price.
PX_SCALE = 1e9

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MAIN_DATABENTO_CACHE_DIR = PROJECT_ROOT / "src" / "data" / "databento_cache"
DEFAULT_REPLAY_CACHE_DIR = PROJECT_ROOT / "src" / "data" / "replay_cache"


def _json_safe(value: Any) -> Any:
    if hasattr(lm, "_json_safe"):
        return lm._json_safe(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str, bytes)) else False:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _ensure_utc(ts_like: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _timeframe_to_rule(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "10m": "10min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[tf]


def _normalize_live_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol not in lm.INSTRUMENTS:
        raise ValueError(f"Unsupported symbol: {symbol}")
    return symbol


def _to_databento_parent_symbol(symbol: str) -> str:
    symbol = _normalize_live_symbol(symbol)
    return f"{symbol}.FUT"


def _record_to_row(record: Any) -> dict[str, Any] | None:
    if not hasattr(record, "ts_event"):
        return None

    needed = ("open", "high", "low", "close", "volume")
    if not all(hasattr(record, field) for field in needed):
        return None

    try:
        ts_event = pd.Timestamp(record.ts_event)
        if ts_event.tzinfo is None:
            ts_event = ts_event.tz_localize("UTC")
        else:
            ts_event = ts_event.tz_convert("UTC")
    except Exception:
        return None

    try:
        open_px = float(record.open) / PX_SCALE
        high_px = float(record.high) / PX_SCALE
        low_px = float(record.low) / PX_SCALE
        close_px = float(record.close) / PX_SCALE
        volume = float(record.volume)
    except Exception:
        return None

    return {
        "ts_event": ts_event,
        "Open": open_px,
        "High": high_px,
        "Low": low_px,
        "Close": close_px,
        "Volume": volume,
    }


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _assert_not_main_cache(path: Path) -> None:
    resolved = path.resolve()
    main_cache = MAIN_DATABENTO_CACHE_DIR.resolve()

    if resolved == main_cache:
        raise ValueError(
            f"Refusing to use main databento cache for parity scratch data: {resolved}"
        )

    if main_cache in resolved.parents:
        raise ValueError(
            f"Refusing to write inside main databento cache tree: {resolved}"
        )


def _build_replay_cache_file(
    replay_cache_dir: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    return replay_cache_dir / (
        f"{symbol}_1m_replay_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}.parquet"
    )


def _load_replay_cache(replay_cache_file: Path) -> pd.DataFrame | None:
    if not replay_cache_file.exists():
        return None

    df = pd.read_parquet(replay_cache_file)
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
            df = df.set_index("ts_event")
        else:
            raise ValueError(
                f"Replay cache file has no DatetimeIndex or ts_event column: {replay_cache_file}"
            )

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df.sort_index()


def _save_replay_cache(df_1m: pd.DataFrame, replay_cache_file: Path) -> None:
    _assert_not_main_cache(replay_cache_file.parent)
    _ensure_dir(replay_cache_file.parent)
    df_1m.to_parquet(replay_cache_file)


def fetch_replay_1m(symbol: str, start: pd.Timestamp, end: pd.Timestamp, max_wait_seconds: int) -> pd.DataFrame:
    symbol = _normalize_live_symbol(symbol)
    db_symbol = _to_databento_parent_symbol(symbol)
    rows: list[dict[str, Any]] = []

    client = db.Live()

    def _on_record(record: Any) -> None:
        row = _record_to_row(record)
        if row is None:
            return
        ts_event = row["ts_event"]
        if start <= ts_event <= end:
            rows.append(row)

    client.add_callback(_on_record)

    client.subscribe(
        dataset=DATASET,
        schema=SCHEMA_1M,
        stype_in=STYPE_IN,
        symbols=[db_symbol],
        start=start.isoformat(),
    )

    try:
        client.start()
        client.block_for_close(timeout=max_wait_seconds)
    finally:
        try:
            client.stop()
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(rows)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.sort_values("ts_event").drop_duplicates(subset=["ts_event"]).set_index("ts_event")
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return df


def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m.copy()

    tf = timeframe.strip().lower()
    if tf == "1m":
        return df_1m.copy()

    rule = _timeframe_to_rule(tf)

    out = (
        df_1m.resample(rule, label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    return out


def _run_pass_on_df(symbol: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = lm.INSTRUMENTS[symbol]
    strategy_cls = lm._make_strategy_class(cfg)
    bt_df = lm._to_bt_index(df)

    bt = lm.Backtest(
        bt_df,
        strategy_cls,
        cash=lm.LIVE_BACKTEST_CASH,
        commission=0.0,
        exclusive_orders=True,
        trade_on_close=False,
    )
    bt.run()

    raw_meta = pd.DataFrame(getattr(strategy_cls, "last_trade_log", []))
    raw_debug = getattr(strategy_cls, "last_debug_counts", {}) or {}

    if raw_meta.empty:
        return pd.DataFrame(), raw_debug

    try:
        meta = lm._prepare_meta(raw_meta, cfg, "type2_baseline", lm.PROP_PROFILE_NAME)
    except TypeError:
        meta = lm._prepare_meta(raw_meta, cfg)

    meta = lm._ensure_meta_columns(meta, symbol, cfg)
    return meta, raw_debug


def _summary_from_run(symbol: str, mode: str, df: pd.DataFrame, meta: pd.DataFrame, debug: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "symbol": symbol,
        "mode": mode,
        "bars": int(len(df)),
        "trade_log_rows": int(len(meta)),
        "latest_bar_utc": pd.Timestamp(df.index[-1]).isoformat() if not df.empty else None,
    }

    for key in sorted(debug.keys()):
        if key.startswith("debug_") or key.startswith("pending_") or key in {
            "partial1_taken",
            "partial2_taken",
            "be_moved",
            "active_risk",
            "open_trade_meta",
            "prev_closed_count",
            "effective_min_stop_points",
            "effective_max_stop_points",
            "manual_min_stop_points",
            "manual_max_stop_points",
        }:
            summary[key] = _json_safe(debug[key])

    return summary


def run_full_pass(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    meta, debug = _run_pass_on_df(symbol, df)
    return _summary_from_run(symbol, "backtest_full_pass", df, meta, debug)


def run_rolling_replay(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    cfg = lm.INSTRUMENTS[symbol]
    min_rows = max(
        lm._required_rows_for_timeframe(str(getattr(cfg, "timeframe", "5m"))),
        lm.SIGNAL_LOOKBACK_BARS + 5,
    )

    if len(df) < min_rows:
        raise ValueError(f"Not enough rows for rolling replay: have {len(df)}, need at least {min_rows}")

    final_meta = pd.DataFrame()
    final_debug: dict[str, Any] = {}
    emitted_signals = 0
    replay_steps = 0

    for end_idx in range(min_rows, len(df) + 1):
        replay_steps += 1
        window = df.iloc[:end_idx].copy()
        meta, debug = _run_pass_on_df(symbol, window)
        final_meta = meta
        final_debug = debug
        emitted_signals += len(lm.extract_fresh_entries(meta, window))

    summary = _summary_from_run(symbol, "live_style_rolling_replay", df, final_meta, final_debug)
    summary["replay_steps"] = replay_steps
    summary["emitted_signals_total"] = emitted_signals
    return summary


def build_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    diff = {}
    for key in keys:
        if a.get(key) != b.get(key):
            diff[key] = {
                "full_pass": a.get(key),
                "rolling_replay": b.get(key),
            }
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare top_bottom_ticking full-pass vs live-style replay on Databento intraday replay bars."
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", required=True, help="UTC/offset-aware start timestamp")
    parser.add_argument("--end", required=True, help="UTC/offset-aware end timestamp")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--replay-cache-dir",
        default=str(DEFAULT_REPLAY_CACHE_DIR),
        help="Scratch cache for parity replay bars only. Must not be inside src/data/databento_cache.",
    )
    parser.add_argument("--max-wait-seconds", type=int, default=DEFAULT_MAX_WAIT_SECONDS)
    parser.add_argument(
        "--refresh-replay-cache",
        action="store_true",
        help="Ignore any existing parity replay cache file and fetch replay bars again.",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper().strip()
    timeframe = args.timeframe.strip().lower()

    if symbol not in lm.INSTRUMENTS:
        raise SystemExit(f"Unsupported symbol: {symbol}")

    start = _ensure_utc(args.start)
    end = _ensure_utc(args.end)

    if end <= start:
        raise SystemExit("--end must be after --start")

    now_utc = pd.Timestamp.now(tz="UTC")
    if start < now_utc - pd.Timedelta(hours=24):
        raise SystemExit(
            "Databento Live intraday replay only supports windows inside the last 24 hours."
        )

    out_dir = Path(args.output_dir).resolve()
    replay_cache_dir = Path(args.replay_cache_dir).resolve()

    _assert_not_main_cache(replay_cache_dir)
    _ensure_dir(out_dir)
    _ensure_dir(replay_cache_dir)

    replay_cache_file = _build_replay_cache_file(
        replay_cache_dir=replay_cache_dir,
        symbol=symbol,
        start=start,
        end=end,
    )

    df_1m: pd.DataFrame | None = None

    if not args.refresh_replay_cache:
        df_1m = _load_replay_cache(replay_cache_file)
        if df_1m is not None:
            print(f"Using parity replay cache: {replay_cache_file}")

    if df_1m is None:
        df_1m = fetch_replay_1m(
            symbol=symbol,
            start=start,
            end=end,
            max_wait_seconds=args.max_wait_seconds,
        )

        if df_1m.empty:
            raise SystemExit("No rows returned from Databento intraday replay for parity check.")

        _save_replay_cache(df_1m, replay_cache_file)
        print(f"Saved parity replay cache -> {replay_cache_file}")

    if df_1m.empty:
        raise SystemExit("Replay rows were returned, but the replay cache dataframe is empty.")

    sliced = resample_ohlcv(df_1m, timeframe)
    sliced = lm._clean_live_df(symbol, sliced)

    if sliced.empty:
        raise SystemExit("Replay rows were returned, but none survived cleaning/resampling.")

    full_summary = run_full_pass(symbol, sliced)
    rolling_summary = run_rolling_replay(symbol, sliced)
    diff = build_diff(full_summary, rolling_summary)

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows_1m": int(len(df_1m)),
        "rows_timeframe": int(len(sliced)),
        "replay_cache_file": str(replay_cache_file),
        "full_pass": full_summary,
        "rolling_replay": rolling_summary,
        "diff": diff,
    }

    stem = f"parity_{symbol}_{timeframe}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}"
    out_path = out_dir / f"{stem}.json"
    bars_path = out_dir / f"{stem}_bars.csv"

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    sliced.reset_index().to_csv(bars_path, index=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved parity report -> {out_path}")
    print(f"Saved replay bars   -> {bars_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())