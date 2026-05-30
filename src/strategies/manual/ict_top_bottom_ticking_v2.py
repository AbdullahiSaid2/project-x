from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Optional

import pandas as pd


# =========================================================
# TOP BOTTOM TICKING V2 — LIMIT RETRACE REBUILD
# =========================================================
#
# Core correction:
#
#   Confirmation is not the entry.
#
# Old broken behaviour:
#
#   sweep
#   → rejection
#   → CE touch
#   → internal sweep
#   → MSS
#   → enter immediately at MSS close
#
# New intended behaviour:
#
#   sweep
#   → rejection
#   → CE touch
#   → internal sweep
#   → MSS confirms reversal
#   → arm retrace entry
#   → enter only if price retraces back to CE / FVG midpoint
#
# This better matches the actual model:
#
#   enter around CE of the rejection block after MSS confirmation,
#   not by chasing the MSS candle.
#
# This file replaces:
#
#   src/strategies/manual/ict_top_bottom_ticking_v2.py
#
# Old original strategy remains untouched:
#
#   src/strategies/manual/ict_top_bottom_ticking.py
# =========================================================


# =========================================================
# CONFIG
# =========================================================

DEFAULT_TARGET_R = 10.0

# Flow timing.
PENDING_SWEEP_EXPIRY_BARS = 40
ZONE_EXPIRY_BARS = 100
CONFIRMATION_WINDOW_BARS = 20
ARMED_ENTRY_EXPIRY_BARS = 20

# CE tolerance for zone interaction.
CE_TOLERANCE = 0.35

# If an FVG exists after MSS, use its midpoint as entry.
# Otherwise use RB CE.
USE_FVG_MIDPOINT_ENTRY = True

# When checking whether a retrace entry filled, allow a small tolerance.
ENTRY_TOUCH_TOLERANCE_POINTS = 0.0

LOCAL_SWEEP_LOOKBACK = 20
MSS_LOOKBACK = 6
LOWER_HIGH_LOOKBACK = 8
HIGHER_LOW_LOOKBACK = 8

# Candle quality.
MIN_RB_ATR_MULTIPLE = 0.65
MIN_RB_BODY_RATIO = 0.35

MIN_MSS_ATR_MULTIPLE = 0.85
MIN_MSS_BODY_RATIO = 0.50

MIN_VOLATILITY_RATIO = 0.45

# Trade throttles.
MAX_SIGNALS_PER_DAY = 3
MIN_BARS_BETWEEN_SIGNALS = 20

# Trend/bias.
EMA_FAST = 50
EMA_SLOW = 200
ENABLE_EMA_BIAS_FILTER = True

# Enable both directions, but both require full confirmation.
ENABLE_LONGS = True
ENABLE_SHORTS = True

# External liquidity only.
ALLOW_LOCAL_EXTERNAL_SWEEP_FALLBACK = False

ALLOWED_LONG_LIQUIDITY_TYPES = {
    "PDL",
    "PSL_LONDON",
    "PSL_NY_AM",
}

ALLOWED_SHORT_LIQUIDITY_TYPES = {
    "PDH",
    "PSH_LONDON",
    "PSH_NY_AM",
}

# ET killzones.
LONDON_KILLZONE = (time(2, 0), time(5, 30))
NY_AM = (time(8, 30), time(11, 30))
NY_PM = (time(13, 30), time(15, 30))

LONDON_SESSION = (time(2, 0), time(5, 30))
NY_AM_SESSION = (time(8, 30), time(11, 30))

# Symbol-aware stop geometry.
SYMBOL_STOP_RULES = {
    "MNQ": {"min_stop": 6.0, "max_stop": 30.0, "buffer": 0.25},
    "MES": {"min_stop": 3.0, "max_stop": 15.0, "buffer": 0.25},
    "MYM": {"min_stop": 20.0, "max_stop": 120.0, "buffer": 1.0},
    "MGC": {"min_stop": 0.8, "max_stop": 6.0, "buffer": 0.1},
    "MCL": {"min_stop": 0.05, "max_stop": 0.6, "buffer": 0.01},
}

DEFAULT_STOP_RULES = {"min_stop": 3.0, "max_stop": 30.0, "buffer": 0.25}

DIAG_DIR = Path("src/backtesting/event_engine/outputs")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_PATH = DIAG_DIR / "top_bottom_ticking_v2_limit_retrace_diagnostics.csv"


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class PendingSweep:
    direction: str
    liquidity_type: str
    swept_level: float
    sweep_i: int
    sweep_ts: object
    used: bool = False

    def expired(self, i: int) -> bool:
        return (i - self.sweep_i) > PENDING_SWEEP_EXPIRY_BARS


@dataclass
class RejectionZone:
    direction: str
    high: float
    low: float
    ce: float
    created_i: int
    created_ts: object
    liquidity_type: str
    swept_level: float
    touched_i: Optional[int] = None
    touched_ts: Optional[object] = None
    internal_swept: bool = False
    internal_sweep_i: Optional[int] = None
    internal_sweep_ts: Optional[object] = None
    used: bool = False

    @property
    def width(self) -> float:
        return self.high - self.low

    def expired(self, i: int) -> bool:
        return (i - self.created_i) > ZONE_EXPIRY_BARS


@dataclass
class ArmedEntry:
    direction: str
    entry: float
    stop: float
    target: float
    risk: float
    target_r: float
    setup: str
    liquidity_type: str
    zone_created_ts: object
    zone_high: float
    zone_low: float
    zone_ce: float
    armed_i: int
    armed_ts: object
    used: bool = False

    def expired(self, i: int) -> bool:
        return (i - self.armed_i) > ARMED_ENTRY_EXPIRY_BARS


# =========================================================
# COLUMN / TIME HELPERS
# =========================================================

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(
            f"top_bottom_ticking_v2 missing OHLC columns: {missing}. "
            f"Available columns: {list(out.columns)}"
        )

    return out


def _to_et_timestamp(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York")


def _add_et_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    et_index = pd.Series([_to_et_timestamp(x) for x in out.index], index=out.index)
    out["_et_ts"] = et_index
    out["_et_date"] = et_index.dt.date
    out["_et_time"] = et_index.dt.time
    return out


def _time_in_window(t, window) -> bool:
    return window[0] <= t <= window[1]


def allowed_session(ts) -> bool:
    ts_et = _to_et_timestamp(ts)
    t = ts_et.time()

    return (
        LONDON_KILLZONE[0] <= t <= LONDON_KILLZONE[1]
        or NY_AM[0] <= t <= NY_AM[1]
        or NY_PM[0] <= t <= NY_PM[1]
    )


def stop_rules_for_symbol(symbol: Optional[str]) -> dict:
    if not symbol:
        return DEFAULT_STOP_RULES
    return SYMBOL_STOP_RULES.get(str(symbol).upper(), DEFAULT_STOP_RULES)


# =========================================================
# INDICATORS / LEVELS
# =========================================================

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def add_bias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    out["bias_long"] = (
        (out["close"] > out["ema_fast"])
        & (out["ema_fast"] > out["ema_slow"])
    )

    out["bias_short"] = (
        (out["close"] < out["ema_fast"])
        & (out["ema_fast"] < out["ema_slow"])
    )

    return out


def add_liquidity_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_et_columns(df)

    daily = out.groupby("_et_date").agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
    )

    daily["pdh"] = daily["day_high"].shift(1)
    daily["pdl"] = daily["day_low"].shift(1)

    out = out.join(daily[["pdh", "pdl"]], on="_et_date")

    out["_session_name"] = ""
    out.loc[out["_et_time"].apply(lambda t: _time_in_window(t, LONDON_SESSION)), "_session_name"] = "london"
    out.loc[out["_et_time"].apply(lambda t: _time_in_window(t, NY_AM_SESSION)), "_session_name"] = "ny_am"

    session_rows = out[out["_session_name"].isin(["london", "ny_am"])].copy()

    if not session_rows.empty:
        session_levels = (
            session_rows
            .groupby(["_session_name", "_et_date"])
            .agg(
                session_high=("high", "max"),
                session_low=("low", "min"),
            )
            .reset_index()
            .sort_values(["_session_name", "_et_date"])
        )

        session_levels["psh"] = session_levels.groupby("_session_name")["session_high"].shift(1)
        session_levels["psl"] = session_levels.groupby("_session_name")["session_low"].shift(1)

        london = session_levels[session_levels["_session_name"] == "london"][
            ["_et_date", "psh", "psl"]
        ].rename(columns={"psh": "prev_london_high", "psl": "prev_london_low"})

        ny = session_levels[session_levels["_session_name"] == "ny_am"][
            ["_et_date", "psh", "psl"]
        ].rename(columns={"psh": "prev_ny_am_high", "psl": "prev_ny_am_low"})

        out = out.join(london.set_index("_et_date"), on="_et_date")
        out = out.join(ny.set_index("_et_date"), on="_et_date")
    else:
        out["prev_london_high"] = pd.NA
        out["prev_london_low"] = pd.NA
        out["prev_ny_am_high"] = pd.NA
        out["prev_ny_am_low"] = pd.NA

    return out


# =========================================================
# CANDLE HELPERS
# =========================================================

def candle_range(candle) -> float:
    return float(candle.high) - float(candle.low)


def candle_body(candle) -> float:
    return abs(float(candle.close) - float(candle.open))


def candle_body_ratio(candle) -> float:
    rng = candle_range(candle)
    if rng <= 0:
        return 0.0
    return candle_body(candle) / rng


def is_bullish(candle) -> bool:
    return float(candle.close) > float(candle.open)


def is_bearish(candle) -> bool:
    return float(candle.close) < float(candle.open)


def volatility_ok(current_atr, rolling_atr) -> bool:
    if pd.isna(current_atr) or pd.isna(rolling_atr):
        return False
    return float(current_atr) >= float(rolling_atr) * MIN_VOLATILITY_RATIO


def bias_allows(direction: str, candle) -> bool:
    if not ENABLE_EMA_BIAS_FILTER:
        return True

    if direction == "long":
        return bool(candle.bias_long)

    return bool(candle.bias_short)


# =========================================================
# EXTERNAL LIQUIDITY SWEEP DETECTION
# =========================================================

def _sweep_level(value, level, direction: str) -> bool:
    if pd.isna(level):
        return False

    if direction == "low":
        return float(value) < float(level)

    return float(value) > float(level)


def external_sweep(candle, direction: str) -> tuple[str, float]:
    if direction == "long":
        if _sweep_level(candle.low, candle.pdl, "low"):
            return "PDL", float(candle.pdl)
        if _sweep_level(candle.low, candle.prev_london_low, "low"):
            return "PSL_LONDON", float(candle.prev_london_low)
        if _sweep_level(candle.low, candle.prev_ny_am_low, "low"):
            return "PSL_NY_AM", float(candle.prev_ny_am_low)
        return "", 0.0

    if _sweep_level(candle.high, candle.pdh, "high"):
        return "PDH", float(candle.pdh)
    if _sweep_level(candle.high, candle.prev_london_high, "high"):
        return "PSH_LONDON", float(candle.prev_london_high)
    if _sweep_level(candle.high, candle.prev_ny_am_high, "high"):
        return "PSH_NY_AM", float(candle.prev_ny_am_high)
    return "", 0.0


def local_internal_sweep(df: pd.DataFrame, i: int, direction: str) -> bool:
    if i - LOCAL_SWEEP_LOOKBACK < 0:
        return False

    candle = df.iloc[i]

    if direction == "long":
        prior_low = float(df.iloc[i - LOCAL_SWEEP_LOOKBACK:i]["low"].min())
        return float(candle.low) < prior_low

    prior_high = float(df.iloc[i - LOCAL_SWEEP_LOOKBACK:i]["high"].max())
    return float(candle.high) > prior_high


def liquidity_allowed(direction: str, liquidity_type: str) -> bool:
    if direction == "long":
        return liquidity_type in ALLOWED_LONG_LIQUIDITY_TYPES

    return liquidity_type in ALLOWED_SHORT_LIQUIDITY_TYPES


# =========================================================
# PENDING SWEEP / RB / CE
# =========================================================

def create_pending_sweep(candle, i: int, direction: str) -> Optional[PendingSweep]:
    if direction == "long" and not ENABLE_LONGS:
        return None
    if direction == "short" and not ENABLE_SHORTS:
        return None

    liquidity_type, swept_level = external_sweep(candle, direction)

    if not liquidity_type:
        return None

    if not liquidity_allowed(direction, liquidity_type):
        return None

    return PendingSweep(
        direction=direction,
        liquidity_type=liquidity_type,
        swept_level=swept_level,
        sweep_i=i,
        sweep_ts=candle.name,
    )


def valid_rejection_after_sweep(candle, pending: PendingSweep, atr: float) -> bool:
    if pd.isna(atr) or atr <= 0:
        return False

    if candle_range(candle) < float(atr) * MIN_RB_ATR_MULTIPLE:
        return False

    if candle_body_ratio(candle) < MIN_RB_BODY_RATIO:
        return False

    if pending.direction == "long":
        return is_bullish(candle) and float(candle.close) > float(pending.swept_level)

    return is_bearish(candle) and float(candle.close) < float(pending.swept_level)


def calculate_ce(high: float, low: float) -> float:
    return (float(high) + float(low)) / 2.0


def build_zone_from_pending(candle, i: int, pending: PendingSweep, atr: float) -> Optional[RejectionZone]:
    if not valid_rejection_after_sweep(candle, pending, atr):
        return None

    if not bias_allows(pending.direction, candle):
        return None

    return RejectionZone(
        direction=pending.direction,
        high=float(candle.high),
        low=float(candle.low),
        ce=calculate_ce(candle.high, candle.low),
        created_i=i,
        created_ts=candle.name,
        liquidity_type=pending.liquidity_type,
        swept_level=pending.swept_level,
    )


def price_touches_zone(candle, zone: RejectionZone) -> bool:
    return float(candle.low) <= zone.high and float(candle.high) >= zone.low


def price_touches_ce(candle, zone: RejectionZone) -> bool:
    if zone.width <= 0:
        return False

    tolerance = zone.width * CE_TOLERANCE

    return (
        float(candle.low) <= zone.ce + tolerance
        and float(candle.high) >= zone.ce - tolerance
    )


def mark_zone_touch(zone: RejectionZone, candle, i: int):
    if zone.touched_i is None:
        zone.touched_i = i
        zone.touched_ts = candle.name


# =========================================================
# CONFIRMATION AFTER CE
# =========================================================

def internal_sweep_after_ce(df: pd.DataFrame, i: int, zone: RejectionZone) -> bool:
    return local_internal_sweep(df, i, zone.direction)


def valid_mss_displacement(candle, atr: float) -> bool:
    if pd.isna(atr) or atr <= 0:
        return False

    return (
        candle_range(candle) >= float(atr) * MIN_MSS_ATR_MULTIPLE
        and candle_body_ratio(candle) >= MIN_MSS_BODY_RATIO
    )


def lower_high_confirmed(df: pd.DataFrame, i: int) -> bool:
    if i - LOWER_HIGH_LOOKBACK < 0:
        return False

    recent_high = float(df.iloc[i - LOWER_HIGH_LOOKBACK:i]["high"].max())
    current_high = float(df.iloc[i]["high"])

    return current_high < recent_high


def higher_low_confirmed(df: pd.DataFrame, i: int) -> bool:
    if i - HIGHER_LOW_LOOKBACK < 0:
        return False

    recent_low = float(df.iloc[i - HIGHER_LOW_LOOKBACK:i]["low"].min())
    current_low = float(df.iloc[i]["low"])

    return current_low > recent_low


def mss_after_ce(df: pd.DataFrame, i: int, zone: RejectionZone, atr: float) -> bool:
    if zone.touched_i is None:
        return False

    if i <= zone.touched_i:
        return False

    if (i - zone.touched_i) > CONFIRMATION_WINDOW_BARS:
        return False

    if i - MSS_LOOKBACK < 0:
        return False

    candle = df.iloc[i]

    if not valid_mss_displacement(candle, atr):
        return False

    if not bias_allows(zone.direction, candle):
        return False

    if zone.direction == "long":
        prior_high = float(df.iloc[i - MSS_LOOKBACK:i]["high"].max())
        return (
            float(candle.close) > prior_high
            and is_bullish(candle)
            and higher_low_confirmed(df, i)
        )

    prior_low = float(df.iloc[i - MSS_LOOKBACK:i]["low"].min())
    return (
        float(candle.close) < prior_low
        and is_bearish(candle)
        and lower_high_confirmed(df, i)
    )


# =========================================================
# FVG / ENTRY PRICE
# =========================================================

def bullish_fvg(c1, c2, c3) -> Optional[dict]:
    if float(c1.high) < float(c3.low):
        return {
            "lower": float(c1.high),
            "upper": float(c3.low),
            "mid": (float(c1.high) + float(c3.low)) / 2.0,
        }
    return None


def bearish_fvg(c1, c2, c3) -> Optional[dict]:
    if float(c1.low) > float(c3.high):
        return {
            "lower": float(c3.high),
            "upper": float(c1.low),
            "mid": (float(c3.high) + float(c1.low)) / 2.0,
        }
    return None


def recent_fvg(df: pd.DataFrame, i: int, direction: str, lookback: int = 5) -> Optional[dict]:
    start = max(2, i - lookback + 1)

    for j in range(i, start - 1, -1):
        if direction == "long":
            fvg = bullish_fvg(df.iloc[j - 2], df.iloc[j - 1], df.iloc[j])
        else:
            fvg = bearish_fvg(df.iloc[j - 2], df.iloc[j - 1], df.iloc[j])

        if fvg:
            return fvg

    return None


def choose_retrace_entry(zone: RejectionZone, fvg: Optional[dict]) -> tuple[float, str]:
    if USE_FVG_MIDPOINT_ENTRY and fvg is not None:
        return float(fvg["mid"]), "fvg_mid"

    return float(zone.ce), "rb_ce"


def entry_touched(candle, armed: ArmedEntry) -> bool:
    entry = float(armed.entry)

    if armed.direction == "long":
        return float(candle.low) <= entry + ENTRY_TOUCH_TOLERANCE_POINTS

    return float(candle.high) >= entry - ENTRY_TOUCH_TOLERANCE_POINTS


# =========================================================
# ORDER PLAN / STOP GEOMETRY
# =========================================================

def adjust_long_stop(entry: float, raw_stop: float, rules: dict) -> Optional[float]:
    min_stop = float(rules["min_stop"])
    max_stop = float(rules["max_stop"])

    risk = entry - raw_stop

    if risk <= 0:
        return None

    if risk < min_stop:
        raw_stop = entry - min_stop
        risk = min_stop

    if risk > max_stop:
        return None

    return raw_stop


def adjust_short_stop(entry: float, raw_stop: float, rules: dict) -> Optional[float]:
    min_stop = float(rules["min_stop"])
    max_stop = float(rules["max_stop"])

    risk = raw_stop - entry

    if risk <= 0:
        return None

    if risk < min_stop:
        raw_stop = entry + min_stop
        risk = min_stop

    if risk > max_stop:
        return None

    return raw_stop


def arm_retrace_entry(
    mss_candle,
    zone: RejectionZone,
    fvg: Optional[dict],
    symbol: Optional[str],
    target_r: float,
    armed_i: int,
) -> Optional[ArmedEntry]:
    rules = stop_rules_for_symbol(symbol)
    buffer = float(rules["buffer"])

    entry, entry_type = choose_retrace_entry(zone, fvg)

    if zone.direction == "long":
        raw_stop = float(zone.low) - buffer
        stop = adjust_long_stop(entry, raw_stop, rules)
        if stop is None:
            return None

        risk = entry - stop
        target = entry + (risk * target_r)

        return ArmedEntry(
            direction="long",
            entry=entry,
            stop=stop,
            target=target,
            risk=risk,
            target_r=target_r,
            setup=(
                "top_bottom_ticking_v2_long_"
                + zone.liquidity_type
                + "_"
                + entry_type
                + "_confirmed_limit_retrace"
            ),
            liquidity_type=zone.liquidity_type,
            zone_created_ts=zone.created_ts,
            zone_high=zone.high,
            zone_low=zone.low,
            zone_ce=zone.ce,
            armed_i=armed_i,
            armed_ts=mss_candle.name,
        )

    raw_stop = float(zone.high) + buffer
    stop = adjust_short_stop(entry, raw_stop, rules)
    if stop is None:
        return None

    risk = stop - entry
    target = entry - (risk * target_r)

    return ArmedEntry(
        direction="short",
        entry=entry,
        stop=stop,
        target=target,
        risk=risk,
        target_r=target_r,
        setup=(
            "top_bottom_ticking_v2_short_"
            + zone.liquidity_type
            + "_"
            + entry_type
            + "_confirmed_limit_retrace"
        ),
        liquidity_type=zone.liquidity_type,
        zone_created_ts=zone.created_ts,
        zone_high=zone.high,
        zone_low=zone.low,
        zone_ce=zone.ce,
        armed_i=armed_i,
        armed_ts=mss_candle.name,
    )


def emit_trade_from_armed(candle, armed: ArmedEntry) -> dict:
    side = "LONG" if armed.direction == "long" else "SHORT"

    return {
        "timestamp": candle.name,
        "side": side,
        "entry": armed.entry,
        "stop": armed.stop,
        "target": armed.target,
        "risk": armed.risk,
        "target_r": armed.target_r,
        "setup": armed.setup,
        "liquidity_type": armed.liquidity_type,
        "zone_created_ts": armed.zone_created_ts,
        "zone_high": armed.zone_high,
        "zone_low": armed.zone_low,
        "zone_ce": armed.zone_ce,
        "armed_ts": armed.armed_ts,
    }


# =========================================================
# DIAGNOSTICS
# =========================================================

def _empty_diag() -> dict:
    return {
        "bars_seen": 0,
        "blocked_session": 0,
        "blocked_atr": 0,
        "blocked_volatility": 0,
        "pending_sweeps_long": 0,
        "pending_sweeps_short": 0,
        "pending_sweeps_expired": 0,
        "zones_created_long": 0,
        "zones_created_short": 0,
        "zones_expired": 0,
        "zone_touches": 0,
        "ce_touches": 0,
        "internal_sweeps_after_ce": 0,
        "mss_after_ce": 0,
        "armed_entries_long": 0,
        "armed_entries_short": 0,
        "armed_entries_expired": 0,
        "rejected_stop_geometry": 0,
        "long_signals": 0,
        "short_signals": 0,
        "target_r_used": 0,
    }


def _write_diag(diag: dict):
    pd.DataFrame([diag]).to_csv(DIAG_PATH, index=False)


# =========================================================
# MAIN SIGNAL GENERATOR
# =========================================================

def generate_top_bottom_ticking_v2_signals(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    target_r: Optional[float] = None,
) -> pd.DataFrame:
    if target_r is None:
        target_r = DEFAULT_TARGET_R

    target_r = float(target_r)

    df = _normalise_columns(df)
    df = add_liquidity_levels(df)
    df = add_bias_columns(df)

    if "atr" not in df.columns:
        df["atr"] = _atr(df, 14)

    df["rolling_atr"] = df["atr"].rolling(50, min_periods=50).mean()

    pending_sweeps: list[PendingSweep] = []
    active_zones: list[RejectionZone] = []
    armed_entries: list[ArmedEntry] = []

    trades: list[dict] = []
    diag = _empty_diag()
    diag["target_r_used"] = target_r

    current_day = None
    signals_today = 0
    last_signal_i = -999999

    for i in range(60, len(df) - 1):
        diag["bars_seen"] += 1

        candle = df.iloc[i]
        ts = candle.name
        ts_et = _to_et_timestamp(ts)

        if current_day != ts_et.date():
            current_day = ts_et.date()
            signals_today = 0

        if not allowed_session(ts):
            diag["blocked_session"] += 1
            continue

        atr = candle.atr
        rolling_atr = candle.rolling_atr

        if pd.isna(atr) or pd.isna(rolling_atr):
            diag["blocked_atr"] += 1
            continue

        if not volatility_ok(atr, rolling_atr):
            diag["blocked_volatility"] += 1
            continue

        # -----------------------------------------------------
        # 0. fill armed retrace entries first
        # -----------------------------------------------------

        still_armed = []
        for armed in armed_entries:
            if armed.used:
                continue

            if armed.expired(i):
                diag["armed_entries_expired"] += 1
                continue

            if signals_today >= MAX_SIGNALS_PER_DAY:
                still_armed.append(armed)
                continue

            if (i - last_signal_i) < MIN_BARS_BETWEEN_SIGNALS:
                still_armed.append(armed)
                continue

            if entry_touched(candle, armed):
                trades.append(emit_trade_from_armed(candle, armed))
                armed.used = True
                signals_today += 1
                last_signal_i = i

                if armed.direction == "long":
                    diag["long_signals"] += 1
                else:
                    diag["short_signals"] += 1
            else:
                still_armed.append(armed)

        armed_entries = still_armed

        # -----------------------------------------------------
        # clean expired pending sweeps
        # -----------------------------------------------------

        new_pending = []
        for p in pending_sweeps:
            if p.used:
                continue
            if p.expired(i):
                diag["pending_sweeps_expired"] += 1
                continue
            new_pending.append(p)
        pending_sweeps = new_pending

        # -----------------------------------------------------
        # clean expired zones
        # -----------------------------------------------------

        new_zones = []
        for z in active_zones:
            if z.used:
                continue
            if z.expired(i):
                diag["zones_expired"] += 1
                continue
            new_zones.append(z)
        active_zones = new_zones

        # -----------------------------------------------------
        # 1. external sweep only creates pending setup
        # -----------------------------------------------------

        long_pending = create_pending_sweep(candle, i, "long")
        if long_pending:
            pending_sweeps.append(long_pending)
            diag["pending_sweeps_long"] += 1

        short_pending = create_pending_sweep(candle, i, "short")
        if short_pending:
            pending_sweeps.append(short_pending)
            diag["pending_sweeps_short"] += 1

        # -----------------------------------------------------
        # 2. after sweep, wait for rejection block
        # -----------------------------------------------------

        for p in pending_sweeps:
            if p.used:
                continue

            if i <= p.sweep_i:
                continue

            zone = build_zone_from_pending(candle, i, p, atr)
            if zone is None:
                continue

            active_zones.append(zone)
            p.used = True

            if zone.direction == "long":
                diag["zones_created_long"] += 1
            else:
                diag["zones_created_short"] += 1

        # -----------------------------------------------------
        # 3. CE touch -> internal sweep -> MSS -> arm entry
        # -----------------------------------------------------

        for zone in active_zones:
            if zone.used:
                continue

            if not price_touches_zone(candle, zone):
                continue

            diag["zone_touches"] += 1

            if not price_touches_ce(candle, zone):
                continue

            diag["ce_touches"] += 1
            mark_zone_touch(zone, candle, i)

            if internal_sweep_after_ce(df, i, zone):
                zone.internal_swept = True
                zone.internal_sweep_i = i
                zone.internal_sweep_ts = ts
                diag["internal_sweeps_after_ce"] += 1

            if not zone.internal_swept:
                continue

            if not mss_after_ce(df, i, zone, atr):
                continue

            diag["mss_after_ce"] += 1

            fvg = recent_fvg(df, i, zone.direction, lookback=5)

            armed = arm_retrace_entry(
                mss_candle=candle,
                zone=zone,
                fvg=fvg,
                symbol=symbol,
                target_r=target_r,
                armed_i=i,
            )

            if armed is None:
                diag["rejected_stop_geometry"] += 1
                zone.used = True
                continue

            armed_entries.append(armed)
            zone.used = True

            if armed.direction == "long":
                diag["armed_entries_long"] += 1
            else:
                diag["armed_entries_short"] += 1

    _write_diag(diag)

    return pd.DataFrame(trades)
