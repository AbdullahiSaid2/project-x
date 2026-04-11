from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def load_external_market(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    try:
        from src.data.fetcher import get_ohlcv
    except Exception:
        return None

    try:
        days_back = max(
            30,
            int((pd.Timestamp.utcnow().tz_localize(None) - start.to_pydatetime()).days) + 5,
        )
        df = get_ohlcv(symbol, exchange="tradovate", timeframe="30m", days_back=days_back)
        if df is None or df.empty:
            return None

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[(df.index >= start.floor("D")) & (df.index <= end.ceil("D"))]
        if df.empty:
            return None
        return df
    except Exception:
        return None


def build_daily_context_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    idx_utc = pd.DatetimeIndex(out.index).tz_localize("UTC")
    idx_ny = idx_utc.tz_convert("America/New_York")
    out["ny_date"] = idx_ny.date

    day_agg = out.groupby("ny_date").agg(
        day_open=("Open", "first"),
        day_high=("High", "max"),
        day_low=("Low", "min"),
        day_close=("Close", "last"),
    )

    day_agg["day_ema20"] = ema(day_agg["day_close"], 20)
    day_agg["day_ema50"] = ema(day_agg["day_close"], 50)
    day_agg["day_eq"] = (day_agg["day_high"] + day_agg["day_low"]) / 2.0

    day_rng = (day_agg["day_high"] - day_agg["day_low"]).replace(0, np.nan)
    day_agg["bull_daily_bias"] = day_agg["day_ema20"] > day_agg["day_ema50"]
    day_agg["bear_daily_bias"] = day_agg["day_ema20"] < day_agg["day_ema50"]
    day_agg["bull_profile"] = (
        (day_agg["day_close"] > day_agg["day_open"])
        & (((day_agg["day_close"] - day_agg["day_low"]) / day_rng) >= 0.6)
    )
    day_agg["bear_profile"] = (
        (day_agg["day_close"] < day_agg["day_open"])
        & (((day_agg["day_high"] - day_agg["day_close"]) / day_rng) >= 0.6)
    )

    out = out.merge(
        day_agg[
            [
                "day_open",
                "day_high",
                "day_low",
                "day_close",
                "day_eq",
                "bull_daily_bias",
                "bear_daily_bias",
                "bull_profile",
                "bear_profile",
            ]
        ],
        left_on="ny_date",
        right_index=True,
        how="left",
    )
    return out


def build_4h_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bars_4h = df.resample("4h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    bars_4h["ema20_4h"] = ema(bars_4h["Close"], 20)
    bars_4h["ema50_4h"] = ema(bars_4h["Close"], 50)
    bars_4h["eq_4h"] = (bars_4h["High"].rolling(8).max() + bars_4h["Low"].rolling(8).min()) / 2.0

    bars_4h["atr14_4h"] = atr(bars_4h, 14)
    body_4h = (bars_4h["Close"] - bars_4h["Open"]).abs()
    rng_4h = (bars_4h["High"] - bars_4h["Low"]).replace(0, np.nan)

    bars_4h["bull_disp_4h"] = (bars_4h["Close"] > bars_4h["Open"]) & (body_4h > bars_4h["atr14_4h"] * 0.8)
    bars_4h["bear_disp_4h"] = (bars_4h["Close"] < bars_4h["Open"]) & (body_4h > bars_4h["atr14_4h"] * 0.8)

    bars_4h["bull_fvg_4h"] = bars_4h["Low"] > bars_4h["High"].shift(2)
    bars_4h["bear_fvg_4h"] = bars_4h["High"] < bars_4h["Low"].shift(2)

    bars_4h["bull_profile_4h"] = (
        (bars_4h["Close"] > bars_4h["Open"])
        & (((bars_4h["Close"] - bars_4h["Low"]) / rng_4h) >= 0.6)
    )
    bars_4h["bear_profile_4h"] = (
        (bars_4h["Close"] < bars_4h["Open"])
        & (((bars_4h["High"] - bars_4h["Close"]) / rng_4h) >= 0.6)
    )

    aligned = bars_4h[
        [
            "ema20_4h",
            "ema50_4h",
            "eq_4h",
            "bull_disp_4h",
            "bear_disp_4h",
            "bull_fvg_4h",
            "bear_fvg_4h",
            "bull_profile_4h",
            "bear_profile_4h",
        ]
    ].reindex(out.index, method="ffill")

    for col in aligned.columns:
        out[col] = aligned[col]

    out["bull_4h_bias"] = out["ema20_4h"] > out["ema50_4h"]
    out["bear_4h_bias"] = out["ema20_4h"] < out["ema50_4h"]
    out["above_4h_eq"] = out["Close"] > out["eq_4h"]
    out["below_4h_eq"] = out["Close"] < out["eq_4h"]

    return out


def detect_ttrades_c2_c3(bars_30m: pd.DataFrame) -> pd.DataFrame:
    out = bars_30m.copy()

    prev_high = out["High"].shift(1)
    prev_low = out["Low"].shift(1)

    out["bull_c2"] = (
        (out["Low"] < prev_low)
        & (out["Close"] > prev_low)
        & (out["Close"] < prev_high)
    )

    out["bear_c2"] = (
        (out["High"] > prev_high)
        & (out["Close"] < prev_high)
        & (out["Close"] > prev_low)
    )

    bull_c2_failed = ((out["Low"] < prev_low) & ~(out["bull_c2"])).astype(bool)
    bear_c2_failed = ((out["High"] > prev_high) & ~(out["bear_c2"])).astype(bool)

    prev2_high = out["High"].shift(2)
    prev2_low = out["Low"].shift(2)

    out["bull_c3"] = (
        bull_c2_failed.shift(1, fill_value=False)
        & (out["Close"] > prev2_low)
        & (out["Close"] < prev2_high)
        & (out["Close"] > out["Open"])
    )

    out["bear_c3"] = (
        bear_c2_failed.shift(1, fill_value=False)
        & (out["Close"] < prev2_high)
        & (out["Close"] > prev2_low)
        & (out["Close"] < out["Open"])
    )

    out["bull_c2_or_c3"] = out["bull_c2"] | out["bull_c3"]
    out["bear_c2_or_c3"] = out["bear_c2"] | out["bear_c3"]

    return out


def build_30m_bridge(df_3m: pd.DataFrame) -> pd.DataFrame:
    out = df_3m.copy()

    bars_30m = df_3m.resample("30min").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    bars_30m["atr14_30m"] = atr(bars_30m, 14)
    body = (bars_30m["Close"] - bars_30m["Open"]).abs()
    rng = (bars_30m["High"] - bars_30m["Low"]).replace(0, np.nan)

    bars_30m["bull_close_strong_30m"] = (
        (bars_30m["Close"] > bars_30m["Open"])
        & (((bars_30m["Close"] - bars_30m["Low"]) / rng) >= 0.65)
    )
    bars_30m["bear_close_strong_30m"] = (
        (bars_30m["Close"] < bars_30m["Open"])
        & (((bars_30m["High"] - bars_30m["Close"]) / rng) >= 0.65)
    )

    bars_30m["bull_disp_30m"] = (bars_30m["Close"] > bars_30m["Open"]) & (body > bars_30m["atr14_30m"] * 0.8)
    bars_30m["bear_disp_30m"] = (bars_30m["Close"] < bars_30m["Open"]) & (body > bars_30m["atr14_30m"] * 0.8)

    bars_30m = detect_ttrades_c2_c3(bars_30m)

    aligned = bars_30m[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "bull_close_strong_30m",
            "bear_close_strong_30m",
            "bull_disp_30m",
            "bear_disp_30m",
            "bull_c2",
            "bear_c2",
            "bull_c3",
            "bear_c3",
            "bull_c2_or_c3",
            "bear_c2_or_c3",
        ]
    ].rename(
        columns={
            "Open": "open_30m",
            "High": "high_30m",
            "Low": "low_30m",
            "Close": "close_30m",
        }
    ).reindex(out.index, method="ffill")

    for col in aligned.columns:
        out[col] = aligned[col]

    return out


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    idx_utc = pd.DatetimeIndex(out.index).tz_localize("UTC")
    idx_ny = idx_utc.tz_convert("America/New_York")
    out["ny_hour"] = idx_ny.hour
    out["ny_minute"] = idx_ny.minute
    out["ny_date"] = idx_ny.date

    out["is_0900_window"] = (out["ny_hour"] == 9) & (out["ny_minute"] < 30)
    out["is_0930_to_1030"] = (
        ((out["ny_hour"] == 9) & (out["ny_minute"] >= 30))
        | (out["ny_hour"] == 10)
    )
    out["is_nyam_valid"] = (
        ((out["ny_hour"] == 9) & (out["ny_minute"] >= 0))
        | (out["ny_hour"] == 10)
        | ((out["ny_hour"] == 11) & (out["ny_minute"] == 0))
    )

    prior_session = out[(out["ny_hour"] < 9) | ((out["ny_hour"] == 9) & (out["ny_minute"] < 30))].copy()
    session_levels = prior_session.groupby("ny_date").agg(
        session_high=("High", "max"),
        session_low=("Low", "min"),
    )
    out = out.merge(session_levels, left_on="ny_date", right_index=True, how="left")

    day_agg = out.groupby("ny_date").agg(
        day_high=("High", "max"),
        day_low=("Low", "min"),
    )
    day_agg["prev_day_high"] = day_agg["day_high"].shift(1)
    day_agg["prev_day_low"] = day_agg["day_low"].shift(1)
    out = out.merge(day_agg[["prev_day_high", "prev_day_low"]], left_on="ny_date", right_index=True, how="left")

    return out


def compute_smt_flags(
    nq_30m: pd.DataFrame,
    es_30m: Optional[pd.DataFrame],
    ym_30m: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = pd.DataFrame(index=nq_30m.index)
    out["bullish_smt"] = False
    out["bearish_smt"] = False

    if es_30m is None and ym_30m is None:
        return out

    nq = nq_30m.copy()
    nq["nq_roll_low"] = nq["Low"].rolling(8).min().shift(1)
    nq["nq_roll_high"] = nq["High"].rolling(8).max().shift(1)

    def ext_flags(other: pd.DataFrame, prefix: str) -> pd.DataFrame:
        tmp = other.copy()
        tmp[f"{prefix}_roll_low"] = tmp["Low"].rolling(8).min().shift(1)
        tmp[f"{prefix}_roll_high"] = tmp["High"].rolling(8).max().shift(1)
        tmp[f"{prefix}_break_low"] = tmp["Low"] < tmp[f"{prefix}_roll_low"]
        tmp[f"{prefix}_break_high"] = tmp["High"] > tmp[f"{prefix}_roll_high"]
        return tmp[[f"{prefix}_break_low", f"{prefix}_break_high"]]

    merged = nq[["Low", "High", "nq_roll_low", "nq_roll_high"]].copy()
    merged["nq_break_low"] = merged["Low"] < merged["nq_roll_low"]
    merged["nq_break_high"] = merged["High"] > merged["nq_roll_high"]

    if es_30m is not None:
        merged = merged.join(ext_flags(es_30m, "es"), how="left")
    if ym_30m is not None:
        merged = merged.join(ext_flags(ym_30m, "ym"), how="left")

    bull_smt = pd.Series(False, index=merged.index)
    bear_smt = pd.Series(False, index=merged.index)

    if "es_break_low" in merged.columns:
        bull_smt = bull_smt | (merged["nq_break_low"] & ~merged["es_break_low"].fillna(False))
    if "ym_break_low" in merged.columns:
        bull_smt = bull_smt | (merged["nq_break_low"] & ~merged["ym_break_low"].fillna(False))

    if "es_break_high" in merged.columns:
        bear_smt = bear_smt | (merged["nq_break_high"] & ~merged["es_break_high"].fillna(False))
    if "ym_break_high" in merged.columns:
        bear_smt = bear_smt | (merged["nq_break_high"] & ~merged["ym_break_high"].fillna(False))

    out["bullish_smt"] = bull_smt.fillna(False)
    out["bearish_smt"] = bear_smt.fillna(False)
    return out


def build_model_frame(df_3m: pd.DataFrame, es_30m: Optional[pd.DataFrame], ym_30m: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df_3m.copy()
    out = build_session_features(out)
    out = build_daily_context_from_intraday(out)
    out = build_4h_context(out)
    out = build_30m_bridge(out)

    out["atr14_3m"] = atr(out, 14)
    body_3m = (out["Close"] - out["Open"]).abs()
    rng_3m = (out["High"] - out["Low"]).replace(0, np.nan)

    out["bull_disp_3m"] = (out["Close"] > out["Open"]) & (body_3m > out["atr14_3m"] * 0.6)
    out["bear_disp_3m"] = (out["Close"] < out["Open"]) & (body_3m > out["atr14_3m"] * 0.6)

    out["bull_close_strong_3m"] = ((out["Close"] - out["Low"]) / rng_3m) >= 0.65
    out["bear_close_strong_3m"] = ((out["High"] - out["Close"]) / rng_3m) >= 0.65

    out["swept_session_low"] = out["Low"] < out["session_low"]
    out["swept_session_high"] = out["High"] > out["session_high"]
    out["reclaimed_session_low"] = out["Close"] > out["session_low"]
    out["rejected_session_high"] = out["Close"] < out["session_high"]

    out["swept_prev_day_low"] = out["Low"] < out["prev_day_low"]
    out["swept_prev_day_high"] = out["High"] > out["prev_day_high"]
    out["reclaimed_prev_day_low"] = out["Close"] > out["prev_day_low"]
    out["rejected_prev_day_high"] = out["Close"] < out["prev_day_high"]

    out["weak_bull_delivery_3m"] = ~out["bull_disp_3m"] | ~out["bull_close_strong_3m"]
    out["weak_bear_delivery_3m"] = ~out["bear_disp_3m"] | ~out["bear_close_strong_3m"]

    nq_30m = df_3m.resample("30min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    smt = compute_smt_flags(nq_30m, es_30m, ym_30m).reindex(out.index, method="ffill")
    out["bullish_smt"] = smt["bullish_smt"].fillna(False)
    out["bearish_smt"] = smt["bearish_smt"].fillna(False)

    return out


@dataclass
class PendingSignal:
    direction: str = ""
    entry_trigger: float = np.nan
    stop_price: float = np.nan
    expiry_bar: int = -1


class ICT_NYAM_NQ_Model_V1(Strategy):
    risk_multiple = 3.0
    stop_buffer_atr = 0.10
    fixed_size = 0.1

    # relaxed from previous version
    use_smt_filter = False
    require_prev_day_context = True

    def init(self):
        df = self.data.df.copy()
        start = pd.to_datetime(df.index.min())
        end = pd.to_datetime(df.index.max())

        es_30m = load_external_market("ES", start, end)
        ym_30m = load_external_market("YM", start, end)

        self.m = build_model_frame(df, es_30m=es_30m, ym_30m=ym_30m)

        self.pending = PendingSignal()
        self.active_risk = np.nan

    def _i(self) -> int:
        return len(self.data) - 1

    def _clear_pending(self):
        self.pending = PendingSignal()

    def _latest_trade(self):
        try:
            if self.trades:
                return self.trades[-1]
        except Exception:
            pass
        return None

    def _manage_trade(self):
        trade = self._latest_trade()
        if trade is None or not self.position or not np.isfinite(self.active_risk):
            return

        price = float(self.data.Close[-1])

        if self.position.is_long:
            r_now = (price - float(trade.entry_price)) / self.active_risk
            if r_now >= 1.0 and trade.sl is not None:
                try:
                    trade.sl = max(float(trade.sl), float(trade.entry_price))
                except Exception:
                    pass

        if self.position.is_short:
            r_now = (float(trade.entry_price) - price) / self.active_risk
            if r_now >= 1.0 and trade.sl is not None:
                try:
                    trade.sl = min(float(trade.sl), float(trade.entry_price))
                except Exception:
                    pass

    def _target_above(self, i: int, entry: float, risk: float) -> float:
        row = self.m.iloc[i]
        candidates = []

        for level_name in ("prev_day_high", "session_high", "day_high"):
            level = row.get(level_name, np.nan)
            if pd.notna(level) and float(level) > entry:
                candidates.append(float(level))

        window_high = float(pd.Series(self.data.High[:]).iloc[max(0, i - 100): i + 1].max())
        if window_high > entry:
            candidates.append(window_high)

        if candidates:
            nearest = min(candidates)
            if nearest > entry + risk * 0.5:
                return nearest

        return entry + risk * self.risk_multiple

    def _target_below(self, i: int, entry: float, risk: float) -> float:
        row = self.m.iloc[i]
        candidates = []

        for level_name in ("prev_day_low", "session_low", "day_low"):
            level = row.get(level_name, np.nan)
            if pd.notna(level) and float(level) < entry:
                candidates.append(float(level))

        window_low = float(pd.Series(self.data.Low[:]).iloc[max(0, i - 100): i + 1].min())
        if window_low < entry:
            candidates.append(window_low)

        if candidates:
            nearest = max(candidates)
            if nearest < entry - risk * 0.5:
                return nearest

        return entry - risk * self.risk_multiple

    def next(self):
        self._manage_trade()
        i = self._i()
        if i < 150:
            return

        row = self.m.iloc[i]
        close_now = float(self.data.Close[-1])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])
        atr3 = float(row["atr14_3m"]) if pd.notna(row["atr14_3m"]) else 0.0

        if self.pending.expiry_bar >= 0 and i > self.pending.expiry_bar:
            self._clear_pending()

        if self.pending.direction == "long" and not self.position:
            if close_now >= self.pending.entry_trigger:
                entry = close_now
                stop = float(self.pending.stop_price)
                risk = entry - stop
                if risk > 0:
                    self.active_risk = risk
                    tp = self._target_above(i, entry, risk)
                    self.buy(size=self.fixed_size, sl=stop, tp=tp)
                self._clear_pending()
                return

        if self.pending.direction == "short" and not self.position:
            if close_now <= self.pending.entry_trigger:
                entry = close_now
                stop = float(self.pending.stop_price)
                risk = stop - entry
                if risk > 0:
                    self.active_risk = risk
                    tp = self._target_below(i, entry, risk)
                    self.sell(size=self.fixed_size, sl=stop, tp=tp)
                self._clear_pending()
                return

        if self.position:
            return

        # invalidation stack
        no_clear_bull_dol = pd.isna(row.get("prev_day_high", np.nan)) and pd.isna(row.get("session_high", np.nan))
        no_clear_bear_dol = pd.isna(row.get("prev_day_low", np.nan)) and pd.isna(row.get("session_low", np.nan))

        bull_eq_invalid = not bool(row["above_4h_eq"])
        bear_eq_invalid = not bool(row["below_4h_eq"])

        bull_structure_invalid = not bool(row["bull_fvg_4h"] or row["bull_disp_4h"] or row["bull_profile_4h"])
        bear_structure_invalid = not bool(row["bear_fvg_4h"] or row["bear_disp_4h"] or row["bear_profile_4h"])

        weak_bull_delivery = bool(row["weak_bull_delivery_3m"])
        weak_bear_delivery = bool(row["weak_bear_delivery_3m"])

        wrong_time = not bool(row["is_nyam_valid"])
        late_time = not bool(row["is_0930_to_1030"])

        bull_smt_invalid = self.use_smt_filter and not bool(row["bullish_smt"])
        bear_smt_invalid = self.use_smt_filter and not bool(row["bearish_smt"])

        # relaxed narrative: daily bias no longer hard-required
        bull_narrative = (
            bool(row["bull_4h_bias"])
            and bool(row["above_4h_eq"])
            and bool(row["bull_profile_4h"] or row["bull_disp_4h"] or row["bull_fvg_4h"])
        )

        bear_narrative = (
            bool(row["bear_4h_bias"])
            and bool(row["below_4h_eq"])
            and bool(row["bear_profile_4h"] or row["bear_disp_4h"] or row["bear_fvg_4h"])
        )

        # relaxed context: session OR previous-day sweep
        bull_context = (
            (bool(row["swept_session_low"]) and bool(row["reclaimed_session_low"]))
            or
            (bool(row["swept_prev_day_low"]) and bool(row["reclaimed_prev_day_low"]))
        )

        bear_context = (
            (bool(row["swept_session_high"]) and bool(row["rejected_session_high"]))
            or
            (bool(row["swept_prev_day_high"]) and bool(row["rejected_prev_day_high"]))
        )

        bull_prevday_invalid = False
        bear_prevday_invalid = False
        if self.require_prev_day_context:
            bull_prevday_invalid = not (
                bool(row["swept_prev_day_low"]) or bool(row["swept_session_low"])
            )
            bear_prevday_invalid = not (
                bool(row["swept_prev_day_high"]) or bool(row["swept_session_high"])
            )

        bull_bridge = (
            bool(row["bull_c2_or_c3"])
            and bool(row["bull_disp_30m"])
            and bool(row["bull_close_strong_30m"])
        )

        bear_bridge = (
            bool(row["bear_c2_or_c3"])
            and bool(row["bear_disp_30m"])
            and bool(row["bear_close_strong_30m"])
        )

        bull_execution = (
            bool(row["is_0930_to_1030"])
            and bool(row["bull_disp_3m"])
            and bool(row["bull_close_strong_3m"])
        )

        bear_execution = (
            bool(row["is_0930_to_1030"])
            and bool(row["bear_disp_3m"])
            and bool(row["bear_close_strong_3m"])
        )

        bull_invalid = (
            no_clear_bull_dol
            or bull_eq_invalid
            or bull_structure_invalid
            or weak_bull_delivery
            or wrong_time
            or late_time
            or bull_smt_invalid
            or bull_prevday_invalid
        )

        bear_invalid = (
            no_clear_bear_dol
            or bear_eq_invalid
            or bear_structure_invalid
            or weak_bear_delivery
            or wrong_time
            or late_time
            or bear_smt_invalid
            or bear_prevday_invalid
        )

        long_setup = bull_narrative and bull_context and bull_bridge and bull_execution and not bull_invalid
        short_setup = bear_narrative and bear_context and bear_bridge and bear_execution and not bear_invalid

        if long_setup:
            entry_trigger = high_now
            stop = low_now - (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
            if stop < entry_trigger:
                self.pending.direction = "long"
                self.pending.entry_trigger = entry_trigger
                self.pending.stop_price = stop
                self.pending.expiry_bar = i + 3
            return

        if short_setup:
            entry_trigger = low_now
            stop = high_now + (atr3 * self.stop_buffer_atr if atr3 > 0 else 0.0)
            if stop > entry_trigger:
                self.pending.direction = "short"
                self.pending.entry_trigger = entry_trigger
                self.pending.stop_price = stop
                self.pending.expiry_bar = i + 3
            return