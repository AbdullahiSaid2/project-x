from __future__ import annotations

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Assume cached futures timestamps are naive UTC
    idx_utc = pd.DatetimeIndex(out.index).tz_localize("UTC")
    idx_ny = idx_utc.tz_convert("America/New_York")
    out["ny_hour"] = idx_ny.hour
    out["ny_minute"] = idx_ny.minute
    out["ny_date"] = idx_ny.date

    # NY morning only
    out["is_ny_morning"] = (
        ((out["ny_hour"] == 9) & (out["ny_minute"] >= 30))
        | (out["ny_hour"] == 10)
        | (out["ny_hour"] == 11)
        | ((out["ny_hour"] == 12) & (out["ny_minute"] == 0))
    )

    # Previous day high/low using NY date grouping
    daily = out.copy()
    day_agg = daily.groupby("ny_date").agg(
        day_high=("High", "max"),
        day_low=("Low", "min"),
    )
    day_agg["prev_day_high"] = day_agg["day_high"].shift(1)
    day_agg["prev_day_low"] = day_agg["day_low"].shift(1)

    out = out.merge(
        day_agg[["prev_day_high", "prev_day_low"]],
        left_on="ny_date",
        right_index=True,
        how="left",
    )

    # 1H bias from 15m bars
    close_1h = out["Close"].resample("1h").last()
    ema20_1h = ema(close_1h, 20)
    ema50_1h = ema(close_1h, 50)

    bias_1h = pd.DataFrame(
        {
            "ema20_1h": ema20_1h,
            "ema50_1h": ema50_1h,
        }
    ).reindex(out.index, method="ffill")

    out["ema20_1h"] = bias_1h["ema20_1h"]
    out["ema50_1h"] = bias_1h["ema50_1h"]

    out["bull_bias"] = out["ema20_1h"] > out["ema50_1h"]
    out["bear_bias"] = out["ema20_1h"] < out["ema50_1h"]

    out["atr14"] = atr(out, 14)

    body = (out["Close"] - out["Open"]).abs()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)

    out["body"] = body
    out["range"] = rng

    out["bull_close_strong"] = (
        (out["Close"] > out["Open"])
        & (((out["Close"] - out["Low"]) / rng) >= 0.7)
    )

    out["bear_close_strong"] = (
        (out["Close"] < out["Open"])
        & (((out["High"] - out["Close"]) / rng) >= 0.7)
    )

    out["bull_displacement"] = (
        (out["Close"] > out["Open"])
        & (body > out["atr14"] * 0.8)
    )

    out["bear_displacement"] = (
        (out["Close"] < out["Open"])
        & (body > out["atr14"] * 0.8)
    )

    out["swept_prev_day_low"] = out["Low"] < out["prev_day_low"]
    out["swept_prev_day_high"] = out["High"] > out["prev_day_high"]

    out["reclaimed_prev_day_low"] = out["Close"] > out["prev_day_low"]
    out["rejected_prev_day_high"] = out["Close"] < out["prev_day_high"]

    return out


class PDHSweepRejectPDLSweepReclaim(Strategy):
    """
    Hand-coded truth-test strategy.

    Long:
    - 1H bullish bias
    - NY morning only
    - bar sweeps previous day low
    - closes back above previous day low
    - bullish displacement candle
    - strong close
    - enter on next bar confirmation above reclaim high

    Short:
    - 1H bearish bias
    - NY morning only
    - bar sweeps previous day high
    - closes back below previous day high
    - bearish displacement candle
    - strong close
    - enter on next bar confirmation below rejection low
    """

    risk_multiple = 2.0
    stop_buffer_atr = 0.15
    fixed_size = 0.1

    def init(self):
        self.f = build_features(self.data.df)

        self.pending_long = False
        self.pending_short = False
        self.pending_entry = np.nan
        self.pending_stop = np.nan
        self.pending_expiry = -1

        self.active_risk = np.nan

    def _i(self) -> int:
        return len(self.data) - 1

    def _clear_pending(self):
        self.pending_long = False
        self.pending_short = False
        self.pending_entry = np.nan
        self.pending_stop = np.nan
        self.pending_expiry = -1

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

    def next(self):
        self._manage_trade()
        i = self._i()
        if i < 50:
            return

        row = self.f.iloc[i]
        close_now = float(self.data.Close[-1])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])
        atr_now = float(row["atr14"]) if pd.notna(row["atr14"]) else 0.0

        # expire pending setups after 2 bars
        if self.pending_expiry >= 0 and i > self.pending_expiry:
            self._clear_pending()

        # trigger pending long on confirmed close through level
        if self.pending_long and not self.position:
            if close_now >= self.pending_entry:
                entry = close_now
                stop = float(self.pending_stop)
                risk = entry - stop
                if risk > 0:
                    tp = entry + risk * self.risk_multiple
                    self.buy(size=self.fixed_size, sl=stop, tp=tp)
                    self.active_risk = risk
                self._clear_pending()
                return

        # trigger pending short on confirmed close through level
        if self.pending_short and not self.position:
            if close_now <= self.pending_entry:
                entry = close_now
                stop = float(self.pending_stop)
                risk = stop - entry
                if risk > 0:
                    tp = entry - risk * self.risk_multiple
                    self.sell(size=self.fixed_size, sl=stop, tp=tp)
                    self.active_risk = risk
                self._clear_pending()
                return

        if self.position:
            return

        # Long setup
        long_setup = (
            bool(row["is_ny_morning"])
            and bool(row["bull_bias"])
            and bool(row["swept_prev_day_low"])
            and bool(row["reclaimed_prev_day_low"])
            and bool(row["bull_displacement"])
            and bool(row["bull_close_strong"])
        )

        if long_setup:
            # require next bar to confirm above the reclaim candle high
            entry = high_now
            stop = low_now - (atr_now * self.stop_buffer_atr if atr_now > 0 else 0.0)

            # only keep valid setups
            if stop < entry:
                self.pending_long = True
                self.pending_short = False
                self.pending_entry = entry
                self.pending_stop = stop
                self.pending_expiry = i + 2
            return

        # Short setup
        short_setup = (
            bool(row["is_ny_morning"])
            and bool(row["bear_bias"])
            and bool(row["swept_prev_day_high"])
            and bool(row["rejected_prev_day_high"])
            and bool(row["bear_displacement"])
            and bool(row["bear_close_strong"])
        )

        if short_setup:
            # require next bar to confirm below the rejection candle low
            entry = low_now
            stop = high_now + (atr_now * self.stop_buffer_atr if atr_now > 0 else 0.0)

            if stop > entry:
                self.pending_short = True
                self.pending_long = False
                self.pending_entry = entry
                self.pending_stop = stop
                self.pending_expiry = i + 2
            return