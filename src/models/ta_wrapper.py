# ============================================================
# 🌙 Technical Analysis Wrapper
# Drop-in replacement for pandas_ta using the `ta` library.
# Works with Python 3.10+ on Mac, Windows, Linux.
#
# Usage:  from src.models.ta_wrapper import ta
# Then use exactly like before: ta.ema(close, 9), ta.rsi(close, 14) etc.
# ============================================================

import pandas as pd
import ta as _ta   # pip install ta


class _TA:
    """Thin wrapper that maps pandas_ta-style calls to the `ta` library."""

    # ── Trend ────────────────────────────────────────────────
    def ema(self, close: pd.Series, length: int = 9, **kwargs) -> pd.Series:
        return _ta.trend.ema_indicator(close, window=length)

    def sma(self, close: pd.Series, length: int = 20, **kwargs) -> pd.Series:
        return _ta.trend.sma_indicator(close, window=length)

    # ── Momentum ─────────────────────────────────────────────
    def rsi(self, close: pd.Series, length: int = 14, **kwargs) -> pd.Series:
        return _ta.momentum.rsi(close, window=length)

    def macd(self, close: pd.Series,
             fast: int = 12, slow: int = 26, signal: int = 9,
             **kwargs):
        """
        Returns a DataFrame with 3 columns matching pandas_ta layout:
          col 0 → MACD line
          col 1 → MACD histogram
          col 2 → Signal line
        """
        macd_line   = _ta.trend.macd(close, window_slow=slow, window_fast=fast)
        signal_line = _ta.trend.macd_signal(close, window_slow=slow,
                                            window_fast=fast, window_sign=signal)
        histogram   = _ta.trend.macd_diff(close, window_slow=slow,
                                          window_fast=fast, window_sign=signal)
        df = pd.DataFrame({
            "MACD":   macd_line,
            "MACDh":  histogram,
            "MACDs":  signal_line,
        })
        return df

    def bbands(self, close: pd.Series, length: int = 20,
               std: float = 2.0, **kwargs):
        """
        Returns a DataFrame matching pandas_ta bbands layout:
          col 0 → Upper band
          col 1 → Middle band
          col 2 → Lower band
          col 3 → %B
          col 4 → Bandwidth
        """
        upper  = _ta.volatility.bollinger_hband(close, window=length, window_dev=std)
        mid    = _ta.volatility.bollinger_mavg(close,  window=length)
        lower  = _ta.volatility.bollinger_lband(close, window=length, window_dev=std)
        pct_b  = _ta.volatility.bollinger_pband(close, window=length, window_dev=std)
        bwidth = _ta.volatility.bollinger_wband(close, window=length, window_dev=std)
        df = pd.DataFrame({
            "BBU": upper, "BBM": mid, "BBL": lower,
            "BBP": pct_b, "BBW": bwidth,
        })
        return df

    def stoch(self, high: pd.Series, low: pd.Series, close: pd.Series,
              k: int = 14, d: int = 3, **kwargs):
        """
        Returns a DataFrame matching pandas_ta stoch layout:
          col 0 → %K
          col 1 → %D
        """
        stoch_k = _ta.momentum.stoch(high, low, close,
                                     window=k, smooth_window=d)
        stoch_d = _ta.momentum.stoch_signal(high, low, close,
                                            window=k, smooth_window=d)
        return pd.DataFrame({"STOCHk": stoch_k, "STOCHd": stoch_d})

    def atr(self, high: pd.Series, low: pd.Series,
            close: pd.Series, length: int = 14, **kwargs) -> pd.Series:
        return _ta.volatility.average_true_range(high, low, close, window=length)

    def adx(self, high: pd.Series, low: pd.Series,
            close: pd.Series, length: int = 14, **kwargs):
        adx_val = _ta.trend.adx(high, low, close, window=length)
        dip     = _ta.trend.adx_pos(high, low, close, window=length)
        din     = _ta.trend.adx_neg(high, low, close, window=length)
        return pd.DataFrame({"ADX": adx_val, "DMP": dip, "DMN": din})

    def obv(self, close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
        return _ta.volume.on_balance_volume(close, volume)

    def vwap(self, high: pd.Series, low: pd.Series,
             close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
        return _ta.volume.volume_weighted_average_price(high, low, close, volume)


# Singleton — import this everywhere instead of pandas_ta
ta = _TA()
