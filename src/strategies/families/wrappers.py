from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, SMAIndicator, MACD


def ind_rsi(x, window: int = 14):
    return RSIIndicator(pd.Series(x), window=window).rsi()


def ind_atr(h, l, c, window: int = 14):
    return AverageTrueRange(
        pd.Series(h),
        pd.Series(l),
        pd.Series(c),
        window=window
    ).average_true_range()


def ind_ema(x, window: int = 20):
    return EMAIndicator(pd.Series(x), window=window).ema_indicator()


def ind_sma(x, window: int = 20):
    return SMAIndicator(pd.Series(x), window=window).sma_indicator()


def ind_macd(x, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
    return MACD(
        pd.Series(x),
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    ).macd()


def ind_macd_signal(x, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
    return MACD(
        pd.Series(x),
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    ).macd_signal()


def ind_bb_low(x, window: int = 20, window_dev: float = 2.0):
    bb = BollingerBands(pd.Series(x), window=window, window_dev=window_dev)
    return bb.bollinger_lband()


def ind_bb_high(x, window: int = 20, window_dev: float = 2.0):
    bb = BollingerBands(pd.Series(x), window=window, window_dev=window_dev)
    return bb.bollinger_hband()


def ind_bb_mid(x, window: int = 20, window_dev: float = 2.0):
    bb = BollingerBands(pd.Series(x), window=window, window_dev=window_dev)
    return bb.bollinger_mavg()