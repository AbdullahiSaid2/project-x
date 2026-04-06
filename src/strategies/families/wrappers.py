import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def ind_rsi(x, window=14):
    return RSIIndicator(pd.Series(x), window=window).rsi()


def ind_atr(h, l, c, window=14):
    return AverageTrueRange(
        pd.Series(h),
        pd.Series(l),
        pd.Series(c),
        window=window
    ).average_true_range()