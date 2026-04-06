from __future__ import annotations
import textwrap
from typing import Any

try:
    from src.strategies.families.schema import StrategySchema
except Exception:
    StrategySchema = object  # fallback to prevent import crash


RUNTIME_IMPORT_BLOCK = """
import numpy as np
import pandas as pd
from backtesting import Strategy
from src.strategies.families.wrappers import ind_rsi, ind_atr
""".strip()


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return repr(v)
    return str(v)


def compile_strategy_class(schema: StrategySchema, class_name: str = "GeneratedStrategy") -> str:
    """
    SAFE minimal compiler — guaranteed to run
    """

    return textwrap.dedent(f"""
class {class_name}(Strategy):

    rsi_window = 14
    atr_window = 14
    sl_atr_mult = 1.0
    tp_r_multiple = 1.5
    fixed_size = 0.1

    def init(self):
        self.rsi = self.I(lambda x: ind_rsi(x, window=self.rsi_window), self.data.Close)
        self.atr = self.I(lambda h,l,c: ind_atr(h,l,c, window=self.atr_window),
                          self.data.High, self.data.Low, self.data.Close)

    def next(self):
        if len(self.data) < 20:
            return

        price = self.data.Close[-1]

        # SIMPLE LOGIC (stable fallback)
        if self.rsi[-1] < 30:
            if not self.position:
                sl = price - self.atr[-1]
                tp = price + (self.atr[-1] * self.tp_r_multiple)
                self.buy(size=self.fixed_size, sl=sl, tp=tp)

        if self.rsi[-1] > 70:
            if not self.position:
                sl = price + self.atr[-1]
                tp = price - (self.atr[-1] * self.tp_r_multiple)
                self.sell(size=self.fixed_size, sl=sl, tp=tp)
""").strip()