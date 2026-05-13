from __future__ import annotations

from typing import Protocol, Any
import pandas as pd
from .models import OrderPlan, SymbolSpec, PropProfile


class StrategyAdapter(Protocol):
    name: str

    def build_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def signal_for_row(self, symbol: str, row: pd.Series, history: pd.DataFrame, spec: SymbolSpec, profile: PropProfile, args: Any) -> OrderPlan | None:
        ...
