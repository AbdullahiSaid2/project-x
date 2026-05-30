from __future__ import annotations

import pandas as pd

from src.strategies.manual.ict_top_bottom_ticking_v2 import (
    generate_top_bottom_ticking_v2_signals,
)


class TopBottomTickingV2Adapter:
    """
    Fast event-engine adapter for Top Bottom Ticking V2.

    Computes all signals once in build_features(), then signal_for_row()
    reads sparse signal columns.

    This version passes args.target_r into the signal generator, so the CLI
    option --target-r now actually controls V2 targets.
    """

    name = "top_bottom_ticking_v2"
    strategy_name = "top_bottom_ticking_v2"

    SIGNAL_SIDE_COL = "_tbt_v2_side"
    SIGNAL_ENTRY_COL = "_tbt_v2_entry"
    SIGNAL_STOP_COL = "_tbt_v2_stop"
    SIGNAL_TARGET_COL = "_tbt_v2_target"
    SIGNAL_TYPE_COL = "_tbt_v2_trade_type"

    def build_features(self, symbol, df):
        """
        Kept for old event-engine compatibility.

        Some run_event_backtest versions call build_features(symbol, df)
        without args. In that case, V2 uses DEFAULT_TARGET_R from the strategy.
        """
        return self.build_features_with_args(symbol=symbol, df=df, args=None)

    def build_features_with_args(self, symbol, df, args=None):
        out = df.copy()

        out[self.SIGNAL_SIDE_COL] = ""
        out[self.SIGNAL_ENTRY_COL] = pd.NA
        out[self.SIGNAL_STOP_COL] = pd.NA
        out[self.SIGNAL_TARGET_COL] = pd.NA
        out[self.SIGNAL_TYPE_COL] = ""

        target_r = None
        if args is not None and hasattr(args, "target_r"):
            target_r = args.target_r

        signals_df = generate_top_bottom_ticking_v2_signals(
            out,
            symbol=symbol,
            target_r=target_r,
        )

        if signals_df is None or signals_df.empty:
            return out

        for _, sig in signals_df.iterrows():
            ts = sig["timestamp"]

            if ts not in out.index:
                continue

            out.at[ts, self.SIGNAL_SIDE_COL] = str(sig["side"]).upper()
            out.at[ts, self.SIGNAL_ENTRY_COL] = float(sig["entry"])
            out.at[ts, self.SIGNAL_STOP_COL] = float(sig["stop"])
            out.at[ts, self.SIGNAL_TARGET_COL] = float(sig["target"])
            out.at[ts, self.SIGNAL_TYPE_COL] = str(
                sig.get("setup", "top_bottom_ticking_v2")
            )

        return out

    def signal_for_row(self, symbol, row, history, spec, profile, args):
        side = row.get(self.SIGNAL_SIDE_COL, "")

        if not side:
            return None

        if pd.isna(row.get(self.SIGNAL_ENTRY_COL)):
            return None

        from src.backtesting.event_engine.models import OrderPlan

        entry = float(row[self.SIGNAL_ENTRY_COL])
        stop = float(row[self.SIGNAL_STOP_COL])
        target = float(row[self.SIGNAL_TARGET_COL])
        trade_type = str(row.get(self.SIGNAL_TYPE_COL, "top_bottom_ticking_v2"))

        if side not in {"LONG", "SHORT"}:
            return None

        if side == "LONG" and not (stop < entry < target):
            return None

        if side == "SHORT" and not (target < entry < stop):
            return None

        return OrderPlan(
            symbol=symbol,
            side=side,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            trade_type=trade_type,
            reason=trade_type,
            strategy_name=self.name,
            setup_score=1.0,
        )

    def generate_signals(self, symbol, df, profile=None, args=None):
        target_r = None
        if args is not None and hasattr(args, "target_r"):
            target_r = args.target_r

        signals_df = generate_top_bottom_ticking_v2_signals(
            df,
            symbol=symbol,
            target_r=target_r,
        )

        signals = []

        if signals_df is None or signals_df.empty:
            return signals

        for _, row in signals_df.iterrows():
            signals.append(
                {
                    "timestamp": row["timestamp"],
                    "symbol": symbol,
                    "side": row["side"],
                    "entry_price": row["entry"],
                    "stop_price": row["stop"],
                    "target_price": row["target"],
                    "trade_type": row["setup"],
                }
            )

        return signals
