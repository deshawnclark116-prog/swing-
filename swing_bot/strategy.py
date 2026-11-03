"""Trading strategies.

A strategy turns OHLCV bars into a signal per bar, evaluated at that bar's close:
    +1 = enter long (on the next bar's open)
    -1 = exit long  (on the next bar's open)
     0 = do nothing
It also supplies the stop-loss and take-profit levels for a new entry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from . import indicators as ind
from .config import StrategyConfig

ENTER, EXIT, HOLD = 1, -1, 0


class Strategy(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with at least `signal` and `atr` columns added."""

    @abstractmethod
    def stop_and_target(self, entry_price: float, row: pd.Series) -> tuple[float, float | None]:
        """Stop-loss and (optional) take-profit for a new long entry."""

    @property
    def warmup(self) -> int:
        """Bars needed before signals are meaningful."""
        return 0


class TrendCrossStrategy(Strategy):
    """Trend-following swing strategy.

    Enter long when the fast EMA crosses above the slow SMA, price is above the
    long-term trend SMA, and RSI isn't already overbought. Exit when the fast EMA
    crosses back below the slow SMA or RSI becomes extremely overbought. Every
    trade carries an ATR-based stop and profit target.
    """

    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = cfg or StrategyConfig()

    @property
    def warmup(self) -> int:
        c = self.cfg
        return max(c.fast_period, c.slow_period, c.trend_period, c.rsi_period, c.atr_period) + 1

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        out["fast"] = ind.ema(out["close"], c.fast_period)
        out["slow"] = ind.sma(out["close"], c.slow_period)
        out["trend"] = ind.sma(out["close"], c.trend_period)
        out["rsi"] = ind.rsi(out["close"], c.rsi_period)
        out["atr"] = ind.atr(out, c.atr_period)

        above = out["fast"] > out["slow"]
        prev_above = above.shift(1, fill_value=False).astype(bool)
        cross_up = above & ~prev_above
        cross_down = ~above & prev_above

        enter = cross_up & (out["close"] > out["trend"]) & (out["rsi"] < c.rsi_entry_max)
        exit_ = cross_down | (out["rsi"] > c.rsi_exit)

        out["signal"] = HOLD
        out.loc[exit_, "signal"] = EXIT
        out.loc[enter, "signal"] = ENTER  # a fresh cross-up wins over an RSI exit
        # Nothing is valid until every indicator is warmed up
        ready = out[["fast", "slow", "trend", "rsi", "atr"]].notna().all(axis=1)
        out.loc[~ready, "signal"] = HOLD
        return out

    def stop_and_target(self, entry_price: float, row: pd.Series) -> tuple[float, float | None]:
        a = float(row["atr"])
        return entry_price - self.cfg.stop_atr_mult * a, entry_price + self.cfg.target_atr_mult * a


STRATEGIES = {"trend": TrendCrossStrategy}
