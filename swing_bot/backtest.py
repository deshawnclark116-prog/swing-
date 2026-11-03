"""Daily-bar portfolio backtester.

Timing model (no look-ahead):
  * Signals are computed from a bar's close.
  * Entries/exits from those signals fill at the NEXT bar's open (plus slippage).
  * Stops and targets are checked against each bar's range. If a bar gaps
    through a level, the fill is at the open. If both the stop and the target
    fall inside one bar, the stop is assumed to hit first (the cautious choice).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BotConfig
from .risk import position_size
from .strategy import ENTER, EXIT, Strategy


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_date: pd.Timestamp
    stop: float
    target: float | None


@dataclass
class Trade:
    symbol: str
    qty: int
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def ret(self) -> float:
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    metrics: dict[str, float] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["symbol", "qty", "entry_date", "entry_price",
                                         "exit_date", "exit_price", "pnl", "ret", "reason"])
        return pd.DataFrame([{**t.__dict__, "pnl": t.pnl, "ret": t.ret} for t in self.trades])


def compute_metrics(equity: pd.Series, trades: list[Trade]) -> dict[str, float]:
    m: dict[str, float] = {}
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    m["start_equity"] = start
    m["end_equity"] = end
    m["total_return"] = end / start - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    m["cagr"] = (end / start) ** (1 / years) - 1 if end > 0 else -1.0
    daily = equity.pct_change().dropna()
    m["sharpe"] = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    m["max_drawdown"] = float((equity / equity.cummax() - 1).min())
    m["num_trades"] = len(trades)
    if trades:
        pnls = np.array([t.pnl for t in trades])
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        m["win_rate"] = len(wins) / len(pnls)
        m["avg_trade_return"] = float(np.mean([t.ret for t in trades]))
        m["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    else:
        m["win_rate"] = m["avg_trade_return"] = m["profit_factor"] = 0.0
    return m


class Backtester:
    def __init__(self, strategy: Strategy, cfg: BotConfig | None = None):
        self.strategy = strategy
        self.cfg = cfg or BotConfig()

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestResult:
        cfg = self.cfg
        slip = cfg.execution.slippage_bps / 10_000
        fee = cfg.execution.commission_per_trade

        frames = {s: self.strategy.generate(df) for s, df in data.items() if not df.empty}
        if not frames:
            raise ValueError("no data to backtest")
        dates = sorted(set().union(*(f.index for f in frames.values())))

        cash = cfg.starting_cash
        positions: dict[str, Position] = {}
        last_close: dict[str, float] = {}
        last_row: dict[str, pd.Series] = {}
        pending_entries: list[str] = []
        pending_exits: list[str] = []
        trades: list[Trade] = []
        equity_curve: list[float] = []

        def close_position(sym: str, price: float, date, reason: str) -> None:
            nonlocal cash
            p = positions.pop(sym)
            cash += p.qty * price - fee
            trades.append(Trade(sym, p.qty, p.entry_date, p.entry_price, date, price, reason))

        def equity() -> float:
            return cash + sum(p.qty * last_close[s] for s, p in positions.items())

        for date in dates:
            bars = {s: f.loc[date] for s, f in frames.items() if date in f.index}

            # 1) Fill yesterday's signals at today's open: exits first, then entries.
            for sym in pending_exits:
                if sym in positions and sym in bars:
                    close_position(sym, bars[sym]["open"] * (1 - slip), date, "signal")
            for sym in pending_entries:
                if sym in positions or sym not in bars or len(positions) >= cfg.risk.max_positions:
                    continue
                price = bars[sym]["open"] * (1 + slip)
                stop, target = self.strategy.stop_and_target(price, last_row[sym])
                qty = position_size(equity(), cash - fee, price, stop, cfg.risk)
                if qty <= 0:
                    continue
                cash -= qty * price + fee
                positions[sym] = Position(sym, qty, price, date, stop, target)
                last_close.setdefault(sym, price)
            pending_entries, pending_exits = [], []

            # 2) Intraday stop / target checks.
            for sym in list(positions):
                if sym not in bars:
                    continue
                p, bar = positions[sym], bars[sym]
                if bar["open"] <= p.stop:
                    close_position(sym, bar["open"] * (1 - slip), date, "stop")
                elif bar["low"] <= p.stop:
                    close_position(sym, p.stop * (1 - slip), date, "stop")
                elif p.target is not None and bar["open"] >= p.target:
                    close_position(sym, bar["open"] * (1 - slip), date, "target")
                elif p.target is not None and bar["high"] >= p.target:
                    close_position(sym, p.target * (1 - slip), date, "target")

            # 3) Mark to market and read today's closing signals.
            for sym, bar in bars.items():
                last_close[sym] = float(bar["close"])
                last_row[sym] = bar
                if bar["signal"] == EXIT and sym in positions:
                    pending_exits.append(sym)
                elif bar["signal"] == ENTER and sym not in positions:
                    pending_entries.append(sym)
            equity_curve.append(equity())

        # Close anything still open at the final close so P&L is realized.
        for sym in list(positions):
            close_position(sym, last_close[sym], dates[-1], "end")

        eq = pd.Series(equity_curve, index=pd.DatetimeIndex(dates, name="date"), name="equity")
        return BacktestResult(eq, trades, compute_metrics(eq, trades))
