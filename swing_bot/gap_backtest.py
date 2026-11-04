"""Backtest the gap-and-go ENTRY/EXIT SIGNAL on the underlying stock.

Free historical options quotes don't exist, so this measures what matters first:
does the stock keep moving in our direction after the signal? If the stock move
has no edge, the options trade won't either - options just amplify it (both
ways) and charge you spread and time decay on top.

Data: Yahoo 5-minute bars incl. premarket (last ~60 days).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .daytrade import DayTradeRules, entry_signal, momentum_lost
from .premarket import premarket_stats, session_slices, to_ny


@dataclass
class GapTrade:
    symbol: str
    day: object
    kind: str
    gap_pct: float
    entry_time: object
    entry: float
    exit_time: object
    exit: float
    reason: str

    @property
    def move_pct(self) -> float:
        """Stock move in the trade's favor (positive = good for the option)."""
        raw = self.exit / self.entry - 1
        return 100 * (raw if self.kind == "call" else -raw)


def simulate_day(sym: str, df: pd.DataFrame, day, min_gap_pct: float,
                 rules: DayTradeRules, stock_stop_pct: float = 1.5) -> GapTrade | None:
    st = premarket_stats(df, day)
    if st is None or abs(st["gap_pct"]) < min_gap_pct:
        return None
    kind = "call" if st["gap_pct"] > 0 else "put"
    _, reg = session_slices(df, day)
    for i in range(len(reg) - 1):
        t = reg.index[i].time()
        if t < rules.entry_start:
            continue
        if t >= rules.entry_end:
            return None
        if entry_signal(reg.iloc[: i + 1], st["pm_high"], st["pm_low"], kind):
            break
    else:
        return None
    e_idx = i + 1                       # fill at the next bar's open
    entry = float(reg["open"].iloc[e_idx])
    stop = entry * (1 - stock_stop_pct / 100) if kind == "call" else entry * (1 + stock_stop_pct / 100)
    for j in range(e_idx, len(reg)):
        bar = reg.iloc[j]
        hit = bar["low"] <= stop if kind == "call" else bar["high"] >= stop
        if hit:
            return GapTrade(sym, day, kind, st["gap_pct"], reg.index[e_idx], entry, reg.index[j], stop, "stop")
        last = j == len(reg) - 1
        if reg.index[j].time() >= rules.exit_time or last:
            return GapTrade(sym, day, kind, st["gap_pct"], reg.index[e_idx], entry, reg.index[j],
                            float(bar["close"]), "time")
        if momentum_lost(reg.iloc[: j + 1], kind) and j + 1 < len(reg):
            return GapTrade(sym, day, kind, st["gap_pct"], reg.index[e_idx], entry, reg.index[j + 1],
                            float(reg["open"].iloc[j + 1]), "lost VWAP")
    return None


def run(bars: dict[str, pd.DataFrame], min_gap_pct: float = 3.0,
        rules: DayTradeRules | None = None, stock_stop_pct: float = 1.5) -> pd.DataFrame:
    rules = rules or DayTradeRules()
    trades = []
    for sym, df in bars.items():
        df = to_ny(df)
        for day in sorted(set(df.index.date)):
            tr = simulate_day(sym, df, day, min_gap_pct, rules, stock_stop_pct)
            if tr:
                trades.append({**tr.__dict__, "move_pct": tr.move_pct})
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No gap-and-go setups triggered."
    m = trades["move_pct"]
    wins, losses = m[m > 0], m[m <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    lines = [
        f"Setups traded:      {len(trades)}  ({trades['symbol'].nunique()} symbols, "
        f"{trades['day'].nunique()} days)",
        f"Win rate:           {len(wins) / len(m):.0%}",
        f"Avg stock move:     {m.mean():+.2f}% per trade (in our favor)",
        f"Avg win / loss:     {wins.mean() if len(wins) else 0:+.2f}% / {losses.mean() if len(losses) else 0:+.2f}%",
        f"Profit factor:      {pf:.2f}",
        f"Exit reasons:       {trades['reason'].value_counts().to_dict()}",
        f"Calls vs puts:      {trades.groupby('kind')['move_pct'].agg(['count', 'mean']).round(2).to_dict('index')}",
    ]
    return "\n".join(lines)
