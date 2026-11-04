"""Backtest the options system's signal on the underlying stock, using the live rules.

Free historical options quotes don't exist, so this measures the stock move the
trade captures. If the stock move has no edge, the options trade won't either -
options amplify the move (both ways) and charge spread + time decay on top.
Rule of thumb: short-dated near-the-money options need roughly +0.5% average stock
move per trade just to cover the spread and decay.

Data: Yahoo 5-minute bars incl. premarket (last ~60 days).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .daytrade import (DayTradeRules, breakout_levels, entry_signal, momentum_lost, opening_rvol,
                       setup_qualifies)
from .premarket import premarket_stats, session_slices, to_ny


@dataclass
class GapTrade:
    symbol: str
    day: object
    kind: str
    gap_pct: float
    rvol: float
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


def simulate_day(sym: str, df: pd.DataFrame, day, rules: DayTradeRules) -> GapTrade | None:
    st = premarket_stats(df, day)
    if st is None or st["gap_pct"] == 0:
        return None
    kind = "call" if st["gap_pct"] > 0 else "put"
    rvol = opening_rvol(df, day, rules.or_minutes)
    if not setup_qualifies(st, kind, rvol, rules):
        return None
    _, reg = session_slices(df, day)
    if len(reg) < 3:
        return None
    bar = reg.index[1] - reg.index[0]
    for i in range(len(reg) - 1):
        end = (reg.index[i] + bar).time()      # the bar is known once it closes
        if end < rules.entry_start:
            continue
        if end > rules.entry_end:
            return None
        levels = breakout_levels(reg.iloc[: i + 1], st, rules)
        if levels and entry_signal(reg.iloc[: i + 1], levels[0], levels[1], kind):
            break
    else:
        return None
    e = i + 1                                    # fill at the next bar's open
    entry = float(reg["open"].iloc[e])
    sign = 1 if kind == "call" else -1
    stop = entry * (1 - sign * rules.stock_stop_pct / 100)
    cutoff = min(rules.call_exit_time if kind == "call" else rules.exit_time, rules.exit_time)
    mk = lambda j, px, why: GapTrade(sym, day, kind, st["gap_pct"], rvol, reg.index[e], entry,
                                     reg.index[j], px, why)
    for j in range(e, len(reg)):
        b = reg.iloc[j]
        if (sign > 0 and b["low"] <= stop) or (sign < 0 and b["high"] >= stop):
            return mk(j, stop, "stop")
        if reg.index[j].time() >= cutoff:
            return mk(j, float(b["open"]), "time")
        if j == len(reg) - 1:
            return mk(j, float(b["close"]), "time")
        if rules.exit_on_vwap and momentum_lost(reg.iloc[: j + 1], kind):
            return mk(j + 1, float(reg["open"].iloc[j + 1]), "lost VWAP")
    return None


def run(bars: dict[str, pd.DataFrame], rules: DayTradeRules | None = None,
        min_price: float = 0.0, max_price: float = float("inf")) -> pd.DataFrame:
    rules = rules or DayTradeRules()
    trades = []
    for sym, df in bars.items():
        df = to_ny(df)
        for day in sorted(set(df.index.date)):
            tr = simulate_day(sym, df, day, rules)
            if tr and min_price <= tr.entry <= max_price:
                trades.append({**tr.__dict__, "move_pct": tr.move_pct})
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No setups triggered."
    m = trades["move_pct"]
    wins, losses = m[m > 0], m[m <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    days = sorted(trades["day"].unique())
    half = days[len(days) // 2]
    first, second = m[trades["day"] < half], m[trades["day"] >= half]
    lines = [
        f"Setups traded:      {len(trades)}  ({trades['symbol'].nunique()} symbols, {len(days)} days)",
        f"Win rate:           {len(wins) / len(m):.0%}",
        f"Avg stock move:     {m.mean():+.2f}% per trade (in our favor; ~+0.5% needed to beat option costs)",
        f"Avg win / loss:     {wins.mean() if len(wins) else 0:+.2f}% / {losses.mean() if len(losses) else 0:+.2f}%",
        f"Profit factor:      {pf:.2f}",
        f"First half / second half: {first.mean():+.2f}% (n={len(first)}) / {second.mean():+.2f}% (n={len(second)})",
        f"Exit reasons:       {trades['reason'].value_counts().to_dict()}",
        f"Calls vs puts:      {trades.groupby('kind')['move_pct'].agg(['count', 'mean']).round(2).to_dict('index')}",
    ]
    return "\n".join(lines)
