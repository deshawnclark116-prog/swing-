"""Live / paper trading loop. Designed to run once per day after the market closes.

Each run:
  1. Fetch recent daily bars for every symbol.
  2. (Paper broker only) enforce stops/targets against the latest bar.
  3. Exit positions whose strategy signal says EXIT.
  4. Enter new positions on ENTER signals, sized by the risk rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .broker import Broker
from .config import BotConfig
from .data import load_yahoo
from .risk import position_size
from .strategy import ENTER, EXIT, Strategy

log = logging.getLogger("swing_bot")


@dataclass
class Action:
    side: str      # "buy" | "sell"
    symbol: str
    qty: float
    price: float
    reason: str
    stop: float | None = None
    target: float | None = None

    def __str__(self) -> str:
        s = f"{self.side.upper():4} {self.symbol:6} qty={self.qty:<6g} @ ~{self.price:.2f}  ({self.reason})"
        if self.stop is not None:
            s += f"  stop={self.stop:.2f}"
        if self.target is not None:
            s += f"  target={self.target:.2f}"
        return s


class SwingBot:
    def __init__(self, strategy: Strategy, broker: Broker, cfg: BotConfig | None = None, loader=None):
        self.strategy = strategy
        self.broker = broker
        self.cfg = cfg or BotConfig()
        self.loader = loader or (lambda sym: load_yahoo(sym, period=f"{self.cfg.lookback_days}d"))

    def _latest(self) -> dict[str, pd.DataFrame]:
        out = {}
        for sym in self.cfg.symbols:
            try:
                df = self.loader(sym)
            except Exception as e:  # one bad ticker shouldn't stop the run
                log.warning("skipping %s: %s", sym, e)
                continue
            if len(df) < self.strategy.warmup:
                log.warning("skipping %s: only %d bars, need %d", sym, len(df), self.strategy.warmup)
                continue
            out[sym] = self.strategy.generate(df)
        return out

    def run_once(self, dry_run: bool = False) -> list[Action]:
        frames = self._latest()
        prices = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
        held = self.broker.positions()
        actions: list[Action] = []

        # Exits
        for sym, h in held.items():
            if sym not in frames:
                continue
            bar = frames[sym].iloc[-1]
            bar_date = str(frames[sym].index[-1].date())
            reason, price = None, prices[sym]
            fresh = h.opened is not None and bar_date <= h.opened
            if not self.broker.manages_exits and not fresh:
                if h.stop is not None and bar["low"] <= h.stop:
                    reason, price = "stop", min(h.stop, float(bar["open"]))
                elif h.target is not None and bar["high"] >= h.target:
                    reason, price = "target", max(h.target, float(bar["open"]))
            if reason is None and bar["signal"] == EXIT:
                reason = "signal"
            if reason:
                actions.append(Action("sell", sym, h.qty, price, reason))

        # Entries
        selling = {a.symbol for a in actions}
        open_slots = self.cfg.risk.max_positions - (len(held) - len(selling))
        equity = self.broker.equity(prices)
        cash = self.broker.cash()
        for sym, f in frames.items():
            if open_slots <= 0:
                break
            if sym in held or f["signal"].iloc[-1] != ENTER:
                continue
            price = prices[sym]
            stop, target = self.strategy.stop_and_target(price, f.iloc[-1])
            qty = position_size(equity, cash, price, stop, self.cfg.risk)
            if qty <= 0:
                log.info("no size for %s (cash=%.2f)", sym, cash)
                continue
            actions.append(Action("buy", sym, qty, price, "signal", stop, target))
            cash -= qty * price
            open_slots -= 1

        for a in actions:
            log.info("%s%s", "[dry-run] " if dry_run else "", a)
            if dry_run:
                continue
            if a.side == "sell":
                self.broker.sell_all(a.symbol, a.price, a.reason)
            else:
                opened = str(frames[a.symbol].index[-1].date())
                self.broker.buy(a.symbol, int(a.qty), a.price, a.stop, a.target, opened)
        if not actions:
            log.info("no actions today")
        return actions
