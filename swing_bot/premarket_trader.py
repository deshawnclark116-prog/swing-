"""System 3: premarket runner bot (shares). FAKE MONEY ONLY until proven.

From 4:00am New York time it scans the whole US market for stocks gapping up
10%+ and buys when one closes a 1-minute bar at a NEW premarket high. It sells
on a 12% stop or at 9:40am, after the opening bell.

In the 60-day full-market test (research/premarket_runners_full_market.py) this
"chase new highs" rule was about breakeven after spreads (+0.1% per trade on the
test half). It's running so we can see real forward results.
"""
from __future__ import annotations

import logging
import math
import time as _time
from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd

from .daytrade import completed_bars
from .premarket import NY, OPEN, full_market_gappers, premarket_stats, to_ny, yahoo_intraday
from .stock_daytrade import StockBroker

log = logging.getLogger("swing_bot.premarket_trader")


@dataclass
class PremarketRules:
    min_gap_pct: float = 10.0
    min_price: float = 0.20
    max_price: float = 20.0
    budget_pct: float = 0.50        # of settled cash per trade
    stop_pct: float = 0.12          # runners swing hard; tight stops get shaken out
    max_trades_per_day: int = 3
    entry_start: time = time(4, 0)
    entry_end: time = time(9, 15)
    exit_time: time = time(9, 40)   # sell everything shortly after the open
    rescan_minutes: int = 15


class PremarketTrader:
    def __init__(self, broker: StockBroker, rules: PremarketRules | None = None,
                 scanner=None, bars_loader=None, dry_run: bool = False):
        self.broker = broker
        self.rules = rules or PremarketRules()
        r = self.rules
        self.scanner = scanner or (lambda: full_market_gappers(r.min_gap_pct, r.min_price, r.max_price))
        self.bars_loader = bars_loader or (lambda syms: yahoo_intraday(syms, period="2d", interval="1m"))
        self.dry_run = dry_run
        self.watch: list[str] = []
        self._last_scan: datetime | None = None

    def rescan(self, now: datetime) -> None:
        df = self.scanner()
        self.watch = list(df["symbol"]) if df is not None and not df.empty else []
        self._last_scan = now
        log.warning("%s scan: %s", now.strftime("%H:%M"),
                    ", ".join(f"{r.symbol} {r.gap_pct:+.0f}%" for r in df.itertuples()) if self.watch else "no runners")

    def step(self, now: datetime) -> list[str]:
        r, events = self.rules, []
        self.broker.start_day(now.date())
        syms = sorted(set(self.watch) | set(self.broker.positions))
        if not syms:
            return events
        data = {s: completed_bars(to_ny(df), now) for s, df in self.bars_loader(syms).items()}

        # ---- exits
        for sym, p in list(self.broker.positions.items()):
            df = data.get(sym)
            if df is None or df.empty:
                continue
            bar = df.iloc[-1]
            opened = pd.Timestamp(p.opened_at).tz_convert(NY).floor("min")
            reason, price = None, float(bar["close"])
            if df.index[-1] >= opened and bar["low"] <= p.stop:
                reason, price = "stop", min(p.stop, float(bar["open"]))
            elif now.time() >= r.exit_time:
                reason = "time"
            if reason:
                msg = f"SELL {p.qty} {sym} @ ~{price:.4g} ({reason})"
                if not self.dry_run:
                    msg += f"  P&L ${self.broker.sell(sym, price, reason, now):+,.2f}"
                events.append(msg)

        # ---- entries: a completed 1-min bar closing at a new premarket high
        if not (r.entry_start <= now.time() <= r.entry_end):
            return events
        for sym in self.watch:
            if self.broker.day["trades"] >= r.max_trades_per_day:
                break
            if sym in self.broker.positions or sym in self.broker.day["traded"] or sym not in data:
                continue
            df = data[sym]
            today = df[(df.index.date == now.date()) & (df.index.time < OPEN)]
            st = premarket_stats(df, now.date())
            if st is None or len(today) < 3 or st["gap_pct"] < r.min_gap_pct:
                continue
            last = float(today["close"].iloc[-1])
            if not (r.min_price <= last <= r.max_price) or last <= float(today["high"].iloc[:-1].max()):
                continue
            qty = math.floor(self.broker.buying_power() * r.budget_pct / (last * 1.01))
            if qty < 1:
                continue
            stop = last * (1 - r.stop_pct)
            msg = f"BUY {qty} {sym} @ ~{last:.4g}  stop {stop:.4g}  [up {st['gap_pct']:+.0f}%, new premarket high]"
            if self.dry_run:
                self.broker.day["traded"].append(sym)
            else:
                self.broker.buy(sym, qty, last, stop, float("inf"), now)
            events.append(msg)
        return events

    def run(self, poll_seconds: int = 60) -> None:
        r = self.rules
        while True:
            now = datetime.now(NY)
            if now.weekday() >= 5:
                log.warning("weekend - market closed")
                return
            if now.time() < r.entry_start:
                _time.sleep(min(60, (datetime.combine(now.date(), r.entry_start, NY) - now).total_seconds() + 1))
                continue
            if now.time() <= r.entry_end and (self._last_scan is None or
                                              (now - self._last_scan).total_seconds() >= r.rescan_minutes * 60):
                self.rescan(now)
            for e in self.step(datetime.now(NY)):
                log.warning(e)
            if now.time() >= r.exit_time and not self.broker.positions:
                log.warning("done. realized P&L today: $%.2f", self.broker.day.get("realized", 0))
                return
            _time.sleep(poll_seconds)
