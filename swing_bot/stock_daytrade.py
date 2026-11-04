"""System 2: small-cap gap-and-go, traded as shares in a cash account.

Entry  - a gap-up stock closes a 1-minute bar above its premarket high and above VWAP.
Stop   - the lower of the signal bar's low and VWAP, but never more than max_stop_pct
         below entry (small caps can drop fast; a wide stop means a tiny position).
Size   - shares so that hitting the stop loses risk_pct of settled cash, capped
         at budget_pct of settled cash.
Exits  - stop; move stop to breakeven once up 1R; take profit at target_r x risk;
         stock closes back below VWAP; or 3:45pm ET.

Small-cap reality check: these stocks get halted, gap through stops, and have wide
spreads. Paper trade this for weeks before any real money.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time as _time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from .cash_account import CashAccount
from .daytrade import completed_bars, entry_signal, momentum_lost, vwap
from .premarket import NY, OPEN, premarket_stats, session_slices, to_ny, yahoo_intraday

log = logging.getLogger("swing_bot.stocks")


@dataclass
class StockRules:
    risk_pct: float = 0.02          # lose at most 2% of settled cash if the stop hits
    budget_pct: float = 0.25        # never more than 25% of settled cash in one stock
    max_stop_pct: float = 0.05      # stop at most 5% below entry
    target_r: float = 2.0           # take profit at 2x the risk
    breakeven_r: float = 1.0        # move stop to entry once up 1x the risk
    max_trades_per_day: int = 3
    daily_loss_limit_pct: float = 0.06
    slippage_pct: float = 0.002     # small caps: assume 0.2% worse fills
    entry_start: time = time(9, 31)  # small caps move in the first minutes
    entry_end: time = time(10, 30)
    exit_time: time = time(15, 45)


def stop_level(reg: pd.DataFrame, entry: float, rules: StockRules) -> float:
    signal_low = float(reg["low"].iloc[-1])
    vw = float(vwap(reg).iloc[-1])
    natural = min(signal_low, vw)
    return max(natural, entry * (1 - rules.max_stop_pct))


def share_size(cash: float, entry: float, stop: float, rules: StockRules) -> int:
    risk = entry - stop
    if risk <= 0 or entry <= 0:
        return 0
    return max(0, math.floor(min(cash * rules.risk_pct / risk, cash * rules.budget_pct / entry)))


@dataclass
class StockPosition:
    symbol: str
    qty: int
    entry: float
    stop: float
    target: float
    risk: float
    opened_at: str


class StockBroker(ABC):
    def __init__(self, state_file):
        self.path = Path(state_file)
        raw = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.positions = {k: StockPosition(**v) for k, v in raw.get("positions", {}).items()}
        self.day: dict = raw.get("day", {})
        self.log: list[dict] = raw.get("log", [])
        self._extra = raw

    def _state(self) -> dict:
        return {}

    def save(self) -> None:
        data = {**self._extra, "positions": {k: asdict(v) for k, v in self.positions.items()},
                "day": self.day, "log": self.log[-1000:], **self._state()}
        # write-then-rename so a reader never sees a half-written file
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp, self.path)

    def start_day(self, today: date) -> None:
        if self.day.get("date") != today.isoformat():
            self.day = {"date": today.isoformat(), "trades": 0, "realized": 0.0,
                        "start_cash": self.buying_power(), "traded": []}
            self.save()

    @abstractmethod
    def buying_power(self) -> float: ...

    @abstractmethod
    def _execute_buy(self, sym: str, qty: int, price: float) -> float: ...

    @abstractmethod
    def _execute_sell(self, p: StockPosition, price: float, today: date) -> float: ...

    def buy(self, sym, qty, price, stop, target, now) -> None:
        fill = self._execute_buy(sym, qty, price)
        self.positions[sym] = StockPosition(sym, qty, fill, stop, target, fill - stop, now.isoformat())
        self.day["trades"] += 1
        self.day["traded"].append(sym)
        self.log.append({"t": now.isoformat(), "side": "buy", "symbol": sym, "qty": qty,
                         "price": round(fill, 4), "stop": round(stop, 4), "target": round(target, 4)})
        self.save()

    def sell(self, sym, price, reason, now) -> float:
        p = self.positions.pop(sym)
        fill = self._execute_sell(p, price, now.date())
        pnl = (fill - p.entry) * p.qty
        self.day["realized"] += pnl
        self.log.append({"t": now.isoformat(), "side": "sell", "symbol": sym, "qty": p.qty,
                         "price": round(fill, 4), "pnl": round(pnl, 2), "reason": reason})
        self.save()
        return pnl


class PaperStockBroker(StockBroker):
    def __init__(self, state_file="stocks_paper_state.json", starting_cash=5_000.0, slippage_pct=0.002):
        super().__init__(state_file)
        self.slip = slippage_pct
        self.account = (CashAccount.from_dict(self._extra["account"]) if "account" in self._extra
                        else CashAccount(starting_cash))

        self.starting_cash = self._extra.get("starting_cash", starting_cash)

    def _state(self) -> dict:
        return {"account": self.account.to_dict(), "starting_cash": self.starting_cash}

    def start_day(self, today: date) -> None:
        self.account.settle(today)
        super().start_day(today)

    def buying_power(self) -> float:
        return self.account.settled

    def _execute_buy(self, sym, qty, price) -> float:
        fill = price * (1 + self.slip)
        self.account.buy(qty * fill)
        return fill

    def _execute_sell(self, p, price, today) -> float:
        fill = price * (1 - self.slip)
        self.account.sell(p.qty * fill, today)
        return fill


class AlpacaStockBroker(StockBroker):
    """Alpaca shares (paper unless ALPACA_PAPER=false). Market orders; stops are managed by the bot."""

    def __init__(self, state_file="stocks_alpaca_state.json"):
        super().__init__(state_file)
        from alpaca.trading.client import TradingClient

        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (see .env.example)")
        self.paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        self.client = TradingClient(key, secret, paper=self.paper)

    def buying_power(self) -> float:
        acct = self.client.get_account()
        # non_marginable_buying_power = settled cash in a cash account
        return float(acct.non_marginable_buying_power or acct.cash)

    def _order(self, sym, qty, side):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        self.client.submit_order(MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY))

    def _execute_buy(self, sym, qty, price) -> float:
        self._order(sym, qty, "buy")
        return price

    def _execute_sell(self, p, price, today) -> float:
        self._order(p.symbol, p.qty, "sell")
        return price


class StockDayTrader:
    def __init__(self, broker: StockBroker, watch: list[str], rules: StockRules | None = None,
                 bars_loader=None, dry_run: bool = False):
        self.broker = broker
        self.watch = watch
        self.rules = rules or StockRules()
        self.bars_loader = bars_loader or (lambda syms: yahoo_intraday(syms, period="2d", interval="1m"))
        self.dry_run = dry_run

    def _halted(self) -> bool:
        d = self.broker.day
        limit = d.get("start_cash", 0) * self.rules.daily_loss_limit_pct
        return limit > 0 and d.get("realized", 0) <= -limit

    def step(self, now: datetime) -> list[str]:
        r, events = self.rules, []
        self.broker.start_day(now.date())
        bars = self.bars_loader(sorted(set(self.watch) | set(self.broker.positions)))
        data = {}
        for sym, df in bars.items():
            df = completed_bars(to_ny(df), now)
            data[sym] = (df, session_slices(df, now.date())[1])

        # ---- exits: check the latest completed bar against each position
        for sym, p in list(self.broker.positions.items()):
            if sym not in data or data[sym][1].empty:
                continue
            reg = data[sym][1]
            # Only judge bars that started after we got in (not the signal bar itself).
            opened = pd.Timestamp(p.opened_at).tz_convert(NY).floor("min")
            if reg.index[-1] < opened:
                continue
            bar = reg.iloc[-1]
            if bar["high"] >= p.entry + r.breakeven_r * p.risk and p.stop < p.entry:
                p.stop = p.entry
                events.append(f"{sym}: up {r.breakeven_r:g}R, stop moved to breakeven {p.entry:.2f}")
            reason, price = None, float(bar["close"])
            if bar["low"] <= p.stop:
                reason, price = "stop", min(p.stop, float(bar["open"]))
            elif bar["high"] >= p.target:
                reason, price = "target", max(p.target, float(bar["open"]))
            elif momentum_lost(reg, "call"):
                reason = "lost VWAP"
            elif now.time() >= r.exit_time:
                reason = "time"
            if reason:
                msg = f"SELL {p.qty} {sym} @ ~{price:.2f} ({reason})"
                if not self.dry_run:
                    msg += f"  P&L ${self.broker.sell(sym, price, reason, now):+,.2f}"
                events.append(msg)
        self.broker.save()

        # ---- entries
        if not (r.entry_start <= now.time() < r.entry_end):
            return events
        if self._halted():
            return events + ["daily loss limit hit - no new trades today"]
        for sym in self.watch:
            if self.broker.day["trades"] >= r.max_trades_per_day:
                break
            if sym in self.broker.day["traded"] or sym in self.broker.positions or sym not in data:
                continue
            full, reg = data[sym]
            st = premarket_stats(full, now.date())
            if st is None or not entry_signal(reg, st["pm_high"], st["pm_low"], "call"):
                continue
            entry = float(reg["close"].iloc[-1])
            stop = stop_level(reg, entry, r)
            qty = share_size(self.broker.buying_power(), entry * (1 + r.slippage_pct), stop, r)
            if qty < 1:
                continue
            target = entry + r.target_r * (entry - stop)
            msg = (f"BUY {qty} {sym} @ ~{entry:.2f}  stop {stop:.2f}  target {target:.2f}  "
                   f"[broke premarket high {st['pm_high']:.2f}, gap {st['gap_pct']:+.1f}%]")
            if self.dry_run:
                self.broker.day["traded"].append(sym)
            else:
                self.broker.buy(sym, qty, entry, stop, target, now)
            events.append(msg)
        return events

    def run(self, poll_seconds: int = 30) -> None:
        while True:
            now = datetime.now(NY)
            if now.weekday() >= 5:
                log.warning("market closed (weekend)")
                return
            if now.time() < OPEN:
                wait = (datetime.combine(now.date(), OPEN, NY) - now).total_seconds()
                log.warning("waiting %.0f min for the open", wait / 60)
                _time.sleep(min(wait, 300))
                continue
            for e in self.step(now):
                log.warning(e)
            if now.time() >= self.rules.entry_end and not self.broker.positions:
                log.warning("done for the day. realized P&L: $%.2f", self.broker.day.get("realized", 0))
                return
            _time.sleep(poll_seconds)
