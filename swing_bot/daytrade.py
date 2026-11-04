"""Gap-and-go options day trader.

Morning routine, automated:
  1. Watch the premarket gappers from the scanner.
  2. After the open, BUY CALLS when a gap-up stock closes a 1-minute bar above its
     premarket high and above VWAP; BUY PUTS when a gap-down stock closes below its
     premarket low and below VWAP.
  3. SELL when any of these hit:
       - premium stop      (option down 30%)
       - profit target     (option up 100%)
       - trailing stop     (after +30%, give back no more than 15% from the peak)
       - momentum lost     (stock closes back through VWAP)
       - time              (3:45pm ET - never hold short-dated options overnight)

Guard rails: only settled cash is used, each trade gets a fixed slice of settled
cash, a max number of trades per day, and a daily loss limit that halts trading.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time as _time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from .cash_account import CashAccount
from .options_chain import Contract, ContractConfig, YahooOptions, pick_contract
from .premarket import NY, OPEN, premarket_stats, session_slices, to_ny, yahoo_intraday

log = logging.getLogger("swing_bot.daytrade")


@dataclass
class DayTradeRules:
    budget_pct: float = 0.10          # max premium per trade, as a fraction of settled cash
    max_trades_per_day: int = 3
    daily_loss_limit_pct: float = 0.10  # stop for the day after losing 10% of the morning's cash
    # Exits. The tested setups exit on a stock-price stop or the clock; the premium
    # stop/target/trail below are wide safety nets on top of that.
    stock_stop_pct: float = 1.5       # exit if the STOCK moves 1.5% against us from entry
    exit_on_vwap: bool = False        # VWAP exits tested worse - off by default
    stop_pct: float = 0.50            # option premium down 50%
    target_pct: float = 2.00          # option premium up 200%
    trail_trigger_pct: float = 1.00   # once up 100%...
    trail_pct: float = 0.30           # ...give back at most 30% from the peak
    # Entry: settings chosen by out-of-sample testing (see research/gap_variants.py)
    entry_mode: str = "orb"           # "orb" = break of the opening range, "pm" = premarket high/low
    or_minutes: int = 15              # opening range = first 15 minutes
    min_gap_call: float = 3.0         # gap-up % needed for calls
    min_gap_put: float = 5.0          # gap-down % needed for puts
    min_rvol_call: float = 2.5        # calls need opening volume >= 2.5x normal
    min_rvol_put: float = 0.0
    entry_start: time = time(9, 45)   # wait for the opening range to form
    entry_end: time = time(11, 0)     # gap-and-go is a morning setup
    call_exit_time: time = time(12, 0)  # call momentum fades after the morning
    exit_time: time = time(15, 45)    # hard close for everything


# ---------------------------------------------------------------- signals
def vwap(bars: pd.DataFrame) -> pd.Series:
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_v = bars["volume"].cumsum()
    return ((tp * bars["volume"]).cumsum() / cum_v.where(cum_v > 0)).fillna(bars["close"])


def _window_mask(idx: pd.DatetimeIndex, minutes: int):
    t = idx.hour * 60 + idx.minute
    return (t >= 9 * 60 + 30) & (t < 9 * 60 + 30 + minutes)


def opening_range(reg: pd.DataFrame, minutes: int = 15) -> tuple[float, float] | None:
    """High/low of the first `minutes` of the session, once that window is complete."""
    if reg.empty:
        return None
    first = reg[_window_mask(reg.index, minutes)]
    last_t = reg.index[-1].hour * 60 + reg.index[-1].minute
    bar_min = max(int((reg.index[1] - reg.index[0]).total_seconds() // 60), 1) if len(reg) > 1 else 1
    if first.empty or last_t + bar_min < 9 * 60 + 30 + minutes:
        return None
    return float(first["high"].max()), float(first["low"].min())


def opening_rvol(df: pd.DataFrame, day: date, minutes: int = 15, lookback: int = 10) -> float:
    """Volume in today's first `minutes` vs the average of the previous `lookback` days."""
    df = df[_window_mask(df.index, minutes)]
    vols = df.groupby(df.index.date)["volume"].sum()
    if day not in vols.index:
        return float("nan")
    prev = vols[vols.index < day].tail(lookback)
    if len(prev) < 5 or prev.mean() <= 0:
        return float("nan")
    return float(vols[day] / prev.mean())


def setup_qualifies(st: dict, kind: str, rvol: float, rules: "DayTradeRules") -> bool:
    gap = st["gap_pct"]
    if kind == "call":
        return gap >= rules.min_gap_call and (rules.min_rvol_call <= 0 or rvol >= rules.min_rvol_call)
    return gap <= -rules.min_gap_put and (rules.min_rvol_put <= 0 or rvol >= rules.min_rvol_put)


def breakout_levels(reg: pd.DataFrame, st: dict, rules: "DayTradeRules") -> tuple[float, float] | None:
    if rules.entry_mode == "orb":
        return opening_range(reg, rules.or_minutes)
    return st["pm_high"], st["pm_low"]


def entry_signal(reg: pd.DataFrame, pm_high: float, pm_low: float, kind: str) -> bool:
    """reg = today's COMPLETED regular-session bars."""
    if reg.empty:
        return False
    close, vw = float(reg["close"].iloc[-1]), float(vwap(reg).iloc[-1])
    if kind == "call":
        return close > pm_high and close > vw
    return close < pm_low and close < vw


def momentum_lost(reg: pd.DataFrame, kind: str) -> bool:
    if reg.empty:
        return False
    close, vw = float(reg["close"].iloc[-1]), float(vwap(reg).iloc[-1])
    return close < vw if kind == "call" else close > vw


def premium_exit(entry: float, peak: float, bid: float, rules: DayTradeRules) -> str | None:
    if bid <= entry * (1 - rules.stop_pct):
        return "stop"
    if bid >= entry * (1 + rules.target_pct):
        return "target"
    if peak >= entry * (1 + rules.trail_trigger_pct) and bid <= peak * (1 - rules.trail_pct):
        return "trail"
    return None


def completed_bars(df: pd.DataFrame, now: datetime, bar: timedelta = timedelta(minutes=1)) -> pd.DataFrame:
    """Drop the still-forming last bar."""
    return df[df.index + bar <= now]


# ---------------------------------------------------------------- brokers
@dataclass
class OptionPosition:
    contract: dict
    qty: int
    entry: float
    peak: float
    opened_at: str
    under_entry: float | None = None  # stock price when we bought

    @property
    def c(self) -> Contract:
        return Contract(**self.contract)


class OptionsBroker(ABC):
    """Holds position metadata and daily counters in a local JSON state file."""

    def __init__(self, state_file: str | Path):
        self.path = Path(state_file)
        raw = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.positions: dict[str, OptionPosition] = {
            k: OptionPosition(**v) for k, v in raw.get("positions", {}).items()}
        self.day: dict = raw.get("day", {})
        self.log: list[dict] = raw.get("log", [])
        self._extra = raw

    def save(self) -> None:
        data = {**self._extra, "positions": {k: asdict(v) for k, v in self.positions.items()},
                "day": self.day, "log": self.log[-1000:]}
        data.update(self._state())
        # write-then-rename so a reader never sees a half-written file
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp, self.path)

    def _state(self) -> dict:
        return {}

    def start_day(self, today: date) -> None:
        if self.day.get("date") != today.isoformat():
            self.day = {"date": today.isoformat(), "trades": 0, "realized": 0.0,
                        "start_cash": self.buying_power(), "traded": []}
            self.save()

    @abstractmethod
    def buying_power(self) -> float: ...

    @abstractmethod
    def quote(self, c: Contract) -> tuple[float, float]: ...

    @abstractmethod
    def _execute_buy(self, c: Contract, qty: int, price: float) -> None: ...

    @abstractmethod
    def _execute_sell(self, p: OptionPosition, price: float, today: date) -> None: ...

    def buy(self, c: Contract, qty: int, price: float, now: datetime, under_price: float | None = None) -> None:
        self._execute_buy(c, qty, price)
        self.positions[c.symbol] = OptionPosition(asdict(c), qty, price, price, now.isoformat(), under_price)
        self.day["trades"] += 1
        self.day["traded"].append(c.underlying)
        self.log.append({"t": now.isoformat(), "side": "buy", "contract": c.symbol, "qty": qty,
                         "price": price})
        self.save()

    def sell(self, symbol: str, price: float, reason: str, now: datetime) -> float:
        p = self.positions.pop(symbol)
        self._execute_sell(p, price, now.date())
        pnl = (price - p.entry) * 100 * p.qty
        self.day["realized"] += pnl
        self.log.append({"t": now.isoformat(), "side": "sell", "contract": symbol, "qty": p.qty,
                         "price": price, "pnl": round(pnl, 2), "reason": reason})
        self.save()
        return pnl


class PaperOptionsBroker(OptionsBroker):
    """Simulated cash account: buys fill at the ask, sells at the bid, T+1 settlement."""

    def __init__(self, state_file="options_paper_state.json", starting_cash=5_000.0, source=None):
        super().__init__(state_file)
        self.source = source or YahooOptions()
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

    def quote(self, c: Contract) -> tuple[float, float]:
        return self.source.quote(c)

    def _execute_buy(self, c, qty, price) -> None:
        self.account.buy(qty * price * 100)

    def _execute_sell(self, p, price, today) -> None:
        self.account.sell(p.qty * price * 100, today)


class AlpacaOptionsBroker(OptionsBroker):
    """Alpaca options trading (paper account unless ALPACA_PAPER=false).

    Buys are marketable limit orders at the ask; sells are market orders so exits
    always go through. Quotes come from Alpaca's options data feed.
    """

    def __init__(self, state_file="options_alpaca_state.json"):
        super().__init__(state_file)
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.trading.client import TradingClient

        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (see .env.example)")
        self.paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        self.client = TradingClient(key, secret, paper=self.paper)
        self.data = OptionHistoricalDataClient(key, secret)

    def buying_power(self) -> float:
        acct = self.client.get_account()
        return float(acct.options_buying_power or acct.cash)

    def quote(self, c: Contract) -> tuple[float, float]:
        from alpaca.data.requests import OptionLatestQuoteRequest

        q = self.data.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=c.symbol))[c.symbol]
        return float(q.bid_price), float(q.ask_price)

    def _execute_buy(self, c, qty, price) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        self.client.submit_order(LimitOrderRequest(symbol=c.symbol, qty=qty, side=OrderSide.BUY,
                                                   time_in_force=TimeInForce.DAY,
                                                   limit_price=round(price, 2)))

    def _execute_sell(self, p, price, today) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        self.client.submit_order(MarketOrderRequest(symbol=p.c.symbol, qty=p.qty, side=OrderSide.SELL,
                                                    time_in_force=TimeInForce.DAY))


# ---------------------------------------------------------------- the trader
class OptionsDayTrader:
    def __init__(self, broker: OptionsBroker, watch: dict[str, str],
                 rules: DayTradeRules | None = None, contract_cfg: ContractConfig | None = None,
                 bars_loader=None, chain_source=None, dry_run: bool = False, hist_loader=None):
        """watch = {symbol: "call" | "put"} from the premarket scan."""
        self.broker = broker
        self.watch = watch
        self.rules = rules or DayTradeRules()
        self.contract_cfg = contract_cfg or ContractConfig()
        self.bars_loader = bars_loader or (lambda syms: yahoo_intraday(syms, period="2d", interval="1m"))
        self.chain_source = chain_source or YahooOptions()
        self.dry_run = dry_run
        self.hist_loader = hist_loader or (lambda syms: yahoo_intraday(syms, period="20d", interval="5m"))
        self._hist: dict[str, pd.DataFrame] | None = None

    def _rvol(self, sym: str, reg: pd.DataFrame, day: date) -> float:
        """Opening relative volume: today's first minutes vs the last 10 sessions."""
        if self._hist is None:
            try:
                self._hist = {s: to_ny(d) for s, d in self.hist_loader(list(self.watch)).items()}
            except Exception as e:
                log.warning("history for relative volume unavailable: %s", e)
                self._hist = {}
        hist = self._hist.get(sym)
        if hist is None:
            return float("nan")
        past = hist[hist.index.date < day]
        combined = pd.concat([past, reg])
        return opening_rvol(combined, day, self.rules.or_minutes)

    def _halted(self) -> bool:
        d = self.broker.day
        limit = d.get("start_cash", 0) * self.rules.daily_loss_limit_pct
        return d.get("realized", 0) <= -limit if limit > 0 else False

    def step(self, now: datetime) -> list[str]:
        """One pass: manage open positions, then look for new entries."""
        events: list[str] = []
        self.broker.start_day(now.date())
        held_under = {p.c.underlying for p in self.broker.positions.values()}
        bars = self.bars_loader(sorted(set(self.watch) | held_under))
        today_reg: dict[str, pd.DataFrame] = {}
        for sym, df in bars.items():
            df = completed_bars(to_ny(df), now)
            today_reg[sym] = (df, session_slices(df, now.date())[1])

        # ---- exits
        for occ, pos in list(self.broker.positions.items()):
            c = pos.c
            try:
                bid, ask = self.broker.quote(c)
            except Exception as e:
                events.append(f"quote failed for {occ}: {e}")
                continue
            pos.peak = max(pos.peak, bid)
            reason = premium_exit(pos.entry, pos.peak, bid, self.rules)
            reg_u = today_reg.get(c.underlying, (None, pd.DataFrame()))[1]
            if reason is None and pos.under_entry and not reg_u.empty:
                last = float(reg_u["close"].iloc[-1])
                against = (pos.under_entry - last) if c.kind == "call" else (last - pos.under_entry)
                if against / pos.under_entry * 100 >= self.rules.stock_stop_pct:
                    reason = "stock stop"
            if reason is None and self.rules.exit_on_vwap and momentum_lost(reg_u, c.kind):
                reason = "lost VWAP"
            cutoff = self.rules.call_exit_time if c.kind == "call" else self.rules.exit_time
            if reason is None and now.time() >= min(cutoff, self.rules.exit_time):
                reason = "time"
            if reason:
                msg = f"SELL {pos.qty} {c} @ {bid:.2f} ({reason})"
                if not self.dry_run:
                    pnl = self.broker.sell(occ, bid, reason, now)
                    msg += f"  P&L ${pnl:+,.2f}"
                events.append(msg)
        self.broker.save()

        # ---- entries
        t = now.time()
        if not (self.rules.entry_start <= t < self.rules.entry_end):
            return events
        if self._halted():
            events.append("daily loss limit hit - no new trades today")
            return events
        for sym, kind in self.watch.items():
            if self.broker.day["trades"] >= self.rules.max_trades_per_day:
                break
            if sym in self.broker.day["traded"] or sym not in today_reg:
                continue
            full, reg = today_reg[sym]
            st = premarket_stats(full, now.date())
            if st is None:
                continue
            rvol = self._rvol(sym, reg, now.date())
            if not setup_qualifies(st, kind, rvol, self.rules):
                self.broker.day["traded"].append(sym)  # gap/volume don't change after the open range
                events.append(f"{sym}: skipped (gap {st['gap_pct']:+.1f}%, opening volume {rvol:.1f}x)")
                continue
            levels = breakout_levels(reg, st, self.rules)
            if levels is None or not entry_signal(reg, levels[0], levels[1], kind):
                continue
            spot = float(reg["close"].iloc[-1])
            budget = self.broker.buying_power() * self.rules.budget_pct
            c = pick_contract(sym, kind, spot, budget, self.contract_cfg, self.chain_source, now.date())
            if c is None:
                events.append(f"{sym}: signal, but no liquid {kind} within ${budget:,.0f}")
                self.broker.day["traded"].append(sym)  # don't retry every minute
                continue
            qty = math.floor(budget / c.cost)
            if qty < 1:
                continue
            msg = f"BUY {qty} {c}  [{sym} {spot:.2f} broke premarket {'high' if kind == 'call' else 'low'}]"
            if self.dry_run:
                self.broker.day["traded"].append(sym)
            else:
                self.broker.buy(c, qty, c.ask, now, under_price=spot)
            events.append(msg)
        return events

    def run(self, poll_seconds: int = 60) -> None:
        """Loop from the open until every position is closed after exit_time."""
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
