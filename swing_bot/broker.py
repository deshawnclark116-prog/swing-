"""Broker adapters used by the live bot.

* PaperBroker  - local simulated account persisted to a JSON file. No API keys needed.
* AlpacaBroker - real orders through Alpaca (paper or live account), using bracket
                 orders so the stop-loss and take-profit live at the broker.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Holding:
    symbol: str
    qty: float
    avg_price: float
    stop: float | None = None
    target: float | None = None
    opened: str | None = None  # date of the bar the entry was based on


class Broker(ABC):
    #: True when the broker enforces stops/targets itself (bracket orders).
    manages_exits: bool = False

    @abstractmethod
    def equity(self, prices: dict[str, float]) -> float: ...

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def positions(self) -> dict[str, Holding]: ...

    @abstractmethod
    def buy(self, symbol: str, qty: int, price: float, stop: float, target: float | None,
            opened: str | None = None) -> None: ...

    @abstractmethod
    def sell_all(self, symbol: str, price: float, reason: str) -> None: ...


class PaperBroker(Broker):
    """Simulated account. Fills immediately at the price given (last close)."""

    def __init__(self, state_file: str | Path = "paper_state.json", starting_cash: float = 100_000.0,
                 slippage_bps: float = 5.0):
        self.path = Path(state_file)
        self.slip = slippage_bps / 10_000
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._cash = raw["cash"]
            self._pos = {s: Holding(**h) for s, h in raw["positions"].items()}
            self.log = raw.get("log", [])
        else:
            self._cash, self._pos, self.log = starting_cash, {}, []

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"cash": self._cash, "positions": {s: asdict(h) for s, h in self._pos.items()},
             "log": self.log[-500:]}, indent=2))

    def equity(self, prices: dict[str, float]) -> float:
        return self._cash + sum(h.qty * prices.get(s, h.avg_price) for s, h in self._pos.items())

    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Holding]:
        return dict(self._pos)

    def buy(self, symbol, qty, price, stop, target, opened=None) -> None:
        fill = price * (1 + self.slip)
        self._cash -= qty * fill
        self._pos[symbol] = Holding(symbol, qty, fill, stop, target, opened)
        self.log.append({"side": "buy", "symbol": symbol, "qty": qty, "price": round(fill, 4)})
        self._save()

    def sell_all(self, symbol, price, reason) -> None:
        h = self._pos.pop(symbol)
        fill = price * (1 - self.slip)
        self._cash += h.qty * fill
        self.log.append({"side": "sell", "symbol": symbol, "qty": h.qty, "price": round(fill, 4),
                         "pnl": round((fill - h.avg_price) * h.qty, 2), "reason": reason})
        self._save()


class AlpacaBroker(Broker):
    """Alpaca adapter (pip install alpaca-py).

    Reads ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_PAPER from the environment.
    Defaults to the PAPER account; real money requires ALPACA_PAPER=false explicitly.
    """

    manages_exits = True

    def __init__(self):
        from alpaca.trading.client import TradingClient

        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (see .env.example)")
        self.paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        self.client = TradingClient(key, secret, paper=self.paper)

    def equity(self, prices) -> float:
        return float(self.client.get_account().equity)

    def cash(self) -> float:
        return float(self.client.get_account().cash)

    def positions(self) -> dict[str, Holding]:
        return {p.symbol: Holding(p.symbol, float(p.qty), float(p.avg_entry_price))
                for p in self.client.get_all_positions()}

    def buy(self, symbol, qty, price, stop, target, opened=None) -> None:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop, 2)),
            take_profit=TakeProfitRequest(limit_price=round(target, 2)) if target else None,
        )
        self.client.submit_order(req)

    def sell_all(self, symbol, price, reason) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        # Cancel the bracket's open stop/target legs first, otherwise the shares are held.
        open_orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for o in open_orders:
            self.client.cancel_order_by_id(o.id)
        self.client.close_position(symbol)
