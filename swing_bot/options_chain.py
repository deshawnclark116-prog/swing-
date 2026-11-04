"""Pick a tradeable options contract for a momentum play.

Rules (all configurable):
  * calls for gap-ups, puts for gap-downs
  * nearest expiration between min_dte and max_dte calendar days
    (default 1-10: skips same-day expiry, whose time decay is brutal)
  * strike nearest the money, optionally a bit in the money for more delta
  * liquidity: real bid, tight bid/ask spread, enough volume or open interest
  * premium must fit the per-trade budget
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

log = logging.getLogger("swing_bot.options")


@dataclass
class ContractConfig:
    min_dte: int = 1
    max_dte: int = 10
    itm_pct: float = 0.0          # 0.02 = prefer a strike ~2% in the money
    max_otm_pct: float = 0.03     # never buy strikes more than 3% out of the money (no lottery tickets)
    max_spread_pct: float = 0.12  # (ask - bid) / mid
    min_open_interest: int = 100
    min_volume: int = 50


@dataclass
class Contract:
    symbol: str          # OCC symbol, e.g. RIOT261113C00013000 (same format Alpaca uses)
    underlying: str
    kind: str            # "call" | "put"
    strike: float
    expiry: str          # YYYY-MM-DD
    bid: float
    ask: float
    volume: int
    open_interest: int
    iv: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else float("inf")

    @property
    def cost(self) -> float:
        """Dollars to buy one contract at the ask."""
        return self.ask * 100

    def __str__(self) -> str:
        return (f"{self.underlying} {self.expiry} {self.strike:g}{self.kind[0].upper()}  "
                f"bid {self.bid:.2f} / ask {self.ask:.2f}  (spread {self.spread_pct:.0%}, "
                f"vol {self.volume}, OI {self.open_interest}, IV {self.iv:.0%})  "
                f"= ${self.cost:,.0f}/contract")


class YahooOptions:
    """Chain source backed by yfinance. Quotes may be delayed ~15 minutes."""

    def expirations(self, symbol: str) -> list[str]:
        import yfinance as yf

        return list(yf.Ticker(symbol).options)

    def chain(self, symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf

        ch = yf.Ticker(symbol).option_chain(expiry)
        return ch.calls, ch.puts

    def quote(self, contract: Contract) -> tuple[float, float]:
        calls, puts = self.chain(contract.underlying, contract.expiry)
        tbl = calls if contract.kind == "call" else puts
        row = tbl[tbl["contractSymbol"] == contract.symbol]
        if row.empty:
            raise LookupError(f"{contract.symbol} not in chain")
        return float(row["bid"].iloc[0]), float(row["ask"].iloc[0])


def pick_contract(underlying: str, kind: str, spot: float, budget: float,
                  cfg: ContractConfig | None = None, source=None,
                  today: date | None = None) -> Contract | None:
    cfg = cfg or ContractConfig()
    source = source or YahooOptions()
    today = today or date.today()

    try:
        exps = source.expirations(underlying)
    except Exception as e:
        log.warning("no options for %s: %s", underlying, e)
        return None
    valid = [e for e in exps if cfg.min_dte <= (date.fromisoformat(e) - today).days <= cfg.max_dte]
    for expiry in sorted(valid):
        calls, puts = source.chain(underlying, expiry)
        tbl = (calls if kind == "call" else puts).copy()
        if tbl.empty:
            continue
        tbl = tbl.fillna({"volume": 0, "openInterest": 0, "impliedVolatility": 0})
        tbl = tbl[(tbl["bid"] > 0) & (tbl["ask"] > tbl["bid"])]
        mid = (tbl["bid"] + tbl["ask"]) / 2
        tbl = tbl[((tbl["ask"] - tbl["bid"]) / mid <= cfg.max_spread_pct)
                  & ((tbl["openInterest"] >= cfg.min_open_interest) | (tbl["volume"] >= cfg.min_volume))
                  & (tbl["ask"] * 100 <= budget)]
        if kind == "call":
            tbl = tbl[tbl["strike"] <= spot * (1 + cfg.max_otm_pct)]
        else:
            tbl = tbl[tbl["strike"] >= spot * (1 - cfg.max_otm_pct)]
        if tbl.empty:
            continue
        # In the money = below spot for calls, above spot for puts.
        target = spot * (1 - cfg.itm_pct) if kind == "call" else spot * (1 + cfg.itm_pct)
        row = tbl.loc[(tbl["strike"] - target).abs().idxmin()]
        return Contract(
            symbol=row["contractSymbol"], underlying=underlying, kind=kind,
            strike=float(row["strike"]), expiry=expiry, bid=float(row["bid"]), ask=float(row["ask"]),
            volume=int(row["volume"]), open_interest=int(row["openInterest"]),
            iv=float(row["impliedVolatility"]),
        )
    return None
