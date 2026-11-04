"""Cash-account bookkeeping with T+1 settlement.

In a cash account there is no day-trade limit, but you may only buy with SETTLED
cash. Options (and stocks) settle one business day after the trade, so money from
a sale is usable again the next business day. Buying only with settled cash means
you can never get a good-faith violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


def next_business_day(d: date) -> date:
    # Weekends only; exchange holidays push settlement one more day in reality.
    return (pd.Timestamp(d) + pd.offsets.BDay(1)).date()


@dataclass
class CashAccount:
    settled: float
    unsettled: list[tuple[str, float]] = field(default_factory=list)  # (settle_date, amount)

    def settle(self, today: date) -> None:
        keep = []
        for d, amt in self.unsettled:
            if date.fromisoformat(d) <= today:
                self.settled += amt
            else:
                keep.append((d, amt))
        self.unsettled = keep

    @property
    def pending(self) -> float:
        return sum(a for _, a in self.unsettled)

    @property
    def total_cash(self) -> float:
        return self.settled + self.pending

    def can_buy(self, cost: float) -> bool:
        return cost <= self.settled + 1e-9

    def buy(self, cost: float) -> None:
        if not self.can_buy(cost):
            raise ValueError(f"cost {cost:.2f} exceeds settled cash {self.settled:.2f} "
                             "(would be a good-faith violation)")
        self.settled -= cost

    def sell(self, proceeds: float, today: date) -> None:
        self.unsettled.append((next_business_day(today).isoformat(), proceeds))

    def to_dict(self) -> dict:
        return {"settled": self.settled, "unsettled": self.unsettled}

    @classmethod
    def from_dict(cls, d: dict) -> "CashAccount":
        return cls(d["settled"], [tuple(x) for x in d.get("unsettled", [])])
