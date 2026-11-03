"""Position sizing."""
from __future__ import annotations

import math

from .config import RiskConfig


def position_size(equity: float, cash: float, price: float, stop: float, cfg: RiskConfig) -> int:
    """Shares to buy so that hitting the stop loses ~risk_per_trade of equity.

    Capped by max_position_pct of equity and by available cash.
    """
    if price <= 0 or equity <= 0:
        return 0
    risk_per_share = price - stop
    if risk_per_share <= 0:
        return 0
    by_risk = (equity * cfg.risk_per_trade) / risk_per_share
    by_alloc = (equity * cfg.max_position_pct) / price
    by_cash = max(cash, 0.0) / price
    return max(0, math.floor(min(by_risk, by_alloc, by_cash)))
