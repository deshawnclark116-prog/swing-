"""Central configuration for strategy, risk, and execution parameters."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    fast_period: int = 20       # fast EMA
    slow_period: int = 50       # slow SMA
    trend_period: int = 200     # long-term trend filter (SMA)
    rsi_period: int = 14
    rsi_entry_max: float = 70.0  # don't buy when already overbought
    rsi_exit: float = 80.0       # take profits when extremely overbought
    atr_period: int = 14
    stop_atr_mult: float = 2.0   # stop = entry - 2 * ATR
    target_atr_mult: float = 4.0  # target = entry + 4 * ATR  (2R)


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01      # risk 1% of equity per trade (entry -> stop)
    max_position_pct: float = 0.20    # never put more than 20% of equity in one name
    max_positions: int = 5            # max concurrent open positions


@dataclass
class ExecutionConfig:
    commission_per_trade: float = 0.0
    slippage_bps: float = 5.0         # 0.05% adverse slippage on every fill


@dataclass
class BotConfig:
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY"])
    starting_cash: float = 100_000.0
    lookback_days: int = 400          # history fetched for live signal generation
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
