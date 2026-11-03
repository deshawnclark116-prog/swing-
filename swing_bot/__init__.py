"""A daily swing-trading bot: strategy, backtester, risk sizing, and broker adapters."""
from .backtest import Backtester, BacktestResult
from .config import BotConfig, ExecutionConfig, RiskConfig, StrategyConfig
from .strategy import TrendCrossStrategy

__all__ = ["Backtester", "BacktestResult", "BotConfig", "ExecutionConfig", "RiskConfig",
           "StrategyConfig", "TrendCrossStrategy"]
