import numpy as np
import pandas as pd
import pytest

from swing_bot import indicators as ind
from swing_bot.backtest import Backtester
from swing_bot.bot import SwingBot
from swing_bot.broker import PaperBroker
from swing_bot.config import BotConfig, RiskConfig, StrategyConfig
from swing_bot.data import synthetic
from swing_bot.risk import position_size
from swing_bot.strategy import ENTER, EXIT, HOLD, Strategy, TrendCrossStrategy


def test_sma_and_rsi_bounds():
    s = pd.Series(np.arange(1, 11, dtype=float))
    assert ind.sma(s, 3).iloc[-1] == pytest.approx(9.0)
    r = ind.rsi(synthetic(300)["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert ind.rsi(s, 3).dropna().eq(100).all()  # monotonic up -> RSI 100


def test_atr_positive():
    a = ind.atr(synthetic(100)).dropna()
    assert (a > 0).all()


def test_no_signals_during_warmup():
    strat = TrendCrossStrategy()
    out = strat.generate(synthetic(400))
    assert (out["signal"].iloc[: strat.cfg.trend_period - 1] == HOLD).all()


def test_signals_have_no_lookahead():
    """Signal at bar t must not change when future bars are appended."""
    df = synthetic(600, seed=3)
    strat = TrendCrossStrategy()
    full = strat.generate(df)["signal"]
    for cut in (300, 450, 599):
        partial = strat.generate(df.iloc[:cut])["signal"]
        pd.testing.assert_series_equal(partial, full.iloc[:cut])


def test_position_size_respects_all_caps():
    cfg = RiskConfig(risk_per_trade=0.01, max_position_pct=0.2)
    # risk cap: 1000 / 2 = 500 shares; alloc cap: 20000/100 = 200 -> 200
    assert position_size(100_000, 100_000, 100, 98, cfg) == 200
    # cash cap
    assert position_size(100_000, 5_000, 100, 98, cfg) == 50
    # risk cap binds: 1000 / 10 = 100
    assert position_size(100_000, 100_000, 100, 90, cfg) == 100
    # invalid stop
    assert position_size(100_000, 100_000, 100, 101, cfg) == 0


class ScriptedStrategy(Strategy):
    """Enter on a chosen bar; fixed stop/target distances."""

    def __init__(self, enter_idx, exit_idx=None, stop=5.0, target=10.0):
        self.enter_idx, self.exit_idx, self.stop, self.target = enter_idx, exit_idx, stop, target

    def generate(self, df):
        out = df.copy()
        out["atr"] = 1.0
        out["signal"] = HOLD
        out.iloc[self.enter_idx, out.columns.get_loc("signal")] = ENTER
        if self.exit_idx is not None:
            out.iloc[self.exit_idx, out.columns.get_loc("signal")] = EXIT
        return out

    def stop_and_target(self, entry_price, row):
        return entry_price - self.stop, entry_price + self.target


def _flat(n=10, price=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 1}, index=idx)


def _cfg():
    cfg = BotConfig(starting_cash=10_000)
    cfg.execution.slippage_bps = 0
    return cfg


def test_entry_fills_next_open_and_exits_on_signal():
    df = _flat()
    df.loc[df.index[3], "open"] = 101.0
    res = Backtester(ScriptedStrategy(enter_idx=2, exit_idx=5), _cfg()).run({"X": df})
    (t,) = res.trades
    assert t.entry_date == df.index[3] and t.entry_price == 101.0
    assert t.exit_date == df.index[6] and t.reason == "signal"


def test_gap_through_stop_fills_at_open():
    df = _flat()
    df.loc[df.index[5], ["open", "high", "low", "close"]] = [90.0, 91.0, 89.0, 90.0]
    res = Backtester(ScriptedStrategy(enter_idx=2), _cfg()).run({"X": df})
    (t,) = res.trades
    assert t.reason == "stop" and t.exit_price == 90.0  # not the 95 stop level


def test_target_hit_intraday():
    df = _flat()
    df.loc[df.index[5], "high"] = 115.0
    res = Backtester(ScriptedStrategy(enter_idx=2), _cfg()).run({"X": df})
    (t,) = res.trades
    assert t.reason == "target" and t.exit_price == 110.0


def test_no_trades_keeps_equity_flat():
    res = Backtester(ScriptedStrategy(enter_idx=0, stop=-1), _cfg()).run({"X": _flat()})
    assert res.trades == [] and (res.equity == 10_000).all()


def test_synthetic_backtest_accounting():
    data = {s: synthetic(800, seed=i) for i, s in enumerate("ABC")}
    res = Backtester(TrendCrossStrategy(), BotConfig()).run(data)
    realized = sum(t.pnl for t in res.trades)
    # zero commission: final equity = start + realized pnl (everything closed at the end)
    assert res.metrics["end_equity"] == pytest.approx(100_000 + realized, rel=1e-9)


def test_bot_paper_cycle(tmp_path):
    df = _flat(10)
    state = tmp_path / "state.json"
    cfg = _cfg()
    cfg.symbols = ["X"]
    broker = PaperBroker(state, starting_cash=10_000, slippage_bps=0)

    bot = SwingBot(ScriptedStrategy(enter_idx=-1), broker, cfg, loader=lambda s: df)
    assert bot.run_once(dry_run=True)[0].side == "buy"
    assert broker.positions() == {}  # dry run places nothing

    (buy,) = bot.run_once()
    assert buy.side == "buy" and "X" in PaperBroker(state).positions()

    # Next day the price collapses through the stop -> bot sells.
    df2 = pd.concat([df, _flat(1, 80.0).set_axis([df.index[-1] + pd.offsets.BDay()])])
    bot2 = SwingBot(ScriptedStrategy(enter_idx=0), PaperBroker(state), cfg, loader=lambda s: df2)
    (sell,) = bot2.run_once()
    assert sell.side == "sell" and sell.reason == "stop"
    assert PaperBroker(state).positions() == {}
