from datetime import date, datetime, time

import pandas as pd
import pytest

from swing_bot.cash_account import CashAccount, next_business_day
from swing_bot.daytrade import (DayTradeRules, OptionsDayTrader, PaperOptionsBroker, entry_signal,
                                opening_range, opening_rvol, premium_exit, setup_qualifies, vwap)
from swing_bot.options_chain import ContractConfig, pick_contract
from swing_bot.premarket import NY, premarket_stats
from swing_bot.stock_daytrade import PaperStockBroker, StockDayTrader, StockRules, share_size, stop_level

DAY = date(2026, 11, 3)


def make_day(pm_prices, reg_prices, prev_close=10.0, day=DAY):
    """1-min bars: one prior-day regular bar, premarket bars from 8:00, regular bars from 9:30."""
    rows = [(pd.Timestamp(datetime.combine(date(2026, 11, 2), time(15, 59)), tz=NY), prev_close, 1000)]
    for i, p in enumerate(pm_prices):
        rows.append((pd.Timestamp(datetime.combine(day, time(8, i)), tz=NY), p, 0))
    for i, p in enumerate(reg_prices):
        rows.append((pd.Timestamp(datetime.combine(day, time(9, 30)), tz=NY) + pd.Timedelta(minutes=i), p, 1000))
    idx = [r[0] for r in rows]
    c = [r[1] for r in rows]
    return pd.DataFrame({"open": c, "high": [x * 1.001 for x in c], "low": [x * 0.999 for x in c],
                         "close": c, "volume": [r[2] for r in rows]}, index=pd.DatetimeIndex(idx))


def at(h, m):
    return datetime.combine(DAY, time(h, m), NY)


# ---------------------------------------------------------------- premarket + signals
def test_premarket_stats():
    df = make_day([11.0, 11.5, 11.2], [11.3])
    st = premarket_stats(df, DAY)
    assert st["prev_close"] == 10.0
    assert st["gap_pct"] == pytest.approx(12.0)
    assert st["pm_high"] == pytest.approx(11.5 * 1.001)
    assert st["pm_volume"] != st["pm_volume"]  # NaN when the feed has no premarket volume


def test_vwap_and_entry_signal():
    reg = make_day([], [10, 11, 12]).iloc[1:]
    assert vwap(reg).iloc[-1] == pytest.approx(11.0, rel=1e-3)
    assert entry_signal(reg, pm_high=11.5, pm_low=9, kind="call")
    assert not entry_signal(reg, pm_high=12.5, pm_low=9, kind="call")
    down = make_day([], [12, 11, 10]).iloc[1:]
    assert entry_signal(down, pm_high=13, pm_low=10.5, kind="put")


def test_premium_exit_rules():
    r = DayTradeRules(stop_pct=0.30, target_pct=1.0, trail_trigger_pct=0.30, trail_pct=0.15)
    assert premium_exit(1.0, 1.0, 0.69, r) == "stop"
    assert premium_exit(1.0, 2.0, 2.0, r) == "target"
    assert premium_exit(1.0, 1.5, 1.27, r) == "trail"      # 15% off a +50% peak
    assert premium_exit(1.0, 1.2, 1.0, r) is None           # trail not armed below +30%


# ---------------------------------------------------------------- cash account
def test_cash_account_settlement():
    a = CashAccount(1000)
    a.buy(600)
    with pytest.raises(ValueError):
        a.buy(500)                         # only 400 settled -> would be a GFV
    a.sell(700, date(2026, 11, 6))         # Friday sale
    assert next_business_day(date(2026, 11, 6)) == date(2026, 11, 9)
    a.settle(date(2026, 11, 7))
    assert a.settled == 400 and a.pending == 700
    a.settle(date(2026, 11, 9))
    assert a.settled == 1100 and a.pending == 0


# ---------------------------------------------------------------- options
class FakeChain:
    def __init__(self, bid=1.0, ask=1.05):
        self.bid, self.ask = bid, ask

    def expirations(self, s):
        return ["2026-11-03", "2026-11-06", "2026-11-13"]

    def chain(self, s, e):
        strikes = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
        tbl = pd.DataFrame({
            "contractSymbol": [f"XYZ261106C{int(k * 1000):08d}" for k in strikes],
            "strike": strikes, "bid": self.bid, "ask": self.ask, "volume": 500,
            "openInterest": 1000, "impliedVolatility": 0.9,
        })
        return tbl, tbl.assign(contractSymbol=tbl["contractSymbol"].str.replace("C", "P", n=1))

    def quote(self, c):
        return self.bid, self.ask


def test_pick_contract_skips_0dte_and_picks_atm():
    c = pick_contract("XYZ", "call", 11.6, budget=500, source=FakeChain(), today=DAY)
    assert c.expiry == "2026-11-06" and c.strike == 11.5


def test_pick_contract_rejects_wide_spread_and_budget():
    assert pick_contract("XYZ", "call", 11.6, 500, source=FakeChain(1.0, 1.5), today=DAY) is None
    assert pick_contract("XYZ", "call", 11.6, 50, source=FakeChain(), today=DAY) is None


def test_options_trader_enters_and_stops_out(tmp_path):
    chain = FakeChain()
    broker = PaperOptionsBroker(tmp_path / "o.json", starting_cash=5000, source=chain)
    df = make_day([11.0, 11.5], [11.4, 11.6, 11.8, 12.0, 12.1, 12.2, 12.3])
    rules = DayTradeRules(entry_mode="pm", min_rvol_call=0, entry_start=time(9, 35), stop_pct=0.30)
    trader = OptionsDayTrader(broker, {"XYZ": "call"}, rules=rules, chain_source=chain,
                              bars_loader=lambda syms: {"XYZ": df}, hist_loader=lambda s: {})
    events = trader.step(at(9, 37))
    assert any(e.startswith("BUY") for e in events)
    assert broker.account.settled == pytest.approx(5000 - 4 * 105)  # 10% budget -> 4 contracts
    chain.bid = 0.6                                     # option drops 40%
    events = trader.step(at(9, 38))
    assert any("(stop)" in e for e in events) and not broker.positions
    assert broker.account.pending == pytest.approx(4 * 60)  # proceeds settle T+1
    # max one trade per symbol per day
    chain.bid = 1.0
    assert not any(e.startswith("BUY") for e in trader.step(at(9, 39)))


# ---------------------------------------------------------------- small-cap shares
def test_share_size_and_stop():
    r = StockRules()
    assert share_size(5000, 10.0, 9.5, r) == 125     # min(100/0.5=200, 1250/10=125)
    reg = make_day([], [10, 10.2, 10.4]).iloc[1:]
    stop = stop_level(reg, 10.4, r)
    assert 10.4 * 0.95 <= stop < 10.4


def test_stock_trader_cycle(tmp_path):
    broker = PaperStockBroker(tmp_path / "s.json", starting_cash=5000, slippage_pct=0)
    base = make_day([5.0, 5.5], [5.4, 5.6, 5.7])
    trader = StockDayTrader(broker, ["ABC"], bars_loader=lambda s: {"ABC": base})
    events = trader.step(at(9, 33))
    assert any(e.startswith("BUY") for e in events), events
    p = broker.positions["ABC"]
    assert p.stop < p.entry < p.target
    # the signal bar itself must not trigger the stop
    assert not any("SELL" in e for e in trader.step(at(9, 33)))
    # next bar crashes through the stop
    crash = pd.concat([base, make_day([], [4.0]).iloc[1:].set_axis(
        [pd.Timestamp(at(9, 33))])])
    trader.bars_loader = lambda s: {"ABC": crash}
    events = trader.step(at(9, 34))
    assert any("(stop)" in e for e in events) and not broker.positions
    assert broker.day["realized"] < 0
    assert broker.account.pending > 0          # unsettled until next business day


# ---------------------------------------------------------------- tested-setup rules
def test_opening_range_waits_for_window():
    reg = make_day([], [10 + 0.1 * i for i in range(20)]).iloc[1:]
    assert opening_range(reg.iloc[:10], 15) is None          # only 10 minutes in
    hi, lo = opening_range(reg, 15)
    assert hi == pytest.approx(11.4 * 1.001) and lo == pytest.approx(10 * 0.999)


def test_opening_rvol():
    frames = []
    for k, d in enumerate(pd.bdate_range("2026-10-20", "2026-11-03")):
        idx = pd.date_range(pd.Timestamp(datetime.combine(d.date(), time(9, 30)), tz=NY), periods=30, freq="1min")
        vol = 3000 if d.date() == DAY else 1000
        frames.append(pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": vol}, index=idx))
    df = pd.concat(frames)
    assert opening_rvol(df, DAY, 15) == pytest.approx(3.0)


def test_setup_qualifies_per_side():
    r = DayTradeRules()
    assert setup_qualifies({"gap_pct": 4.0}, "call", 3.0, r)
    assert not setup_qualifies({"gap_pct": 4.0}, "call", 1.2, r)      # volume too light
    assert not setup_qualifies({"gap_pct": 4.0}, "call", float("nan"), r)
    assert setup_qualifies({"gap_pct": -6.0}, "put", float("nan"), r)
    assert not setup_qualifies({"gap_pct": -4.0}, "put", 5.0, r)      # puts need a 5% gap


def test_options_stock_stop_and_noon_exit(tmp_path):
    chain = FakeChain()
    broker = PaperOptionsBroker(tmp_path / "o.json", starting_cash=5000, source=chain)
    rules = DayTradeRules(entry_mode="pm", min_rvol_call=0, entry_start=time(9, 35))
    df = make_day([11.0, 11.5], [11.4, 11.6, 11.8, 12.0, 12.1, 12.2, 12.3])
    trader = OptionsDayTrader(broker, {"XYZ": "call"}, rules=rules, chain_source=chain,
                              bars_loader=lambda s: {"XYZ": df}, hist_loader=lambda s: {})
    trader.step(at(9, 37))
    (pos,) = broker.positions.values()
    assert pos.under_entry == pytest.approx(12.3)
    # stock drops ~4% -> stock stop even though the option quote hasn't moved
    drop = pd.concat([df, make_day([], [11.75]).iloc[1:].set_axis([pd.Timestamp(at(9, 37))])])
    trader.bars_loader = lambda s: {"XYZ": drop}
    assert any("(stock stop)" in e for e in trader.step(at(9, 38)))

    broker2 = PaperOptionsBroker(tmp_path / "o2.json", starting_cash=5000, source=chain)
    t2 = OptionsDayTrader(broker2, {"XYZ": "call"}, rules=rules, chain_source=chain,
                          bars_loader=lambda s: {"XYZ": df}, hist_loader=lambda s: {})
    t2.step(at(9, 37))
    assert not any("SELL" in e for e in t2.step(at(11, 59)))
    assert any("(time)" in e for e in t2.step(at(12, 0)))            # calls are out by noon
