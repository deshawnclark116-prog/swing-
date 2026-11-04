"""Command-line interface.

    python -m swing_bot backtest --symbols AAPL MSFT NVDA --start 2018-01-01
    python -m swing_bot backtest --source synthetic
    python -m swing_bot run --dry-run              # show what the bot would do today
    python -m swing_bot run                        # paper-trade with local simulated account
    python -m swing_bot run --broker alpaca        # Alpaca (paper account by default)
    python -m swing_bot status                     # show paper account
"""
from __future__ import annotations

import argparse
import logging
import sys

from .backtest import Backtester
from .broker import AlpacaBroker, PaperBroker
from .config import BotConfig
from .data import load_many
from .strategy import STRATEGIES


def _fmt_metrics(m: dict) -> str:
    pct = {"total_return", "cagr", "max_drawdown", "win_rate", "avg_trade_return"}
    lines = []
    for k, v in m.items():
        if k in pct:
            lines.append(f"  {k:18} {v:>10.2%}")
        elif k == "num_trades":
            lines.append(f"  {k:18} {int(v):>10d}")
        else:
            lines.append(f"  {k:18} {v:>10.2f}")
    return "\n".join(lines)


def cmd_backtest(args, cfg: BotConfig) -> int:
    kwargs = {}
    if args.source == "yahoo":
        kwargs = {"start": args.start, "end": args.end}
    data = load_many(cfg.symbols, source=args.source, **kwargs)
    strat = STRATEGIES[args.strategy](cfg.strategy)
    res = Backtester(strat, cfg).run(data)
    print(f"Backtest: {', '.join(cfg.symbols)}  ({res.equity.index[0].date()} -> {res.equity.index[-1].date()})")
    print(_fmt_metrics(res.metrics))

    # Buy-and-hold benchmark: equal weight across the same symbols
    closes = [df["close"] for df in data.values()]
    bh = sum((c.iloc[-1] / c.iloc[0]) for c in closes) / len(closes) - 1
    print(f"  {'buy_and_hold':18} {bh:>10.2%}  (equal-weight benchmark)")

    tf = res.trades_frame()
    if not tf.empty:
        print("\nBy symbol:")
        print(tf.groupby("symbol").agg(trades=("pnl", "size"), pnl=("pnl", "sum"),
                                       win_rate=("pnl", lambda x: (x > 0).mean())).round(2).to_string())
        print("\nExit reasons:", tf["reason"].value_counts().to_dict())
    if args.trades_csv:
        tf.to_csv(args.trades_csv, index=False)
        print(f"\nTrades written to {args.trades_csv}")
    return 0


def cmd_run(args, cfg: BotConfig) -> int:
    from .bot import SwingBot

    if args.broker == "alpaca":
        broker = AlpacaBroker()
        print(f"Alpaca account: {'PAPER' if broker.paper else '*** LIVE MONEY ***'}")
    else:
        broker = PaperBroker(args.state, starting_cash=cfg.starting_cash,
                             slippage_bps=cfg.execution.slippage_bps)
    bot = SwingBot(STRATEGIES[args.strategy](cfg.strategy), broker, cfg)
    actions = bot.run_once(dry_run=args.dry_run)
    for a in actions:
        print(("[dry-run] " if args.dry_run else "") + str(a))
    if not actions:
        print("No actions today.")
    return 0


def cmd_status(args, cfg: BotConfig) -> int:
    b = PaperBroker(args.state, starting_cash=cfg.starting_cash)
    print(f"Cash: {b.cash():,.2f}")
    for h in b.positions().values():
        print(f"  {h.symbol:6} qty={h.qty:<6g} avg={h.avg_price:.2f} stop={h.stop or 0:.2f} "
              f"target={h.target or 0:.2f} opened={h.opened}")
    for entry in b.log[-10:]:
        print("  log:", entry)
    return 0


# ====================================================================== day trading
def _scan_both(args):
    from .premarket import OPTIONS_SCAN, SMALLCAP_SCAN, scan

    opt = scan(OPTIONS_SCAN, with_news=not args.no_news)
    small = scan(SMALLCAP_SCAN, with_news=not args.no_news)
    return opt, small


def _print_scan(title, df, extra=None):
    print(f"\n=== {title} ===")
    if df.empty:
        print("  nothing qualifies right now")
        return
    print(f"  (premarket session of {df['date'].iloc[0]})")
    for _, r in df.iterrows():
        print(f"  {r.symbol:6} {r.gap_pct:+6.1f}%  ${r.pm_last:<8.2f} PM high {r.pm_high:.2f} / low {r.pm_low:.2f}"
              f"   avg vol {r.avg_volume / 1e6:.1f}M")
        if extra:
            line = extra(r)
            if line:
                print(f"         {line}")
        if isinstance(r.get("news"), str) and r["news"]:
            for h in r["news"].split(" | "):
                print(f"         news: {h}")


def cmd_premarket(args, cfg) -> int:
    from .options_chain import pick_contract

    opt, small = _scan_both(args)
    budget = args.cash * 0.10

    def contract_line(r):
        c = pick_contract(r.symbol, r.direction, r.pm_last, budget)
        return f"{r.direction.upper()}: {c}" if c else f"{r.direction.upper()}: no liquid contract within ${budget:,.0f}"

    _print_scan("SYSTEM 1 - OPTIONS (liquid names, calls on gap-ups / puts on gap-downs)", opt, contract_line)
    _print_scan("SYSTEM 2 - SMALL-CAP SHARES (gap-ups, $1-$20)", small)
    print("\nEntry trigger after 9:30: a 1-min close above the premarket high (calls/shares) "
          "or below the premarket low (puts), on the right side of VWAP.")
    return 0


def _loop_or_once(trader, args):
    from datetime import datetime

    from .premarket import NY

    if args.once:
        events = trader.step(datetime.now(NY))
        print("\n".join(events) if events else "no signals / no exits right now")
    else:
        trader.run()
    return 0


def cmd_options_trade(args, cfg) -> int:
    from .daytrade import AlpacaOptionsBroker, OptionsDayTrader, PaperOptionsBroker
    from .premarket import OPTIONS_SCAN, scan

    watch = scan(OPTIONS_SCAN, with_news=False)
    watch_map = dict(zip(watch["symbol"], watch["direction"])) if not watch.empty else {}
    print(f"Watching: {watch_map or 'nothing'}")
    broker = AlpacaOptionsBroker() if args.broker == "alpaca" else PaperOptionsBroker(starting_cash=args.cash)
    return _loop_or_once(OptionsDayTrader(broker, watch_map, dry_run=args.dry_run), args)


def cmd_stocks_trade(args, cfg) -> int:
    from .premarket import SMALLCAP_SCAN, scan
    from .stock_daytrade import AlpacaStockBroker, PaperStockBroker, StockDayTrader

    watch = scan(SMALLCAP_SCAN, with_news=False)
    syms = list(watch["symbol"]) if not watch.empty else []
    print(f"Watching: {syms or 'nothing'}")
    broker = AlpacaStockBroker() if args.broker == "alpaca" else PaperStockBroker(starting_cash=args.cash)
    return _loop_or_once(StockDayTrader(broker, syms, dry_run=args.dry_run), args)


def cmd_gap_backtest(args, cfg) -> int:
    from . import gap_backtest
    from .premarket import OPTIONS_SCAN, SMALLCAP_SCAN, screener_quotes, yahoo_intraday

    sc = OPTIONS_SCAN if args.system == "options" else SMALLCAP_SCAN
    universe = sorted(set(args.symbols or []) or (set(screener_quotes(sc.screeners)) | set(sc.watchlist)))
    print(f"Backtesting gap-and-go on {len(universe)} symbols, 5-min bars, last ~60 days...")
    bars = yahoo_intraday(universe, period="60d", interval="5m")
    # apply the system's price filter using each symbol's latest price
    bars = {s: d for s, d in bars.items() if sc.min_price <= d["close"].iloc[-1] <= sc.max_price}
    trades = gap_backtest.run(bars, min_gap_pct=sc.min_gap_pct, stock_stop_pct=args.stock_stop)
    if args.system == "smallcap" and not trades.empty:
        trades = trades[trades["kind"] == "call"]  # long only
    print(gap_backtest.summarize(trades))
    print("\nNote: the universe is today's movers, so this is optimistic (survivorship/selection bias).")
    if args.trades_csv and not trades.empty:
        trades.to_csv(args.trades_csv, index=False)
    return 0


def cmd_accounts(args, cfg) -> int:
    from .daytrade import PaperOptionsBroker
    from .stock_daytrade import PaperStockBroker

    for title, b in (("OPTIONS paper account", PaperOptionsBroker()), ("STOCKS paper account", PaperStockBroker())):
        a = b.account
        print(f"\n=== {title} ===\n  settled cash ${a.settled:,.2f}   unsettled ${a.pending:,.2f}")
        print(f"  today: {b.day}")
        for k, p in b.positions.items():
            print(f"  open: {k} {asdict_safe(p)}")
        for e in b.log[-10:]:
            print(f"  log: {e}")
    return 0


def asdict_safe(obj):
    from dataclasses import asdict

    return {k: v for k, v in asdict(obj).items() if k != "contract"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swing_bot", description="Daily swing-trading bot")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--symbols", nargs="+", help="tickers (default: config list)")
        sp.add_argument("--strategy", default="trend", choices=sorted(STRATEGIES))
        sp.add_argument("--cash", type=float, help="starting cash")
        sp.add_argument("--risk", type=float, help="fraction of equity risked per trade, e.g. 0.01")
        sp.add_argument("--max-positions", type=int)

    bt = sub.add_parser("backtest", help="backtest the strategy on historical data")
    common(bt)
    bt.add_argument("--source", default="yahoo", choices=["yahoo", "synthetic", "csv"])
    bt.add_argument("--start", default="2018-01-01")
    bt.add_argument("--end")
    bt.add_argument("--trades-csv", help="write the trade list to this CSV")
    bt.set_defaults(func=cmd_backtest)

    run = sub.add_parser("run", help="generate today's signals and place orders")
    common(run)
    run.add_argument("--broker", default="paper", choices=["paper", "alpaca"])
    run.add_argument("--state", default="paper_state.json", help="paper account state file")
    run.add_argument("--dry-run", action="store_true", help="print orders without placing them")
    run.set_defaults(func=cmd_run)

    st = sub.add_parser("status", help="show the local paper account")
    st.add_argument("--state", default="paper_state.json")
    st.set_defaults(func=cmd_status, symbols=None, cash=None, risk=None, max_positions=None)

    none = dict(symbols=None, cash=None, risk=None, max_positions=None)

    pm = sub.add_parser("premarket", help="morning scan: options plays + small-cap plays + news")
    pm.add_argument("--cash", type=float, default=5_000, help="account size (sets option budget)")
    pm.add_argument("--no-news", action="store_true")
    pm.set_defaults(func=cmd_premarket, **{**none, "cash": 5_000})

    for name, fn, helptext in (("options-trade", cmd_options_trade, "System 1: day trade options"),
                               ("stocks-trade", cmd_stocks_trade, "System 2: day trade small-cap shares")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--broker", default="paper", choices=["paper", "alpaca"])
        sp.add_argument("--cash", type=float, default=5_000, help="starting cash for the paper account")
        sp.add_argument("--dry-run", action="store_true", help="show trades without placing them")
        sp.add_argument("--once", action="store_true", help="run one check instead of looping all day")
        sp.set_defaults(func=fn, **{**none, "cash": 5_000})

    gb = sub.add_parser("gap-backtest", help="test the gap-and-go signal on the last ~60 days")
    gb.add_argument("--system", default="options", choices=["options", "smallcap"])
    gb.add_argument("--symbols", nargs="+")
    gb.add_argument("--stock-stop", type=float, default=1.5, help="stop, %% move against the stock")
    gb.add_argument("--trades-csv")
    gb.set_defaults(func=cmd_gap_backtest, cash=None, risk=None, max_positions=None)

    ac = sub.add_parser("accounts", help="show both day-trading paper accounts")
    ac.set_defaults(func=cmd_accounts, **none)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = BotConfig()
    if args.symbols:
        cfg.symbols = [s.upper() for s in args.symbols]
    if args.cash:
        cfg.starting_cash = args.cash
    if args.risk:
        cfg.risk.risk_per_trade = args.risk
    if args.max_positions:
        cfg.risk.max_positions = args.max_positions
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
