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
