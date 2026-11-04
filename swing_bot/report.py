"""Plain-English results report for the paper (fake money) accounts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")


def account_section(title: str, state_file: str | Path, multiplier: int) -> str:
    path = Path(state_file)
    if not path.exists():
        return f"## {title}\n\nNo trades yet. The bot hasn't run.\n"
    st = json.loads(path.read_text())
    acct = st.get("account", {})
    settled = acct.get("settled", 0.0)
    unsettled = sum(a for _, a in acct.get("unsettled", []))
    positions = st.get("positions", {})
    open_cost = sum(p["entry"] * multiplier * p["qty"] for p in positions.values())
    value = settled + unsettled + open_cost
    start = st.get("starting_cash") or value or 1.0
    sells = [e for e in st.get("log", []) if e.get("side") == "sell"]
    wins = [e for e in sells if e.get("pnl", 0) > 0]
    total_pnl = sum(e.get("pnl", 0) for e in sells)

    lines = [
        f"## {title}", "",
        "| | |", "|---|---|",
        f"| Started with | ${start:,.2f} |",
        f"| Account value now | **${value:,.2f}** ({(value / start - 1) * 100:+.1f}%) |",
        f"| Cash ready to trade | ${settled:,.2f} |",
        f"| Cash settling (usable next business day) | ${unsettled:,.2f} |",
        f"| Money in open trades | ${open_cost:,.2f} |",
        f"| Trades closed | {len(sells)} |",
        f"| Winners | {len(wins)} ({len(wins) / len(sells):.0%}) |" if sells else "| Winners | - |",
        f"| Total profit / loss | ${total_pnl:+,.2f} |", "",
    ]
    if positions:
        lines += ["**Open trades**", "", "| What | Qty | Paid |", "|---|---|---|"]
        lines += [f"| {k} | {p['qty']} | ${p['entry']:.2f} |" for k, p in positions.items()]
        lines.append("")
    log = st.get("log", [])
    if log:
        lines += ["**Trade history (newest first)**", "",
                  "| Time | Action | What | Qty | Price | P&L | Why |", "|---|---|---|---|---|---|---|"]
        for e in reversed(log[-50:]):
            t = e.get("t", "")[:16].replace("T", " ")
            pnl = f"${e['pnl']:+,.2f}" if "pnl" in e else ""
            lines.append(f"| {t} | {e['side'].upper()} | {e.get('contract', e.get('symbol', ''))} | {e['qty']} | "
                         f"${e['price']:.2f} | {pnl} | {e.get('reason', 'signal')} |")
        lines.append("")
    return "\n".join(lines)


def options_report(state_file: str | Path) -> str:
    return f"# Paper trading results (fake money)\n\n_Updated {_now()}_\n\n" + \
        account_section("Options bot", state_file, 100).replace("## Options bot\n\n", "")


def full_report(options_state: str | Path, stocks_state: str | Path | None = None,
                premarket_state: str | Path | None = None) -> str:
    parts = [f"# Paper trading results (fake money)\n\n_Updated {_now()}_\n",
             account_section("Options bot (System 1)", options_state, 100)]
    if stocks_state:
        parts.append(account_section("Small-cap stock bot (System 2)", stocks_state, 1))
    if premarket_state:
        parts.append(account_section("Premarket runner bot (System 3)", premarket_state, 1))
    return "\n".join(parts)
