"""Plain-English results report for the paper options account."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def options_report(state_file: str | Path) -> str:
    path = Path(state_file)
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")
    if not path.exists():
        return f"# Paper trading results\n\n_Updated {now}_\n\nNo trades yet. The bot hasn't run.\n"
    st = json.loads(path.read_text())
    acct = st.get("account", {})
    settled = acct.get("settled", 0.0)
    unsettled = sum(a for _, a in acct.get("unsettled", []))
    open_cost = sum(p["entry"] * 100 * p["qty"] for p in st.get("positions", {}).values())
    value = settled + unsettled + open_cost
    start = st.get("starting_cash", value)
    sells = [e for e in st.get("log", []) if e.get("side") == "sell"]
    wins = [e for e in sells if e.get("pnl", 0) > 0]
    total_pnl = sum(e.get("pnl", 0) for e in sells)

    lines = [
        "# Paper trading results (fake money)",
        "",
        f"_Updated {now}_",
        "",
        "| | |",
        "|---|---|",
        f"| Started with | ${start:,.2f} |",
        f"| Account value now | **${value:,.2f}** ({(value / start - 1) * 100:+.1f}%) |" if start else "",
        f"| Cash ready to trade | ${settled:,.2f} |",
        f"| Cash settling (usable next business day) | ${unsettled:,.2f} |",
        f"| Money in open trades | ${open_cost:,.2f} |",
        f"| Trades closed | {len(sells)} |",
        f"| Winners | {len(wins)} ({len(wins) / len(sells):.0%}) |" if sells else "| Winners | - |",
        f"| Total profit / loss | ${total_pnl:+,.2f} |",
        "",
    ]
    if st.get("positions"):
        lines += ["## Open trades", "", "| Contract | Qty | Paid |", "|---|---|---|"]
        for k, p in st["positions"].items():
            lines.append(f"| {k} | {p['qty']} | ${p['entry']:.2f} |")
        lines.append("")
    log = st.get("log", [])
    if log:
        lines += ["## Trade history (newest first)", "", "| Time | Action | Contract | Qty | Price | P&L | Why |",
                  "|---|---|---|---|---|---|---|"]
        for e in reversed(log[-50:]):
            t = e.get("t", "")[:16].replace("T", " ")
            pnl = f"${e['pnl']:+,.2f}" if "pnl" in e else ""
            lines.append(f"| {t} | {e['side'].upper()} | {e.get('contract', e.get('symbol', ''))} | {e['qty']} | "
                         f"${e['price']:.2f} | {pnl} | {e.get('reason', 'signal')} |")
    return "\n".join(l for l in lines if l is not None) + "\n"
