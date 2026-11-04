"""Premarket scanner: find stocks gapping before the open, and why.

Universe = Yahoo's movers/most-active lists + a watchlist of names with liquid options.
For each name we compare the latest premarket price with the prior regular-session
close, record the premarket high/low (the levels the day trader watches), and pull
recent headlines.

Note: Yahoo's free intraday data reports 0 volume for premarket bars, so premarket
volume is only available when the data source provides it (e.g. Alpaca).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger("swing_bot.premarket")
NY = ZoneInfo("America/New_York")
OPEN, CLOSE, PM_START = time(9, 30), time(16, 0), time(4, 0)

# Liquid, heavily-traded options names that frequently gap on news.
DEFAULT_WATCHLIST = [
    "TSLA", "NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "NFLX", "AVGO",
    "PLTR", "SMCI", "COIN", "HOOD", "MSTR", "MARA", "RIOT", "SOFI", "UBER", "SHOP",
    "INTC", "MU", "BA", "DIS", "NIO", "RIVN", "LCID", "SNAP", "PYPL", "ARM",
    "SPY", "QQQ", "IWM",
]


@dataclass
class ScanConfig:
    min_price: float = 3.0           # options on sub-$3 stocks are usually untradeable
    max_price: float = 1_000.0
    min_gap_pct: float = 3.0         # absolute % move vs prior close
    min_avg_volume: float = 1_000_000  # average daily shares (liquidity proxy)
    max_results: int = 10
    include_gap_downs: bool = True   # gap downs = put candidates
    screeners: tuple[str, ...] = ("day_gainers", "day_losers", "most_actives", "small_cap_gainers")
    watchlist: list[str] = field(default_factory=lambda: list(DEFAULT_WATCHLIST))


def to_ny(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC").tz_convert(NY) if idx.tz is None else idx.tz_convert(NY)
    return df


def session_slices(df: pd.DataFrame, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(premarket, regular) bars for `day`. df must be NY-indexed."""
    d = df[df.index.date == day]
    t = d.index.time
    return d[(t >= PM_START) & (t < OPEN)], d[(t >= OPEN) & (t < CLOSE)]


def prior_close(df: pd.DataFrame, day: date) -> float | None:
    prev = df[df.index.date < day]
    reg = prev[(prev.index.time >= OPEN) & (prev.index.time < CLOSE)]
    return float(reg["close"].iloc[-1]) if not reg.empty else None


def premarket_stats(df: pd.DataFrame, day: date | None = None) -> dict | None:
    """Gap statistics for one symbol's intraday (pre/post-inclusive) bars."""
    if df.empty:
        return None
    df = to_ny(df)
    day = day or df.index[-1].date()
    pm, reg = session_slices(df, day)
    pc = prior_close(df, day)
    if pm.empty or pc is None:
        return None
    last = float(pm["close"].iloc[-1])
    vol = float(pm["volume"].sum())
    return {
        "date": day,
        "prev_close": pc,
        "pm_last": last,
        "gap_pct": (last / pc - 1) * 100,
        "pm_high": float(pm["high"].max()),
        "pm_low": float(pm["low"].min()),
        "pm_volume": vol if vol > 0 else float("nan"),
        "open": float(reg["open"].iloc[0]) if not reg.empty else float("nan"),
    }


# ---------------------------------------------------------------- data access
def screener_quotes(names: tuple[str, ...], count: int = 50) -> dict[str, dict]:
    import yfinance as yf

    out: dict[str, dict] = {}
    for name in names:
        try:
            for q in yf.screen(name, count=count)["quotes"]:
                if q.get("quoteType") == "EQUITY":
                    out.setdefault(q["symbol"], q)
        except Exception as e:
            log.warning("screener %s failed: %s", name, e)
    return out


def _split_download(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0)
                d = raw[sym] if sym in lvl0 else raw.xs(sym, axis=1, level=1)
            else:
                d = raw
        except KeyError:
            continue
        d = d.rename(columns=str.lower).dropna(subset=["close"])
        if not d.empty:
            out[sym] = d[["open", "high", "low", "close", "volume"]]
    return out


def yahoo_intraday(symbols: list[str], period: str = "5d", interval: str = "5m") -> dict[str, pd.DataFrame]:
    import yfinance as yf

    if not symbols:
        return {}
    raw = yf.download(symbols, period=period, interval=interval, prepost=True,
                      progress=False, group_by="ticker", threads=True)
    return _split_download(raw, symbols)


def headlines(symbol: str, n: int = 3) -> list[str]:
    """Most recent news titles (best effort - Yahoo's news feed is flaky)."""
    try:
        import yfinance as yf

        items = yf.Ticker(symbol).news or []
    except Exception as e:
        log.info("news for %s unavailable: %s", symbol, e)
        return []
    titles = []
    for it in items[:n]:
        content = it.get("content", it)
        if content.get("title"):
            titles.append(content["title"])
    return titles


# ---------------------------------------------------------------- the scan
def scan(cfg: ScanConfig | None = None, intraday_loader=None, quote_loader=None,
         with_news: bool = True) -> pd.DataFrame:
    """Rank today's (or the latest session's) premarket gappers."""
    cfg = cfg or ScanConfig()
    quotes = (quote_loader or screener_quotes)(cfg.screeners)
    universe = sorted(set(quotes) | set(cfg.watchlist))
    bars = (intraday_loader or yahoo_intraday)(universe)

    rows = []
    for sym, df in bars.items():
        st = premarket_stats(df)
        if st is None:
            continue
        q = quotes.get(sym, {})
        avg_vol = q.get("averageDailyVolume10Day") or q.get("averageDailyVolume3Month")
        if avg_vol is None:  # not from a screener -> estimate from regular-session bars
            reg = to_ny(df)
            reg = reg[(reg.index.time >= OPEN) & (reg.index.time < CLOSE)]
            days = max(len(set(reg.index.date)), 1)
            avg_vol = float(reg["volume"].sum()) / days
        st.update(symbol=sym, avg_volume=float(avg_vol), name=q.get("shortName", ""))
        rows.append(st)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    price_ok = df["pm_last"].between(cfg.min_price, cfg.max_price)
    gap_ok = df["gap_pct"] >= cfg.min_gap_pct
    if cfg.include_gap_downs:
        gap_ok |= df["gap_pct"] <= -cfg.min_gap_pct
    df = df[price_ok & gap_ok & (df["avg_volume"] >= cfg.min_avg_volume)].copy()
    df["direction"] = df["gap_pct"].apply(lambda g: "call" if g > 0 else "put")
    df = df.reindex(df["gap_pct"].abs().sort_values(ascending=False).index).head(cfg.max_results)
    if with_news:
        df["news"] = [" | ".join(headlines(s)) for s in df["symbol"]]
    cols = ["symbol", "direction", "gap_pct", "pm_last", "prev_close", "pm_high", "pm_low",
            "pm_volume", "avg_volume", "date", "name"] + (["news"] if with_news else [])
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------- the two systems
# System 1: options on liquid names. Calls on gap-ups, puts on gap-downs.
OPTIONS_SCAN = ScanConfig(
    min_price=5.0, max_price=1_000.0, min_gap_pct=3.0, min_avg_volume=1_000_000,
    include_gap_downs=True, screeners=("day_gainers", "day_losers", "most_actives"),
)

# System 2: low-priced small-cap runners traded as shares. Long only - a cash
# account can't short - so gap-ups only, and bigger gaps required.
SMALLCAP_SCAN = ScanConfig(
    min_price=1.0, max_price=20.0, min_gap_pct=10.0, min_avg_volume=300_000,
    include_gap_downs=False,
    screeners=("small_cap_gainers", "aggressive_small_caps", "day_gainers", "most_actives"),
    watchlist=[],
)
