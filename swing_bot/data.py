"""Market data loading: Yahoo Finance, CSV files, or synthetic data for tests/demos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) columns even for a single ticker
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing columns: {missing}")
    df = df[COLUMNS].dropna(subset=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def load_yahoo(symbol: str, start: str | None = None, end: str | None = None,
               period: str | None = None) -> pd.DataFrame:
    """Daily, split/dividend-adjusted OHLCV from Yahoo Finance."""
    import yfinance as yf

    kwargs = {"interval": "1d", "auto_adjust": True, "progress": False}
    if start or end:
        kwargs.update(start=start, end=end)
    else:
        kwargs["period"] = period or "5y"
    df = yf.download(symbol, **kwargs)
    if df is None or df.empty:
        raise ValueError(f"no data returned for {symbol}")
    return _normalize(df)


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV with a date column (or index) and open/high/low/close/volume columns."""
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        raise ValueError(f"{path}: no date column found")
    df = df.set_index(date_col)
    return _normalize(df)


def synthetic(n_days: int = 750, seed: int = 0, drift: float = 0.0004,
              vol: float = 0.015, start_price: float = 100.0) -> pd.DataFrame:
    """Random-walk OHLCV data, useful for offline tests and demos."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n_days)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(rng.normal(0, vol / 4, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days)
    idx = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def load_many(symbols: list[str], source: str = "yahoo", **kwargs) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        if source == "yahoo":
            out[sym] = load_yahoo(sym, **kwargs)
        elif source == "synthetic":
            out[sym] = synthetic(seed=i, **kwargs)
        elif source == "csv":
            out[sym] = load_csv(Path(kwargs.get("csv_dir", "data")) / f"{sym}.csv")
        else:
            raise ValueError(f"unknown data source: {source}")
    return out
