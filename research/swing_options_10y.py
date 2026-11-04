"""Daily swing setups for options, 2014-2026. Train 2014-2020, test 2021+.
Option P&L estimated with Black-Scholes: ~30-day ATM option, IV = 1.15 x 20-day realized vol,
buy at mid +3% (spread), sell at mid -3%, time decay charged for each day held."""
import math, pickle, sys
import numpy as np, pandas as pd
S = sys.argv[1] if len(sys.argv) > 1 else "research_data"  # folder with daily10y.pkl
data = pickle.load(open(f"{S}/daily10y.pkl", "rb"))

N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
def bs(S0, K, T, vol, call):
    if T <= 0: return max(S0 - K, 0) if call else max(K - S0, 0)
    d1 = (math.log(S0 / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T)); d2 = d1 - vol * math.sqrt(T)
    return S0 * N(d1) - K * N(d2) if call else K * N(-d2) - S0 * N(-d1)
def option_ret(p0, p1, days, vol, call, dte=30):
    vol = min(max(vol, 0.15), 2.5); K = p0
    a = bs(p0, K, dte / 365, vol, call) * 1.03
    b = bs(p1, K, max(dte - days * 1.4, 0) / 365, vol, call) * 0.97   # calendar days ~ 1.4 x trading days
    return (b / a - 1) * 100 if a > 0 else np.nan

def rsi(c, n):
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)

def setups(x):
    c, o, h, l, v = x.close, x.open, x.high, x.low, x.volume
    sma200, sma50, sma5 = c.rolling(200).mean(), c.rolling(50).mean(), c.rolling(5).mean()
    r2 = rsi(c, 2); vavg = v.rolling(20).mean().shift(1)
    gap = o / c.shift(1) - 1; rng = (c - l) / (h - l).replace(0, np.nan)
    hi20 = h.rolling(20).max().shift(1); lo20 = l.rolling(20).min().shift(1)
    return {
        "rsi2_dip_call":       (c > sma200) & (r2 < 10),
        "rsi2_pop_put":        (c < sma200) & (r2 > 90),
        "gap_go_call":         (gap >= 0.04) & (rng >= 0.75) & (v > 2 * vavg),
        "gap_down_put":        (gap <= -0.04) & (rng <= 0.25) & (v > 2 * vavg),
        "big_gap_call":        (gap >= 0.08) & (c > o) & (v > 3 * vavg),
        "breakout20_call":     (c > hi20) & (c > sma200) & (v > 1.5 * vavg),
        "breakdown20_put":     (c < lo20) & (c < sma200) & (v > 1.5 * vavg),
        "trend_pullback_call": (c > sma50) & (sma50 > sma200) & (c < sma5) & (r2 < 25) & (c.shift(1) < c.shift(2)),
    }, sma5

HOLDS = {"rsi2_dip_call": ["sma5", 3, 5], "rsi2_pop_put": ["sma5", 3, 5], "trend_pullback_call": ["sma5", 3, 5]}
rows = []
for sym, x in data.items():
    x = x[x.index >= "2013-06-01"]
    if len(x) < 260: continue
    sig, sma5 = setups(x)
    c = x.close.values; vol20 = (np.log(x.close).diff().rolling(20).std() * math.sqrt(252) * 1.15).values
    idx = x.index; s5 = sma5.values
    for name, cond in sig.items():
        call = name.endswith("call")
        hits = np.where(cond.fillna(False).values)[0]
        last_exit = -1
        for hold in HOLDS.get(name, [1, 3, 5, 10]):
            last_exit = -1
            for i in hits:
                if i < 200 or i + 1 >= len(c) or i <= last_exit or idx[i].year < 2014: continue
                if hold == "sma5":
                    j = i + 1
                    while j < len(c) - 1 and j - i < 7 and not ((c[j] > s5[j]) if call else (c[j] < s5[j])): j += 1
                else:
                    j = min(i + hold, len(c) - 1)
                last_exit = j
                mv = (c[j] / c[i] - 1) * (1 if call else -1) * 100
                rows.append((name, str(hold), sym, idx[i], mv, option_ret(c[i], c[j], j - i, vol20[i], call), j - i))
df = pd.DataFrame(rows, columns=["setup", "hold", "sym", "date", "stock", "option", "days"])
df["period"] = np.where(df.date < "2021-01-01", "train", "test")
pickle.dump(df, open(f"{S}/swing_trades.pkl", "wb"))
g = df.groupby(["setup", "hold", "period"]).agg(n=("option", "size"), stock=("stock", "mean"),
        opt=("option", "mean"), opt_med=("option", "median"), win=("option", lambda s: (s > 0).mean())).unstack("period")
g.columns = [f"{a}_{b}" for a, b in g.columns]
g = g[["n_train", "stock_train", "opt_train", "win_train", "n_test", "stock_test", "opt_test", "opt_med_test", "win_test"]]
pd.set_option("display.width", 220)
print("Per-trade averages. stock = % move in our favor; opt = estimated option % return after spread + decay\n")
print(g.sort_values("opt_train", ascending=False).round(2).to_string())
