"""Premarket-runner strategies tested on EVERY gap/runner day of the last ~60 days (full US market)."""
import pickle, sys
from datetime import time
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from swing_bot.premarket import to_ny, prior_close
S = sys.argv[1] if len(sys.argv) > 1 else "research_data"  # runners.pkl + runners_5m.pkl
bars = pickle.load(open(f"{S}/runners_5m.pkl", "rb"))
runners = pickle.load(open(f"{S}/runners.pkl", "rb"))

def cost(price, premarket):            # one-way spread + slippage
    if price < 1: return 0.015 if premarket else 0.010
    if price < 5: return 0.0075 if premarket else 0.004
    return 0.004 if premarket else 0.002

def tmin(t): return t.hour * 60 + t.minute

days = []
for sym, df in bars.items():
    df = to_ny(df)
    for d in runners.get(sym, []):
        x = df[df.index.date == d]
        if x.empty: continue
        pc = prior_close(df, d)
        if pc is None or not (0.1 <= pc <= 20): continue
        tm = np.array([tmin(t) for t in x.index.time])
        x = x[(tm >= 240) & (tm < 16 * 60)]; tm = tm[(tm >= 240) & (tm < 16 * 60)]
        if (tm < 570).sum() < 6 or (tm >= 570).sum() < 20: continue
        days.append(dict(sym=sym, day=d, pc=pc, tm=tm, o=x.open.values, h=x.high.values,
                         l=x.low.values, c=x.close.values, v=x.volume.values.astype(float)))
print(f"{len(days)} runner days with premarket + regular data")

def entry_index(D, kind, U=0.2, B=60):
    tm, c, h, pc = D["tm"], D["c"], D["h"], D["pc"]
    pm = tm < 570
    if kind in ("chase", "base", "early"):
        last_hi_t, run_hi = tm[0], h[0]
        for i in range(1, pm.sum() - 1):
            if kind == "early" and tm[i] > 4 * 60 + 45: return None
            if tm[i] > 9 * 60 + 15: return None
            up = c[i] / pc - 1 >= U
            if kind == "early" and up: return i + 1
            if c[i] > run_hi and up:
                if kind == "chase" or (tm[i] - last_hi_t >= B and tm[i] >= 5 * 60): return i + 1
            if h[i] > run_hi: run_hi, last_hi_t = h[i], tm[i]
        return None
    # after-the-open entries
    reg = np.where(~pm)[0]; r0 = reg[0]
    if c[r0 - 1] / pc - 1 < U: return None          # must still be up U% into the open
    pm_hi = h[:r0].max(); or_hi = h[r0:r0 + 3].max()
    tp = (h + D["l"] + c) / 3; cv = np.cumsum(D["v"][r0:]); vw = np.cumsum(tp[r0:] * D["v"][r0:]) / np.maximum(cv, 1)
    for k, i in enumerate(reg[:-1]):
        if tm[i] >= 11 * 60: return None
        if kind == "orb" and k >= 3 and c[i] > or_hi and c[i] > vw[k]: return i + 1
        if kind == "pmbreak" and c[i] > pm_hi and c[i] > vw[k]: return i + 1
    return None

def exit_ret(D, e, stop_pct, mode):
    tm, o, h, l, c = D["tm"], D["o"], D["h"], D["l"], D["c"]
    px = o[e]; pre_in = tm[e] < 570
    entry = px * (1 + cost(px, pre_in)); stop = px * (1 - stop_pct); best = px
    end = {"preopen": 9 * 60 + 25, "open15": 9 * 60 + 45, "noon": 12 * 60, "trail": 11 * 60}[mode]
    if tm[e] >= end and mode in ("preopen", "open15"): return None
    for j in range(e, len(tm)):
        if l[j] <= stop: ex = min(stop, o[j]); break
        if tm[j] >= end: ex = o[j]; break
        if mode == "trail":
            best = max(best, c[j]); stop = max(stop, best * (1 - stop_pct))
    else: ex = c[-1]
    return (ex * (1 - cost(ex, tm[j] < 570)) / entry - 1) * 100

ENTRIES = [("chase", U, 0) for U in (0.1, 0.2, 0.4)] + \
          [("base", U, B) for U in (0.1, 0.2, 0.4) for B in (45, 90)] + \
          [("early", U, 0) for U in (0.1, 0.2, 0.4)] + [("orb", 0.1, 0), ("pmbreak", 0.1, 0)]
EXITS = [(s, m) for s in (0.05, 0.08, 0.12) for m in ("preopen", "open15", "noon", "trail")]
alld = sorted({D["day"] for D in days}); split = alld[len(alld) // 2]
rows = []
for kind, U, B in ENTRIES:
    ents = [(D, entry_index(D, kind, U, B)) for D in days]
    ents = [(D, e) for D, e in ents if e is not None]
    for sp, mode in EXITS:
        res = {"train": [], "test": []}; bucket = {"<$1": [], "$1-5": [], "$5-20": []}
        for D, e in ents:
            r = exit_ret(D, e, sp, mode)
            if r is None: continue
            res["train" if D["day"] < split else "test"].append(r)
            p = D["o"][e]; bucket["<$1" if p < 1 else "$1-5" if p < 5 else "$5-20"].append(r)
        row = dict(entry=kind, up=U, base=B, stop=sp, exit=mode)
        for k, v in res.items():
            v = np.array(v); row[k + "_n"] = len(v); row[k + "_avg"] = v.mean() if len(v) else np.nan
            row[k + "_med"] = np.median(v) if len(v) else np.nan; row[k + "_win"] = (v > 0).mean() if len(v) else np.nan
        for k, v in bucket.items(): row[k] = np.mean(v) if v else np.nan
        rows.append(row)
df = pd.DataFrame(rows)
pickle.dump(df, open(f"{S}/runners_results.pkl", "wb"))
pd.set_option("display.width", 250)
print(f"train < {split} <= test   (returns are % per trade AFTER spread/slippage)")
print("\nTop 15 by TRAIN average (min 30 train trades):")
print(df[df.train_n >= 30].sort_values("train_avg", ascending=False).head(15).round(2).to_string(index=False))
print("\nBest exit per entry type, judged on TEST:")
print(df[df.test_n >= 30].sort_values("test_avg", ascending=False).groupby("entry").head(1).round(2).to_string(index=False))
print(f"\nAll {len(df)} combos on test: mean {df.test_avg.mean():+.2f}%, positive {(df.test_avg > 0).sum()}/{df.test_avg.notna().sum()}")
