"""Premarket momentum research for shares (train/test split by date)."""
import itertools, pickle, sys
from datetime import time
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from swing_bot.premarket import to_ny, prior_close
S = sys.argv[1] if len(sys.argv) > 1 else "research_data"  # folder with options.pkl / smallcap.pkl
bars = {**pickle.load(open(f"{S}/options.pkl", "rb")), **pickle.load(open(f"{S}/smallcap.pkl", "rb"))}
SLIP = 0.005  # 0.5% each way: premarket spreads are wide

days_data = []  # (sym, day, prev_close, pm bars, reg bars)
for sym, df in bars.items():
    if sym in ("SPY", "QQQ"): continue
    df = to_ny(df)
    for d in sorted(set(df.index.date)):
        x = df[df.index.date == d]
        pm = x[(x.index.time >= time(4)) & (x.index.time < time(9, 30))]
        reg = x[(x.index.time >= time(9, 30)) & (x.index.time < time(16))]
        pc = prior_close(df, d)
        if pc is None or len(pm) < 6 or len(reg) < 10: continue
        if not (1 <= pc <= 20): continue
        if pm.high.max() / pc - 1 < 0.05: continue   # never moved 5%+ premarket: skip early
        days_data.append((sym, d, pc, pm, reg))
print(len(days_data), "symbol-days with a 5%+ premarket move, $1-$20")

def sim(sym, d, pc, pm, reg, min_up, start, stop_pct, exit_mode):
    t = pm.index.time; c, o, h, l = pm.close.values, pm.open.values, pm.high.values, pm.low.values
    for i in range(1, len(pm) - 1):
        if t[i] < start or t[i] > time(9, 15): continue
        if c[i] / pc - 1 >= min_up and c[i] > h[:i].max():   # up X% and closing at a new PM high
            break
    else: return None
    px = o[i + 1] * (1 + SLIP); stop = px * (1 - stop_pct); best = px
    allb = pd.concat([pm.iloc[i + 1:], reg])
    for j in range(len(allb)):
        b = allb.iloc[j]; tt = allb.index[j].time()
        if b.low <= stop: ex = min(stop, b.open); break
        if exit_mode == "preopen" and tt >= time(9, 25): ex = b.close; break
        if exit_mode == "open15" and tt >= time(9, 45): ex = b.open; break
        if exit_mode == "noon" and tt >= time(12): ex = b.open; break
        if exit_mode.startswith("trail"):
            best = max(best, b.close); stop = max(stop, best * (1 - stop_pct))
            if tt >= time(12): ex = b.open; break
    else: ex = allb.close.iloc[-1]
    return (ex * (1 - SLIP) / px - 1) * 100

alldays = sorted({x[1] for x in days_data}); split = alldays[len(alldays) // 2]
rows = []
for min_up, start, stop_pct, exit_mode in itertools.product(
        [0.10, 0.20, 0.30], [time(4), time(7), time(8)], [0.04, 0.07, 0.10], ["preopen", "open15", "noon", "trail"]):
    res = {"train": [], "test": []}
    for x in days_data:
        m = sim(*x, min_up, start, stop_pct, exit_mode)
        if m is not None: res["train" if x[1] < split else "test"].append(m)
    r = dict(up=min_up, start=start.strftime("%H:%M"), stop=stop_pct, exit=exit_mode)
    for k, v in res.items():
        v = np.array(v); r[k + "_n"] = len(v); r[k + "_avg"] = v.mean() if len(v) else np.nan
        r[k + "_win"] = (v > 0).mean() if len(v) else np.nan
    rows.append(r)
df = pd.DataFrame(rows)
print(f"train < {split} <= test.  Each trade pays ~1% round-trip in premarket spread/slippage.")
print(df[df.train_n >= 15].sort_values("train_avg", ascending=False).head(10).round(2).to_string(index=False))
print(f"\nALL {len(df)} variants on test: mean {df.test_avg.mean():+.2f}%, positive {(df.test_avg > 0).sum()}/{df.test_avg.notna().sum()}")
print(f"same, IGNORING spread costs: mean {df.test_avg.mean() + 1.0:+.2f}%")
