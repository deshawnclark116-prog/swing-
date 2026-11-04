"""Gap-and-go variant research with a chronological train/test split."""
import itertools, pickle, sys
from datetime import time
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from swing_bot.premarket import to_ny, session_slices, prior_close

OPEN_T, EXIT_T = time(9, 30), time(15, 45)

def vwap_arr(reg):
    tp = (reg.high + reg.low + reg.close).values / 3; v = reg.volume.values.astype(float)
    cv = np.cumsum(v); return np.where(cv > 0, np.cumsum(tp * v) / np.maximum(cv, 1), reg.close.values)

def prep(bars, min_gap=3.0, min_price=1, max_price=1e9, gap_up_only=False):
    spy = to_ny(bars["SPY"]); spy_days = {}
    for d in sorted(set(spy.index.date)):
        _, r = session_slices(spy, d)
        if len(r): spy_days[d] = pd.Series(r.close.values > vwap_arr(r), index=r.index)
    setups = []
    for sym, df in bars.items():
        if sym in ("SPY", "QQQ"): continue
        df = to_ny(df); days = sorted(set(df.index.date)); or_vols = {}
        for d in days:
            pm, reg = session_slices(df, d)
            if len(reg) < 20: continue
            or_vols[d] = reg.volume.iloc[:3].sum()
            pc = prior_close(df, d)
            if pc is None or pm.empty: continue
            last = pm.close.iloc[-1]; gap = (last / pc - 1) * 100
            if abs(gap) < min_gap or not (min_price <= last <= max_price): continue
            if gap_up_only and gap < 0: continue
            prev = [or_vols[x] for x in list(or_vols)[:-1][-10:]]
            rvol = or_vols[d] / np.mean(prev) if len(prev) >= 5 and np.mean(prev) > 0 else np.nan
            setups.append(dict(sym=sym, day=d, gap=gap, kind="call" if gap > 0 else "put",
                               pm_high=pm.high.max(), pm_low=pm.low.min(), reg=reg, vw=vwap_arr(reg),
                               rvol=rvol, spy=spy_days.get(d)))
    return setups

def simulate(s, entry, exit_, spy_align, stop_pct):
    reg, vw = s["reg"], s["vw"]; sign = 1 if s["kind"] == "call" else -1
    c, o, h, l = reg.close.values, reg.open.values, reg.high.values, reg.low.values
    times = reg.index.time; or_hi, or_lo = h[:3].max(), l[:3].min()
    start = 3 if entry == "orb15" else 1
    lvl_hi, lvl_lo = (or_hi, or_lo) if entry == "orb15" else (s["pm_high"], s["pm_low"])
    for i in range(start, len(reg) - 1):
        if times[i] >= time(11, 0): return None
        brk = c[i] > lvl_hi and c[i] > vw[i] if sign > 0 else c[i] < lvl_lo and c[i] < vw[i]
        if not brk: continue
        if spy_align and s["spy"] is not None:
            sp = s["spy"]; st = sp[sp.index <= reg.index[i]]
            if len(st) and (bool(st.iloc[-1]) != (sign > 0)): return None
        break
    else: return None
    e = i + 1; px = o[e]; stop = px * (1 - sign * stop_pct / 100); best = px
    for j in range(e, len(reg)):
        if (sign > 0 and l[j] <= stop) or (sign < 0 and h[j] >= stop):
            ex = stop; break
        if times[j] >= (time(12, 0) if exit_ == "noon" else EXIT_T) or j == len(reg) - 1:
            ex = c[j]; break
        if exit_ == "vwap" and (c[j] - vw[j]) * sign < 0 and j + 1 < len(reg):
            ex = o[j + 1]; break
        if exit_ == "trail":
            best = max(best, c[j]) if sign > 0 else min(best, c[j])
            stop = max(stop, best * (1 - stop_pct / 100)) if sign > 0 else min(stop, best * (1 + stop_pct / 100))
    return sign * (ex / px - 1) * 100

GRID = dict(entry=["pmbreak", "orb15"], gap=[3, 5], rvol=[0, 1.5, 2.5], spy=[False, True],
            exit=["vwap", "hold", "trail", "noon"])

def evaluate(setups, split_day, stop_pct, side=None):
    rows = []
    for entry, gap, rvol, spy, exit_ in itertools.product(*GRID.values()):
        res = {"train": [], "test": []}
        for s in setups:
            if abs(s["gap"]) < gap or (rvol and not (s["rvol"] >= rvol)): continue
            if side and s["kind"] != side: continue
            m = simulate(s, entry, exit_, spy, stop_pct)
            if m is not None: res["train" if s["day"] < split_day else "test"].append(m)
        row = dict(entry=entry, gap=gap, rvol=rvol, spy=spy, exit=exit_)
        for k, v in res.items():
            v = np.array(v); row[f"{k}_n"] = len(v)
            row[f"{k}_avg"] = v.mean() if len(v) else np.nan
            row[f"{k}_win"] = (v > 0).mean() if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    S = sys.argv[2] if len(sys.argv) > 2 else "research_data"
    which = sys.argv[1]
    bars = pickle.load(open(f"{S}/{which}.pkl", "rb"))
    if which == "options":
        setups = prep(bars, 3.0, 5, 1e9); stop = 1.5; sides = [None, "call", "put"]
    else:
        bars["SPY"] = pickle.load(open(f"{S}/options.pkl", "rb"))["SPY"]
        setups = prep(bars, 10.0, 1, 20, gap_up_only=True); stop = 4.0; sides = [None]
    days = sorted({s["day"] for s in setups}); split = days[len(days) // 2]
    print(f"{which}: {len(setups)} gap setups over {len(days)} days; train < {split} <= test")
    for side in sides:
        r = evaluate(setups, split, stop, side)
        pickle.dump(r, open(f"{S}/res_{which}_{side}.pkl", "wb"))
        ok = r[r.train_n >= (30 if which == "options" else 8)].sort_values("train_avg", ascending=False)
        print(f"\n--- side={side or 'both'}: top 8 by TRAIN avg move %, with their TEST results ---")
        print(ok.head(8).round(2).to_string(index=False))
        print(f"test avg across ALL {len(r)} variants: {r.test_avg.mean():+.2f}%  "
              f"(variants positive on test: {(r.test_avg > 0).sum()}/{r.test_avg.notna().sum()})")
