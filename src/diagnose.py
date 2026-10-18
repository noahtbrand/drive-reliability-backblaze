"""Sanity checks to run after analyze.py and forecast.py.

  1. Failures and AFR per quarter  (compare with Backblaze's published quarterly numbers)
  2. AFR by drive age, next to the AFR the fitted Weibull implies  (does the fitted shape match the data?)
  3. Model-level backtest results   (which models drive the forecast error)

Usage (from the project root):   python src/diagnose.py
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
import common as c

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
c.OUT.mkdir(exist_ok=True)

frames = c.load_quarters()
d = c.clean(c.build_drives(frames), verbose=False)
keep = set(d.model.unique())

# ------------------------------------------------------------------ 1. per-quarter failures / AFR
print("=" * 100)
print("1) FAILURES AND AFR PER QUARTER (analysed models only)")
print("   Backblaze's published quarterly AFR: Q1 2026 = 1.24% (1,030 failures), Q2 2026 = 1.73% (1,498 failures).")
print("   Expect the same ballpark, not an exact match: they also drop boot drives and small models.\n")
rows = []
for f in frames:
    a = f[f.model.isin(keep)]
    days = (a.exit_days - a.entry_days).clip(lower=0).sum()
    fails = int(a.failed.sum())
    rows.append(dict(quarter=f.quarter.iloc[0], drives=len(a), failures=fails,
                     drive_days=round(days), afr_pct=round(100 * fails / (days / 365), 2),
                     failures_all_models=int(f.failed.sum())))
q = pd.DataFrame(rows)
print(q.to_string(index=False))
q.to_csv(c.OUT / "quarterly_failures.csv", index=False)


# ------------------------------------------------------------------ 2. AFR by age band vs Weibull-implied
EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15]          # drive age in years
MIN_EXPOSURE_YRS = 50                                 # hide cells with too little data to trust


def afr_by_age(g, edges=EDGES):
    e, x, f = g.entry_days.values, g.exit_days.values, g.failed.values.astype(bool)
    (k, lam), _ = c.weibull_fit(e, x, f)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = a * 365.0, b * 365.0
        expo = np.clip(np.minimum(x, hi) - np.maximum(e, lo), 0, None)
        fails = int(((x >= lo) & (x < hi) & f).sum())
        yrs = expo.sum() / 365.0
        if yrs <= 0:
            continue
        implied = ((hi / lam) ** k - (lo / lam) ** k) / (b - a)      # average hazard per year over the band
        out.append(dict(age_band=f"{a}-{b}y", age_mid_years=(a + b) / 2, drives=int((expo > 0).sum()),
                        exposure_years=round(yrs, 1), failures=fails,
                        afr_pct=100 * fails / yrs,
                        afr_lo_pct=100 * (chi2.ppf(0.025, 2 * fails) / 2 if fails else 0) / yrs,
                        afr_hi_pct=100 * chi2.ppf(0.975, 2 * fails + 2) / 2 / yrs,
                        weibull_implied_afr_pct=100 * implied,
                        weibull_shape=k))
    return pd.DataFrame(out)


print("\n" + "=" * 100)
print("2) AFR BY DRIVE AGE vs the AFR the fitted Weibull implies (top models by failures)")
print("   If the two columns track each other, the fitted shape is describing the data. If AFR is flat or")
print("   jumps around while the Weibull curve climbs, don't claim wear-out for that model.\n")
tops = d.groupby("model").failed.sum().sort_values(ascending=False).head(8).index
allrows = []
for m in tops:
    t = afr_by_age(d[d.model == m])
    t.insert(0, "model", m)
    allrows.append(t)
    shown = t[t.exposure_years >= MIN_EXPOSURE_YRS]
    print(f"--- {m}  (Weibull shape {t.weibull_shape.iloc[0]:.2f})")
    print(shown[["age_band", "drives", "exposure_years", "failures", "afr_pct", "afr_lo_pct", "afr_hi_pct",
                 "weibull_implied_afr_pct"]].round(2).to_string(index=False), "\n")
fleet = afr_by_age(d)
fleet.insert(0, "model", "ALL MODELS (pooled)")
allrows.append(fleet)
pd.concat(allrows).round(3).to_csv(c.OUT / "afr_by_age.csv", index=False)
print("Wrote outputs/afr_by_age.csv  (use it for a 'bathtub curve' view in Tableau)\n")

# ------------------------------------------------------------------ 3. model-level backtest
print("=" * 100)
print("3) BACKTEST BY MODEL (from outputs/backtest.csv)")
try:
    bt = pd.read_csv(c.OUT / "backtest.csv")
    m = bt[~bt.scope.str.startswith("FLEET")]
    cum = m.groupby("scope").agg(actual=("actual", "sum"), forecast_p50=("p50", "sum"),
                                 quarters_in_band=("in_band", "sum"), n=("in_band", "count")).reset_index()
    cum["forecast_over_actual"] = (cum.forecast_p50 / cum.actual.replace(0, np.nan)).round(2)
    print("Cumulative over the held-out quarters (ratio > 1 means the forecast was too high):")
    print(cum.sort_values("actual", ascending=False).round(1).to_string(index=False))
    tot_a, tot_f = cum.actual.sum(), cum.forecast_p50.sum()
    print(f"\nAll modelled drives: actual {tot_a:.0f} vs forecast {tot_f:.0f}  ({100 * (tot_f - tot_a) / tot_a:+.1f}%)")
    print("\nPer quarter:")
    print(m.sort_values(["quarter_ahead", "actual"], ascending=[True, False])
           [["scope", "quarter_ahead", "actual", "p5", "p50", "p95", "in_band"]].round(0).to_string(index=False))
except FileNotFoundError:
    print("outputs/backtest.csv not found - run python src/forecast.py first.")
