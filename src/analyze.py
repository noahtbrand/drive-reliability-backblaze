"""AFR tables, failure Pareto, Weibull fits and survival curves -> CSVs in outputs/ (for Tableau).

Usage (from the project root):   python src/analyze.py
"""
import numpy as np
import pandas as pd
import common as c

MIN_FAILURES_FOR_FIT = 30      # skip Weibull fits for models with fewer failures than this

c.OUT.mkdir(exist_ok=True)
frames = c.load_quarters()
print(f"Loaded {len(frames)} quarters: {[f.quarter.iloc[0] for f in frames]}")
d = c.clean(c.build_drives(frames))
print(f"\nFleet: {len(d):,} drives, {int(d.failed.sum()):,} failures, "
      f"{d.span_days.sum() / 365:,.0f} drive-years, {d.model.nunique()} models\n")

# ---------------------------------------------------------------- AFR (by model and manufacturer)
afr_model = c.afr_table(d, "model")
afr_model["manufacturer"] = afr_model.model.map(c.manufacturer)
afr_mfr = c.afr_table(d, "manufacturer")
fleet_afr = 100 * d.failed.sum() / (d.span_days.sum() / 365)
afr_model.to_csv(c.OUT / "afr_by_model.csv", index=False)
afr_mfr.to_csv(c.OUT / "afr_by_manufacturer.csv", index=False)
print("AFR by manufacturer (95% exact Poisson CI):")
print(afr_mfr.round(2).to_string(index=False))
print(f"\nFleet AFR: {fleet_afr:.2f}%   <- typical published fleet AFRs are roughly 1-3%; "
      f"a value far above ~10% usually means a data problem\n")

# ---------------------------------------------------------------- Pareto of failures
par = afr_model[["model", "manufacturer", "drives", "failures"]].copy()
par["pct_of_failures"] = 100 * par.failures / par.failures.sum()
par["cum_pct"] = par.pct_of_failures.cumsum()
par.to_csv(c.OUT / "failure_pareto.csv", index=False)
print("Top 5 models by failures (Pareto):")
print(par.head(5).round(1).to_string(index=False), "\n")

# ---------------------------------------------------------------- Weibull per model and per manufacturer
rows = []
for scope, col in (("model", "model"), ("manufacturer", "manufacturer")):
    for name, g in d.groupby(col):
        if g.failed.sum() < MIN_FAILURES_FOR_FIT:
            continue
        r = c.weibull_fit_ci(g.entry_days, g.exit_days, g.failed)
        r.update(scope=scope, name=name, drives=len(g), failures=int(g.failed.sum()),
                 max_age_days=g.exit_days.max())
        rows.append(r)
wb = pd.DataFrame(rows)
if len(wb):
    wb["verdict_vs_beta1"] = np.where(wb.shape_lo > 1, "wear-out (CI above 1)",
                              np.where(wb.shape_hi < 1, "early-life (CI below 1)", "inconclusive (CI includes 1)"))
    wb["scale_years"] = wb.scale_days / 365
    wb.to_csv(c.OUT / "weibull_fits.csv", index=False)
    show = wb[["scope", "name", "drives", "failures", "shape", "shape_lo", "shape_hi", "scale_years", "verdict_vs_beta1"]]
    print("Weibull fits (shape = beta; entry-adjusted for left truncation):")
    print(show.round(2).to_string(index=False), "\n")

# cross-check my fit against lifelines on the largest model (should agree closely)
try:
    from lifelines import WeibullFitter
    top = d.groupby("model").failed.sum().idxmax()
    g = d[d.model == top]
    wf = WeibullFitter().fit(g.exit_days, g.failed, entry=g.entry_days)
    (k, lam), _ = c.weibull_fit(g.entry_days, g.exit_days, g.failed)
    print(f"Cross-check on {top}: lifelines rho={wf.rho_:.3f} lambda={wf.lambda_:.0f} | "
          f"this script k={k:.3f} lambda={lam:.0f}  (should match within ~1%)\n")
except Exception as e:                                     # never let the cross-check kill the run
    print("lifelines cross-check skipped:", e)

# ---------------------------------------------------------------- survival curves by manufacturer
curves = []
for m, g in d.groupby("manufacturer"):
    if g.failed.sum() < MIN_FAILURES_FOR_FIT:
        continue
    k = c.km_delayed(g.entry_days, g.exit_days, g.failed, min_at_risk=100)
    k["manufacturer"] = m
    curves.append(k)
if curves:
    km = pd.concat(curves)
    km["age_years"] = km.age_days / 365
    km.to_csv(c.OUT / "survival_curves.csv", index=False)
    print(f"Wrote survival_curves.csv ({len(km)} points). Curves are conditional on reaching the first age "
          f"with >=100 drives under observation, so they start at 1.0 at different ages.")
print("\nWrote:", ", ".join(sorted(p.name for p in c.OUT.glob('*.csv'))))
