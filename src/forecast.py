"""Monte Carlo failure forecast with bootstrap parameter uncertainty, plus a backtest.

Usage (from the project root):
    python src/forecast.py                    # backtest on the last 2 quarters, then forecast 8 quarters ahead
    python src/forecast.py --top 10 --boot 100 --sims 3000     # faster test run

Method: for each model, refit the Weibull on bootstrap resamples of its drives (this carries the
uncertainty in beta/lambda and their correlation). Each simulation uses one bootstrap fit, and for every
drive still running it draws a failure age from the Weibull conditional on having survived to its current
age t:  T = lam * ((t/lam)^k - ln U)^(1/k).  Failures are then counted in 91.25-day buckets.
"""
import argparse
import numpy as np
import pandas as pd
import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=15, help="forecast the N models with the most failures")
ap.add_argument("--min-failures", type=int, default=30)
ap.add_argument("--boot", type=int, default=200, help="bootstrap refits per model")
ap.add_argument("--sims", type=int, default=5000)
ap.add_argument("--horizon", type=int, default=8, help="quarters to forecast")
ap.add_argument("--holdout", type=int, default=2, help="quarters held out for the backtest")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
c.OUT.mkdir(exist_ok=True)


def bootstrap_params(g, B, rng):
    (k0, l0), _ = c.weibull_fit(g.entry_days, g.exit_days, g.failed)
    e, x, f = g.entry_days.values, g.exit_days.values, g.failed.values
    out = []
    for _ in range(B):
        i = rng.integers(0, len(g), len(g))
        if f[i].sum() < 5:
            continue
        (k, lam), _ = c.weibull_fit(e[i], x[i], f[i], start=np.array([k0, l0]))
        out.append((k, lam))
    return np.array(out)


def simulate(ages, params, quarters, n_sims, rng):
    """Failures per quarter-ahead for the given surviving drives. Returns (n_sims, quarters)."""
    edges = np.arange(0, quarters + 1) * c.QUARTER_DAYS
    out = np.zeros((n_sims, quarters), int)
    for s in range(n_sims):
        k, lam = params[s % len(params)]
        u = rng.random(len(ages))
        t_fail = lam * ((ages / lam) ** k - np.log(u)) ** (1.0 / k)
        out[s] = np.histogram(t_fail - ages, bins=edges)[0]
    return out


def survivors(drives, cutoff_slack_days=7):
    """Drives still running at the end of the data: not failed and seen within a week of the last date."""
    end = drives.last_date.max()
    return drives[(drives.failed == 0) & (drives.last_date >= end - pd.Timedelta(days=cutoff_slack_days))]


def pct(a, q):
    return np.percentile(a, q, axis=0)


def eligible_models(d):
    fails = d.groupby("model").failed.sum().sort_values(ascending=False)
    return list(fails[fails >= args.min_failures].head(args.top).index)


frames = c.load_quarters()
if len(frames) <= args.holdout + 1:
    raise SystemExit(f"Need more than {args.holdout + 1} quarters for a backtest; have {len(frames)}. "
                     "Ingest more quarters first.")

# Use exactly the same model set as analyze.py (cleaning on the full window), so every result in the
# project describes the same fleet. (A model can pass the age/calendar check on a short window and fail
# it on a longer one, e.g. drives that were powered down or moved.)
d_all = c.clean(c.build_drives(frames), verbose=False)
KEEP = set(d_all.model.unique())

# ================================================================= BACKTEST
train, hold = frames[:-args.holdout], frames[-args.holdout:]
d_tr = c.clean(c.build_drives(train), verbose=False)
d_tr = d_tr[d_tr.model.isin(KEEP)]
models = eligible_models(d_tr)
print(f"Backtest: fit on {[f.quarter.iloc[0] for f in train][0]} .. {[f.quarter.iloc[0] for f in train][-1]}, "
      f"forecast {args.holdout} quarters, {len(models)} models.")
surv_tr = survivors(d_tr)
hold_fail = pd.concat([f[f.failed == 1][["model", "serial_number"]].assign(q=i + 1)
                       for i, f in enumerate(hold)], ignore_index=True)

bt_rows, fleet_sims, fleet_actual = [], np.zeros((args.sims, args.holdout), int), np.zeros(args.holdout, int)
for m in models:
    g = d_tr[d_tr.model == m]
    params = bootstrap_params(g, args.boot, rng)
    s = surv_tr[surv_tr.model == m]
    sims = simulate(s.exit_days.values, params, args.holdout, args.sims, rng)
    actual = (hold_fail[hold_fail.serial_number.isin(s.serial_number)]
              .groupby("q").size().reindex(range(1, args.holdout + 1), fill_value=0).values)
    fleet_sims += sims
    fleet_actual += actual
    for q in range(args.holdout):
        lo, med, hi = pct(sims[:, q], [5, 50, 95])
        bt_rows.append(dict(scope=m, quarter_ahead=q + 1, actual=int(actual[q]), p5=lo, p50=med, p95=hi,
                            in_band=bool(lo <= actual[q] <= hi), survivors=len(s)))
for q in range(args.holdout):
    lo, med, hi = pct(fleet_sims[:, q], [5, 50, 95])
    bt_rows.append(dict(scope="FLEET (modelled)", quarter_ahead=q + 1, actual=int(fleet_actual[q]),
                        p5=lo, p50=med, p95=hi, in_band=bool(lo <= fleet_actual[q] <= hi),
                        survivors=int(len(surv_tr[surv_tr.model.isin(models)]))))
bt = pd.DataFrame(bt_rows)
bt["pct_error_of_median"] = 100 * (bt.p50 - bt.actual) / bt.actual.replace(0, np.nan)
bt.round(1).to_csv(c.OUT / "backtest.csv", index=False)
print(bt[bt.scope.str.startswith("FLEET")].round(1).to_string(index=False))
per_model = bt[~bt.scope.str.startswith("FLEET")]
print(f"\nModel-level 90% band coverage: {per_model.in_band.mean():.0%} of {len(per_model)} model-quarters "
      f"(a calibrated 90% band should land near 90%; much lower means the forecast is overconfident or biased).")
print(f"Median forecast is {'above' if (per_model.p50 > per_model.actual).mean() > 0.5 else 'below'} actual "
      f"in {max((per_model.p50 > per_model.actual).mean(), (per_model.p50 <= per_model.actual).mean()):.0%} of model-quarters.\n")

# ================================================================= FORECAST
models = eligible_models(d_all)
surv = survivors(d_all)
print(f"Forecast: {len(models)} models, {args.horizon} quarters ahead, {args.sims} simulations each.")
rows, fleet = [], np.zeros((args.sims, args.horizon))
for m in models:
    g = d_all[d_all.model == m]
    params = bootstrap_params(g, args.boot, rng)
    s = surv[surv.model == m]
    sims = simulate(s.exit_days.values, params, args.horizon, args.sims, rng)
    fleet += sims
    for q in range(args.horizon):
        lo, med, hi = pct(sims[:, q], [5, 50, 95])
        rows.append(dict(scope=m, quarter_ahead=q + 1, p5=lo, p50=med, p95=hi, mean=sims[:, q].mean(),
                         survivors=len(s)))
for q in range(args.horizon):
    lo, med, hi = pct(fleet[:, q], [5, 50, 95])
    rows.append(dict(scope="FLEET (modelled)", quarter_ahead=q + 1, p5=lo, p50=med, p95=hi,
                     mean=fleet[:, q].mean(), survivors=int(len(surv[surv.model.isin(models)]))))
fc = pd.DataFrame(rows).round(1)
fc.to_csv(c.OUT / "forecast.csv", index=False)
print(fc[fc.scope.str.startswith("FLEET")].to_string(index=False))
print("\nAssumptions to state in your README: forecast covers only drives running today (no new installs), "
      "uses a constant fitted Weibull per model, and ignores drives retired for non-failure reasons "
      "(which makes it lean high).\nWrote outputs/backtest.csv and outputs/forecast.csv")
