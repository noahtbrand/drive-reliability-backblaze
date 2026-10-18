"""Reshape the result CSVs into Tableau-friendly files in outputs/tableau/.

Run after analyze.py, diagnose.py and forecast.py:   python src/tableau_prep.py
"""
import pathlib
import pandas as pd

OUT = pathlib.Path("outputs")
TAB = OUT / "tableau"
TAB.mkdir(parents=True, exist_ok=True)
MIN_EXPOSURE_YRS = 50      # hide age bands with too little exposure to trust
# Quarterly fleet AFR (%) published by Backblaze in its Q1 2026 and Q2 2026 Drive Stats posts
BACKBLAZE_PUBLISHED = {"2026-Q1": 1.24, "2026-Q2": 1.73}


def save(df, name):
    df.to_csv(TAB / name, index=False)
    print(f"  {name:28s} {len(df):5d} rows   columns: {', '.join(df.columns)}")


print("Writing outputs/tableau/ ...")

# 1. AFR by age ("bathtub"): one row per model x age band x series
a = pd.read_csv(OUT / "afr_by_age.csv")
a = a[a.exposure_years >= MIN_EXPOSURE_YRS]
names = {"afr_pct": "Observed AFR", "afr_lo_pct": "95% CI low", "afr_hi_pct": "95% CI high",
         "weibull_implied_afr_pct": "Weibull fit"}
bath = a.melt(id_vars=["model", "age_band", "age_mid_years", "drives", "exposure_years", "failures", "weibull_shape"],
              value_vars=list(names), var_name="series", value_name="afr_value_pct")
bath["series"] = bath.series.map(names)
save(bath, "bathtub.csv")

# 2. Quarterly AFR, with Backblaze's published values alongside
q = pd.read_csv(OUT / "quarterly_failures.csv")
q["quarter_sort"] = q.quarter.str.replace(r"Q(\d)_(\d{4})", r"\2-Q\1", regex=True)
mine = q[["quarter_sort", "afr_pct", "failures"]].assign(series="This analysis")
pub = pd.DataFrame([dict(quarter_sort=k, afr_pct=v, series="Backblaze published")
                    for k, v in BACKBLAZE_PUBLISHED.items() if k in set(q.quarter_sort)])
save(pd.concat([mine, pub], ignore_index=True), "quarterly.csv")

# 3. Forecast: one row per scope x quarter ahead x percentile
f = pd.read_csv(OUT / "forecast.csv")
fl = f.melt(id_vars=["scope", "quarter_ahead", "survivors"], value_vars=["p5", "p50", "p95"],
            var_name="series", value_name="failures")
fl["series"] = fl.series.map({"p5": "5th percentile", "p50": "Median", "p95": "95th percentile"})
fl["quarter_label"] = "Q+" + fl.quarter_ahead.astype(str)
save(fl, "forecast.csv")

# 4. Backtest by model: actual vs forecast, summed over the held-out quarters
b = pd.read_csv(OUT / "backtest.csv")
b = b[~b.scope.str.startswith("FLEET")]
cum = b.groupby("scope").agg(Actual=("actual", "sum"), Forecast=("p50", "sum")).reset_index()
cum["forecast_over_actual"] = (cum.Forecast / cum.Actual.replace(0, float("nan"))).round(2)
bt = cum.melt(id_vars=["scope", "forecast_over_actual"], value_vars=["Actual", "Forecast"],
              var_name="series", value_name="failures")
bt["series"] = bt.series.replace({"Forecast": "Forecast (median)"})
save(bt, "backtest_by_model.csv")

# 5. Pareto (top 15 models) and manufacturer AFR (drop the tiny 'Other' group)
save(pd.read_csv(OUT / "failure_pareto.csv").head(15), "pareto.csv")
m = pd.read_csv(OUT / "afr_by_manufacturer.csv")
save(m[m.manufacturer != "Other"], "manufacturer_afr.csv")
print("Done. Connect Tableau to the files in outputs/tableau/")
