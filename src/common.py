"""Shared helpers: loading the per-quarter parquet files, cleaning, AFR, Weibull fitting, Kaplan-Meier.

Run every script from the project root (the folder that contains src/ and data/).
"""
import re
import pathlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

PROCESSED = pathlib.Path("data/processed")
OUT = pathlib.Path("outputs")          # small result CSVs (commit these; data/ is git-ignored)
QUARTER_DAYS = 91.25


# ----------------------------------------------------------------- loading
def quarter_key(p):
    m = re.match(r"Q(\d)_(\d{4})", pathlib.Path(p).stem)
    return (int(m.group(2)), int(m.group(1)))


def quarter_files():
    return sorted(PROCESSED.glob("Q*_*.parquet"), key=quarter_key)


def read_quarter(path):
    d = pd.read_parquet(path)
    for c in ("first_date", "last_date"):
        d[c] = pd.to_datetime(d[c])
    d["quarter"] = pathlib.Path(path).stem
    return d


def load_quarters(files=None):
    files = list(files) if files is not None else quarter_files()
    if not files:
        raise SystemExit("No parquet files in data/processed/. Run src/ingest.py first.")
    return [read_quarter(f) for f in files]          # list of per-quarter DataFrames, oldest first


def build_drives(frames):
    """One row per drive across the given quarters."""
    q = pd.concat(frames, ignore_index=True)
    return q.groupby(["model", "serial_number"], as_index=False).agg(
        entry_days=("entry_days", "min"), exit_days=("exit_days", "max"),
        first_date=("first_date", "min"), last_date=("last_date", "max"),
        failed=("failed", "max"))


def manufacturer(model):
    m = str(model).upper()
    if m.startswith("ST") or "SEAGATE" in m:
        return "Seagate"
    if m.startswith("HGST"):
        return "HGST"
    if m.startswith("WDC") or m.startswith("WD"):
        return "WDC"
    if m.startswith("TOSHIBA"):
        return "Toshiba"
    if m.startswith("HITACHI"):
        return "Hitachi"
    if m.startswith("SAMSUNG"):
        return "Samsung"
    return "Other"


def clean(drives, min_drives=500, max_mismatch=0.05, verbose=True):
    """Drop unusable rows, then drop models whose power-on hours disagree with calendar time."""
    d = drives.copy()
    n0, f0 = len(d), int(d.failed.sum())
    d = d.dropna(subset=["entry_days", "exit_days"])
    d["failed"] = d["failed"].astype(int)
    # keep drives with positive exposure, plus failures seen only once (e.g. dead on arrival)
    d = d[(d.entry_days >= 0) & (d.exit_days > 0) &
          ((d.exit_days > d.entry_days) | (d.failed == 1))].copy()
    if verbose:
        print(f"[clean] dropped {n0 - len(d)} unusable drives "
              f"({f0 - int(d.failed.sum())} of them failures) -> {len(d)} drives")
    cal = (d.last_date - d.first_date).dt.days + 1
    d["mismatch"] = ((d.exit_days - d.entry_days) - cal).abs() > 30
    stats = d.groupby("model").agg(n=("serial_number", "count"), mismatch=("mismatch", "mean"),
                                   failures=("failed", "sum"))
    bad = stats[(stats.n < min_drives) | (stats.mismatch > max_mismatch)]
    if verbose and len(bad):
        print(f"[clean] excluded {len(bad)} models (fewer than {min_drives} drives, or "
              f"> {max_mismatch:.0%} of drives where power-on time disagrees with calendar time). "
              f"Worst 5 by mismatch rate:")
        print(bad.sort_values("mismatch", ascending=False).head(5).round(3).to_string())
    d = d[d.model.isin(stats.index.difference(bad.index))].copy()
    d["manufacturer"] = d.model.map(manufacturer)
    d["span_days"] = d.exit_days - d.entry_days
    return d.drop(columns=["mismatch"])


# ----------------------------------------------------------------- AFR
def afr_table(d, by):
    g = d.groupby(by).agg(drives=("serial_number", "count"), failures=("failed", "sum"),
                          drive_days=("span_days", "sum")).reset_index()
    yrs = g.drive_days / 365.0
    f = g.failures.values
    g["afr_pct"] = 100 * f / yrs
    g["afr_lo_pct"] = 100 * np.where(f > 0, chi2.ppf(0.025, 2 * np.maximum(f, 1)) / 2, 0) / yrs
    g["afr_hi_pct"] = 100 * chi2.ppf(0.975, 2 * f + 2) / 2 / yrs
    return g.sort_values("failures", ascending=False)


# ----------------------------------------------------------------- Weibull (left-truncated, right-censored)
def _nll(p, entry, exit_, ev):
    k, lam = np.exp(p)
    with np.errstate(all="ignore"):
        lx = k * (np.log(exit_) - np.log(lam))
        le = np.where(entry > 0, k * (np.log(np.maximum(entry, 1e-12)) - np.log(lam)), -np.inf)
        ll = np.where(ev, np.log(k / lam) + (k - 1) * (np.log(exit_) - np.log(lam)), 0.0) \
            - np.exp(lx) + np.exp(le)
        out = -ll.sum()
    return out if np.isfinite(out) else 1e18


def weibull_fit(entry, exit_, event, start=None):
    """MLE of (shape k, scale lam in days). `entry` handles drives already old when first observed."""
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    ev = np.asarray(event, bool)
    x0 = np.log(start) if start is not None else np.array([0.0, np.log(2 * exit_.max())])
    r = minimize(_nll, x0, args=(entry, exit_, ev), method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
    return np.exp(r.x), r


def weibull_fit_ci(entry, exit_, event):
    """Fit plus 95% Wald CIs (delta method on the log scale, numeric Hessian)."""
    (k, lam), r = weibull_fit(entry, exit_, event)
    args = (np.asarray(entry, float), np.asarray(exit_, float), np.asarray(event, bool))
    x = r.x
    h = 1e-3
    H = np.zeros((2, 2))
    f = lambda z: _nll(z, *args)
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    z = 1.96
    return dict(shape=k, shape_lo=k * np.exp(-z * se[0]), shape_hi=k * np.exp(z * se[0]),
                scale_days=lam, scale_lo=lam * np.exp(-z * se[1]), scale_hi=lam * np.exp(z * se[1]))


# ----------------------------------------------------------------- Kaplan-Meier with delayed entry
def km_delayed(entry, exit_, event, min_at_risk=100):
    """Survival curve for left-truncated data, conditional on reaching the first age that has
    at least `min_at_risk` drives under observation. Returns age, n_at_risk, events, survival."""
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    ev = np.asarray(event, bool)
    keep = exit_ > entry                      # zero-exposure rows can't be in a risk set
    entry, exit_, ev = entry[keep], exit_[keep], ev[keep]
    t_ev, d = np.unique(exit_[ev], return_counts=True)
    ent, ex = np.sort(entry), np.sort(exit_)
    n = np.searchsorted(ent, t_ev, side="left") - np.searchsorted(ex, t_ev, side="left")
    ok = np.where(n >= min_at_risk)[0]
    if len(ok) == 0:
        return pd.DataFrame(columns=["age_days", "n_at_risk", "events", "survival"])
    lo, hi = ok[0], ok[-1]
    t_ev, d, n = t_ev[lo:hi + 1], d[lo:hi + 1], n[lo:hi + 1]
    s = np.cumprod(1 - d / n)
    return pd.DataFrame({"age_days": t_ev, "n_at_risk": n, "events": d, "survival": s})
