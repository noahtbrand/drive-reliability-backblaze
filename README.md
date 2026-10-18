# drive-reliability-backblaze
# Hard Drive Reliability Analysis

I used [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) to study how hard drive failure rates change with age and vary between models. I also fitted Weibull survival models to forecast failures, then checked the predictions against two quarters of data held out of the fit.

The analysis covers Q1 2025 through Q2 2026: about 345,000 drives, 455,000 drive-years, and 6,177 failures across 30 drive models.

[View the Tableau dashboard](https://public.tableau.com/views/Drive_Reliability_Project/DriveReliability)

![Drive reliability dashboard](dashboard_project.png)

## Results

- TOSHIBA MG07ACA14TA showed a sharp increase in observed AFR: 0.66% at ages 4–5, 1.47% at ages 5–6, and 5.25% at ages 6–7. The last increase was about 3.6 times the previous rate.
- For the backtest, I trained through Q4 2025 and predicted failures in Q1 and Q2 2026. The median forecast total was 1,829 failures, compared with 1,639 actual failures, or 11.6% high. Six of the ten models were within 10% of their actual totals.
- The 90% prediction bands covered only about 50% of model-quarter observations. The forecast was much less reliable at describing its uncertainty than the nominal coverage suggests.
- As a check against the published data, my Q1 2026 failure count matched Backblaze's count of 1,030. My 2025 fleet AFR also matched their reported 1.36%.

The age plots did not follow one smooth pattern for every model. Some fitted Weibull shapes suggested wear-out that was not supported by the observed rates. I used the age-band results alongside the fitted curves when interpreting the models.

## How it works

- `src/ingest.py` uses DuckDB to turn the daily CSVs into one row per drive per quarter, saved as Parquet. Each row records the first and last observation dates, entry and exit ages, and failure status. Drive age comes from SMART-9 power-on hours.
- `src/common.py` contains the cleaning rules and Weibull fitting code. The cleaning excludes impossible ages, models with fewer than 500 drives, and models where more than 5% of drives have power-on-hour changes that disagree with elapsed calendar time by more than 30 days.
- `src/analyze.py` calculates AFR as failures divided by observed drive-years, with exact Poisson 95% confidence intervals. It produces model, manufacturer, and age-band tables, failure contributions by model, and Kaplan–Meier curves with delayed entry. I used AFR estimates and confidence intervals for model comparisons and did not use log-rank tests.
- The Weibull fit uses maximum likelihood with left truncation and right censoring. Confidence intervals come from a numerical Hessian on the log scale. I cross-checked the fitting against `lifelines`.
- `src/forecast.py` refits the Weibull model on bootstrap samples and runs Monte Carlo simulations. For each surviving drive, it samples a failure age conditional on the drive having reached its current age. It then counts failures by quarter. The same script runs the backtest using the last two quarters as held-out data.
- `src/diagnose.py` compares quarterly results with Backblaze's published figures, observed age-band AFR with the Weibull-implied rates, and forecasts with actual failures by model.
- `src/tableau_prep.py` reshapes the results into CSVs in `outputs/tableau/` for the dashboard.

Accounting for entry age matters because many drives were already several years old when they first appeared in the dataset. In a simulation with a known Weibull shape of 1.40, ignoring this gave an estimate of 1.77. The fit that accounted for entry age recovered the known shape.

## Running the project

Download the quarterly ZIP files from Backblaze into `data/raw/`, then run these commands from the project folder:

```bash
pip install -r requirements.txt
python src/ingest.py
python src/analyze.py
python src/forecast.py --horizon 4
python src/diagnose.py
python src/tableau_prep.py
```

The raw data in `data/` is not committed because of its size. The results in `outputs/` are included.

## Limitations

- The prediction bands are too narrow in the backtest. The simulation treats drives as independent, so it may miss shared batch, cohort, or data-center effects. I have not established which effects explain the poor coverage.
- The forecast does not model retirement for non-failure reasons. That could contribute to the 11.6% overprediction by keeping drives in the forecast population longer than they remain in service.
- I only backtested two quarters. The displayed forecast extends four quarters, so the longer horizon has not been validated.
- The forecast covers drives running at the end of the observation period. It does not include future installations.
- The AFR analysis excludes models that fail the size or age-consistency filters, including ST8000DM002 and ST8000NM0055. The saved backtest predates applying that filter to the backtest and still includes ST8000NM0055. Its model list therefore differs slightly from the AFR analysis.
- My filtering does not exactly match Backblaze's published tables, which also exclude boot drives and small models. Some quarterly figures differ as a result.

Analysis and code by Noah Brand.
