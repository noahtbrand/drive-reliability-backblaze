import duckdb
from lifelines import WeibullFitter

# Connect to DuckDB and create a view mapping your processed parquet file
con = duckdb.connect()
con.sql("""
CREATE VIEW drives AS
SELECT model, serial_number, 
       MIN(entry_days) AS entry_days, 
       MAX(exit_days) AS exit_days,
       MIN(first_date) AS first_date, 
       MAX(last_date) AS last_date, 
       MAX(failed) AS failed
FROM 'data/processed/*.parquet' 
GROUP BY model, serial_number
""")

# Load the data into a pandas dataframe
df = con.sql("SELECT model, entry_days, exit_days, failed FROM drives").df()

# Find the drive model with the highest total failures to analyze first
top = df.groupby("model").failed.sum().idxmax()
d = df[(df.model == top) & (df.exit_days > df.entry_days)]

print("model:", top, "drives:", len(d), "failures:", int(d.failed.sum()))

# Fit the Weibull distribution (accounting for left truncation with entry=)
wf = WeibullFitter().fit(d.exit_days, d.failed, entry=d.entry_days)
wf.print_summary()