import sys, zipfile, tempfile, pathlib, duckdb

zp = pathlib.Path(sys.argv[1])
q = zp.stem.replace("data_", "")
out = pathlib.Path("data/processed") / f"{q}.parquet"

with tempfile.TemporaryDirectory() as tmp:
    zipfile.ZipFile(zp).extractall(tmp)
    files = [p.as_posix() for p in pathlib.Path(tmp).rglob("*.csv") 
             if "__MACOSX" not in p.parts and not p.name.startswith("._")]
    
    duckdb.sql(f"""
      COPY (
        SELECT serial_number, model, 
               MIN(date) AS first_date, MAX(date) AS last_date,
               MIN(smart_9_raw)/24.0 AS entry_days,
               MAX(smart_9_raw)/24.0 AS exit_days,
               MAX(failure) AS failed
        FROM read_csv({files}, union_by_name=true, 
             types={{'date':'DATE','failure':'INTEGER','smart_9_raw':'DOUBLE'}})
        GROUP BY serial_number, model
      ) TO '{out.as_posix()}' (FORMAT PARQUET)
    """)

print("wrote", out)
