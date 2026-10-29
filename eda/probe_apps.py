import pandas as pd, numpy as np, glob, os
D = os.path.join(os.path.dirname(__file__), "..", "dataset 2")
def uid(p): return os.path.basename(p).split("_")[-1].split(".")[0]
stats=[]; pk=pd.Series(dtype=float)
for p in sorted(glob.glob(f"{D}/app_usage/*.csv")):
    df=pd.read_csv(p, encoding="utf-8-sig", usecols=["timestamp","RUNNING_TASKS_id","RUNNING_TASKS_topActivity_mPackage"])
    ts=df.timestamp.drop_duplicates().sort_values()
    gap=ts.diff().dropna()
    # the first task listed per snapshot ~ foreground
    first=df.groupby("timestamp",sort=False).head(1)
    stats.append(dict(uid=uid(p), rows=len(df), snapshots=len(ts), rows_per_snap=len(df)/len(ts),
        start=pd.to_datetime(ts.min(),unit='s'), end=pd.to_datetime(ts.max(),unit='s'),
        gap_med_s=gap.median(), gap_p90_s=gap.quantile(.9), gaps_over_1h=(gap>3600).sum(),
        null_pkg=df.RUNNING_TASKS_topActivity_mPackage.isna().mean()))
    pk=pk.add(first.RUNNING_TASKS_topActivity_mPackage.value_counts(), fill_value=0)
s=pd.DataFrame(stats); pd.set_option("display.width",250)
print(s.to_string()); print(s.describe().T)
print("\nTop 'first-listed' packages:\n", pk.sort_values(ascending=False).head(30))
