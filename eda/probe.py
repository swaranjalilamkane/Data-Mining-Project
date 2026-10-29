import pandas as pd, numpy as np, glob, os, json
D = os.path.join(os.path.dirname(__file__), "..", "dataset 2")
def uid(p): return os.path.basename(p).split("_")[-1].split(".")[0]

# interval streams
for s in ["phonelock","dark","phonecharge","conversation"]:
    durs=[]; users=0; empty=[]
    for p in sorted(glob.glob(f"{D}/sensing/{s}/*.csv")):
        df=pd.read_csv(p, index_col=False); df.columns=[c.strip() for c in df.columns]
        if len(df)==0: empty.append(uid(p)); continue
        users+=1; durs.append((df.iloc[:,1]-df.iloc[:,0]).values)
    d=np.concatenate(durs)/60
    print(f"{s}: users={users} intervals={len(d)} empty={empty} dur_min: min={d.min():.1f} p5={np.percentile(d,5):.1f} med={np.median(d):.1f} p95={np.percentile(d,95):.1f} max={d.max():.0f}")

# time coverage per user (activity)
rows=[]
for p in sorted(glob.glob(f"{D}/sensing/activity/*.csv")):
    t=pd.read_csv(p, usecols=[0]).iloc[:,0]
    rows.append((uid(p), len(t), pd.to_datetime(t.min(),unit='s'), pd.to_datetime(t.max(),unit='s')))
cov=pd.DataFrame(rows,columns=["uid","n","start","end"]); cov["days"]=(cov.end-cov.start).dt.days
print(cov.describe(include='all').T[["count","mean","min","max"]] if False else cov.sort_values("days").to_string())
