import pandas as pd, numpy as np, glob, os, json
D = os.path.join(os.path.dirname(__file__), "..", "dataset 2")
for s in ["activity","audio"]:
    vc=pd.Series(dtype=float); gaps=[]
    for u in ["u00","u19","u59"]:
        df=pd.read_csv(f"{D}/sensing/{s}/{s}_{u}.csv"); df.columns=["t","v"]
        vc=vc.add(df.v.value_counts(),fill_value=0)
        g=df.t.diff().dropna(); gaps.append(g)
        print(s,u,len(df),"gap s quantiles", g.quantile([.25,.5,.75,.95,.99]).round(1).tolist(), "dup ts",df.t.duplicated().sum())
    print(s,"value share",(vc/vc.sum()).round(3).to_dict())
gps=pd.read_csv(f"{D}/sensing/gps/gps_u00.csv",index_col=False)
print("\nGPS dtypes\n",gps.dtypes, "\nmissing\n", gps.isna().mean().round(3))
w=pd.read_csv(f"{D}/sensing/wifi_location/wifi_location_u00.csv",index_col=False)
print(w.location.str.extract(r"^(\w+)\[")[0].value_counts())
# EMA
for e in ["Sleep","Stress","PAM","Social"]:
    n=[];keys=pd.Series(dtype=float)
    for p in glob.glob(f"{D}/EMA/response/{e}/*.json"):
        r=json.load(open(p)); n.append(len(r))
        for x in r: keys=keys.add(pd.Series(1,index=list(x.keys())),fill_value=0)
    n=np.array(n); print(f"\nEMA {e}: users={len(n)} zero={int((n==0).sum())} median resp={np.median(n)} max={n.max()} keys={keys.to_dict()}")
dfn=json.load(open(f"{D}/EMA/EMA_definition.json"))
for q in dfn:
    if q.get("name") in ("Sleep","Stress"): print(json.dumps(q)[:900])
# surveys
for p in sorted(glob.glob(f"{D}/survey/*.csv"))+[f"{D}/education/grades.csv"]:
    df=pd.read_csv(p); tc=df["type"].value_counts().to_dict() if "type" in df else {}
    print(os.path.basename(p), df.shape, "uids",df.iloc[:,0].nunique(), tc, "missing cells %.1f%%"%(100*df.isna().mean().mean()))
