"""
StudentLife preprocessing for Codewalk 1.

Reads the raw per-student files in `dataset 2/`, cleans them, and aggregates
everything onto one shared grid: one row per (student, local hour).
Each student's file is read, reduced to hourly numbers, and discarded before
the next one is read, so peak memory stays at one student's raw data (~150 MB)
even though the raw folder is 2.8 GB.

Outputs (in eda/processed/):
  hourly.parquet   one row per uid x local hour, all sensing + app features
  daily.parquet    hourly rolled up to uid x local day, plus EMA sleep/stress
  users.parquet    one row per student: survey scores (pre/post) + GPA
Each table is also written as .csv next to it, for opening in a spreadsheet or editor.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "dataset 2")
OUT = os.path.join(HERE, "processed")
TZ = "America/New_York"            # timestamps are UTC epoch; Dartmouth is EDT (UTC-4)
TERM_START = pd.Timestamp("2013-03-27", tz=TZ)   # Dartmouth spring term, 10 weeks + finals
TERM_END = pd.Timestamp("2013-06-01", tz=TZ)     # anything after this is a phone left running
NIGHT_HOURS = range(0, 6)          # 00:00-05:59 local counts as "late night"

# Apps that appear in the running-task list but are not "use" by the student:
# the study's own EMA app, launchers/home screens, lock screens and system UI.
NOT_USAGE = re.compile(
    r"apps\.paco|launcher|\.home$|trebuchet|systemui|deskclock|screenlock|"
    r"^android$|com\.android\.settings|^edu\.dartmouth"
)
SOCIAL = re.compile(
    r"facebook|twitter|instagram|snapchat|tumblr|kakao|whatsapp|textplus|"
    r"com\.android\.mms|talk|messag|skype|viber|tencent|pinterest|reddit|vine"
)


def uid_of(path):
    return re.search(r"(u\d\d)", os.path.basename(path)).group(1)


def to_local_hour(epoch_s):
    return pd.to_datetime(epoch_s, unit="s", utc=True).dt.tz_convert(TZ).dt.floor("h")


def read_csv_clean(path, **kw):
    # Raw files have a UTF-8 BOM (app/sms/call), leading spaces in headers
    # (" activity inference") and a trailing comma on every row (gps, wifi_location)
    # which pandas would otherwise read as an extra unnamed index column.
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=False, **kw)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- point streams
def categorical_hourly(path, n_classes, prefix):
    """activity (0 stationary,1 walking,2 running,3 unknown) and
    audio (0 silence,1 voice,2 noise) are ~1-3 s samples of an inference label.
    Reduce to: share of samples in each class per hour, and sample count."""
    df = read_csv_clean(path)
    df.columns = ["t", "v"]
    df = df.drop_duplicates("t")              # ~5% of audio rows repeat a timestamp
    df["hour"] = to_local_hour(df["t"])
    counts = pd.crosstab(df["hour"], df["v"]).reindex(columns=range(n_classes), fill_value=0)
    out = counts.div(counts.sum(axis=1), axis=0)
    out.columns = [f"{prefix}_share_{c}" for c in out.columns]
    out[f"{prefix}_samples"] = counts.sum(axis=1)
    return out


# ------------------------------------------------------------- interval streams
def interval_minutes_hourly(path, name):
    """phonelock / dark / phonecharge / conversation are (start, end) intervals.
    Spread each interval over the clock hours it covers -> minutes per hour."""
    df = read_csv_clean(path)
    df.columns = ["start", "end"]
    df = df[df["end"] > df["start"]]
    if df.empty:
        return pd.DataFrame(columns=[f"{name}_min"])
    minutes = np.concatenate([np.arange(s, e, 60) for s, e in zip(df["start"], df["end"])])
    hours = to_local_hour(pd.Series(minutes))
    return hours.value_counts().sort_index().rename(f"{name}_min").to_frame()


# --------------------------------------------------------------------- app usage
def app_hourly(path):
    """running_app_uXX.csv is a snapshot of the running-task list every 20 min,
    NOT a log of foreground events. First row of a snapshot = most recent task.
    Usage proxy: the top task CHANGED since the previous snapshot -> the student
    interacted with the phone in that 20-min window."""
    df = read_csv_clean(path, usecols=["timestamp", "RUNNING_TASKS_topActivity_mPackage"])
    df.columns = ["t", "pkg"]
    top = df.groupby("t", sort=True).head(1).sort_values("t").reset_index(drop=True)
    top["usage_app"] = ~top["pkg"].str.contains(NOT_USAGE)
    top["social_app"] = top["pkg"].str.contains(SOCIAL) & top["usage_app"]
    top["changed"] = top["pkg"].ne(top["pkg"].shift())
    # A change into or out of the study's EMA app (paco) is prompted by the study,
    # not the student's own phone use, so it does not count as an interaction.
    paco = top["pkg"].str.contains("apps.paco", regex=False)
    top["interaction"] = top["changed"] & ~paco & ~paco.shift(fill_value=False)
    top["hour"] = to_local_hour(top["t"])
    g = top.groupby("hour")
    return pd.DataFrame({
        "app_snapshots": g.size(),
        "app_switches": g["changed"].sum(),
        "app_switches_real": g["interaction"].sum(),                      # interaction proxy
        "app_social_snaps": g["social_app"].sum(),
        "app_distinct": top[top["usage_app"]].groupby("hour")["pkg"].nunique(),
    })


# ------------------------------------------------------------------ comms logs
def comm_hourly(path, name):
    # Rows with an empty payload are sync "pings" from the logger, not messages/calls.
    # For about half the students the file holds ONLY pings (3 columns, no payload):
    # their texting/calling was never captured, so return None -> stays NaN, not 0.
    df = read_csv_clean(path)
    if df.shape[1] <= 3:
        return None
    df = df[df.iloc[:, 3].notna()]
    return to_local_hour(df["timestamp"]).value_counts().rename(f"{name}_n").to_frame()


def build_user(u):
    s = os.path.join(RAW, "sensing")
    parts = [
        categorical_hourly(f"{s}/activity/activity_{u}.csv", 4, "act"),
        categorical_hourly(f"{s}/audio/audio_{u}.csv", 3, "aud"),
        interval_minutes_hourly(f"{s}/phonelock/phonelock_{u}.csv", "lock"),
        interval_minutes_hourly(f"{s}/dark/dark_{u}.csv", "dark"),
        interval_minutes_hourly(f"{s}/phonecharge/phonecharge_{u}.csv", "charge"),
        interval_minutes_hourly(f"{s}/conversation/conversation_{u}.csv", "convo"),
        app_hourly(f"{RAW}/app_usage/running_app_{u}.csv"),
        comm_hourly(f"{RAW}/sms/sms_{u}.csv", "sms"),
        comm_hourly(f"{RAW}/call_log/call_log_{u}.csv", "call"),
    ]
    unlogged = [n for n, p in zip(["sms", "call"], parts[-2:]) if p is None]
    parts = [p for p in parts if p is not None]
    # Full hourly grid over the term so "no data" is an explicit row, not a missing one.
    grid = pd.date_range(TERM_START, TERM_END, freq="h", inclusive="left", name="hour")
    df = pd.DataFrame(index=grid).join(parts, how="left")
    for n in unlogged:
        df[f"{n}_n"] = np.nan
    # Interval & count streams: absence inside an observed stream means zero.
    zero_fill = [c for c in df if c.endswith(("_min", "_n")) and c[:-2] not in unlogged] + \
                ["app_switches", "app_switches_real", "app_social_snaps"]
    df[zero_fill] = df[zero_fill].fillna(0)
    df["sensing_ok"] = df["act_samples"].fillna(0) > 0     # phone was alive and logging
    df.insert(0, "uid", u)
    return df.reset_index()


# --------------------------------------------------------------------------- EMA
def load_ema(name, field, recode):
    """EMA JSON: one list per student. Entries with only a 'null' key hold a
    'lat,lon' string — the prompt fired but the student never answered.
    `recode` maps raw answer codes to a meaningful scale; it is applied per
    response BEFORE averaging, and codes outside the questionnaire are dropped."""
    rows = []
    for p in glob.glob(f"{RAW}/EMA/response/{name}/*.json"):
        for r in json.load(open(p)):
            if field in r:
                rows.append((uid_of(p), r["resp_time"], r[field]))
    df = pd.DataFrame(rows, columns=["uid", "t", field])
    df[field] = pd.to_numeric(df[field], errors="coerce").map(recode)
    df["day"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(TZ).dt.normalize()
    return df.groupby(["uid", "day"])[field].mean()


# ------------------------------------------------------------------------ surveys
PHQ = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}


def load_users():
    sv = os.path.join(RAW, "survey")
    phq = pd.read_csv(f"{sv}/PHQ-9.csv")
    items = phq.columns[2:11]                                 # 9 symptom items; last col is a difficulty rating
    phq["phq9"] = phq[items].apply(lambda c: c.str.strip().map(PHQ)).sum(axis=1, min_count=9)
    pss = pd.read_csv(f"{sv}/PerceivedStressScale.csv")
    pss_map = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
    pss_items = pss.iloc[:, 2:12].apply(lambda c: c.str.strip().map(pss_map))
    reverse = [3, 4, 6, 7]                                    # positively worded PSS items (0-indexed)
    pss_items.iloc[:, reverse] = 4 - pss_items.iloc[:, reverse]
    pss["pss"] = pss_items.sum(axis=1, min_count=10)
    lon = pd.read_csv(f"{sv}/LonelinessScale.csv")
    lon_map = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}
    lon_items = lon.iloc[:, 2:22].apply(lambda c: c.str.strip().map(lon_map))
    lon_reverse = [0, 4, 5, 8, 9, 14, 15, 18, 19]            # positively worded UCLA items (0-indexed)
    lon_items.iloc[:, lon_reverse] = 5 - lon_items.iloc[:, lon_reverse]
    lon["loneliness"] = lon_items.sum(axis=1, min_count=20)
    wide = []
    for df, col in [(phq, "phq9"), (pss, "pss"), (lon, "loneliness")]:
        w = df.pivot_table(index="uid", columns="type", values=col)
        w.columns = [f"{col}_{c}" for c in w.columns]
        wide.append(w)
    gr = pd.read_csv(f"{RAW}/education/grades.csv")
    gr.columns = [c.strip() for c in gr.columns]
    wide.append(gr.set_index("uid")[["gpa all"]].rename(columns={"gpa all": "gpa"}))
    return pd.concat(wide, axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    uids = sorted(uid_of(p) for p in glob.glob(f"{RAW}/sensing/activity/*.csv"))
    hourly = []
    for u in uids:
        hourly.append(build_user(u))
        print(u, "done", flush=True)
    hourly = pd.concat(hourly, ignore_index=True)
    hourly["hour"] = hourly["hour"].dt.tz_localize(None)   # store as naive local time
    hourly.to_parquet(f"{OUT}/hourly.parquet")

    h = hourly.assign(day=hourly["hour"].dt.normalize(), hod=hourly["hour"].dt.hour)
    night = h[h["hod"].isin(NIGHT_HOURS)]
    daily = h.groupby(["uid", "day"]).agg(
        sensing_hours=("sensing_ok", "sum"),
        app_snapshots=("app_snapshots", "sum"),
        app_switches_real=("app_switches_real", "sum"),
        app_social_snaps=("app_social_snaps", "sum"),
        lock_min=("lock_min", "sum"), dark_min=("dark_min", "sum"),
        convo_min=("convo_min", "sum"),
        # min_count=1 keeps unlogged students NaN; a plain sum() would turn them into 0
        sms_n=("sms_n", lambda s: s.sum(min_count=1)), call_n=("call_n", lambda s: s.sum(min_count=1)),
        act_moving=("act_share_1", "mean"), aud_voice=("aud_share_1", "mean"),
    ).join(night.groupby(["uid", "day"]).agg(
        night_switches=("app_switches_real", "sum"),
        night_lock_min=("lock_min", "sum")))
    daily.index = daily.index.set_levels(daily.index.levels[1].tz_localize(TZ), level=1)
    # Sleep "hour" is a category code: 1 = <3 h, 2 = 3.5 h ... 19 = 12 h  ->  hours = 3 + 0.5*(code-1)
    sleep_h = {c: 3 + 0.5 * (c - 1) for c in range(1, 20)}
    # Stress codes are NOT ordinal (4/5 = feeling good/great). Recode to 1 (great) .. 5 (stressed out).
    stress = {5: 1, 4: 2, 1: 3, 2: 4, 3: 5}
    daily = daily.join(load_ema("Sleep", "hour", sleep_h).rename("ema_sleep_h")) \
                 .join(load_ema("Stress", "level", stress).rename("ema_stress"))
    daily = daily.reset_index()
    daily["day"] = daily["day"].dt.tz_localize(None)
    daily.to_parquet(f"{OUT}/daily.parquet")
    users = load_users()
    users.to_parquet(f"{OUT}/users.parquet")
    # CSV copies are for reading only; the analysis code loads the parquet files.
    hourly.to_csv(f"{OUT}/hourly.csv", index=False)
    daily.to_csv(f"{OUT}/daily.csv", index=False)
    users.to_csv(f"{OUT}/users.csv")
    print(hourly.shape, daily.shape)


if __name__ == "__main__":
    main()
