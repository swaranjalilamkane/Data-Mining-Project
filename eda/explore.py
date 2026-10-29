"""
Basic exploration of the processed tables (run build_features.py first).
Prints shape / dtypes / missingness / summary stats and writes figures to eda/figures/.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "processed")
FIG = os.path.join(HERE, "figures")
RAW = os.path.join(HERE, "..", "dataset 2")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=150)
    plt.close(fig)


hourly = pd.read_parquet(f"{P}/hourly.parquet")
daily = pd.read_parquet(f"{P}/daily.parquet")
users = pd.read_parquet(f"{P}/users.parquet")

print("== shapes", hourly.shape, daily.shape, users.shape)
print("\n== hourly dtypes\n", hourly.dtypes.value_counts())
print("\n== hourly missingness (% of uid-hours)\n", (100 * hourly.isna().mean()).round(1).sort_values(ascending=False).to_string())
print("\n== daily summary\n", daily.describe().T.round(2).to_string())
print("\n== users summary\n", users.describe().T.round(2).to_string())
print("\n== users with each survey/GPA\n", users.notna().sum().to_string())

# ------------------------------------------------------------ 1. data coverage
cov = hourly.assign(day=hourly.hour.dt.normalize()).groupby(["uid", "day"])["sensing_ok"].mean().unstack()
cov = cov.loc[cov.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(cov.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
ax.set_yticks(range(len(cov)), cov.index, fontsize=6)
ticks = range(0, cov.shape[1], 7)
ax.set_xticks(ticks, [cov.columns[i].strftime("%b %d") for i in ticks], fontsize=8)
ax.grid(False)
ax.set_title("Sensing coverage: share of each day's hours with activity data")
fig.colorbar(im, ax=ax, fraction=0.025, label="share of hours logged")
save(fig, "01_coverage_heatmap")
user_cov = cov.mean(axis=1)
print("\n== per-student coverage: median %.2f, min %.2f (%s), students <70%%: %s"
      % (user_cov.median(), user_cov.min(), user_cov.idxmin(), list(user_cov[user_cov < .7].index)))

# --------------------------------------------- 2. phonelock interval durations
durs = []
for p in glob.glob(f"{RAW}/sensing/phonelock/*.csv"):
    d = pd.read_csv(p)
    durs.append((d.iloc[:, 1] - d.iloc[:, 0]).values / 60)
durs = np.concatenate(durs)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(durs, bins=np.arange(0, 660, 15), color=BLUE, edgecolor="#fcfcfb", linewidth=1)
ax.axvline(60, color=INK2, ls="--", lw=1)
ax.text(64, ax.get_ylim()[1] * .9, "no interval shorter than 60 min", color=INK2, fontsize=9)
ax.set_xlabel("lock interval length (minutes)")
ax.set_ylabel("intervals")
ax.set_title(f"Phone-lock intervals (n={len(durs):,}): only locks >= 1 h were logged, capped at 10 h")
save(fig, "02_phonelock_durations")

# -------------------------------------------- 3. hour-of-day profile (2 panels)
h = hourly[hourly.sensing_ok].assign(hod=lambda d: d.hour.dt.hour)
prof = h.groupby(["uid", "hod"])[["app_switches_real", "lock_min"]].mean().groupby("hod")
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
for ax, col, color, label in [(axes[0], "app_switches_real", BLUE, "app switches per hour"),
                              (axes[1], "lock_min", ORANGE, "minutes in a >=1 h lock")]:
    med = prof[col].median()
    q1, q3 = prof[col].quantile(.25), prof[col].quantile(.75)
    ax.fill_between(med.index, q1, q3, color=color, alpha=.18, lw=0)
    ax.plot(med.index, med, color=color, lw=2)
    ax.set_ylabel(label)
    ax.axvspan(-0.5, 5.5, color=GRID, alpha=.5, lw=0)
axes[0].set_title("Typical day (local time): median student, band = middle 50% of students")
axes[1].set_xlabel("hour of day (shaded = late night 00-06)")
axes[1].set_xticks(range(0, 24, 2))
save(fig, "03_hour_of_day_profile")

# ------------------------------------- 4. between-student variation, sorted box
d_ok = daily[daily.sensing_hours >= 18]          # days with at least 18 h of sensing
order = d_ok.groupby("uid").app_switches_real.median().sort_values().index
fig, ax = plt.subplots(figsize=(11, 4))
ax.boxplot([d_ok.loc[d_ok.uid == u, "app_switches_real"] for u in order], tick_labels=order,
           widths=.6, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor=BLUE, alpha=.35, edgecolor=BLUE),
           medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
ax.tick_params(axis="x", labelsize=7, rotation=90)
ax.set_ylabel("app switches per day")
ax.set_title("Daily phone-interaction proxy per student (days with >= 18 h sensing)")
save(fig, "04_per_student_daily_switches")

# ---------------------------------------- 5. late-night use prevalence by student
night = d_ok.assign(any_night=d_ok.night_switches > 0).groupby("uid").any_night.mean().sort_values()
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(range(len(night)), night.values, color=BLUE, width=.8)
ax.set_xticks(range(len(night)), night.index, fontsize=7, rotation=90)
ax.set_ylabel("share of nights")
ax.set_title("Share of nights with an app switch between 00:00 and 06:00")
ax.axhline(night.median(), color=INK2, ls="--", lw=1)
ax.text(0, night.median() + .02, f"median student {night.median():.0%}", color=INK2, fontsize=9)
save(fig, "05_late_night_prevalence")

# ------------------------------------------------ 6. across the term (2 panels)
wk = d_ok.assign(week=((d_ok.day - d_ok.day.min()).dt.days // 7) + 1)
wk_u = wk.groupby(["uid", "week"])[["app_switches_real", "ema_stress"]].mean()
wk_m = wk_u.groupby("week").median()
n_stress = wk.groupby("week").ema_stress.count()
fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
axes[0].plot(wk_m.index, wk_m.app_switches_real, color=BLUE, lw=2, marker="o", ms=5)
axes[0].set_ylabel("app switches / day")
axes[0].set_title("Across the term: median student per week")
axes[1].plot(wk_m.index, wk_m.ema_stress, color=ORANGE, lw=2, marker="o", ms=5)
axes[1].set_ylabel("EMA stress (1 great - 5 stressed)")
axes[1].set_xlabel("term week")
for w, n in n_stress.items():
    axes[1].annotate(f"n={n}", (w, wk_m.ema_stress.get(w, np.nan)), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7, color=INK2)
save(fig, "06_term_trend")
print("\n== EMA stress answers per week\n", n_stress.to_string())

# -------------------------------------------------- 7. missingness of daily table
miss = daily.drop(columns=["uid", "day"]).isna().mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(miss.index, miss.values * 100, color=BLUE, height=.7)
for i, v in enumerate(miss.values):
    ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=8, color=INK2)
ax.set_xlabel("% of student-days missing")
ax.set_xlim(0, 105)
ax.set_title("Missingness per feature (student-day table)")
save(fig, "07_missingness")

# ------------------------------------------------------- 8. top apps by snapshots
tops = []
for p in glob.glob(f"{RAW}/app_usage/*.csv"):
    a = pd.read_csv(p, encoding="utf-8-sig", usecols=["timestamp", "RUNNING_TASKS_topActivity_mPackage"])
    tops.append(a.groupby("timestamp").head(1).RUNNING_TASKS_topActivity_mPackage)
tops = pd.concat(tops).value_counts()
top15 = tops.head(15)[::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
colors = [ORANGE if ("paco" in k or "launcher" in k or "home" in k) else BLUE for k in top15.index]
ax.barh(top15.index, top15.values / tops.sum() * 100, color=colors, height=.7)
ax.set_xlabel("% of all 20-min snapshots where the app is on top")
ax.set_title("Most frequent top-of-stack app (orange = study app / home screen, excluded)")
ax.tick_params(axis="y", labelsize=8)
save(fig, "08_top_apps")
print("\n== top app share: paco %.1f%%, any launcher/home %.1f%%" % (
    100 * tops[tops.index.str.contains("paco")].sum() / tops.sum(),
    100 * tops[tops.index.str.contains("launcher|home|trebuchet")].sum() / tops.sum()))

# -------------------------------- 9. sanity check: proxy vs self-reported sleep
s = d_ok.dropna(subset=["ema_sleep_h"])
print("\n== Spearman: night lock minutes vs EMA sleep hours (n=%d): %.2f" % (
    len(s), s.night_lock_min.corr(s.ema_sleep_h, method="spearman")))
print("== Spearman: daily app switches vs EMA stress (n=%d): %.2f" % (
    d_ok.ema_stress.notna().sum(), d_ok.app_switches_real.corr(d_ok.ema_stress, method="spearman")))
