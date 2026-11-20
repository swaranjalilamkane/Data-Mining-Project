"""
Checkpoint 5: early methodology on the processed tables (run eda/build_features.py first).

  1. per-student usage features      (one row per student, from good-quality days)
  2. density estimation (KDE)        bandwidth from Silverman's and Scott's rules
  3. clustering                      k-means on the standardised features, k chosen by silhouette
  4. cluster profiles                what the students in each cluster look like
  5. timing                          how each step scales

Figures go to analysis/figures/, printed results to the terminal.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "eda", "processed")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 0

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
CLUSTER_COLORS = [BLUE, ORANGE, AQUA, PURPLE]
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


timings = {}


def timed(name):
    """Small context helper: with timed("kde"): ... records wall time per step."""
    class T:
        def __enter__(self):
            self.t = time.perf_counter()

        def __exit__(self, *a):
            timings[name] = time.perf_counter() - self.t
    return T()


hourly = pd.read_parquet(f"{P}/hourly.parquet")
daily = pd.read_parquet(f"{P}/daily.parquet")

# ------------------------------------------------- 1. per-student usage features
# A day only counts if the phone was sensing >= 18 h and the app logger took >= 36 of
# its 72 snapshots; otherwise a "quiet" day may just be a phone that was off.
with timed("features"):
    good = daily[(daily.sensing_hours >= 18) & (daily.app_snapshots >= 36)]
    good_days = good[["uid", "day"]]
    h = hourly.assign(day=hourly.hour.dt.normalize(), hod=hourly.hour.dt.hour) \
              .merge(good_days, on=["uid", "day"])
    part = pd.cut(h.hod, [0, 6, 12, 18, 24], right=False, labels=["night", "morning", "afternoon", "evening"])
    by_part = h.groupby(["uid", part], observed=True).app_switches_real.sum().unstack()
    frac = by_part.div(by_part.sum(axis=1), axis=0)

    g = good.groupby("uid")
    feats = pd.DataFrame({
        "switches_per_day": g.app_switches_real.median(),
        "nights_with_use": g.night_switches.apply(lambda s: (s > 0).mean()),
        "night_frac": frac["night"],
        "morning_frac": frac["morning"],
        "evening_frac": frac["evening"],      # afternoon is left out: the 4 parts sum to 1
        "social_frac": g.app_social_snaps.sum() / g.app_snapshots.sum(),
        "long_lock_h": g.lock_min.median() / 60,
    })
    # Context is NOT used for clustering; it describes the clusters afterwards.
    context = pd.DataFrame({
        "convo_h": g.convo_min.median() / 60,
        "moving": g.act_moving.mean(),
        "voice": g.aud_voice.mean(),
        "good_days": g.size(),
        # day-to-day variability; kept out of clustering because it is ~ -0.94 correlated
        # with switches_per_day (a low mean with some zero days mechanically gives a high CV)
        "switches_cv": g.app_switches_real.std() / g.app_switches_real.mean(),
    })
print("== feature table", feats.shape, "| good days per student: min %d, median %d"
      % (context.good_days.min(), context.good_days.median()))
print(feats.describe().T.round(3).to_string())
print("\n== Spearman correlation between features\n", feats.corr("spearman").round(2).to_string())

# Counts are right-skewed (a few heavy users), so log them before standardising;
# otherwise one student dominates the distances k-means works with.
X = feats.copy()
X["switches_per_day"] = np.log1p(X.switches_per_day)
X["long_lock_h"] = np.log1p(X.long_lock_h)
Z = StandardScaler().fit_transform(X)

# ------------------------------------------------------- 2. density estimation
def scott(x):
    """Scott's rule: assumes roughly bell-shaped data."""
    return 1.06 * x.std() * len(x) ** (-1 / 5)


def silverman(x):
    """Silverman's rule: like Scott's, but uses the IQR when it is smaller, so a skewed
    tail or outliers do not widen the bandwidth."""
    return 0.9 * min(x.std(), np.subtract(*np.percentile(x, [75, 25])) / 1.34) * len(x) ** (-1 / 5)


def modes(kde, xs):
    y = np.exp(kde.score_samples(xs[:, None]))
    return np.round(xs[1:-1][np.diff(np.sign(np.diff(y))) < 0], 2)


with timed("kde"):
    day_sw = np.log1p(good.app_switches_real.values)
    nw = feats.nights_with_use.values
    bws = {}
    for name, x in [("daily switches", day_sw), ("nights with use", nw)]:
        bws[name] = {"Silverman": silverman(x), "Scott": scott(x)}

print("\n== KDE bandwidths (share of zero days %.2f)" % (good.app_switches_real == 0).mean())
xs_n = np.linspace(-.1, 1.1, 600)
for name, row in bws.items():
    for rule, bw in row.items():
        x = day_sw if name == "daily switches" else nw
        grid = np.linspace(x.min() - .1, x.max() + .1, 600)
        peaks = modes(KernelDensity(bandwidth=bw).fit(x[:, None]), grid)
        if name == "daily switches":          # back from log(1 + x) to switches per day
            peaks = np.round(np.expm1(peaks), 1)
        print("   %-16s %-10s %.3f  peaks at %s" % (name, rule, bw, peaks))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
xs = np.linspace(0, day_sw.max() + .3, 300)
axes[0].hist(day_sw, bins=30, density=True, color=BLUE, alpha=.3, edgecolor="#fcfcfb")
for rule, color, ls in [("Silverman", BLUE, "-"), ("Scott", INK2, "--")]:
    bw = bws["daily switches"][rule]
    y = np.exp(KernelDensity(bandwidth=bw).fit(day_sw[:, None]).score_samples(xs[:, None]))
    axes[0].plot(xs, y, color=color, lw=2, ls=ls, label=f"{rule} ({bw:.2f})")
ticks = [0, 1, 3, 10, 30, 100]
axes[0].set_xticks(np.log1p(ticks), ticks)
axes[0].set_xlabel("app switches per day (log scale)")
axes[0].set_ylabel("density")
axes[0].set_title(f"Daily phone use, all {len(day_sw):,} good student-days")
axes[0].legend(frameon=False, fontsize=8, title="bandwidth rule", title_fontsize=8)
axes[1].hist(nw, bins=12, range=(0, 1), density=True, color=ORANGE, alpha=.3, edgecolor="#fcfcfb")
for rule, color, ls in [("Silverman", ORANGE, "-"), ("Scott", INK2, "--")]:
    bw = bws["nights with use"][rule]
    y = np.exp(KernelDensity(bandwidth=bw).fit(nw[:, None]).score_samples(xs_n[:, None]))
    axes[1].plot(xs_n, y, color=color, lw=2, ls=ls, label=f"{rule} ({bw:.2f})")
axes[1].plot(nw, np.full_like(nw, -0.05), "|", color=INK2, ms=10)
axes[1].set_xlabel("share of nights with phone use (00-06)")
axes[1].set_title("Late-night use across the 49 students")
axes[1].legend(frameon=False, fontsize=8, title="bandwidth rule", title_fontsize=8)
save(fig, "01_kde")

# ------------------------------------------------------------- 3. clustering
# k-means runs directly on the 7 standardised features; k is chosen by silhouette.
S = Z
with timed("kmeans_k_search"):
    ks = range(2, 7)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=50, random_state=SEED).fit(S)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(S, km.labels_))
best_k = ks[int(np.argmax(sil))]
km = KMeans(best_k, n_init=50, random_state=SEED).fit(S)
labels = km.labels_
# order clusters by switches per day so the numbering is stable between runs
order = np.argsort([feats.switches_per_day.values[labels == c].mean() for c in range(best_k)])
labels = np.argsort(order)[labels]
print("\n== k-means: silhouette by k", {k: round(float(v), 3) for k, v in zip(ks, sil)}, "-> k =", best_k)
print("   inertia by k", {k: round(float(v), 1) for k, v in zip(ks, inertia)})
print("   cluster sizes", np.bincount(labels).tolist())
print("   good days per cluster (median):", context.groupby(labels + 1).good_days.median().to_dict(),
      "| students with < 40 good days per cluster:", context.groupby(labels + 1).good_days
      .apply(lambda s: (s < 40).sum()).to_dict())

# Names come from reading the profile table below (only meaningful for k = 3).
GROUP_NAMES = ["Light, irregular", "Light, regular sleepers", "Heavy late-night"]
names = [f"{c + 1}. {GROUP_NAMES[c]}" if best_k == 3 else f"cluster {c + 1}" for c in range(best_k)]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(list(ks), inertia, color=INK2, marker="o", ms=7, lw=2)
axes[0].set_xticks(list(ks))
axes[0].set_xlabel("number of groups k")
axes[0].set_ylabel("inertia (lower = tighter groups)")
axes[0].set_title("Elbow: falls steadily, no clear bend")
axes[1].plot(list(ks), sil, color=BLUE, marker="o", ms=7, lw=2)
axes[1].plot(best_k, sil[list(ks).index(best_k)], "o", ms=13, mfc="none", mec=ORANGE, mew=2)
axes[1].set_xticks(list(ks))
axes[1].set_ylim(0, .5)      # full scale, so small differences don't look big
axes[1].set_xlabel("number of groups k")
axes[1].set_ylabel("silhouette (higher = more separate)")
axes[1].set_title("Silhouette: k = %d is highest, but all are close" % best_k)
save(fig, "02_choosing_k")

# 7 features can't be drawn at once, so the scatter shows the two that separate the groups most visibly
px, py = np.log1p(feats.switches_per_day.values), feats.nights_with_use.values
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for c in range(best_k):
    m = labels == c
    ax.scatter(px[m], py[m], s=70, color=CLUSTER_COLORS[c], edgecolor="#fcfcfb", linewidth=1.5,
               label=f"{names[c]} ({m.sum()} students)", zorder=3)
sw_ticks = [0, 1, 3, 10, 30]
ax.set_xticks(np.log1p(sw_ticks), sw_ticks)
ax.set_xlabel("app switches per day (log scale)")
ax.set_ylabel("share of nights with phone use")
ax.set_title("Each dot is one student, coloured by k-means group")
ax.legend(frameon=False, fontsize=9, loc="upper left")
save(fig, "03_clusters")

# ------------------------------------------------- 4. what each cluster looks like
prof = feats.assign(cluster=labels + 1).join(context).groupby("cluster")
print("\n== cluster profiles (median per cluster)\n",
      prof.median().T.round(3).to_string())

# Profile table as a heatmap: the text is the group's median in real units, the colour is how far
# that is from the average student (in standardised units), so one picture shows all 7 features.
rows = [("switches_per_day", "App switches per day", "{:.0f}"),
        ("nights_with_use", "Nights with phone use", "{:.0%}"),
        ("night_frac", "Share of use at night (00-06)", "{:.0%}"),
        ("morning_frac", "Share of use in the morning", "{:.0%}"),
        ("evening_frac", "Share of use in the evening", "{:.0%}"),
        ("social_frac", "Social-app snapshots", "{:.1%}"),
        ("long_lock_h", "Long-lock hours per day", "{:.0f} h")]
Zdf = pd.DataFrame(Z, index=feats.index, columns=feats.columns)
heat = np.array([[Zdf.loc[labels == c, col].median() for c in range(best_k)] for col, _, _ in rows])
fig, ax = plt.subplots(figsize=(8, 4.4))
im = ax.imshow(heat, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
for i, (col, _, fmt) in enumerate(rows):
    for c in range(best_k):
        ax.text(c, i, fmt.format(feats.loc[labels == c, col].median()), ha="center", va="center",
                fontsize=10, color=INK)
ax.set_xticks(range(best_k), [f"{n}\n({(labels == c).sum()} students)" for c, n in enumerate(names)], fontsize=9)
ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=9)
ax.tick_params(length=0)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("What a typical student in each group looks like (median)")
cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
cb.set_label("below average  <->  above average", fontsize=8)
cb.outline.set_visible(False)
save(fig, "04_cluster_profiles")

# ------------------------------------------------------------------ 5. timing
print("\n== timing (s):", {k: round(v, 3) for k, v in timings.items()})

feats.assign(cluster=labels + 1).join(context).to_csv(os.path.join(HERE, "student_features.csv"))
