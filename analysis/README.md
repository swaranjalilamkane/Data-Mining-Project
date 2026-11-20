# Checkpoint 5: Full Guide

This is the detailed, step-by-step guide to the Checkpoint 5 analysis: what each step does, the idea
behind it, the code, the decisions and why, and the results. At the end there is a list of questions
with their answers, for the codewalk.

The short write-up for submission is [CHECKPOINT5_SUMMARY.md](CHECKPOINT5_SUMMARY.md).

## Files

| File | What it is |
|---|---|
| `student_profiles.py` | the whole Checkpoint 5 analysis |
| `output.txt` | everything the script prints |
| `student_features.csv` | one row per student: the 7 features, the group, and context |
| `figures/01_kde.png` | KDE curves |
| `figures/02_choosing_k.png` | elbow and silhouette |
| `figures/03_clusters.png` | students coloured by group |
| `figures/04_cluster_profiles.png` | what a typical student in each group looks like |
| `CHECKPOINT5_SUMMARY.md` | the short write-up |

Run it from the `Data-Mining-Project` folder (after `eda/build_features.py` has been run once):

```bash
.venv/bin/python analysis/student_profiles.py
```

## The pipeline in one line

**good days → one row per student → KDE → log and standardise → k-means**

My question: **do students fall into different types of phone users?** Every step below serves that
question.

---

## Step 1: Keep only good days

### The idea
Before measuring behaviour, make sure every day really measured behaviour. A day where the phone was off
is not the same as a day where the student didn't use the phone, but in the data both show "0 switches".

### Where the numbers come from

Two columns, built in Checkpoint 4 from two different sensors:

| Column | Built from | Code | Meaning |
|---|---|---|---|
| `sensing_hours` | activity sensor (`sensing/activity/`) | `eda/build_features.py:154` and `:222` | hours (0–24) that day where the phone wrote at least one activity reading. The activity sensor writes every 2–3 seconds when the phone is on, so it is the best sign the phone was alive |
| `app_snapshots` | app logger (`app_usage/`) | `eda/build_features.py:110` and `:223` | how many app snapshots were taken that day. One every 20 minutes, so a full day has 72 |

Why two checks: all my phone-use features come from the app snapshots. The phone can be alive while the
app logger is broken, and that day would wrongly look like zero use.

### The code (`student_profiles.py:68-70`)
```python
good = daily[(daily.sensing_hours >= 18) & (daily.app_snapshots >= 36)]   # keep a day only if both are true
good_days = good[["uid", "day"]]                                          # list of good (student, day) pairs
h = hourly.assign(...).merge(good_days, on=["uid", "day"])                # keep only hours from good days
```
The hourly table is filtered too, because the time-of-day shares are built from it. That way both
tables use the same days.

### Real example: student u39

| Day | Sensing hours | Snapshots | Switches | Kept? |
|---|---|---|---|---|
| Apr 6 | 20 | 58 | 0 | yes: phone on most of the day, 0 is real |
| Apr 7 | 0 | 0 | 0 | no: phone off, 0 means nothing |
| Apr 9 | 6 | 17 | 2 | no: only 6 hours recorded |
| Apr 13 | 24 | 72 | 1 | yes: full day |

April 7 and April 10 both show 0 switches. On April 7 the phone was off; on April 10 it was on all day
and the student really didn't switch apps. Without the filter they look the same.

### Decision: why 18 hours and 36 snapshots

How many hours the phone recorded, over all 3,234 student-days:

| Sensing hours | Days |
|---|---|
| 0 (phone off all day) | 368 |
| 1–17 | 373 |
| 18–23 | 292 |
| 24 (full day) | 2,201 |

- Most days are either full or empty, so the cut-off doesn't have to be exact.
- 18 hours = 75% of the day. Mostly recorded, but a short gap (a dead battery for an hour) doesn't throw
  away an otherwise good day.
- 36 snapshots = half of 72. That's enough app data to count switches. It removes 33 extra days where the
  phone was on but the app logger wasn't working.
- A stricter rule (a full 24 hours) would remove about 290 more days and leave some students with very
  few days, which matters with only 49 students.

### Result
3,234 − 741 (under 18 hours) − 33 (too few snapshots) = **2,460 good days**. Every student keeps at least
17 (u39 has the fewest), the median student has 54 of 66, so nobody is dropped.

---

## Step 2: One row per student

### The idea
I'm grouping **students**, so each student needs exactly one row of numbers that describes their habits.
Their 17–66 good days are summarised into 7 phone-use features. This is called feature engineering.

### Where the features come from

| Feature | Table | Column | How | Plain meaning |
|---|---|---|---|---|
| `switches_per_day` | daily | `app_switches_real` | median over good days | how often they use the phone on a normal day |
| `nights_with_use` | daily | `night_switches` | share of days where it's > 0 | on how many nights (00–06) they use the phone |
| `night_frac` | hourly | `app_switches_real`, hours 0–6 | share of all their switches | how much of their use is at night |
| `morning_frac` | hourly | same, hours 6–12 | share | how much is in the morning |
| `evening_frac` | hourly | same, hours 18–24 | share | how much is in the evening |
| `social_frac` | daily | `app_social_snaps` ÷ `app_snapshots` | totals over good days | how often a social app was on screen |
| `long_lock_h` | daily | `lock_min` | median ÷ 60 | hours a day the phone sat locked for an hour or more, mostly sleep |

The time-of-day shares need the hourly table, because the daily table only has whole-day and night
totals.

### The code (`student_profiles.py:72-95`)
```python
part = pd.cut(h.hod, [0, 6, 12, 18, 24], right=False,
              labels=["night", "morning", "afternoon", "evening"])            # put each hour in a part of the day
by_part = h.groupby(["uid", part], observed=True).app_switches_real.sum().unstack()   # switches per part
frac = by_part.div(by_part.sum(axis=1), axis=0)                               # turn into shares that add up to 1

g = good.groupby("uid")                                                       # one group per student
feats = pd.DataFrame({
    "switches_per_day": g.app_switches_real.median(),
    "nights_with_use":  g.night_switches.apply(lambda s: (s > 0).mean()),
    "night_frac":   frac["night"],
    "morning_frac": frac["morning"],
    "evening_frac": frac["evening"],
    "social_frac":  g.app_social_snaps.sum() / g.app_snapshots.sum(),
    "long_lock_h":  g.lock_min.median() / 60,
})
```

### Worked example: student u00 (54 good days)

| Feature | Calculation | Result |
|---|---|---|
| `switches_per_day` | median of 54 days | 11.5 |
| `nights_with_use` | 28 nights with use ÷ 54 | 0.52 |
| `night_frac` | 81 night switches ÷ 689 total | 0.12 |
| `morning_frac` | 204 ÷ 689 | 0.30 |
| `evening_frac` | 166 ÷ 689 | 0.24 (the other 34% is afternoon) |
| `social_frac` | 14 social snapshots ÷ 3,763 | 0.004 |
| `long_lock_h` | 615.5 min ÷ 60 | 10.3 h |

For comparison, u02 has 19 switches a day and uses the phone on 82% of nights (a heavy, social user), and
u39 has 0 switches a day and uses it on 6% of nights (very light).

### Decisions and why

1. **These 7, because each matches part of the proposal question** ("profiles by session frequency,
   time-of-day usage and nighttime activity"): how often → switches; late night → nights with use, night
   share; time of day → morning, evening share; social media (the original motivation) → social share;
   sleep and idle time → long-lock hours.
2. **Median, not mean.** One unusual day shouldn't change a student's profile. u00's mean is 12.8 but the
   median is 11.5, because a few busy days pull the mean up.
3. **Two night features, because they measure different things.** `nights_with_use` is how *regularly*
   (on how many nights). `night_frac` is how *much* of their use happens at night.
4. **Shares for time of day, not counts.** Counts would mostly repeat "how much" (a heavy user has more of
   everything). Shares keep **when** separate from **how much**.
5. **Afternoon left out.** The 4 shares add up to 1, so afternoon = 1 − the other three. It adds nothing.
6. **`social_frac` divided by snapshots, not days**, so it adjusts for how much app data each student has.
7. **Movement, conversation and audio kept aside** (the `context` table). They describe the student's
   surroundings, not their phone use. Keeping them out means I can later ask "do heavy users move less?"
   without circular reasoning.
8. **Day-to-day variability (`switches_cv`) dropped.** It moved almost exactly opposite to switches per day
   (correlation −0.94): a low average with some zero days automatically gives a high value. It said the
   same thing twice.
9. **SMS and calls left out**, because nothing was logged for 26 and 29 students.
10. **Stress and sleep surveys left out**, because those are what I want to *compare* the groups with later,
    so they can't be used to *make* the groups.

**Caveat:** for very light users, the time-of-day shares rest on only a few switches (u39's evening share
of 0.75 comes from a handful), so they are less reliable than a heavy user's.

### How many columns were available, and why only 7

The daily table has 15 features and the hourly table has 21. Every one has a role:

| Used for | Columns |
|---|---|
| filtering good days / hours | `sensing_hours`, `app_snapshots`, `sensing_ok` |
| building the 7 features | `app_switches_real`, `night_switches`, `app_social_snaps`, `lock_min` |
| describing the groups (context) | `convo_min`, `act_moving`, `aud_voice` |
| not used, and why | `dark_min`, `night_lock_min` (overlap with `lock_min`); `sms_n`, `call_n` (too much missing); `ema_sleep_h`, `ema_stress` (for comparing later); the other activity/audio codes, `charge_min`, `app_switches` (includes the Paco survey app), `app_distinct` (82% missing) |

Why only 7: they're about phone use only, have little missing data, don't repeat each other, and each one
matches a research question. With only 49 students, a few clear features work better than 20 noisy ones.

---

## Step 3: KDE (look at the shape of the data first)

### The idea
A **density curve** shows where values are concentrated: high where many days or students have that
value. The area under the curve is 1.

A **histogram** counts values in bars, but its shape changes depending on how wide the bars are and
where they start. **KDE (kernel density estimation)** instead:

1. puts a small bell curve (a "kernel") on top of every data point
2. adds them all up and divides by the number of points

$$\hat f(x) = \frac{1}{n\,h}\sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)$$

$K$ is the bell shape (Gaussian), $x_i$ are the data points, and $h$ is the **bandwidth**: how wide each
bell is.

| Bandwidth | Result |
|---|---|
| too small | a spike for every point (shows noise) |
| too large | one flat blob (hides real bumps) |
| right | the real shape |

### Choosing the bandwidth: Scott's and Silverman's rules
Both are standard formulas, based on how spread out the data is and how many points there are. Both
assume the data is roughly bell-shaped.

- **Scott:** h = 1.06 × σ × n^(−1/5). Uses the standard deviation.
- **Silverman:** h = 0.9 × min(σ, IQR ÷ 1.34) × n^(−1/5). Uses the interquartile range instead when it is
  smaller, so a long tail or outliers don't make the bandwidth too wide. That suits lopsided data.

### The code (`student_profiles.py:109-165`)
```python
def scott(x):
    return 1.06 * x.std() * len(x) ** (-1 / 5)

def silverman(x):
    return 0.9 * min(x.std(), np.subtract(*np.percentile(x, [75, 25])) / 1.34) * len(x) ** (-1 / 5)

day_sw = np.log1p(good.app_switches_real.values)    # daily switches, on a log(1 + x) scale
nw = feats.nights_with_use.values                   # share of nights with use, one per student
```
Then `KernelDensity(bandwidth=...)` draws each curve.

### Decisions and why
- **KDE before anything else**, to see whether the data is lopsided, has many zeros or more than one bump.
  That decides how to prepare the data and whether looking for groups makes sense.
- **KDE over just a histogram**, because the smoothness comes from a rule, not from where I cut the bars.
  The histogram is still drawn underneath as a check.
- **Gaussian kernel**, because it's smooth and standard; with enough data the bandwidth matters far more
  than the kernel shape.
- **Silverman's rule, checked against Scott's.** Both gave almost the same curves, so the result doesn't
  depend on that choice:

  | Data | Silverman (used) | Scott | Peaks |
  |---|---|---|---|
  | daily switches | 0.21 | 0.25 | at 0 and around 12 switches a day |
  | share of nights with use | 0.11 | 0.13 | one peak at about 16% |

- **Log scale for daily switches.** Most days have under 20 switches, but one has 93. Without the log,
  everything from 0 to 20 would be squashed into the left of the chart. I use log(1 + x) because 549 days
  are exactly 0 and log(0) doesn't exist. The axis labels (0, 1, 3, 10, 30, 100) are tick marks I chose,
  placed at log(1 + label), to show the real values.

### Reading the plot (`figures/01_kde.png`)

**Left panel: daily phone use, all 2,460 good days**
- x axis: app switches per day (log scale). y axis: density.
- Light bars: histogram. Solid line: Silverman's KDE. Dashed line: Scott's KDE.

**Right panel: late-night use, 49 students**
- x axis: share of a student's nights with phone use, from 0 (never) to 1 (every night).
- Each small tick at the bottom is one student.

### What "density" on the y axis means
- **Bars:** bar height = (points in the bar) ÷ (total points × bar width). Dividing like this makes the
  total area of all bars = 1, so the bars are on the same scale as the KDE curve and can be drawn
  together. The area of a bar is the share of points in it.
  - Left: the first bar has 549 days. 549 ÷ (2,460 × 0.151) = 1.47. Its area, 1.47 × 0.151 = 0.22, is
    the 22% of zero days.
  - Right: 12 bars, each 1/12 ≈ 0.083 wide. The tallest bar is 2.2, which is 9 students
    (2.2 × 49 × 0.083 ≈ 9) who used the phone on 8–17% of their nights.
- **Curve:** at any point x, add up the heights of all the bells there and divide by the number of
  points. Checked by hand: at 0 switches the curve is 0.427, at 12 switches it is 0.340, exactly what the
  library gives.
- **Why density can be above 1:** it isn't a percentage. It's "share per unit of x". The right x axis only
  runs from 0 to 1, so to make the total area 1, some parts must be above 1.
- **Why the curve is lower than the bar at 0 (left):** all 549 zero days sit on one exact value. KDE
  spreads each into a bell, so the height gets shared with the neighbouring area.

I tried showing counts instead of density on the y axis, but kept density because it puts the bars and
curves on the same scale. For speaking, counts are still useful ("the tallest bar is 9 students").

### The bars on the right panel

| Share of nights | Students |
|---|---|
| 0.00 – 0.08 | 7 |
| 0.08 – 0.17 | 9 |
| 0.17 – 0.25 | 4 |
| 0.25 – 0.33 | 6 |
| 0.33 – 0.42 | 3 |
| 0.42 – 0.50 | 3 |
| 0.50 – 0.58 | 3 |
| 0.58 – 0.67 | 4 |
| 0.67 – 0.75 | 2 |
| 0.75 – 0.83 | 5 |
| 0.83 – 0.92 | 1 |
| 0.92 – 1.00 | 2 |

### Conclusions from KDE

1. **On many days students don't use the phone at all.** 549 of 2,460 good days (22%, about 1 in 5) had no
   app switches, even though the phone was recording. So "no use" is common, not rare. → Students are
   described by their median day and their share of nights, not one average that hides the zero days.
2. **On days with use, a typical day is about 12 switches.** After the zero peak the curve dips, then rises
   into a broad hill around 10–12. So there are "no use" days and "normal use" days, with few in between.
3. **Phone use is very lopsided (right-skewed).** Most days are under 20 switches, but a few go up to 93.
   → I take the log before k-means, so heavy users don't dominate.
4. **Most students rarely use the phone at night, but some do on most nights.** The peak is at about 16%
   of nights. About 24 students are below 30%, and about 14 use it on 60% or more of their nights, with 5
   around 75–83%. Students are clearly not all alike, which is what the profile question is looking for.
5. **KDE can't say whether the late-night users are a separate group.** There is one clear peak and a long
   tail, not two peaks. It could be two types of students, or one sliding scale. KDE only looks at one
   feature at a time, and with 49 students the tail rests on a few points. → I test it with k-means, which
   looks at all 7 features together.

Also: Silverman's and Scott's curves lie on top of each other, so these conclusions come from the data,
not from a setting I picked.

**Known quirk:** the right curve spills below 0 and above 1, although a share can't be negative. This is a
known boundary problem of KDE: the bells near the edges hang over.

---

## Step 4: Log and standardise

### The idea
k-means groups students by **how close they are** across all 7 numbers. That only works if every feature
is on a fair scale.

**Problem 1, different number sizes:**

| Feature | Range across students |
|---|---|
| switches per day | 0 to 27 |
| long-lock hours | 0 to 20 |
| nights with use | 0 to 0.97 |
| night share | 0 to 0.28 |

A difference of 10 switches looks huge next to a difference of 0.1 in night share, even if both matter
equally. Like comparing people by height in centimetres and shoe size: height wins every time just
because its numbers are bigger.

**Problem 2, extreme students:** most students have 0–10 switches a day, but one has 27. That student would
pull the groups toward them.

### The two fixes
- **Log, log(1 + x):** squeezes big numbers closer. 0 → 0, 3 → 1.4, 10 → 2.4, 27 → 3.3. The heaviest user
  is still the highest, just not far away on their own.
- **Standardise (z-score):** z = (value − average) ÷ typical spread. 0 means average, +1 a bit above, +2 well
  above, −1 a bit below. After this, every feature has an equal say.

### The code (`student_profiles.py:103-106`)
```python
X = feats.copy()
X["switches_per_day"] = np.log1p(X.switches_per_day)   # squeeze big values
X["long_lock_h"] = np.log1p(X.long_lock_h)
Z = StandardScaler().fit_transform(X)                  # every feature: average 0, spread 1
```

### Example

| Student | Raw switches | After log | After standardising |
|---|---|---|---|
| u31 (heaviest) | 27 | 3.33 | +1.96 |
| u02 | 19 | 3.00 | +1.60 |
| u39 (lightest) | 0 | 0.00 | −1.59 |

### A choice to revisit
Measuring how lopsided each feature is (skewness: near 0 = balanced) showed that the log fixed
`switches_per_day` (1.31 → 0.08), but `long_lock_h` was already balanced (−0.28) and the log made it more
lopsided (−2.36), while `social_frac` is the most lopsided feature (3.04) and isn't transformed. A better
choice would be: log switches, no log on long-lock hours, square root on the social share. A test showed
this keeps 3 groups and mostly the same students (agreement 0.75), with clearer separation. Not applied
yet.

---

## Step 5: k-means (look for groups)

### Why k-means, and why after KDE
1. **My research question asks for groups:** "Can distinct behavioural profiles be identified by clustering
   students?" Clustering was always the plan; KDE came first to check it was worth trying.
2. **KDE showed there is something to find:** students differ a lot at night. If everyone sat in one tight
   bump, there would be nothing to group.
3. **But KDE couldn't answer the question:** it looks at one feature at a time, and a "type" of student is a
   mix of features. It also showed one peak with a tail, which is unclear.
4. **So I needed a method that looks at all 7 features together and finds groups.**

Why k-means out of the clustering methods:

| Method | What it gives | For my data |
|---|---|---|
| **k-means** | each student in one group, each group with a centre | the centre is a typical student I can describe; simple, fast, planned in my proposal |
| GMM | soft groups (a probability of belonging to each) | harder to describe; a good check later |
| Hierarchical | a tree of groups | good for small data; a good check later |
| DBSCAN | dense clumps, the rest labelled noise | 49 students is too few to find dense areas reliably |

### How k-means works
Picture the 49 students as dots, placed by their 7 numbers, and k flags among them.

1. Place k centres (flags) at random.
2. Every student goes to the nearest centre. That makes k groups.
3. Move each centre to the middle (the average) of its group.
4. Repeat steps 2 and 3 until no student changes group.

The final centres describe a typical student in each group. k-means tries to make the total "walking
distance" from students to their centre as small as possible.

**Weakness:** the random start can change the result. **Fix:** I run it 50 times from different starts
(`n_init=50`) and keep the best. `random_state=SEED` makes the answer the same every run.

### Choosing k (how many groups)

k-means doesn't decide how many groups there are; I have to tell it. So I try several and score them.

**The code (`student_profiles.py:171-177`)**
```python
ks = range(2, 7)                       # try k = 2, 3, 4, 5, 6
inertia, sil = [], []                  # lists to store the two scores
for k in ks:
    km = KMeans(k, n_init=50, random_state=SEED).fit(S)   # run k-means with k groups, best of 50 starts
    inertia.append(km.inertia_)                           # score 1: for the elbow
    sil.append(silhouette_score(S, km.labels_))           # score 2: silhouette
best_k = ks[int(np.argmax(sil))]       # pick the k with the highest silhouette
```
Why 2 to 6: fewer than 2 isn't a grouping. More than 6 would mean groups of about 8 students, too small to
describe.

#### Score 1: inertia and the elbow

**Inertia** = for every student, the squared distance to their own group's centre, all added up.

$$\text{inertia} = \sum_{\text{all students}} (\text{distance to own group's centre})^2$$

- Lower means tighter groups. Checked by hand for k = 3: 188.8, the same as the library.
- **The catch:** more groups always lowers it. With 49 groups (one per student), it would be 0. So you
  can't just pick the lowest.
- **The elbow:** plot inertia against k and look for the point where the line suddenly flattens. Before it,
  each extra group splits a real group (big drop). After it, extra groups just cut real groups into pieces
  (small drop).

| k | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| inertia | 245 | 189 | 155 | 134 | 112 |
| drop from previous | – | 56 | 34 | 21 | 22 |

It falls steadily with **no clear bend**, so the elbow can't choose k for my data.

#### Score 2: silhouette

For each student:
- **a** = average distance to the other students in **their own group** (smaller is better)
- **b** = average distance to the students in the **nearest other group** (bigger is better)

$$s = \frac{b - a}{\max(a, b)}$$

| Score | Meaning |
|---|---|
| near +1 | much closer to own group: clearly in the right group |
| around 0 | equally close to two groups: on the border |
| below 0 | closer to another group: probably in the wrong group |

The average over all 49 students gives one score per k. It's better than inertia because it also checks
that groups are **separate from each other**, not just tight, and it doesn't automatically improve with
more groups, so you can simply pick the highest.

| k | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| silhouette | 0.255 | **0.267** | 0.254 | 0.244 | 0.255 |

**Decision: k = 3.** It's the highest, but all values are close (0.24–0.27). 0.27 is low to moderate;
above about 0.5 would mean clearly separate groups. So the groups exist, but they overlap.

**Worked examples (k = 3):**
- u31 (group 3): a = 4.94, b = 6.72 → (6.72 − 4.94) ÷ 6.72 = 0.27. In the right group, though not tightly.
- u00 (group 2): a = 2.76, b = 3.10 → (3.10 − 2.76) ÷ 3.10 = 0.11. Close to the border.
- Clearest students: u30 (0.53), u09 (0.52), u05 (0.50).
- Border students with a slightly negative score: u35 (−0.07), u23 (−0.04), u01 (−0.01).

| Group | Average silhouette | Meaning |
|---|---|---|
| 1 | 0.14 | weakest: its students are close to group 2 |
| 2 | 0.33 | clearest group |
| 3 | 0.28 | fairly clear |

**How the two scores work together:**

| | Elbow (inertia) | Silhouette |
|---|---|---|
| asks | are the groups tight? | are groups tight and separate? |
| with more groups | always goes down | can go up or down |
| how to read | look for a bend | pick the highest |
| my result | no clear bend, can't decide | k = 3 is best, but close |

### Reading the k-means output (`output.txt`)

```
cluster sizes [11, 21, 17]
```
3 groups of 11, 21 and 17 students. No tiny group, so the sizes are reasonable.

```
good days per cluster (median): {1: 37, 2: 56, 3: 55}
students with < 40 good days per cluster: {1: 7, 2: 1, 3: 3}
```
Group 1 has much less data: 7 of its 11 students have fewer than 40 good days. So group 1 might partly be
"students with less data", not only "light users".

**Group profiles (median student in each group, real units):**

| Feature | Group 1 | Group 2 | Group 3 |
|---|---|---|---|
| switches per day | 1 | 2 | 12 |
| nights with use | 16% | 21% | 71% |
| night share of use | 16% | 9% | 20% |
| morning share | 9% | 23% | 16% |
| evening share | 39% | 33% | 33% |
| social apps | 0% | 0.4% | 4.3% |
| long-lock hours | 6 h | 13 h | 11 h |

Read across each row for the number that stands out:

- **Group 3, heavy late-night (17 students):** 12 switches a day (about 6 times the others), phone use on 71%
  of nights, the most social-app use.
- **Group 2, light regular sleepers (21 students):** light use, long overnight locks (13 h, the phone sits
  still for a long time, probably while sleeping), the most morning use and the least night share. An
  early-bird routine.
- **Group 1, light irregular (11 students):** the lightest use, short locks (6 h), the least morning use,
  with more of their little use in the evening. And less data.

**Key insight:** groups 1 and 2 use the phone about equally little. What separates them is **when** and
**how regularly**. k-means found this because it looks at all 7 features together.

### Reading the charts

**`figures/02_choosing_k.png`: how many groups**
- Left (elbow): the line falls evenly from k = 2 to 6, no sudden bend.
- Right (silhouette): the orange circle marks k = 3, the highest. The y axis starts at 0, so you can see
  honestly that all the values are about the same.

**`figures/03_clusters.png`: the students and their groups**
- Each dot is one student. Left to right: switches per day (log scale). Bottom to top: share of nights
  with use.
- Green, top-right: heavy late-night users, clearly separate.
- Blue and orange, bottom-left: the two light groups. They overlap on this chart, because what separates
  them (long locks and morning use) isn't on these two axes.
- All the dots form one diagonal band, so students go smoothly from light to heavy. That's why the
  silhouette is only 0.27.

**`figures/04_cluster_profiles.png`: what each group looks like**
- Columns are the 3 groups, rows are the 7 features.
- The number in each box is the typical student (median) in real units.
- The colour shows how that compares with the average student: red = above, blue = below, white = about
  average.
- Groups 1 and 2 are both blue at the top (both light), but differ in the bottom rows (long locks, morning
  use). Group 3 is dark red at the top.

---

## Efficiency and scale

- The heavy work was in Checkpoint 4: over 100 million raw rows became small tables in about 20 seconds.
- This analysis works on 49 students and 2,460 days. The whole script takes about 2 seconds, and the
  analysis itself about 0.1 seconds (from `output.txt`):

  | Step | Time |
  |---|---|
  | building features | ~0.02 s |
  | KDE (bandwidth rules are one formula each) | ~0 s |
  | k-means search (5 values of k × 50 starts) | ~0.1 s |

- Drawing a KDE curve compares every grid point with every data point, so it grows with the data, but
  2,460 days is still instant.
- k-means grows roughly in step with the number of students × k × iterations, so it would stay fast with
  many more students. For very large data I would use mini-batch k-means, which works on small samples
  at a time.

---

## What I can try next and why

- **Check whether the groups are stable** by re-running k-means on random resamples of the students. The
  silhouette is only 0.27 and the other k values are close.
- **Compare with other clustering methods from class** (GMM, hierarchical clustering, k-medoids). If they
  find the same groups, the groups are real. If not, students may sit on a sliding scale.
- **Re-run without the students who have few good days**, because most of group 1 has fewer than 40.
- **Compare the groups with stress, depression, loneliness and GPA**, the wellbeing part of my question.
- **Add PCA before k-means.** Some features overlap: students who switch apps a lot also use the phone on
  more nights (correlation 0.87), and social share goes with both (about 0.75). So k-means counts that idea
  more than once. PCA merges overlapping features into new ones that don't overlap, so each idea is
  counted once.
- **Fix which features get the log** (see "A choice to revisit" in step 4).
- **Add location (`wifi_location`) and run association rules**, to see which situations go together with
  phone use.

---

## Questions and answers

**How are good days calculated, from which data, and why?**
From two daily columns built in Checkpoint 4: `sensing_hours` (from the activity sensor) and
`app_snapshots` (from the app logger). A day is kept if the phone recorded at least 18 hours and the app
logger took at least 36 of its 72 snapshots. Without this, a phone that was off looks like a student who
didn't use it. See step 1.

**How was 3,234 days calculated for 49 students?**
49 students × 66 days = 3,234. The term runs from March 27 to May 31, 2013 (5 + 30 + 31 = 66 days). In
Checkpoint 4 I built a full calendar of every hour of the term for every student, so every student has
all 66 days, even days where the phone recorded nothing. That keeps missing days visible.

**What does "every student has at least 17 good days" mean?**
After removing bad days, the student with the fewest good days (u39) still has 17. The median student has
54. So nobody had to be dropped.

**How were the 7 features decided, and from which table?**
Four from the daily table (switches, nights with use, social share, long locks) and three from the hourly
table (night, morning, evening shares). Each matches part of the research question. See step 2.

**How many features were in the daily and hourly tables, and why only 7?**
15 in the daily table and 21 in the hourly table. The rest are used for filtering, for describing the
groups, or left out because they overlap, have too much missing data, or are for comparing later. See
step 2.

**Did we use wifi and wifi_location?**
Not yet. `wifi_location` (which building the phone was in) is the planned next step for the location
question. Raw `wifi` and `gps` give the same kind of information in a messier form, and `bluetooth`,
`calendar`, `dinning`, the other 24 EMA types and 5 of the 8 surveys aren't needed for the current
questions.

**Why KDE, and why Silverman's and Scott's rules?**
To see the shape of the data before choosing methods. The two rules are standard formulas for the
bandwidth; Silverman's handles lopsided data a little better, and both gave almost the same curves.

**Was cross-validation in the syllabus?**
Not by name. The syllabus lists histograms, KDE and MLE. I first used cross-validation to choose the
bandwidth, then removed it and kept only the rules of thumb, which are simpler and gave the same
conclusions.

**Where does the 22% come from?**
549 of the 2,460 good days had zero switches. 549 ÷ 2,460 = 0.223.

**Where does the 0 to 100 on the left x axis come from?**
The data goes from 0 to 93 switches a day. The labels 0, 1, 3, 10, 30, 100 are tick marks I chose. The
curve is drawn on a log(1 + x) scale, and each label is placed at log(1 + label).

**How is density calculated, and what is the y axis on the second curve?**
Both panels show density: bar height = (points in the bar) ÷ (total points × bar width), so the total
area is 1 and the bars match the KDE curve. On the right panel it is a density of students; the tallest
bar (2.2) is 9 students. See step 3.

**What do we conclude from the KDE curves?**
Many days have no use (22%); use is very lopsided, so take the log; most students rarely use the phone at
night, but some do on most nights; and KDE can't say whether those are a separate group, so test with
k-means. See step 3.

**Why k-means after KDE?**
KDE showed students differ, but it looks at one feature at a time and couldn't tell whether there are
separate types. k-means looks at all 7 together and gives describable group centres, which is what the
research question asks for. See step 5.

**Why isn't PCA used?**
I tried it, but took it out for now to keep the pipeline simple. k-means runs directly on the 7
standardised features, and found the same 3 groups. PCA is a planned next step, because some features
overlap.

**What are the elbow and the silhouette?**
Two ways to choose how many groups. The elbow looks for where adding groups stops helping; mine had no
bend. The silhouette checks whether each student is closer to their own group than the next one; k = 3
scored highest (0.27). See step 5.

**What are `act_share_0`, `aud_share_1` and similar columns?**
From Checkpoint 4. They are the share of an hour the phone guessed each activity (0 still, 1 walking,
2 running, 3 unknown) or each sound (0 silence, 1 voice, 2 noise). `_samples` is how many readings that
hour. They are named in `eda/build_features.py:72-73`.

---

## One-minute version for the professor

"I keep only days where the phone was really recording, then describe each student with 7 phone-use
numbers: how often, how late, when in the day, how social, and how long the phone is put down. KDE showed
that daily use is very lopsided with many zero days, so I take the log, and that most students rarely use
the phone at night while some do on most nights, but it couldn't say whether those are separate groups.
So I standardised the features and ran k-means. The elbow had no clear bend, so I chose k by silhouette:
k = 3 was best, at 0.27, but the values were close, so the groups overlap. The three groups are heavy
late-night users, light regular sleepers and light irregular users. The two light groups differ in their
daily rhythm, not in how much they use the phone."
