# Codewalk 1: StudentLife data checkpoint
Swaranjali Lamkane · CS 6220 · 10/30

## What I've done so far
I explored the StudentLife dataset (Dartmouth, spring 2013) and built a preprocessing pipeline
(`eda/build_features.py`, `eda/explore.py`). The raw data has **49 students** and covers **65 days
(Mar 27 – Jun 1, 2013)**. It is split into about 1,500 per-student files across 10 passive sensing
streams, app/SMS/call logs, 26 EMA question types (JSON), 8 pre/post surveys, and grades for 30 students.
The sensor files are inference labels, not raw signals. Audio has about 99 M rows (one label per second:
silence, voice or noise). Activity has about 23 M rows (one label every 2–3 s: stationary, walking,
running or unknown). App usage has 2 M rows. The pipeline reduces all of this to one shared table with
one row per student per local hour (77,616 rows × 23 columns, 14 MB). From that it builds a
student-day table (3,234 rows) and a per-student table of survey scores and GPA.

## Preprocessing decisions (and why)
- **Time:** timestamps are UTC epoch seconds, so I converted them to America/New_York. Without this,
  "late night" would be shifted by 4 hours. Each day of the term starts exactly at 04:00 UTC, which is
  local midnight. I clipped the data to the term, because some phones kept logging until Jan 2014.
- **Formatting:** the files have several quirks. Some start with a UTF-8 BOM, some headers have
  leading spaces, and every GPS and wifi row ends with a trailing comma. Audio and activity have a
  blank line after every row, so `wc -l` doubles the count. The app files are not in time order, so
  I sort them. About 5% of audio timestamps are duplicates, which I drop.
- **App usage is not a usage log.** It is a snapshot of the running-task list every 20 minutes. My
  interaction proxy counts how often the foreground app changes between snapshots. I don't count
  changes caused by the study's own survey app (Paco), which is on top in 40% of snapshots.
- **Missing vs. zero:** for sensing hours with no data, I keep the value missing; they are not zero
  activity. 26 students' SMS files and 29 call logs contain only logger "pings" with no message or call
  content. I mark those students as unobserved, not "0 texts". (A first version of the pipeline turned
  them into 0 by accident when summing by day. I caught that in the missingness plot.)
- **Recoding:** the EMA stress codes are not in order (4–5 mean "feeling good"), so I recoded them to
  a 1 (great) to 5 (stressed out) scale. Sleep answers are category codes, which I converted to hours.
  Survey scores reverse the positively worded items on the PSS and UCLA loneliness scales.

## Scale vs. compute
I process one student at a time: read each file with only the columns I need, reduce it to hourly
values, then discard the raw rows. The whole build takes **18 s with 1.4 GB peak memory** on my
laptop, so I didn't need to sample. Later techniques (PCA, k-means, association rules) will run on
the small hourly and daily tables, not on the raw rows.

## What surprised me
1. **I can't measure session length directly.** Phone-lock intervals were only logged when the phone
   was locked for at least 60 minutes, and they are capped at 10 hours (all 9,275 intervals fall in
   that range). App data only has 20-minute resolution. So my first research question, the
   distribution of session lengths, has to be reframed. I'll work with interaction frequency and
   long idle periods instead of session length.
2. **The data is smaller and sparser than I assumed.** My local copy is 2.8 GB, not the 52.6 GB in my
   proposal. Social media rarely shows up (Facebook is in under 1% of snapshots, since this is 2013).
   12 students have less than 70% sensing coverage. EMA responses drop off sharply, from about 200 a
   week at the start of term to 8 in week 10.
3. **A sanity check came back null.** Nightly locked minutes don't line up with self-reported sleep
   hours (Spearman ρ ≈ 0.00, n = 1,133 days). The lock proxy alone is a weak sleep measure.

## Where things stand / next
The cleaned tables and 8 EDA figures are in `eda/processed/` and `eda/figures/`. Phone interaction
varies a lot between students: the median ranges from 0 to 26 app switches a day. The share of
nights with any phone use between 00:00 and 06:00 ranges from 0% to 93% (median 32%). Next I'll
build per-student features (frequency, time of day, night use, context shares), then apply
PCA → k-means → association rules.
