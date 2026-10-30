# Codewalk 1: StudentLife data checkpoint
Swaranjali Lamkane · CS 6220 Data Mining · 10/30

**What I've done.** I explored the StudentLife dataset (Dartmouth, spring 2013). The raw data covers
**49 students over 65 days**, in about 1,500 per-student files. There are 10 sensing streams (sensor
inference labels, not raw signals: about 99 M audio rows and 23 M activity rows), app snapshots
(2 M rows), SMS and call logs, 26 EMA question types (JSON), 8 pre/post surveys, and GPA for 30
students. I merged everything into three tables: one row per **student-hour** (77,616 × 23), one
per **student-day** with the EMA answers (3,234 rows), and one per **student** with survey scores
and GPA (49 rows). The streams have very different sampling (1 s labels, 20-minute snapshots,
start–end intervals, events), so I couldn't join on raw timestamps. I summarised each stream per
hour and left-joined them onto a complete hourly grid, so hours with no data stay visible as missing.

**Preprocessing decisions.**
- Converted timestamps from UTC to Dartmouth local time, otherwise "late night" is 4 hours off.
  Clipped the data to the term.
- Fixed formatting quirks: BOM characters, spaces in headers, trailing commas, a blank line after
  every row, unsorted app files, and about 5% duplicate audio timestamps.
- The app data is a snapshot every 20 minutes, not a usage log. My usage measure counts how often
  the front app changes, excluding the study's own survey app (on top in 40% of snapshots).
- Missing is not zero: hours with no sensing stay missing, and the 26 SMS and 29 call logs that have
  no content are marked unobserved. A first version summed those to 0 by mistake; the missingness
  plot caught it.
- Recoded EMA answers per response (the stress codes are out of order; sleep answers are codes,
  not hours) and reverse-scored the positively worded PSS and loneliness items.

**Scale vs. compute.** The pipeline processes one student at a time and keeps only per-hour summaries,
so it runs in **about 20 s with 1.4 GB peak memory** on my laptop. No sampling was needed, and later
methods (PCA, k-means, association rules) will run on the small tables.

**What surprised me.** (1) Phone-lock periods were only recorded when they lasted **1 to 10 hours**,
and app data only has 20-minute resolution. So session length can't be measured, and my questions on
session length (Q1) and within-session context (Q6) must be reframed around interaction frequency.
(2) My copy is 2.8 GB, not the 52.6 GB in my proposal. Facebook is in under 1% of snapshots, 12
students have less than 70% sensing coverage, and EMA answers drop from about 200 a week to 8 by
week 10. (3) Nighttime lock minutes don't track self-reported sleep (Spearman ρ ≈ 0.00).

**Where things stand.** The cleaned tables and 8 figures are done. Students differ widely: median
daily app switches range from 0 to 26, and the share of nights with phone use between 00:00 and 06:00
ranges from 0% to 93%. Next: add location (`wifi_location`), build per-student features, then
PCA → k-means → association rules.
