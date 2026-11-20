# StudentLife: smartphone usage patterns of college students

CS 6220 Data Mining project. Exploration and preprocessing of the
[StudentLife dataset](https://www.kaggle.com/datasets/dartweichen/student-life)
(Dartmouth, spring 2013: 49 students, 65 days of phone sensing, EMA and surveys).

## Layout

```
eda/
  build_features.py    raw files -> hourly / daily / per-student tables (eda/processed/)
  explore.py           shape, dtypes, missingness, summary stats, figures (eda/figures/)
  probe*.py            first-pass structure checks of the raw files
  CODEWALK1_SUMMARY.md checkpoint write-up
  explore_output.txt   saved output of explore.py
analysis/
  student_profiles.py  per-student features -> KDE -> k-means
  output.txt           saved output of student_profiles.py
  figures/             KDE, choosing k, cluster and profile plots
  CHECKPOINT5_SUMMARY.md  checkpoint 5 write-up
  README.md            full step-by-step guide with questions and answers
```

## Setup

The dataset is not in this repo (2.8 GB). Download it from Kaggle and put the
extracted folder next to `eda/`, named `dataset 2`:

```
Data-Mining-Project/
  dataset 2/   EMA/ app_usage/ calendar/ call_log/ dinning/ education/ sensing/ sms/ survey/
  eda/
```

Create the environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python eda/build_features.py   # ~20 s, ~1.4 GB RAM
.venv/bin/python eda/explore.py
.venv/bin/python analysis/student_profiles.py   # a few seconds
```
