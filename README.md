# Data Science Salary Dashboard & Predictor

An interactive dashboard and salary prediction model built from three
combined Data Science salary datasets, covering how salary relates to
experience level, employment type, job title, company size, work year,
and location — plus a `RandomForestRegressor` model that estimates salary
for a given profile.

## Contents

| File | Description |
|---|---|
| `EDA.ipynb` | Original exploratory notebook — cleaning + first-pass matplotlib/seaborn charts |
| `dashboard.py` | Loads and merges all three datasets, cleans/normalizes them, builds every chart, and combines them into one interactive HTML dashboard. Also calls `prediction.py` to power the in-dashboard prediction feature |
| `prediction.py` | Trains the `RandomForestRegressor` salary model (one-hot encoded features, 80/20 split) and returns metrics plus per-feature salary adjustments used by the dashboard |
| `ds_salary_dashboard.html` | Generated dashboard — open directly in a browser, no setup needed |
| `Data Science Salary 2021 to 2023.csv` | Dataset 1 (2021–2023) |
| `data_science_salaries.csv` | Dataset 2 |
| `salaries.csv` | Dataset 3 (includes 2025 data, plus a `remote_ratio` column) |
| `README.md` | This file |

## Dashboard

`dashboard.py` loads all three CSVs, concatenates them, normalizes column
differences between sources (e.g. `employee_residence` → `company_location`
where needed), drops duplicates, and standardizes category labels across
datasets before charting. It builds:

- Overall salary distribution
- Salary by experience level (line)
- Salary by employment type (bar)
- Salary trend across work years (line)
- Salary by job title (bar)
- Experience level × employment type → salary
- Employment type × work year → salary
- Job title × employment type → salary
- Company size × employment type → salary
- Average salary by year, employment type, experience level, and company size
- Employment type × year trend (line)
- World map of job locations by country (via `pycountry` for ISO code lookup)

All charts are interactive — hover, zoom, pan, legend toggle — via
[Plotly](https://plotly.com/python/).

## Salary prediction

`prediction.py` exposes `build_salary_predictor(frame, sample_ratio=None)`,
which:

- Trains a `RandomForestRegressor` (scikit-learn) inside a `Pipeline` with
  one-hot encoding for categorical features
- **Features:** work year, experience level, employment type, job title,
  company size, company location
- **Split:** 80/20 train/test, `random_state=42`
- **Metric:** R² on the held-out test set
- Supports an optional `sample_ratio` (or `SAMPLE_RATIO` env var) to train
  on a subset for quicker iteration, scaling `n_estimators` down accordingly
- Also returns per-feature average-salary adjustments (job title, experience,
  employment type, company size, year, location) relative to the overall
  mean — used to power quick what-if comparisons in the dashboard without
  rerunning the full model

**Note:** this currently reports feature-level average adjustments, not
per-prediction model explainability (e.g. SHAP). If real explainability is
still on your list, that would be a further addition on top of this.

## Data cleaning

- Loads and concatenates all three source CSVs
- Reconciles column naming differences between sources
- Removes duplicate rows
- Normalizes inconsistent category labels across datasets (experience level,
  employment type, company size) so codes/abbreviations and full words from
  different sources map to the same values

## Running it yourself

```bash
pip install pandas plotly pycountry scikit-learn

python dashboard.py    # loads all 3 CSVs, trains the model, builds ds_salary_dashboard.html
```

## Possible next steps

- Add SHAP (or similar) explainability for individual predictions, not just
  feature-level average adjustments
- Add cross-filtering to the dashboard (year / experience / employment type
  / company size) via Dash or Streamlit instead of a static HTML export
- Retrain periodically as new salary-year data becomes available
