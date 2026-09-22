# Data Science Salary Dashboard & Predictor

An interactive dashboard and salary prediction model built from five
combined salary and AI job datasets, covering how salary relates to
experience level, employment type, job title, company size, work year,
location, education level, work setting, industry, and skills — plus a
`RandomForestRegressor` model that estimates salary for a given profile.

## Contents

| File | Description |
|---|---|
| `streamlit_app.py` | Interactive Streamlit dashboard with real model predictions |
| `prediction.py` | Trains `RandomForestRegressor` salary model |
| `notebooks/EDA.ipynb` | Original exploratory notebook |
| `notebooks/preprocessing.ipynb` | Loads 5 CSVs, normalizes schemas, merges, feature engineering |
| `notebooks/prediction.ipynb` | Model experimentation notebook |
| `data/raw/` | Source CSVs (see Data Sources below) |
| `data/processed/` | Cleaned/merged outputs |
| `misc/DS Dashboard.txt` | Original feature wishlist |

## Data Sources

| File | Rows | Key Columns |
|---|---|---|
| `Data Science Salary 2021 to 2023.csv` | 3,761 | work_year, experience_level, job_title, salary_in_usd |
| `data_science_salaries.csv` | 6,599 | work_year, experience_level, job_title, work_setting |
| `salaries.csv` | 151,445 | work_year, experience_level, job_title, remote_ratio |
| `global_ai_tech_salaries_2020_2025.csv` | 12,000 | education_level, industry, primary_skills |
| `global_ai_jobs.csv` | 90,000 | education_level, industry, experience_years, job scores |

## Dashboard

`streamlit_app.py` runs an interactive web app with real model predictions:

- Sidebar prediction form — select job title, experience, education, work setting,
  industry, company size, year, and location → get an instant salary estimate
  from the trained RandomForest model
- **Salary distribution** — histogram with mean/median lines
- **Salary by experience level** — horizontal bar chart
- **Salary by work setting** — Onsite vs Hybrid vs Remote
- **Salary trend over years** — line chart with value labels
- **Top 10 job titles by salary** — horizontal bar chart
- **Top 10 industries by salary** — horizontal bar chart

> **Note:** An earlier version used a static HTML dashboard (`dashboard.py`) that
> generated a standalone Plotly HTML file with adjustment-based predictions. This
> was replaced with the Streamlit app to enable real model predictions via a
> Python backend.

## Salary Prediction

`prediction.py` exposes `build_salary_predictor(frame, sample_ratio=None)`,
which:

- Trains a `RandomForestRegressor` (scikit-learn) inside a `Pipeline` with
  one-hot encoding for categorical features
- **Features:** work year, experience level, employment type, job title,
  company size, company location, education level, work setting, industry,
  experience years
- **Split:** 80/20 train/test, `random_state=42`
- **Metric:** R² on the held-out test set
- Supports an optional `sample_ratio` (or `SAMPLE_RATIO` env var) to train
  on a subset for quicker iteration, scaling `n_estimators` down accordingly
- Also returns per-feature average-salary adjustments (job title, experience,
  employment type, company size, year, location, education, work setting,
  industry) relative to the overall mean — useful for quick what-if analysis

## Preprocessing

`notebooks/preprocessing.ipynb` handles the full data pipeline:

- Loads all 5 source CSVs
- Normalizes column names across sources (e.g. `job_role` → `job_title`,
  `salary_usd` → `salary_in_usd`)
- Maps experience level abbreviations (EN/MI/SE/EX → Entry/Mid/Senior/Executive)
- Maps employment type codes (FT/PT/CT/FL → Full-time/Part-time/Contract/Freelance)
- Maps company size codes (S/M/L → Small/Medium/Large/Enterprise)
- Maps work setting labels (On-site → Onsite)
- Deduplicates across original datasets
- Splits into two outputs:
  - **Dataset A** (`salary_original_only.csv`): 173K rows, 7 original columns
  - **Dataset B** (`salary_model_ready.csv`): 102K rows, 41 columns with enriched features
- Feature engineering: top-15 skill binary columns, skill count

## Running It Yourself

```bash
pip install pandas numpy plotly pycountry scikit-learn streamlit

# Optional: run preprocessing if you need to regenerate processed data
# jupyter notebook notebooks/preprocessing.ipynb

streamlit run streamlit_app.py
```

Set `SAMPLE_RATIO=0.1` to train on 10% of data for faster iteration.

## Possible Next Steps

- Add SHAP (or similar) explainability for individual predictions
- Retrain periodically as new salary-year data becomes available
- Add cross-filtering between charts (click one chart to filter others)
