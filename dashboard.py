"""
Data Science Salary Dashboard
Full pipeline: load -> clean -> build every chart -> combine into one
interactive HTML dashboard.

Run with:  python ds_salary_dashboard.py
Output:    ds_salary_dashboard.html  (open in any browser)
"""
import json
import os

import pandas as pd
import plotly.express as px
import pycountry
from prediction import build_salary_predictor

# ===========================================================
# 1. LOAD DATA
# ===========================================================
# Load multiple datasets (base + the two additional files) and concatenate
sources = [
    "Data Science Salary 2021 to 2023.csv",
    "data_science_salaries.csv",
    "salaries.csv",
]
frames = []
for src in sources:
    path = os.path.join(os.getcwd(), src)
    if os.path.exists(path):
        try:
            df_tmp = pd.read_csv(path)
            frames.append(df_tmp)
            print(f"Loaded {src}: {df_tmp.shape[0]} rows")
        except Exception as e:
            print(f"Failed to load {src}: {e}")
    else:
        print(f"File not found, skipping: {src}")

if not frames:
    raise SystemExit("No dataset files found. Place at least one source CSV in the folder.")

# Concatenate and normalize column names
df = pd.concat(frames, ignore_index=True, sort=False)

# Normalize column names that differ between sources
if "employee_residence" in df.columns and "company_location" not in df.columns:
    df["company_location"] = df["employee_residence"]

# Drop duplicates
df.drop_duplicates(inplace=True)

# Normalizers for categorical fields: make them consistent across datasets

def normalize_experience(val):
    if pd.isna(val):
        return val
    s = str(val).strip().lower()
    if s in ("en",) or "entry" in s:
        return "Entry Level"
    if s in ("mi",) or "mid" in s:
        return "Mid Level"
    if s in ("se",) or "senior" in s:
        return "Senior Level"
    if s in ("ex",) or "expert" in s:
        return "Expert Level"
    return val


def normalize_employment(val):
    if pd.isna(val):
        return val
    s = str(val).strip().lower()
    if s in ("ft",) or "full" in s:
        return "Full Time"
    if s in ("pt",) or "part" in s:
        return "Part Time"
    if s in ("ct",) or "contract" in s:
        return "Contract"
    if s in ("fl",) or "freelance" in s:
        return "Freelance"
    return val


def normalize_company_size(val):
    if pd.isna(val):
        return val
    s = str(val).strip().lower()
    if s in ("l", "large"):
        return "Large"
    if s in ("m", "medium", "mid"):
        return "Mid"
    if s in ("s", "small"):
        return "Small"
    return val

# Map company_location -> keep an alpha2 code column and also full country name

def get_alpha2(code_or_name):
    if pd.isna(code_or_name):
        return None
    s = str(code_or_name).strip()
    # If looks like alpha2 code
    if len(s) == 2 and s.isalpha():
        return s.upper()
    # try to find by country name (fuzzy)
    try:
        res = pycountry.countries.search_fuzzy(s)
        if res:
            return res[0].alpha_2
    except Exception:
        return None

# apply normalizers
if "experience_level" in df.columns:
    df["experience_level"] = df["experience_level"].apply(normalize_experience)
if "employment_type" in df.columns:
    df["employment_type"] = df["employment_type"].apply(normalize_employment)
if "company_size" in df.columns:
    df["company_size"] = df["company_size"].apply(normalize_company_size)

# produce company_location_code (alpha2) and full name for display
if "company_location" in df.columns:
    df["company_location_code"] = df["company_location"].apply(get_alpha2)
    # where code is known, convert to full name; otherwise keep original string
    def code_to_name(row):
        code = row["company_location_code"]
        orig = row["company_location"]
        if code:
            try:
                return pycountry.countries.get(alpha_2=code).name
            except Exception:
                return orig
        return orig
    df["company_location"] = df.apply(code_to_name, axis=1)

EXP_ORDER = ["Entry Level", "Mid Level", "Senior Level", "Expert Level"]
if "experience_level" in df.columns:
    df["experience_level"] = pd.Categorical(df["experience_level"], categories=EXP_ORDER, ordered=True)

print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Allow quick test sampling via env var SAMPLE_RATIO (e.g. 0.1 for 10%)
try:
    sample_ratio = float(os.getenv("SAMPLE_RATIO", "1"))
except Exception:
    sample_ratio = 1.0

if 0 < sample_ratio < 1.0:
    orig_n = len(df)
    df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
    print(f"Sampled dataset: {len(df)} rows (ratio={sample_ratio}) from {orig_n} rows")

TEMPLATE = "plotly_white"
figs = []  # collected in order for the dashboard



def add(fig, title):
    fig.update_layout(template=TEMPLATE, title=title, title_x=0.02)
    figs.append(fig)
    return fig


# ===========================================================
# CHART: Overall salary distribution (dashboard overview)
# ===========================================================
fig = px.histogram(df, x="salary_in_usd", nbins=40, color_discrete_sequence=["#4C78A8"])
add(fig, "Distribution of Data Science Salaries (USD)")

# ===========================================================
# CHART: Experience level vs salary (line)
# ===========================================================
avg_sal_exp = df.groupby("experience_level", observed=True)["salary_in_usd"].mean().reset_index()
avg_sal_exp = avg_sal_exp.sort_values("experience_level")
fig = px.line(avg_sal_exp, x="experience_level", y="salary_in_usd", markers=True,
              color_discrete_sequence=["teal"])
add(fig, "Experience Level vs Salary")

# ===========================================================
# CHART: Employment type vs salary (bar)
# ===========================================================
avg_sal_emp = df.groupby("employment_type", observed=True)["salary_in_usd"].mean().reset_index()
avg_sal_emp = avg_sal_emp.sort_values("salary_in_usd", ascending=False)
fig = px.bar(avg_sal_emp, x="employment_type", y="salary_in_usd", color="employment_type")
add(fig, "Employment Type vs Salary")

# ===========================================================
# CHART: Salary trend across years (timeline line)
# ===========================================================
avg_sal_year = df.groupby("work_year")["salary_in_usd"].mean().reset_index()
fig = px.line(avg_sal_year, x="work_year", y="salary_in_usd", markers=True,
              color_discrete_sequence=["darkorange"])
fig.update_xaxes(dtick=1)
add(fig, "Average Salary Over Time")

# ===========================================================
# CHART: Job title vs salary (top 15 by frequency, bar)
# ===========================================================
top_titles = df["job_title"].value_counts().head(15).index
avg_sal_title = (df[df["job_title"].isin(top_titles)]
                  .groupby("job_title")["salary_in_usd"].mean()
                  .reset_index().sort_values("salary_in_usd", ascending=False))
fig = px.bar(avg_sal_title, x="job_title", y="salary_in_usd", color="salary_in_usd",
             color_continuous_scale="Blues")
fig.update_xaxes(tickangle=45)
add(fig, "Average Salary by Job Title (Top 15 most common)")

# ===========================================================
# CHART: Experience level x Employment type -> salary
# ===========================================================
avg_sal_exp_emp = (df.groupby(["experience_level", "employment_type"], observed=True)["salary_in_usd"]
                    .mean().reset_index())
fig = px.bar(avg_sal_exp_emp, x="experience_level", y="salary_in_usd", color="employment_type",
             barmode="group")
add(fig, "Experience Level x Employment Type -> Salary")

# ===========================================================
# CHART: Employment type x Work year -> salary
# ===========================================================
avg_sal_emp_year = (df.groupby(["employment_type", "work_year"])["salary_in_usd"]
                     .mean().reset_index())
fig = px.bar(avg_sal_emp_year, x="work_year", y="salary_in_usd", color="employment_type",
             barmode="group")
fig.update_xaxes(dtick=1)
add(fig, "Employment Type x Work Year -> Salary")

# ===========================================================
# CHART: Job title x Employment type (bar, top 10 titles)
# ===========================================================
top10_titles = df["job_title"].value_counts().head(10).index
sub = df[df["job_title"].isin(top10_titles)]
avg_sal_title_emp = (sub.groupby(["job_title", "employment_type"])["salary_in_usd"]
                      .mean().reset_index())
fig = px.bar(avg_sal_title_emp, x="job_title", y="salary_in_usd", color="employment_type",
             barmode="group")
fig.update_xaxes(tickangle=45)
add(fig, "Job Title x Employment Type -> Salary (Top 10 titles)")

# ===========================================================
# CHART: Company size x Employment type -> salary
# ===========================================================
avg_sal_size_emp = (df.groupby(["company_size", "employment_type"])["salary_in_usd"]
                     .mean().reset_index())
fig = px.bar(avg_sal_size_emp, x="company_size", y="salary_in_usd", color="employment_type",
             barmode="group")
add(fig, "Company Size x Employment Type -> Salary")

# ===========================================================
# CHART: Average salary per year (KPI-style line)
# ===========================================================
fig = px.line(avg_sal_year, x="work_year", y="salary_in_usd", markers=True,
              color_discrete_sequence=["#59A14F"])
fig.update_xaxes(dtick=1)
add(fig, "Average Salary per Year")

# ===========================================================
# CHART: Average salary by employment type
# ===========================================================
fig = px.bar(avg_sal_emp, x="employment_type", y="salary_in_usd", color="employment_type")
add(fig, "Average Salary by Employment Type")

# ===========================================================
# CHART: Average salary by experience level
# ===========================================================
fig = px.bar(avg_sal_exp, x="experience_level", y="salary_in_usd", color="experience_level")
add(fig, "Average Salary by Experience Level")

# ===========================================================
# CHART: Average salary by company size
# ===========================================================
avg_sal_size = df.groupby("company_size")["salary_in_usd"].mean().reset_index()
size_order = ["Small", "Mid", "Large"]
avg_sal_size["company_size"] = pd.Categorical(avg_sal_size["company_size"], categories=size_order, ordered=True)
avg_sal_size = avg_sal_size.sort_values("company_size")
fig = px.bar(avg_sal_size, x="company_size", y="salary_in_usd", color="company_size")
add(fig, "Average Salary by Company Size")

# ===========================================================
# CHART: Average salary by employment type and year
# ===========================================================
fig = px.line(avg_sal_emp_year, x="work_year", y="salary_in_usd", color="employment_type",
              markers=True)
fig.update_xaxes(dtick=1)
add(fig, "Average Salary by Employment Type and Year")

# ===========================================================
# MAP: job locations (avg salary + job count by country)
# ===========================================================
def iso2_to_iso3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except Exception:
        return None

loc_stats = (df.groupby("company_location")["salary_in_usd"]
             .agg(avg_salary="mean", job_count="count").reset_index())
loc_stats["company_location_code"] = loc_stats["company_location"].apply(
    lambda name: next((code for code, full in df[["company_location_code", "company_location"]].drop_duplicates().itertuples(index=False, name=None) if full == name), None)
)
loc_stats["iso3"] = loc_stats["company_location_code"].apply(iso2_to_iso3)
loc_stats = loc_stats.dropna(subset=["iso3"])

fig = px.scatter_geo(
    loc_stats,
    locations="iso3",
    size="job_count",
    color="avg_salary",
    color_continuous_scale="Viridis",
    hover_name="company_location",
    hover_data={"avg_salary": ":.0f", "job_count": True, "iso3": False, "company_location_code": False},
    projection="natural earth",
)
add(fig, "Job Locations - Blip Size = Job Count, Color = Avg Salary (USD)")

# ===========================================================
# MODEL: salary prediction profile for dashboard input
# ===========================================================
prediction_model = build_salary_predictor(df)

# ===========================================================
# COMBINE EVERYTHING INTO ONE DASHBOARD HTML
# ===========================================================
html_parts = ["""
<html>
<head>
<meta charset="utf-8">
<title>Data Science Salary Dashboard</title>
<style>
  body { font-family: Arial, sans-serif; background:#f5f6f8; margin:0; padding:20px; }
  h1 { text-align:center; color:#222; }
  .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:20px; max-width:1400px; margin:0 auto; }
  .card { background:white; border-radius:10px; padding:10px; box-shadow:0 1px 4px rgba(0,0,0,0.15); }
  .full { grid-column: 1 / -1; }
  .predict-header { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
  .predict-form { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; align-items:end; }
  label { display:flex; flex-direction:column; font-size:12px; color:#4a4a4a; font-weight:600; }
  select, button { padding:8px 10px; border:1px solid #dfe3e8; border-radius:8px; font-size:14px; }
  button { background:#2f6fed; color:white; border:none; cursor:pointer; }
  .salary-result { margin-top:16px; font-size:28px; font-weight:700; color:#0d4ba3; }
  .model-metrics { display:flex; gap:20px; flex-wrap:wrap; color:#475467; font-size:14px; }
</style>
</head>
<body>
<h1>Data Science Salary Dashboard (2021-2023)</h1>
<div class="grid">
"""]

job_titles = sorted(prediction_model["job_titles"].keys())
location_options = sorted(df["company_location"].dropna().unique().tolist())

prediction_card = f"""
<div class="card full">
  <div class="predict-header">
    <h2>Salary Prediction Model</h2>
    <div class="model-metrics">
      <span><strong>Model R²:</strong> {prediction_model['r2']}</span>
      <span><strong>Average salary:</strong> ${prediction_model['mean_salary']:,.0f} USD</span>
    </div>
  </div>

  <div class="predict-form">
    <label>Job title
      <select id="pred-job-title">
        {''.join(f'<option value="{title}">{title}</option>' for title in job_titles)}
      </select>
    </label>
    <label>Experience level
      <select id="pred-experience">
        <option value="Entry Level">Entry Level</option>
        <option value="Mid Level">Mid Level</option>
        <option value="Senior Level" selected>Senior Level</option>
        <option value="Expert Level">Expert Level</option>
      </select>
    </label>
    <label>Employment type
      <select id="pred-employment">
        <option value="Full Time" selected>Full Time</option>
        <option value="Part Time">Part Time</option>
        <option value="Contract">Contract</option>
        <option value="Freelance">Freelance</option>
      </select>
    </label>
    <label>Company size
      <select id="pred-company-size">
        <option value="Small">Small</option>
        <option value="Mid" selected>Mid</option>
        <option value="Large">Large</option>
      </select>
    </label>
    <label>Work year
      <select id="pred-work-year">
        <option value="2021">2021</option>
        <option value="2022" selected>2022</option>
        <option value="2023">2023</option>
      </select>
    </label>
    <label>Company location
      <select id="pred-location">
        {''.join(f'<option value="{loc}">{loc}</option>' for loc in location_options)}
      </select>
    </label>
  </div>

  <div style="margin-top:14px; display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
    <button type="button" onclick="predictSalary()">Predict Salary</button>
    <div id="salary-result" class="salary-result">$0 USD</div>
  </div>

  <script>
    const salaryModel = {json.dumps(prediction_model)};

    function safeValue(map, key, fallback = 0) {{
      const value = map[String(key)];
      return typeof value === 'number' ? value : fallback;
    }}

    function predictSalary() {{
      const jobTitle = document.getElementById('pred-job-title').value;
      const experience = document.getElementById('pred-experience').value;
      const employment = document.getElementById('pred-employment').value;
      const companySize = document.getElementById('pred-company-size').value;
      const workYear = document.getElementById('pred-work-year').value;
      const location = document.getElementById('pred-location').value;

      let estimate = salaryModel.mean_salary;
      estimate *= (1 + safeValue(salaryModel.job_titles, jobTitle, 0));
      estimate *= (1 + safeValue(salaryModel.experience, experience, 0));
      estimate *= (1 + safeValue(salaryModel.employment, employment, 0));
      estimate *= (1 + safeValue(salaryModel.company_size, companySize, 0));
      estimate *= (1 + safeValue(salaryModel.year, String(workYear), 0));
      estimate *= (1 + safeValue(salaryModel.location, location, 0));

      if (estimate < 0) estimate = 0;
      document.getElementById('salary-result').textContent = '$' + Math.round(estimate).toLocaleString() + ' USD';
    }}

    document.addEventListener('DOMContentLoaded', predictSalary);
  </script>
</div>
"""

html_parts.append(prediction_card)

full_width_titles = {"Distribution of Data Science Salaries (USD)",
                      "Job Locations - Blip Size = Job Count, Color = Avg Salary (USD)",
                      "Average Salary by Job Title (Top 15 most common)",
                      "Job Title x Employment Type -> Salary (Top 10 titles)"}

first = True
for fig in figs:
    div = fig.to_html(full_html=False, include_plotlyjs=("cdn" if first else False))
    first = False
    css_class = "card full" if fig.layout.title.text in full_width_titles else "card"
    html_parts.append(f'<div class="{css_class}">{div}</div>')

html_parts.append("</div></body></html>")

with open("ds_salary_dashboard.html", "w", encoding="utf-8") as f:
    f.write("".join(html_parts))

print(f"\nDashboard built with {len(figs)} charts -> ds_salary_dashboard.html")