"""
Data Science Salary Dashboard — Streamlit version
Interactive dashboard with real model predictions via RandomForest.

Run with:  streamlit run streamlit_app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prediction import build_salary_predictor

st.set_page_config(page_title="Data Science Salary Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ===========================================================
# LOAD DATA & TRAIN MODEL (cached)
# ===========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "salary_model_ready.csv"), low_memory=False)
    return df

@st.cache_resource
def train_model(df):
    return build_salary_predictor(df)

df = load_data()
model_data = train_model(df)
model = model_data["model"]
feature_cols = model_data["feature_cols"]

EXP_ORDER = ["Entry", "Mid", "Senior", "Lead", "Executive"]

# ===========================================================
# HEADER
# ===========================================================
st.title("Data Science Salary Dashboard")
st.caption(f"Based on {len(df):,} data science & AI job records across 5 datasets")

# ===========================================================
# SIDEBAR: PREDICTION FORM
# ===========================================================
st.sidebar.header("Salary Predictor")
st.sidebar.write(f"**Model R\u00b2:** {model_data['r2']}  ")
st.sidebar.write(f"**MAE:** ${model_data['mae']:,.0f}  ")
st.sidebar.write(f"**Avg Salary:** ${model_data['mean_salary']:,.0f}")

job_titles = sorted(df["job_title"].dropna().unique().tolist())
location_options = sorted(df["company_location"].dropna().unique().tolist())
education_options = sorted(df["education_level"].dropna().unique().tolist())
industry_options = sorted(df["industry"].dropna().unique().tolist())
work_settings = ["Onsite", "Hybrid", "Remote"]
years = sorted(df["work_year"].unique().tolist())
company_sizes = ["Small", "Medium", "Large", "Enterprise"]

pred_job_title = st.sidebar.selectbox("Job Title", job_titles, index=job_titles.index("Data Scientist") if "Data Scientist" in job_titles else 0)
pred_experience = st.sidebar.selectbox("Experience Level", EXP_ORDER, index=EXP_ORDER.index("Senior"))
pred_education = st.sidebar.selectbox("Education Level", education_options, index=education_options.index("Master's") if "Master's" in education_options else 0)
pred_work_setting = st.sidebar.selectbox("Work Setting", work_settings, index=work_settings.index("Remote"))
pred_industry = st.sidebar.selectbox("Industry", industry_options)
pred_company_size = st.sidebar.selectbox("Company Size", company_sizes, index=company_sizes.index("Medium"))
pred_work_year = st.sidebar.selectbox("Work Year", years, index=len(years) - 1)
pred_location = st.sidebar.selectbox("Company Location", location_options)

if st.sidebar.button("Predict Salary", type="primary"):
    input_df = pd.DataFrame([{
        "work_year": pred_work_year,
        "experience_level": pred_experience,
        "employment_type": "Full-time",
        "job_title": pred_job_title,
        "company_size": pred_company_size,
        "company_location": pred_location,
        "education_level": pred_education,
        "work_setting": pred_work_setting,
        "industry": pred_industry,
        "experience_years": {"Entry": 1, "Mid": 4, "Senior": 8, "Lead": 12, "Executive": 17}.get(pred_experience, 5),
    }])

    prediction = model.predict(input_df[feature_cols])[0]
    st.sidebar.metric("Predicted Salary", f"${prediction:,.0f}")

# ===========================================================
# MAIN CONTENT: CHARTS
# ===========================================================

# Row 1: Salary Distribution (full width)
st.subheader("Salary Distribution")
mean_sal = df["salary_in_usd"].mean()
median_sal = df["salary_in_usd"].median()

fig = px.histogram(df, x="salary_in_usd", nbins=40,
                   color_discrete_sequence=["#4C78A8"])
fig.add_vline(x=mean_sal, line_dash="dash", line_color="red",
              annotation_text=f"Mean: ${mean_sal:,.0f}",
              annotation_position="top right")
fig.add_vline(x=median_sal, line_dash="dash", line_color="green",
              annotation_text=f"Median: ${median_sal:,.0f}",
              annotation_position="top right")
fig.update_layout(xaxis_title="Salary (USD)", yaxis_title="Count", height=350)
st.plotly_chart(fig, use_container_width=True)

# Row 2: Experience Level + Work Setting (two columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("By Experience Level")
    avg_sal_exp = df.groupby("experience_level", observed=True)["salary_in_usd"].agg(["mean", "count"]).reset_index()
    avg_sal_exp = avg_sal_exp.sort_values("mean", ascending=True)
    avg_sal_exp["label"] = avg_sal_exp.apply(
        lambda r: f"{r['experience_level']}  ({r['count']:,} jobs)", axis=1)

    fig = px.bar(avg_sal_exp, x="mean", y="label", orientation="h",
                 color="mean", color_continuous_scale="Blues",
                 labels={"mean": "Avg Salary (USD)", "label": ""})
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("By Work Setting")
    avg_sal_ws = df.groupby("work_setting")["salary_in_usd"].agg(["mean", "count"]).reset_index()
    avg_sal_ws = avg_sal_ws[avg_sal_ws["work_setting"].isin(work_settings)]
    avg_sal_ws["work_setting"] = pd.Categorical(avg_sal_ws["work_setting"], categories=work_settings, ordered=True)
    avg_sal_ws = avg_sal_ws.sort_values("mean", ascending=True)
    avg_sal_ws["label"] = avg_sal_ws.apply(
        lambda r: f"{r['work_setting']}  ({r['count']:,} jobs)", axis=1)

    fig = px.bar(avg_sal_ws, x="mean", y="label", orientation="h",
                 color="mean", color_continuous_scale="Teal",
                 labels={"mean": "Avg Salary (USD)", "label": ""})
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", height=300)
    st.plotly_chart(fig, use_container_width=True)

# Row 3: Salary Trend + Top Job Titles (two columns)
col3, col4 = st.columns(2)

with col3:
    st.subheader("Salary Trend Over Time")
    avg_sal_year = df.groupby("work_year")["salary_in_usd"].agg(["mean", "count"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_sal_year["work_year"], y=avg_sal_year["mean"],
        mode="lines+markers+text",
        text=avg_sal_year["mean"].apply(lambda v: f"${v/1000:.0f}K"),
        textposition="top center",
        line=dict(color="#4C78A8", width=3),
        marker=dict(size=10),
        hovertemplate="Year: %{x}<br>Average: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(xaxis_title="Year", yaxis_title="Avg Salary (USD)",
                      xaxis=dict(dtick=1), height=300)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Top 10 Job Titles by Salary")
    top_titles = df["job_title"].value_counts().head(10).index
    avg_sal_title = (df[df["job_title"].isin(top_titles)]
                      .groupby("job_title")["salary_in_usd"].agg(["mean", "count"])
                      .reset_index().sort_values("mean", ascending=True))

    fig = px.bar(avg_sal_title, x="mean", y="job_title", orientation="h",
                 color="mean", color_continuous_scale="Oranges",
                 labels={"mean": "Avg Salary (USD)", "job_title": ""})
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", height=300)
    st.plotly_chart(fig, use_container_width=True)

# Row 4: Top Industries (full width)
st.subheader("Top 10 Industries by Salary")
ind_data = df.groupby("industry")["salary_in_usd"].agg(["mean", "count"]).reset_index()
ind_data = ind_data[ind_data["count"] >= 100].sort_values("mean", ascending=True).tail(10)

fig = px.bar(ind_data, x="mean", y="industry", orientation="h",
             color="mean", color_continuous_scale="Greens",
             labels={"mean": "Avg Salary (USD)", "industry": ""})
fig.update_layout(coloraxis_showscale=False, yaxis_title="", height=350)
st.plotly_chart(fig, use_container_width=True)
