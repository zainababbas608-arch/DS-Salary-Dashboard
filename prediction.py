"""
prediction.py

Contains the salary model build function used by the dashboard.
"""
import os
from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_salary_predictor(frame, sample_ratio: Optional[float] = None):
    """Train a simple RandomForest on the provided DataFrame and return
    a small JSON-serializable profile used by the dashboard prediction UI.

    Args:
        frame: pandas.DataFrame already cleaned/normalized and containing
               salary_in_usd and feature columns.
        sample_ratio: optional override for quick runs (<1 for sampling).
    Returns:
        dict with model metrics and per-feature multiplicative adjustments.
    """
    import pandas as pd

    df = frame.copy()
    # ensure target exists
    if "salary_in_usd" not in df.columns:
        raise ValueError("salary_in_usd column is required for training")

    # drop rows without target
    df = df.dropna(subset=["salary_in_usd"]).reset_index(drop=True)

    # optional sampling override (also respects SAMPLE_RATIO env var if sample_ratio is None)
    if sample_ratio is None:
        try:
            sample_ratio = float(os.getenv("SAMPLE_RATIO", "1"))
        except Exception:
            sample_ratio = 1.0

    if 0 < sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)

    feature_cols = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "company_size",
        "company_location",
    ]
    # keep only those that exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns available for the model after checking DataFrame columns")

    X = df[feature_cols].copy()
    y = df["salary_in_usd"]

    numeric_cols = [c for c in ["work_year"] if c in X.columns]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        # use sparse_output=False where available; recent sklearns support it
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # older sklearn versions use `sparse`
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # choose estimator size depending on dataset size and whether we sampled
    n_rows = len(df)
    if sample_ratio and sample_ratio < 1.0:
        n_estimators = 50
    else:
        n_estimators = 450 if n_rows <= 50000 else 150

    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("regressor", RandomForestRegressor(n_estimators=n_estimators, random_state=42, min_samples_leaf=2))])

    # train/test split and fit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model_score = r2_score(y_test, preds)

    # compute additional regression diagnostics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    import math
    rmse = float(math.sqrt(mse))

    overall_mean = float(y.mean())

    def group_adjust(col):
        if col not in df.columns:
            return {}
        return df.groupby(col)["salary_in_usd"].mean().div(overall_mean).sub(1).to_dict()

    job_title_adjustments = group_adjust("job_title")
    experience_adjustments = group_adjust("experience_level")
    employment_adjustments = group_adjust("employment_type")
    size_adjustments = group_adjust("company_size")
    year_adjustments = group_adjust("work_year")
    location_adjustments = group_adjust("company_location")

    # residuals histogram and percentiles for quick diagnostic
    import numpy as np
    residuals = (y_test - preds)
    # convert to numpy
    residuals_np = np.asarray(residuals)
    hist_counts, hist_bins = np.histogram(residuals_np, bins=30)
    percentiles = {p: float(np.percentile(residuals_np, p)) for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)}

    # format return payload
    return {
        "r2": round(float(model_score), 3),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mean_salary": round(overall_mean, 2),
        "residuals_hist": {"bins": [float(b) for b in hist_bins.tolist()], "counts": hist_counts.tolist()},
        "residuals_percentiles": {k: round(v, 2) for k, v in percentiles.items()},
        "job_titles": {str(k): round(float(v), 4) for k, v in job_title_adjustments.items()},
        "experience": {str(k): round(float(v), 4) for k, v in experience_adjustments.items()},
        "employment": {str(k): round(float(v), 4) for k, v in employment_adjustments.items()},
        "company_size": {str(k): round(float(v), 4) for k, v in size_adjustments.items()},
        "year": {str(int(k)): round(float(v), 4) for k, v in year_adjustments.items()},
        "location": {str(k): round(float(v), 4) for k, v in location_adjustments.items()},
    }
