"""
Train lightweight models for Streamlit Community Cloud deployment.

Why:
- The full thesis models are too large to commit (GB-scale joblib).
- Streamlit Cloud needs small models that can be shipped with the repo.

Input:
- data/sample/trips_sample_rawschema.csv (public-safe, raw schema)

Output:
- outputs/models_cloud/clf_cloud.joblib
- outputs/models_cloud/reg_cloud.joblib

Models:
- Classifier: LogisticRegression (fast, small)
- Regressor: Ridge regression (fast, small)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.config import ColumnMap, ModelingConfig
from src.data import preprocess_trips
from src.modeling import SplitConfig, make_feature_lists, make_preprocessor, make_train_test


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "sample" / "trips_sample_rawschema.csv"
    out_dir = project_root / "outputs" / "models_cloud"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_path.exists():
        raise SystemExit(f"Sample not found at: {sample_path}")

    cols = ColumnMap()
    cfg = ModelingConfig()

    # Load + feature engineer from sample (no filters to keep rows)
    df, _meta = preprocess_trips(
        data_path=sample_path,
        cols=cols,
        cfg=cfg,
        nrows=None,
    )

    # Targets expected by pipeline
    if "surge_event" not in df.columns:
        raise SystemExit("Expected 'surge_event' column after preprocessing.")
    if "surge_multiplier" not in df.columns:
        raise SystemExit("Expected 'surge_multiplier' column after preprocessing.")

    # Build feature lists and preprocessor
    numeric, categorical = make_feature_lists(df)
    pre = make_preprocessor(numeric, categorical)

    # --- Classifier (surge_event) ---
    split_cfg = SplitConfig(test_size=0.2, random_state=42, time_aware=("ts_utc" in df.columns))
    X_train, X_test, y_train, y_test = make_train_test(df, target_col="surge_event", cfg=split_cfg)

    clf = Pipeline(
        steps=[
            ("pre", pre),
            ("model", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)

    # --- Regressor (surge_multiplier) ---
    # Train only on rows with valid multiplier
    df_reg = df.dropna(subset=["surge_multiplier"]).copy()
    Xr_train, Xr_test, yr_train, yr_test = make_train_test(df_reg, target_col="surge_multiplier", cfg=split_cfg)
    reg = Pipeline(
        steps=[
            ("pre", pre),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    reg.fit(Xr_train, yr_train)

    # Save models
    joblib.dump(clf, out_dir / "clf_cloud.joblib", compress=3)
    joblib.dump(reg, out_dir / "reg_cloud.joblib", compress=3)

    # Print quick sanity stats
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    y_pred = (proba >= 0.5).astype(int) if proba is not None else clf.predict(X_test)
    acc = float((y_pred == y_test.to_numpy()).mean())
    rmse = float(np.sqrt(np.mean((reg.predict(Xr_test) - yr_test.to_numpy()) ** 2)))
    print("Saved cloud models to:", out_dir)
    print("Classifier holdout accuracy:", round(acc, 4))
    print("Regressor holdout RMSE:", round(rmse, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

