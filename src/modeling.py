from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    val_ratio: float = 0.15  # fraction of training period used as validation (last part of train)
    random_state: int = 42
    time_aware: bool = True  # split chronologically using ts_utc


def make_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical feature lists used for modelling.

    Only columns that are actually present in `df` are kept, so the same list
    works whether or not weather / spatial features have been merged in.
    """
    numeric = [
        "total_distance",
        "total_time",
        "trip_haversine_km",
        "trip_bearing_deg",
        "duration_hr",
        "speed_kmh",
        "distance_vs_haversine_ratio",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        # Weather
        "rain_flag",
        "precip_mm",
        "temp_c",
        "humidity",
        "wind_kph",
        # Spatial policy flag (1 if pickup in CBD / Westlands / Airport)
        "high_demand_zone",
    ]
    categorical = [
        "service_type_id",
        "dayofweek",
        "hour",
        "is_weekend",
        "is_morning_peak",
        "is_evening_peak",
    ]

    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Keep one-hot stable + bounded (important if you have high-cardinality IDs).
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=200, max_categories=2000)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def time_aware_split(df: pd.DataFrame, y: pd.Series, cfg: SplitConfig):
    df = df.sort_values("ts_utc").reset_index(drop=True)
    n_test = int(np.ceil(len(df) * cfg.test_size))
    split_idx = max(1, len(df) - n_test)
    X_train = df.iloc[:split_idx].copy()
    X_test = df.iloc[split_idx:].copy()
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]
    return X_train, X_test, y_train, y_test


def make_train_test(
    df: pd.DataFrame, target_col: str, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = df[target_col]
    if cfg.time_aware and "ts_utc" in df.columns:
        return time_aware_split(df, y, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if y.nunique() <= 10 else None,
    )
    return X_train, X_test, y_train, y_test


def make_train_val_test(
    df: pd.DataFrame, target_col: str, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Time-aware split into train / val / test. Test is last test_size of data;
    of the remainder (train period), last val_ratio is validation, rest is train.
    So: [ ... train ... | ... val ... | ... test ... ]
    """
    if not (cfg.time_aware and "ts_utc" in df.columns):
        raise ValueError("make_train_val_test requires time_aware=True and ts_utc column")
    df = df.sort_values("ts_utc").reset_index(drop=True)
    y = df[target_col]
    n = len(df)
    n_test = int(np.ceil(n * cfg.test_size))
    n_val = int(np.ceil((n - n_test) * cfg.val_ratio))
    n_train = n - n_test - n_val
    n_train = max(1, n_train)
    # train: [0 : n_train], val: [n_train : n_train+n_val], test: [n_train+n_val :]
    X_train = df.iloc[:n_train].copy()
    X_val = df.iloc[n_train : n_train + n_val].copy()
    X_test = df.iloc[n_train + n_val :].copy()
    y_train = y.loc[X_train.index]
    y_val = y.loc[X_val.index]
    y_test = y.loc[X_test.index]
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_classification(y_true, y_pred, y_proba=None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Attempts to recover transformed feature names for feature importance reporting.
    """
    out: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                out.extend(list(trans.get_feature_names_out(cols)))
                continue
            except Exception:
                pass
        # fallback
        if isinstance(cols, list):
            out.extend(cols)
    return out

