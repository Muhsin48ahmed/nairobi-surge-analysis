from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import ColumnMap, ModelingConfig
from .utils_geo import bearing_deg, haversine_km, latlon_grid_id


RAW_DTYPES = {
    "city": "category",
    "is_provider_status": "int64",
    "source_address": "string",
    "destination_address": "string",
    "sourceLocation[0]": "float64",
    "sourceLocation[1]": "float64",
    "destinationLocation[0]": "float64",
    "destinationLocation[1]": "float64",
    "total_distance": "float64",
    "total_time": "float64",
    "created_at": "string",
    "service_type_id": "string",
    "user_id": "string",
    "driver_id": "string",
    "unique_id": "string",
}


def read_trips_csv(csv_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Reads the raw CSV with stable dtypes (faster + lower memory).
    Maps 'surged' to 'surge_multiplier' if present (same as JSON loader).
    """
    df = pd.read_csv(csv_path, dtype=RAW_DTYPES, nrows=nrows)
    if "surged" in df.columns and "surge_multiplier" not in df.columns:
        df["surge_multiplier"] = pd.to_numeric(df["surged"], errors="coerce")
    return df


def read_trips_json(json_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Reads trip data from JSON format (MongoDB export style).
    Handles MongoDB date format: {"$date": "2024-01-11T00:00:57.476Z"}
    Extracts lat/lon from sourceLocation/destinationLocation arrays.
    Maps 'surged' column to 'surge_multiplier'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        records = list(data.values())
    else:
        records = data
    
    # Limit rows if requested
    if nrows is not None:
        records = records[:nrows]
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Parse MongoDB date format
    if 'created_at' in df.columns:
        df['created_at'] = df['created_at'].apply(
            lambda x: x.get('$date', x) if isinstance(x, dict) else x
        )
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    
    # Extract lat/lon from arrays
    if 'sourceLocation' in df.columns:
        df['sourceLocation[0]'] = df['sourceLocation'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
        df['sourceLocation[1]'] = df['sourceLocation'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else np.nan)
        df = df.drop(columns=['sourceLocation'])
    
    if 'destinationLocation' in df.columns:
        df['destinationLocation[0]'] = df['destinationLocation'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
        df['destinationLocation[1]'] = df['destinationLocation'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else np.nan)
        df = df.drop(columns=['destinationLocation'])
    
    # Map 'surged' to 'surge_multiplier' if present
    if 'surged' in df.columns and 'surge_multiplier' not in df.columns:
        df['surge_multiplier'] = pd.to_numeric(df['surged'], errors='coerce')
    
    return df


def parse_timestamp_utc(series: pd.Series) -> pd.Series:
    # Example: '2024-01-01T00:00:03.418Z'
    return pd.to_datetime(series, utc=True, errors="coerce")


def add_temporal_features(
    df: pd.DataFrame,
    ts_col: str,
    local_tz: str = "Africa/Nairobi",
) -> pd.DataFrame:
    """
    Derives calendar / peak-hour / cyclical features.

    IMPORTANT: hour, dayofweek, month, and peak flags are computed in the
    **local timezone** (default Africa/Nairobi, UTC+3), not UTC. Nairobi
    morning peak (07:00–10:00 local) is 04:00–07:00 UTC; without this
    conversion the "morning peak" flag misfires and Figure 4.2 reads
    inverted. ts_utc is still stored in UTC for downstream joins.
    """
    ts_utc = df[ts_col]
    # Ensure tz-aware UTC first. If naive, assume UTC (raw data uses Zulu time).
    if not pd.api.types.is_datetime64tz_dtype(ts_utc):
        ts_utc = pd.to_datetime(ts_utc, utc=True, errors="coerce")

    ts_local = ts_utc.dt.tz_convert(local_tz)

    df["ts_utc"] = ts_utc
    df["ts_local"] = ts_local
    df["date"] = ts_local.dt.date.astype("string")
    df["year"] = ts_local.dt.year.astype("int16")
    df["month"] = ts_local.dt.month.astype("int8")
    df["day"] = ts_local.dt.day.astype("int8")
    df["dayofweek"] = ts_local.dt.dayofweek.astype("int8")  # Mon=0
    df["hour"] = ts_local.dt.hour.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    # Peak heuristics for Nairobi ride-hailing (local time):
    df["is_morning_peak"] = df["hour"].between(7, 10).astype("int8")
    df["is_evening_peak"] = df["hour"].between(16, 20).astype("int8")

    # Cyclical encodings (based on LOCAL hour / dayofweek / month)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_spatial_features(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    df["trip_haversine_km"] = haversine_km(
        df[cols.src_lat], df[cols.src_lon], df[cols.dst_lat], df[cols.dst_lon]
    )
    df["trip_bearing_deg"] = bearing_deg(
        df[cols.src_lat], df[cols.src_lon], df[cols.dst_lat], df[cols.dst_lon]
    )

    # Lightweight pickup/dropoff bins for heatmaps + neighborhood-like aggregation
    df["pickup_grid"] = latlon_grid_id(df[cols.src_lat], df[cols.src_lon], cell_size_deg=0.005)
    df["dropoff_grid"] = latlon_grid_id(df[cols.dst_lat], df[cols.dst_lon], cell_size_deg=0.005)
    return df


def add_high_demand_zone(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    """
    Flags pickup in high-demand zones: CBD, Westlands, JKIA Airport.
    Proposal Table 3.5.4: high_demand_zone = 1 if pickup in CBD, Westlands, Airport.
    Uses coordinate bounding boxes for Nairobi (lat negative, lon ~36.8).
    """
    lat = df[cols.src_lat].astype("float64")
    lon = df[cols.src_lon].astype("float64")

    # Nairobi bounding boxes (approximate; proposal Table 3.6.3.B)
    # CBD: central business district
    cbd = lat.between(-1.295, -1.275) & lon.between(36.800, 36.835)
    # Westlands: business/nightlife district
    westlands = lat.between(-1.275, -1.255) & lon.between(36.795, 36.830)
    # JKIA Airport
    airport = lat.between(-1.335, -1.315) & lon.between(36.910, 36.945)

    df["high_demand_zone"] = (cbd | westlands | airport).astype("int8")

    # Fallback: if source_address exists, match strings (case-insensitive)
    if "source_address" in df.columns:
        addr = df["source_address"].astype(str).str.lower()
        addr_match = (
            addr.str.contains("cbd|westlands|airport|jkia|jkia ", na=False, regex=True)
            | addr.str.contains("kilimani|upper hill", na=False, regex=True)
        )
        df["high_demand_zone"] = (df["high_demand_zone"] | addr_match).astype("int8")
    return df


def add_trip_quality_features(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    df["duration_hr"] = (df[cols.duration_min] / 60.0).astype("float64")
    df["speed_kmh"] = np.where(
        df["duration_hr"] > 0,
        df[cols.distance_km] / df["duration_hr"],
        np.nan,
    )
    df["distance_vs_haversine_ratio"] = np.where(
        df["trip_haversine_km"] > 0,
        df[cols.distance_km] / df["trip_haversine_km"],
        np.nan,
    )
    return df


def apply_iqr_capping(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    """
    IQR-based capping for distance and duration (proposal Table 3.5.3).
    Caps values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR to retain data while reducing outlier influence.
    """
    df = df.copy()
    for col, key in [(cols.distance_km, "distance_km"), (cols.duration_min, "duration_min")]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=max(low, 0), upper=high)
    return df


def clean_outliers(
    df: pd.DataFrame,
    cols: ColumnMap,
    speed_cap_kmh: float = 140.0,
) -> pd.DataFrame:
    """
    Conservative filters that remove obvious sensor/ingestion errors.
    Applies IQR-based capping (proposal) before hard bounds and an
    explicit speed cap (default 140 km/h) to remove residual
    division-by-near-zero artefacts (raw max observed ~123,404 km/h).
    """
    df = apply_iqr_capping(df, cols)

    mask = np.ones(len(df), dtype=bool)

    # Nairobi-ish bounds (wide box, keeps almost all real trips)
    mask &= df[cols.src_lat].between(-2.0, 0.0)
    mask &= df[cols.dst_lat].between(-2.0, 0.0)
    mask &= df[cols.src_lon].between(36.0, 38.0)
    mask &= df[cols.dst_lon].between(36.0, 38.0)

    # Distance/time sanity
    mask &= df[cols.distance_km].between(0.1, 200.0)
    mask &= df[cols.duration_min].between(0.5, 600.0)

    # Speed sanity: derived speed above this is almost certainly a
    # duration-near-zero artefact, not a real trip.
    if "speed_kmh" in df.columns:
        mask &= df["speed_kmh"].le(speed_cap_kmh) | df["speed_kmh"].isna()

    df = df.loc[mask].copy()
    return df


def add_target_columns(
    df: pd.DataFrame, cols: ColumnMap, cfg: ModelingConfig
) -> tuple[pd.DataFrame, dict]:
    """
    Adds surge_multiplier (from real column) and surge_event (binary).
    surge_event = 1 when surge_multiplier > threshold (0 = any positive, per data provider).
    """
    meta = {"modeling_config": asdict(cfg)}

    # Use surge_multiplier or surged column (JSON/CSV may use different names)
    surge_col = cols.surge_multiplier if cols.surge_multiplier in df.columns else None
    if surge_col is None and "surged" in df.columns:
        surge_col = "surged"
    if surge_col is None:
        raise ValueError(
            "surge_multiplier (or 'surged') column is required. "
            f"Found columns: {list(df.columns)}"
        )

    df["surge_multiplier"] = pd.to_numeric(df[surge_col], errors="coerce")
    meta["target_mode_used"] = "real_surge"

    # Primary ("any uplift") definition, per data provider: surge > 0
    df["surge_event"] = (df["surge_multiplier"] > cfg.surge_event_threshold).astype("int8")

    # Secondary ("economically material") definition: multiplier >= 1.2
    # Reported side-by-side so readers can see how the picture shifts when
    # only real surges are counted.
    material_thr = getattr(cfg, "surge_event_threshold_material", 1.2)
    df["surge_event_material"] = (df["surge_multiplier"] >= material_thr).astype("int8")
    meta["surge_event_threshold"] = float(cfg.surge_event_threshold)
    meta["surge_event_threshold_material"] = float(material_thr)
    return df, meta


def preprocess_trips(
    data_path: Path,
    cols: ColumnMap,
    cfg: ModelingConfig,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    End-to-end preprocessing used by notebooks and Streamlit.
    Automatically detects JSON vs CSV format.
    """
    if data_path.suffix.lower() == '.json':
        df = read_trips_json(data_path, nrows=nrows)
    else:
        df = read_trips_csv(data_path, nrows=nrows)
    
    df = df.copy()

    # Ensure timestamp is parsed (JSON loader already does this, but CSV needs it)
    if not pd.api.types.is_datetime64_any_dtype(df[cols.ts]):
        df[cols.ts] = parse_timestamp_utc(df[cols.ts])
    df = df.dropna(subset=[cols.ts, cols.src_lat, cols.src_lon, cols.dst_lat, cols.dst_lon])

    df, meta = preprocess_trips_df(df, cols=cols, cfg=cfg, apply_filters=True)

    # Light categorical cleanup
    if cols.service_type in df.columns:
        df[cols.service_type] = df[cols.service_type].astype("category")

    return df, meta


def preprocess_trips_df(
    df: pd.DataFrame,
    *,
    cols: ColumnMap,
    cfg: ModelingConfig,
    apply_filters: bool = True,
    add_target: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Preprocesses an *already loaded* trips dataframe.

    This is used for:
    - notebook experimentation without re-reading CSV
    - Streamlit single-row inference (apply_filters=False, add_target=False)
    """
    meta: dict = {}
    df = df.copy()

    # Timestamp parsing
    if not (pd.api.types.is_datetime64_any_dtype(df[cols.ts]) or pd.api.types.is_datetime64tz_dtype(df[cols.ts])):
        df[cols.ts] = parse_timestamp_utc(df[cols.ts])
    df = df.dropna(subset=[cols.ts, cols.src_lat, cols.src_lon, cols.dst_lat, cols.dst_lon])

    local_tz = getattr(cfg, "local_tz", "Africa/Nairobi")
    speed_cap = float(getattr(cfg, "speed_cap_kmh", 140.0))

    df = add_temporal_features(df, cols.ts, local_tz=local_tz)
    df = add_spatial_features(df, cols)
    df = add_high_demand_zone(df, cols)
    df = add_trip_quality_features(df, cols)

    if apply_filters:
        df = clean_outliers(df, cols, speed_cap_kmh=speed_cap)

    if add_target:
        df, meta = add_target_columns(df, cols, cfg)
    else:
        meta["target_mode_used"] = None

    # Light categorical cleanup
    if cols.service_type in df.columns:
        df[cols.service_type] = df[cols.service_type].astype("category")

    return df, meta


def maybe_merge_weather_hourly(
    df: pd.DataFrame,
    weather_csv: Path,
    *,
    ts_col: str = "ts_utc",
) -> tuple[pd.DataFrame, dict]:
    """
    Optional weather merge. Expected schema for weather_csv (hourly):
    - ts_utc (ISO timestamp)
    - temp_c, precip_mm, wind_kph, humidity, ...

    If file doesn't exist, returns df unchanged with a note.
    """
    meta: dict = {}
    if not weather_csv.exists():
        meta["weather_merge"] = f"SKIPPED (file not found): {weather_csv}"
        return df, meta

    w = pd.read_csv(weather_csv)
    if "ts_utc" not in w.columns:
        raise ValueError("weather_csv must include a 'ts_utc' column for merging.")

    w["ts_utc"] = pd.to_datetime(w["ts_utc"], utc=True, errors="coerce")
    w = w.dropna(subset=["ts_utc"]).copy()

    # Round trip timestamps down to hour for merge
    df = df.copy()
    df["ts_hour"] = df[ts_col].dt.floor("h")
    w["ts_hour"] = w["ts_utc"].dt.floor("h")

    w = w.drop(columns=["ts_utc"])
    df = df.merge(w, on="ts_hour", how="left")
    meta["weather_merge"] = f"OK (merged on ts_hour) from {weather_csv.name}"
    return df, meta


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

