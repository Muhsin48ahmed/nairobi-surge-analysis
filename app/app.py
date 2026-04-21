from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Allow running from repo root: streamlit run app/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure `src/` imports work when Streamlit runs from a different working directory.
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ColumnMap, ModelingConfig  # noqa: E402
from src.data import preprocess_trips, preprocess_trips_df  # noqa: E402
from src.spatial_maps import build_folium_heatmap  # noqa: E402
try:
    from src.spatial_maps import build_folium_prediction_map  # noqa: E402
except ImportError:
    # Fallback when build_folium_prediction_map import fails (e.g. missing folium plugins)
    def build_folium_prediction_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, map_tiles="CartoDB positron"):
        import folium
        center_lat = (pickup_lat + dropoff_lat) / 2
        center_lon = (pickup_lon + dropoff_lon) / 2
        m = folium.Map(location=(center_lat, center_lon), zoom_start=14, tiles=map_tiles)
        folium.Marker((pickup_lat, pickup_lon), popup="Pickup", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        folium.Marker((dropoff_lat, dropoff_lon), popup="Dropoff", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        return m


st.set_page_config(
    page_title="Nairobi Surge Prediction & Hotspots",
    layout="wide",
)


@st.cache_data(show_spinner=True)
def load_processed_data(max_rows: int | None = 200_000) -> pd.DataFrame:
    """
    Loads a processed dataset. If a saved parquet exists, use it. Otherwise
    preprocess from JSON or CSV (optionally limited for faster local demos).
    """
    parquet_path = PROJECT_ROOT / "outputs" / "trips_processed.parquet"
    sample_csv_path = PROJECT_ROOT / "data" / "sample" / "trips_sample_rawschema.csv"
    json_path = PROJECT_ROOT / "data" / "New Files" / "trip_analysis_data.json"
    csv_path = PROJECT_ROOT / "data" / "trips_raw.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    cols = ColumnMap()
    cfg = ModelingConfig()
    # Prefer a public-safe sample (for Streamlit Cloud), then full JSON, then CSV fallback.
    if sample_csv_path.exists():
        data_path = sample_csv_path
    else:
        data_path = json_path if json_path.exists() else csv_path
    df, _meta = preprocess_trips(data_path=data_path, cols=cols, cfg=cfg, nrows=max_rows)
    return df


@st.cache_resource(show_spinner=True)
def load_models():
    models_dir = PROJECT_ROOT / "outputs" / "models"
    metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"
    paths = {
        "clf": models_dir / "clf_best.joblib",
        "reg": models_dir / "reg_best.joblib",
    }
    out = {}
    for k, p in paths.items():
        out[k] = joblib.load(p) if p.exists() else None
    out["regression_target"] = "raw"
    if metrics_path.exists():
        try:
            meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            out["regression_target"] = meta.get("regression_target", "raw")
        except Exception:
            pass
    return out


def main():
    st.title("Nairobi Surge Decision-Support Dashboard")
    st.caption(
        "Explore surge patterns, hotspot maps, and get surge probability and multiplier for a trip."
    )
    st.divider()

    df = load_processed_data()
    models = load_models()

    # ---- Sidebar: prediction form ----
    with st.sidebar:
        st.subheader("Surge prediction")
        st.caption("Enter trip details and click Predict. Results appear in the **Predict** tab.")
        now_utc = datetime.now(timezone.utc)
        if "trip_date" not in st.session_state:
            st.session_state.trip_date = now_utc.date()
        if "trip_time" not in st.session_state:
            st.session_state.trip_time = now_utc.time()
        d = st.date_input(
            "Trip request date (UTC)",
            value=st.session_state.trip_date,
            key="trip_date",
        )
        t = st.time_input(
            "Trip request time (UTC)",
            value=st.session_state.trip_time,
            key="trip_time",
        )
        dt = datetime.combine(d, t, tzinfo=timezone.utc)
        service_type_id = st.text_input("service_type_id (optional)", value="")
        st.write("Pickup / dropoff (Nairobi)")
        p_lat = st.number_input("Pickup lat", value=-1.2864, format="%.6f")
        p_lon = st.number_input("Pickup lon", value=36.8172, format="%.6f")
        d_lat = st.number_input("Dropoff lat", value=-1.2921, format="%.6f")
        d_lon = st.number_input("Dropoff lon", value=36.8219, format="%.6f")
        dist_km = st.number_input("Estimated trip distance (km)", value=5.0, min_value=0.1, step=0.1)
        dur_min = st.number_input(
            "Estimated trip duration (min)",
            value=15.0,
            min_value=0.5,
            step=0.5,
        )
        raining = st.radio("Raining?", options=["No", "Yes"], horizontal=True)
        rain_flag = 1 if raining == "Yes" else 0

        if st.button("Predict"):
            if models["clf"] is None or models["reg"] is None:
                st.error("Models not found. Run `notebooks/02_modeling.ipynb` first.")
            else:
                row_raw = pd.DataFrame(
                    [
                        {
                            "created_at": dt.isoformat().replace("+00:00", "Z"),
                            "service_type_id": service_type_id or "unknown",
                            "sourceLocation[0]": p_lat,
                            "sourceLocation[1]": p_lon,
                            "destinationLocation[0]": d_lat,
                            "destinationLocation[1]": d_lon,
                            "total_distance": dist_km,
                            "total_time": dur_min,
                            "city": "Nairobi",
                            "is_provider_status": 9,
                            "source_address": "",
                            "destination_address": "",
                            "user_id": "",
                            "driver_id": "",
                            "unique_id": "ad_hoc",
                        }
                    ]
                )
                cols = ColumnMap()
                cfg = ModelingConfig()
                row_feat, _ = preprocess_trips_df(
                    row_raw,
                    cols=cols,
                    cfg=cfg,
                    apply_filters=False,
                    add_target=False,
                )
                row_feat["rain_flag"] = rain_flag
                for col, default in (
                    ("temp_c", np.nan),
                    ("precip_mm", np.nan),
                    ("wind_kph", np.nan),
                    ("humidity", np.nan),
                ):
                    if col not in row_feat.columns:
                        row_feat[col] = default
                try:
                    proba = float(models["clf"].predict_proba(row_feat)[0, 1])
                    pred_mult = float(models["reg"].predict(row_feat)[0])
                    if models.get("regression_target") == "log1p":
                        pred_mult = float(np.expm1(pred_mult))
                except (ValueError, KeyError) as e:
                    st.error(
                        "Prediction failed: model expects different features. "
                        "Re-run `notebooks/02_modeling.ipynb` to retrain."
                    )
                    st.caption(f"Technical: {e!s}")
                else:
                    st.session_state["last_prediction_summary"] = {
                        "datetime_utc": str(dt),
                        "pickup_lat": p_lat,
                        "pickup_lon": p_lon,
                        "dropoff_lat": d_lat,
                        "dropoff_lon": d_lon,
                        "distance_km": dist_km,
                        "duration_min": dur_min,
                        "raining": raining,
                        "surge_probability": round(proba, 4),
                        "surge_multiplier": round(pred_mult, 4),
                    }
                    st.success("Done. Open the **Predict** tab for results.")

    # ---- Main area: tabs ----
    tab_overview, tab_hotspot, tab_predict = st.tabs(["Overview", "Hotspot map", "Predict"])

    with tab_overview:
        st.subheader("Exploratory view: patterns & hotspots")
        avg_mult = float(df["surge_multiplier"].mean())
        surge_pct = 100 * df["surge_event"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Trips loaded", f"{len(df):,}")
        c2.metric("Avg surge multiplier", f"{avg_mult:.2f}", delta=round(avg_mult - 1.0, 2))
        c3.metric("Surge event %", f"{surge_pct:.1f}%")
        surge_above_105 = 100 * (df["surge_multiplier"] > 1.05).mean()
        st.caption(f"{surge_above_105:.1f}% of trips in this dataset had surge multiplier > 1.05.")
        st.divider()
        st.write("**Surge multiplier by hour**")
        st.caption("Mean surge multiplier (y-axis) by hour of day (UTC).")
        by_hour = df.groupby("hour", observed=True)["surge_multiplier"].mean().reset_index()
        st.line_chart(by_hour, x="hour", y="surge_multiplier", use_container_width=True)

    with tab_hotspot:
        st.subheader("Pickup heatmap by hour")
        st.divider()
        hour = st.slider("Hour of day (UTC)", min_value=0, max_value=23, value=8)
        st.caption("Pickup heatmap for this hour.")
        df_h = df[df["hour"] == hour].sample(min(len(df[df["hour"] == hour]), 30_000), random_state=42)
        try:
            m = build_folium_heatmap(df_h, lat_col="sourceLocation[0]", lon_col="sourceLocation[1]")
            st.components.v1.html(m._repr_html_(), height=520)  # noqa: SLF001
        except Exception as e:
            st.info(f"Map render skipped (missing folium plugins?): {e}")

    with tab_predict:
        st.subheader("Surge prediction")
        st.caption("Enter trip details in the sidebar and click **Predict** to see results here.")
        if st.session_state.get("last_prediction_summary"):
            summary = st.session_state["last_prediction_summary"]
            cfg = ModelingConfig()
            surge_thresh = cfg.surge_decision_threshold
            proba = summary["surge_probability"]
            surge_expected = proba >= surge_thresh
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Surge probability", f"{proba:.2%}")
            with m2:
                st.metric("Predicted multiplier", f"{summary['surge_multiplier']:.2f}")
            with m3:
                st.metric("Surge expected", "Yes" if surge_expected else "No")
            st.caption(f"Surge/no-surge decision uses threshold {surge_thresh} (best F1).")
            st.divider()
            with st.expander("View inputs used", expanded=True):
                st.caption(
                    f"Date/time (UTC): {summary['datetime_utc']} · "
                    f"Pickup: ({summary['pickup_lat']:.4f}, {summary['pickup_lon']:.4f}) · "
                    f"Dropoff: ({summary['dropoff_lat']:.4f}, {summary['dropoff_lon']:.4f}) · "
                    f"Distance: {summary['distance_km']} km · Duration: {summary['duration_min']} min · "
                    f"Raining: {summary['raining']}"
                )
                try:
                    pred_map = build_folium_prediction_map(
                        summary["pickup_lat"], summary["pickup_lon"],
                        summary["dropoff_lat"], summary["dropoff_lon"],
                    )
                    st.components.v1.html(pred_map._repr_html_(), height=350)
                except Exception:
                    pass
            st.download_button(
                label="Download last prediction (JSON)",
                data=json.dumps(summary, indent=2),
                file_name="surge_prediction_summary.json",
                mime="application/json",
            )
            metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"
            if metrics_path.exists():
                st.download_button(
                    label="Download model metrics (JSON)",
                    data=metrics_path.read_text(encoding="utf-8"),
                    file_name="metrics.json",
                    mime="application/json",
                    key="download_metrics",
                )
        else:
            st.info("Use the **sidebar** to enter trip details and click **Predict**. Results will appear here.")


if __name__ == "__main__":
    main()
