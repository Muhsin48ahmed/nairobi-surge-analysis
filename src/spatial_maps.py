from __future__ import annotations

from pathlib import Path

import pandas as pd


def nairobi_map_center() -> tuple[float, float]:
    # Rough center of Nairobi
    return (-1.286389, 36.817223)


def to_heatmap_points(df: pd.DataFrame, lat_col: str, lon_col: str, weight_col: str | None = None):
    """
    Returns data in the format expected by folium.plugins.HeatMap.
    """
    if weight_col is None:
        return df[[lat_col, lon_col]].dropna().values.tolist()
    pts = df[[lat_col, lon_col, weight_col]].dropna().values.tolist()
    return pts


def build_folium_heatmap(
    df: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    weight_col: str | None = None,
    map_tiles: str = "CartoDB positron",
    radius: int = 12,
    blur: int = 18,
    max_zoom: int = 15,
) -> "folium.Map":
    import folium
    from folium.plugins import HeatMap

    m = folium.Map(location=nairobi_map_center(), zoom_start=12, tiles=map_tiles)
    HeatMap(
        to_heatmap_points(df, lat_col, lon_col, weight_col),
        radius=radius,
        blur=blur,
        max_zoom=max_zoom,
    ).add_to(m)
    return m


def build_folium_prediction_map(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    map_tiles: str = "CartoDB positron",
) -> "folium.Map":
    """Build a simple map with pickup (green) and dropoff (red) markers."""
    import folium

    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lon + dropoff_lon) / 2
    m = folium.Map(
        location=(center_lat, center_lon),
        zoom_start=14,
        tiles=map_tiles,
    )
    folium.Marker(
        (pickup_lat, pickup_lon),
        popup="Pickup",
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(m)
    folium.Marker(
        (dropoff_lat, dropoff_lon),
        popup="Dropoff",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    return m


def save_folium_map(m: "folium.Map", out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def grid_hotspots(
    df: pd.DataFrame,
    *,
    grid_col: str,
    value_col: str,
    min_trips: int = 50,
) -> pd.DataFrame:
    """
    Aggregates by a grid id (e.g., pickup_grid) and computes:
    - trips
    - avg value (e.g., avg surge_multiplier)

    Returned df is sorted by avg value then trips.
    """
    agg = (
        df.groupby(grid_col, observed=True)
        .agg(trips=(grid_col, "size"), avg_value=(value_col, "mean"))
        .reset_index()
    )
    agg = agg[agg["trips"] >= min_trips].copy()
    return agg.sort_values(["avg_value", "trips"], ascending=[False, False]).reset_index(drop=True)


def attach_grid_centroids(
    hotspots: pd.DataFrame,
    *,
    grid_col: str,
    cell_size_deg: float = 0.005,
) -> pd.DataFrame:
    """
    Converts our simple 'latbin_lonbin' grid to approximate centroid coordinates.
    """
    parts = hotspots[grid_col].str.split("_", expand=True)
    lat_bin = parts[0].astype("int64")
    lon_bin = parts[1].astype("int64")
    hotspots = hotspots.copy()
    hotspots["lat"] = (lat_bin + 0.5) * cell_size_deg
    hotspots["lon"] = (lon_bin + 0.5) * cell_size_deg
    return hotspots


def build_hotspot_marker_map(
    hotspots: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    label_col: str = "avg_value",
    size_col: str = "trips",
    map_tiles: str = "CartoDB positron",
) -> "folium.Map":
    import folium

    m = folium.Map(location=nairobi_map_center(), zoom_start=12, tiles=map_tiles)
    max_trips = float(hotspots[size_col].max()) if len(hotspots) else 1.0

    for _, r in hotspots.iterrows():
        radius = 4 + 10 * (float(r[size_col]) / max_trips)
        folium.CircleMarker(
            location=(float(r[lat_col]), float(r[lon_col])),
            radius=radius,
            color="#d62728",
            fill=True,
            fill_opacity=0.6,
            popup=f"{label_col}: {r[label_col]:.3f}<br>{size_col}: {int(r[size_col])}",
        ).add_to(m)
    return m


def kmeans_cluster_surge_hotspots(
    df: pd.DataFrame,
    *,
    n_clusters: int = 8,
    lat_col: str = "lat",
    lon_col: str = "lon",
    value_col: str = "avg_value",
    random_state: int = 42,
) -> tuple[pd.DataFrame, "KMeans"]:
    """
    K-means clustering on (lat, lon, value) to identify surge-prone zones.
    Proposal Table 3.6.3: spatial clustering (K-means, DBSCAN) for surge-prone neighborhoods.
    Expects df from grid_hotspots + attach_grid_centroids (has lat, lon, avg_value, trips).
    Returns (df with cluster column, fitted KMeans model).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not in df")
    if len(df) < n_clusters:
        return df, KMeans(n_clusters=n_clusters, random_state=random_state)

    X = df[[lat_col, lon_col, value_col]].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    return df, km


def kmeans_surge_clusters(
    df: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    value_col: str = "surge_multiplier",
    n_clusters: int = 8,
    min_samples: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    K-means clustering on (lat, lon, avg_surge) to identify surge-prone zones.
    Proposal Table 3.6.3: spatial clustering (K-means, DBSCAN) for surge-prone neighborhoods.
    Returns cluster centers with mean surge and trip count.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Aggregate by grid to reduce noise and get avg surge per location
    agg = df.groupby([lat_col, lon_col], observed=True).agg(
        avg_value=(value_col, "mean"),
        trips=(value_col, "size"),
    ).reset_index()
    agg = agg[agg["trips"] >= min_samples]

    if len(agg) < n_clusters:
        return pd.DataFrame()

    # Features: lat, lon, avg_value (normalize for K-means)
    X = agg[[lat_col, lon_col, "avg_value"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    agg["cluster_id"] = km.fit_predict(X_scaled)

    cluster_summary = (
        agg.groupby("cluster_id")
        .agg(
            lat=(lat_col, "mean"),
            lon=(lon_col, "mean"),
            avg_surge=("avg_value", "mean"),
            trips=("trips", "sum"),
        )
        .reset_index()
    )
    return cluster_summary.sort_values("avg_surge", ascending=False)

