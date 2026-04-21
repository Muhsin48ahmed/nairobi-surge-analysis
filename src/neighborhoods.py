from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_neighborhood_boundaries(path: Path) -> gpd.GeoDataFrame:
    """
    Loads a Nairobi neighborhood boundary file (GeoJSON / Shapefile / GPKG).

    Expected: a polygon layer with a neighborhood name column (varies by source).
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        # Most admin boundary files are WGS84; if unknown, assume EPSG:4326.
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return gdf


def detect_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    """
    Heuristic to pick a readable neighborhood name column.
    Returns None if no suitable column is found (caller then uses index-based names).
    """
    candidates = [
        "name", "NAME", "Name",
        "NAME_1", "NAME_2", "NAME_3",
        "neighborhood", "Neighbourhood", "NEIGHBOURHOOD",
        "NEIGHBORHOOD", "NBHD_NAME", "EST_NAME", "ESTATE",
        "ward", "WARD", "WARD_NAME",
        "subcounty", "SUBCOUNTY", "SUB_COUNTY",
        "constituency", "CONSTITUENCY", "CONST",
    ]
    for c in candidates:
        if c in gdf.columns:
            # reject anonymised id-like columns
            sample = gdf[c].astype(str).str.lower().head(5).tolist()
            if any(s.startswith("neighborhood_") for s in sample):
                continue
            return c
    # Fallback: first non-geometry object column that doesn't look anonymised
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        if gdf[c].dtype == "object":
            sample = gdf[c].astype(str).str.lower().head(5).tolist()
            if any(s.startswith("neighborhood_") for s in sample):
                continue
            return c
    return None


def spatial_join_points_to_polygons(
    df: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    polygons: gpd.GeoDataFrame,
    polygon_name_col: str | None = None,
    out_col: str = "pickup_neighborhood",
) -> pd.DataFrame:
    """
    Adds a neighborhood/area label to each point using a polygon spatial join.
    If no name column exists, uses index-based names (neighborhood_0, neighborhood_1, ...).
    """
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=4326,
    )

    name_col = polygon_name_col or detect_name_column(polygons)
    
    # If no name column, create one from index
    if name_col is None:
        polygons = polygons.copy()
        name_col = out_col
        polygons[name_col] = polygons.index.astype(str).map(lambda x: f"neighborhood_{x}")
    
    keep_cols = [name_col, polygons.geometry.name]
    polygons_small = polygons[keep_cols].rename(columns={name_col: out_col})

    joined = gpd.sjoin(gdf_points, polygons_small, how="left", predicate="within")
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    return pd.DataFrame(joined)


def export_hotspots_to_geojson(
    hotspots: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    out_geojson: Path,
    crs_epsg: int = 4326,
) -> None:
    """
    Exports hotspot points (with metrics) to GeoJSON for QGIS.
    """
    gdf = gpd.GeoDataFrame(
        hotspots.copy(),
        geometry=gpd.points_from_xy(hotspots[lon_col], hotspots[lat_col]),
        crs=crs_epsg,
    )
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")

