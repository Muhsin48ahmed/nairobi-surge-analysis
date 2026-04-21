from __future__ import annotations

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance in kilometers.

    Inputs can be scalars or array-like; output is a numpy array.
    """
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    r = 6371.0088  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Initial bearing from point 1 to point 2 in degrees [0, 360).
    """
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    theta = np.arctan2(x, y)
    deg = (np.degrees(theta) + 360) % 360
    return deg


def latlon_grid_id(lat: np.ndarray, lon: np.ndarray, cell_size_deg: float = 0.005) -> np.ndarray:
    """
    Simple spatial binning without external dependencies.

    cell_size_deg ~ 0.005 degrees ≈ 500–600m in Nairobi latitude.
    Output is a string grid id like "latbin_lonbin".
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    lat_bin = np.floor(lat / cell_size_deg).astype("int64")
    lon_bin = np.floor(lon / cell_size_deg).astype("int64")
    return lat_bin.astype(str) + "_" + lon_bin.astype(str)

