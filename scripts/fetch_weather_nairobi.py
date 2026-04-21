"""
Fetch hourly weather for Nairobi from Open-Meteo Archive API and save as
data/weather_nairobi_hourly.csv (ts_utc, temp_c, precip_mm, wind_kph, humidity).
Overlaps trip date range (e.g. 2024-01-01 to 2025-05-31). Fetches in 6-month
chunks to avoid timeouts.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

# Nairobi (project uses same coords in spatial_maps)
LAT, LON = -1.286389, 36.817223
# Chunks to avoid API timeout (each ~4k rows)
CHUNKS = [
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-05-31"),
]
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "weather_nairobi_hourly.csv"
TEMP_PATH = DATA_DIR / "weather_nairobi_hourly_new.csv"


def fetch_weather(start: str, end: str) -> dict:
    url = (
        f"{BASE_URL}?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.load(resp)


def hourly_to_rows(hourly: dict) -> list[dict]:
    times = hourly["time"]
    n = len(times)
    rows = []
    for i in range(n):
        ts = times[i]
        if len(ts) == 16 and not ts.endswith("Z") and "+" not in ts:
            ts_utc = f"{ts}+00:00"
        else:
            ts_utc = ts
        rows.append({
            "ts_utc": ts_utc,
            "temp_c": hourly["temperature_2m"][i],
            "precip_mm": hourly["precipitation"][i],
            "wind_kph": hourly["wind_speed_10m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
        })
    return rows


def main() -> None:
    all_rows = []
    for start, end in CHUNKS:
        print(f"Fetching {start} to {end}...")
        data = fetch_weather(start, end)
        hourly = data.get("hourly", {})
        if not hourly:
            raise RuntimeError(f"No hourly data for {start}–{end}")
        all_rows.extend(hourly_to_rows(hourly))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Write to temp file first in case weather_nairobi_hourly.csv is open/locked
    with open(TEMP_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts_utc", "temp_c", "precip_mm", "wind_kph", "humidity"])
        w.writeheader()
        w.writerows(all_rows)
    try:
        TEMP_PATH.replace(OUT_PATH)
        print(f"Saved {len(all_rows):,} hourly rows to {OUT_PATH}")
    except OSError:
        print(f"Saved {len(all_rows):,} rows to {TEMP_PATH}. Copy it to {OUT_PATH.name} when the file is not in use.")


if __name__ == "__main__":
    main()
