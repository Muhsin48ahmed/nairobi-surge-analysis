# Nairobi Ride-Hailing Surge Pricing Prediction & Spatial Mapping

A data science project analysing surge pricing dynamics in Nairobi's ride-hailing industry using machine learning and spatial analysis.

## What This Project Does

This project predicts surge pricing events and multipliers for ride-hailing trips in Nairobi by analysing temporal patterns, spatial hotspots, and trip characteristics. It covers:

- **Data preprocessing** with feature engineering (temporal, spatial, weather integration).
- **Exploratory data analysis** with visualisations and spatial mapping.
- **Machine learning models** (Logistic Regression, Random Forest, XGBoost, MLP) for both classification (surge yes/no) and regression (surge multiplier).
- **Interactive Streamlit dashboard** for exploring results and making single-trip predictions.
- **GIS mapping** of surge hotspots across Nairobi neighbourhoods.
- **Qualitative triangulation** from interview coding used in the thesis discussion chapter.

## Data

- **Trip data:** Ride-hailing platform trip-level records (2023–2024), anonymised. Stored as `data/New Files/trip_analysis_data.json` (or a CSV fallback). Each row has a timestamp, GPS pickup/dropoff, distance, duration, service type, and surge multiplier. The implementation uses the full ~3.7M trips end-to-end.
- **Weather:** Hourly weather for Nairobi from the [Open-Meteo Archive API](https://open-meteo.com), merged to trips by hour. Features used: `rain_flag` (1 if precipitation > 0), `precip_mm`, `temp_c`, `humidity`, and `wind_kph`.
- **Timezone:** All hour / peak / cyclical features are derived in **Africa/Nairobi local time (UTC+3)**, not UTC. The conversion lives in `src.data.add_temporal_features`.
- **Speed cap:** `speed_kmh` is capped at 140 km/h during cleaning; raw values above this are division-by-near-zero artefacts.
- **Neighbourhoods:** The shipped shapefile contains polygon boundaries joined to each pickup point in `src.neighborhoods.spatial_join_points_to_polygons`.

**Reproducibility.** Use Python 3.9 or 3.10. Key random seeds are set to `42`. For exact reproducibility install from `requirements-pinned.txt`: `pip install -r requirements-pinned.txt`. Run settings (sample size, train/val/test split) are in `src.config.RunConfig`.

## Live app (Streamlit Cloud)

- Dashboard: https://nairobi-surge-analysis-czq7b3kpgxgdttuq4csvud.streamlit.app/

## Repository structure

- `app/app.py`: Streamlit dashboard entry point
- `src/`: data preprocessing + feature engineering + modeling utilities
- `notebooks/`: analysis and modeling notebooks used for the thesis/report
- `data/sample/trips_sample_rawschema.csv`: public-safe sample dataset used for the cloud demo
- `outputs/`: generated artifacts (some are ignored by git due to size)

## Getting Started

### Prerequisites

Python 3.9 or 3.10. The project uses standard data science libraries (pandas, scikit-learn, xgboost, lightgbm, geopandas, folium, streamlit).

### Installation

1. **Clone or extract the project** to your local machine.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-pinned.txt
   ```

## How to Use

### Option 1: Use Pre-Generated Results (Quickest)

If you just want to explore the results without re-running training:

1. **View the report**: Open `report/Ahmed_Muhsin_ProjectReport.pdf` (submission copy) or `report/Ahmed_Muhsin_ProjectReport.docx` for the written analysis.

2. **Explore visualisations**: PNG figures live in `outputs/figures/` (EDA and model diagnostics).

3. **View interactive maps**: Open any HTML file in `outputs/maps/` in your browser, e.g.:

   - `pickup_heatmap_weighted.html` — pickup density heatmap
   - `dropoff_heatmap.html` — dropoff density heatmap
   - `pickup_hotspots_top250.html` — top surge hotspots

4. **Run the dashboard**:
   ```bash
   streamlit run app/app.py
   ```
   Then open the URL shown (usually `http://localhost:8501`). The dashboard uses a single page with three tabs:
   - **Overview** — headline KPIs (trips loaded, mean surge multiplier, surge event %) and mean surge by hour.
   - **Hotspot map** — pickup heatmap filtered by hour of day.
   - **Predict** — single-trip prediction results (surge probability, predicted multiplier, map) using the sidebar form.

### Option 2: Run the Full Analysis (Reproduce Results)

To regenerate all outputs from scratch, run the notebooks in order.

1. **Start Jupyter**:

   ```bash
   jupyter notebook
   ```

2. **Run the notebooks in order**:

   - **Notebook 01** (`notebooks/01_data_eda.ipynb`)

     - Loads and preprocesses trip data from `data/New Files/trip_analysis_data.json`.
     - Creates temporal and spatial features.
     - Merges weather data.
     - Generates EDA visualisations → `outputs/figures/`.
     - Creates spatial maps → `outputs/maps/`.
     - Saves the processed dataset → `outputs/trips_processed.parquet`.

   - **Notebook 02** (`notebooks/02_modeling.ipynb`)

     - Loads the processed data.
     - Trains Logistic Regression, Random Forest, XGBoost, and MLP for classification and regression.
     - Evaluates with ROC, precision–recall, calibration, and regression diagnostics.
     - Saves the best models → `outputs/models/`.
     - Saves metrics → `outputs/metrics.json`.

   - **Notebook 03** (`notebooks/03_streamlit_logic.ipynb`)

     - Documents the Streamlit dashboard architecture and verifies required artefacts exist.

   - **Notebook 04** (`notebooks/04_qualitative_interviews.ipynb`)
     - Processes the interview coding CSV.
     - Produces frequency tables and figures for the qualitative chapter.
     - Exports thematic artefacts → `outputs/qualitative/`.

3. **Run the dashboard** (same as Option 1).

## Project Structure

```
.
├── app/
│   └── app.py                    # Streamlit dashboard
├── notebooks/
│   ├── 01_data_eda.ipynb         # Data preprocessing & EDA
│   ├── 02_modeling.ipynb         # Model training & evaluation
│   ├── 03_streamlit_logic.ipynb  # Dashboard architecture
│   └── 04_qualitative_interviews.ipynb  # Interview thematic analysis
├── scripts/
│   ├── fetch_weather_nairobi.py  # Fetch hourly weather from Open-Meteo
│   ├── export_notebooks_to_pdf.py
│   ├── patch_notebooks.py
│   ├── run_qualitative_analysis.py
│   └── README.md
├── src/                          # Core Python modules
│   ├── config.py                 # Configuration dataclasses
│   ├── data.py                   # Data loading & preprocessing
│   ├── modeling.py               # Model training utilities
│   ├── neighborhoods.py          # GIS / spatial joins
│   ├── spatial_maps.py           # Map generation
│   └── utils_geo.py              # Haversine, bearing, etc.
├── data/
│   └── New Files/
│       ├── trip_analysis_data.json   # Main trip data (~3.7M rows)
│       ├── nairobi_neighborhoods.shp # Neighbourhood boundaries
│       └── weather_nairobi_hourly.csv
├── outputs/
│   ├── figures/                  # Visualisation PNGs
│   ├── maps/                     # Interactive HTML maps
│   ├── models/                   # Trained models (.joblib)
│   ├── qualitative/              # Interview frequency tables
│   ├── metrics.json              # Model performance metrics
│   └── trips_processed.parquet   # Processed dataset
└── report/
    ├── Abdule_Muhsin_ProjectReport.pdf    # Submission copy (PDF)
    └── Abdule_Muhsin_ProjectReport.docx   # Source Word document
```

## Key Features

### Data Processing

- Handles ~3.7M trip records with real surge multipliers.
- Extracts temporal features (hour, day of week, peak periods) in Nairobi local time.
- Computes spatial features (Haversine distance, bearing).
- Merges hourly weather data.
- Joins pickup points to Nairobi neighbourhood polygons.

### Models Trained

- **Logistic Regression** — classification baseline, `class_weight="balanced"`.
- **Random Forest** — classification + regression, `class_weight="balanced"`.
- **XGBoost** — classification + regression, `scale_pos_weight` set from train prevalence.
- **MLP (neural net)** — classification + regression on numeric features, early stopping.

The current test metrics (time-based test window, last 20% by calendar time) are written to `outputs/metrics.json`. Treat `metrics.json` as the source of truth rather than hard-coded numbers in this README.

### Class Imbalance

The dataset has extreme class imbalance (~253 non-surge trips per surge event). Handled per model:

- **Logistic Regression & Random Forest**: `class_weight="balanced"`.
- **XGBoost**: `scale_pos_weight` ≈ 253.
- **MLP**: early stopping + balanced batches.

### Spatial Insights

- Surge hotspots cluster in high-traffic areas (CBD, Westlands, the airport corridor).
- Clear morning- and evening-peak temporal patterns.
- Interactive heatmaps in `outputs/maps/` allow spatial exploration.

## Notes

- The project uses **real surge multipliers** from the `surged` column in the JSON data.
- Notebook 02 trains on a sample by default for faster iteration; adjust `MODEL_N` in the notebook to train on the full dataset.
- Weather data may not overlap with every trip date, so some weather features may be null.
- All generated outputs land in `outputs/`.

## Data Access Statement

The full ride-hailing trip dataset used in this project contains granular spatio-temporal trip traces (pickup/dropoff locations and timestamps) and is therefore not included in this public repository.

The full ride-hailing trip dataset used in this project is approximately 1.4GB (JSON) and is not stored in this GitHub repository due to GitHub file-size limits and privacy/data‑protection considerations.

- Access: Available to supervisor

- After downloading, place the file at:
  `data/New Files/trip_analysis_data.json`

A public, de‑identified sample is included in this repository for demonstration/testing and Streamlit deployment:

- `data/sample/trips_sample_rawschema.csv`

## Troubleshooting

- **Import errors?** Activate your virtual environment and re-install requirements.
- **Notebooks won't run?** Make sure you launch Jupyter from the project root so relative paths resolve.
- **Dashboard won't start?** Check that model files exist in `outputs/models/` and `outputs/trips_processed.parquet` is present.
- **Out of memory?** Reduce `NROWS` in Notebook 01 or `MODEL_N` in Notebook 02.

## AI Use Disclosure

This project used AI assistants in a supervised capacity during development for:

- Suggesting code structure for Python helpers (`src/`), notebooks, and the Streamlit dashboard; all code was reviewed, executed, and verified by the author.
- Debugging error messages and proposing fixes.

No AI was used to fabricate data, results, or references. Every citation in the report has been verified to exist. A full AI Use Disclosure section appears at the end of `report/Abdule_Muhsin_ProjectReport.pdf`.

## Submission

- **Student:** Muhsin Ahmed (ID 665629)
- **Programme:** BSc Data Science, USIU-Africa
- **Module:** DSA 4900A Final-Year Project
- **Report file:** `report/Abdule_Muhsin_ProjectReport.pdf`
- **Repository:** this GitHub repository (link submitted with hardbound thesis)

## Contact

For questions about methodology or results, refer to `report/Abdule_Muhsin_ProjectReport.pdf`.
