from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def maps_dir(self) -> Path:
        return self.outputs_dir / "maps"

    @property
    def models_dir(self) -> Path:
        return self.outputs_dir / "models"


@dataclass(frozen=True)
class ColumnMap:
    """
    Central place to map raw dataset column names to standard names used across
    notebooks/models/app.

    If your client's dataset has different names, update these here and
    everything should keep running.
    """

    ts: str = "created_at"

    # Coordinates in this dataset:
    # - sourceLocation[0] ~ latitude  (-1.28...)
    # - sourceLocation[1] ~ longitude (36.84...)
    src_lat: str = "sourceLocation[0]"
    src_lon: str = "sourceLocation[1]"
    dst_lat: str = "destinationLocation[0]"
    dst_lon: str = "destinationLocation[1]"

    distance_km: str = "total_distance"
    duration_min: str = "total_time"

    # Surge multiplier: can be 'surge_multiplier' (from CSV) or 'surged' (from JSON)
    surge_multiplier: str = "surge_multiplier"

    # IDs / categories
    service_type: str = "service_type_id"


@dataclass(frozen=True)
class ModelingConfig:
    """
    Defines target behavior. Real surge_multiplier column is required
    (from data provider; mapped from 'surged' in JSON if needed).
    """

    target_mode: str = "real_surge"  # Uses real surge column from data

    # Classification target (per data provider): surge_event when surge > 0
    # Primary ("any uplift") threshold – what the provider sees.
    surge_event_threshold: float = 0.0

    # Secondary ("economically material") threshold – industry convention.
    # Reported side-by-side with the primary threshold in the leaderboard so
    # readers can compare "any uplift" vs "real surge" definitions.
    surge_event_threshold_material: float = 1.2

    # Surge/no-surge decision threshold: probability >= this → predict "surge"
    surge_decision_threshold: float = 0.4  # Best F1 at 0.4; better recall than 0.5

    # ---- Data-quality filters ---------------------------------------------
    # Hard cap for the derived speed_kmh column. Values above this indicate
    # residual division-by-near-zero artefacts (raw max was ~123,404 km/h).
    speed_cap_kmh: float = 140.0

    # Local timezone used to derive hour / peak flags from the UTC timestamp.
    # Nairobi is UTC+3 year-round.
    local_tz: str = "Africa/Nairobi"


@dataclass(frozen=True)
class RunConfig:
    """
    Centralized run settings for notebooks and reproducibility.
    For final thesis run, set model_n=None (use full data).
    """
    model_n: Optional[int] = None  # None = use all data (final run); int = use last N rows by time
    test_size: float = 0.2  # fraction of data for test (last in time)
    val_ratio: float = 0.15  # fraction of *training* period used as validation (last part of train)
    random_state: int = 42


def default_paths() -> ProjectPaths:
    # src/ is inside the project root, so root is parent of this file's parent.
    project_root = Path(__file__).resolve().parents[1]
    return ProjectPaths(project_root=project_root)

