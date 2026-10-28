"""Socioeconomic suitability scoring for rice cultivation.

Socioeconomic factors affect the practical feasibility of rice farming:
- Proximity to roads (input transport, output to market)
- Proximity to markets/towns (selling produce)
- Population density (labor availability)
- Infrastructure access
"""

import numpy as np
from pathlib import Path
from typing import Optional

from scipy.ndimage import distance_transform_edt
import rasterio


def score_distance_to_roads(distance: np.ndarray, max_distance: float = 50000) -> np.ndarray:
    """Score proximity to road network.

    Closer to roads = easier transport of inputs and outputs.
    Uses exponential decay.
    """
    norm_dist = np.clip(distance / max_distance, 0, 1)
    score = np.exp(-4 * norm_dist)
    return score.astype(np.float32)


def score_distance_to_markets(distance: np.ndarray, max_distance: float = 100000) -> np.ndarray:
    """Score proximity to markets/urban centers.

    Closer to markets = better prices, less spoilage.
    """
    norm_dist = np.clip(distance / max_distance, 0, 1)
    score = np.exp(-3 * norm_dist)
    return score.astype(np.float32)


def score_population_density(pop: np.ndarray) -> np.ndarray:
    """Score population density for labor availability.

    Too sparse: insufficient labor
    Too dense: land competition
    Optimal: moderate density (50-200 people/km²)
    """
    score = np.zeros_like(pop, dtype=np.float32)

    score[pop < 10] = 0.2       # Very sparse
    score[pop >= 10] = 0.4
    score[pop >= 30] = 0.7
    score[pop >= 50] = 0.9      # Optimal range
    score[pop >= 200] = 0.8     # Getting crowded
    score[pop >= 500] = 0.5     # Very urban
    score[pop >= 1000] = 0.2    # Dense urban

    return score


def score_socioeconomic(
    road_distance_path: Optional[Path] = None,
    market_distance_path: Optional[Path] = None,
    population_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    cell_size_meters: float = 1000.0,
) -> np.ndarray:
    """Compute combined socioeconomic suitability score.

    Uses available socioeconomic data layers.

    Returns:
        2D numpy array of scores (0-1)
    """
    scores = []
    meta = None

    if road_distance_path and Path(road_distance_path).exists():
        with rasterio.open(road_distance_path) as src:
            dist = src.read(1).astype(np.float32)
            meta = src.meta.copy()
        dist_meters = dist * cell_size_meters
        scores.append(("roads", score_distance_to_roads(dist_meters), 0.35))

    if market_distance_path and Path(market_distance_path).exists():
        with rasterio.open(market_distance_path) as src:
            dist = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
        dist_meters = dist * cell_size_meters
        scores.append(("markets", score_distance_to_markets(dist_meters), 0.35))

    if population_path and Path(population_path).exists():
        with rasterio.open(population_path) as src:
            pop = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
        scores.append(("population", score_population_density(pop), 0.30))

    if not scores:
        # If no socioeconomic data available, return neutral score
        # This prevents the criterion from being ignored in the overlay
        if meta is None:
            raise ValueError("At least one socioeconomic data source must be provided")
        return np.full((meta["height"], meta["width"]), 0.5, dtype=np.float32)

    # Weighted combination
    total_weight = sum(w for _, _, w in scores)
    combined = np.zeros_like(scores[0][1])
    for name, s, w in scores:
        combined += (w / total_weight) * s

    combined = np.clip(combined, 0.0, 1.0)

    if output_path:
        meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(combined, 1)

    return combined
