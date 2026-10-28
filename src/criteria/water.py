"""Water availability suitability scoring for rice cultivation.

Rice is water-intensive and requires:
- Proximity to water sources (rivers, dams, reservoirs)
- High water table or floodplain areas
- Access to irrigation infrastructure
"""

import numpy as np
from pathlib import Path
from typing import Optional

from scipy.ndimage import distance_transform_edt
import rasterio


def score_distance_to_water(distance: np.ndarray, max_distance: float = 50000) -> np.ndarray:
    """Score proximity to water bodies.

    Distance is in meters. Closer to water = more suitable.
    Uses exponential decay function.
    """
    # Normalize distance
    norm_dist = np.clip(distance / max_distance, 0, 1)

    # Exponential decay: score drops quickly as distance increases
    # At 0 distance: score = 1.0
    # At max_distance: score ≈ 0.05
    score = np.exp(-3 * norm_dist)

    return score.astype(np.float32)


def score_water_occurrence(water_occurrence: np.ndarray) -> np.ndarray:
    """Score based on surface water occurrence frequency.

    Global Surface Water data: 0 = never water, 100 = always water.
    Occasional flooding is good for rice.
    """
    score = np.zeros_like(water_occurrence, dtype=np.float32)

    score[water_occurrence > 0] = 0.3    # Rarely wet
    score[water_occurrence > 10] = 0.6   # Occasionally wet
    score[water_occurrence > 30] = 0.8   # Often wet
    score[water_occurrence > 50] = 1.0   # Frequently wet
    score[water_occurrence > 80] = 0.9   # Permanent water (slightly less ideal than seasonal)
    score[water_occurrence > 95] = 0.7   # Permanent water body (not ideal for paddy)

    return score


def score_water(
    water_distance_path: Optional[Path] = None,
    water_occurrence_path: Optional[Path] = None,
    water_mask_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    cell_size_meters: float = 1000.0,
) -> np.ndarray:
    """Compute combined water availability suitability score.

    Can use either:
        - Pre-computed distance to water raster
        - Water mask from which distance is computed
        - Surface water occurrence data
        - Any combination of the above

    Returns:
        2D numpy array of scores (0-1)
    """
    scores = []

    # Score from distance to water
    if water_distance_path and Path(water_distance_path).exists():
        with rasterio.open(water_distance_path) as src:
            dist = src.read(1).astype(np.float32)
            meta = src.meta.copy()
        # Convert from pixel units to meters
        dist_meters = dist * cell_size_meters
        scores.append(("distance", score_distance_to_water(dist_meters), 0.6))

    # Score from water mask (compute distance on the fly)
    elif water_mask_path and Path(water_mask_path).exists():
        with rasterio.open(water_mask_path) as src:
            water = src.read(1)
            meta = src.meta.copy()

        # Invert: water=0, non-water=1
        water_binary = (water > 0).astype(np.uint8)
        dist = distance_transform_edt(water_binary) * cell_size_meters
        scores.append(("distance", score_distance_to_water(dist), 0.6))

    # Score from water occurrence
    if water_occurrence_path and Path(water_occurrence_path).exists():
        with rasterio.open(water_occurrence_path) as src:
            occurrence = src.read(1).astype(np.float32)
            if "meta" not in dir():
                meta = src.meta.copy()
        scores.append(("occurrence", score_water_occurrence(occurrence), 0.4))

    if not scores:
        raise ValueError("At least one water data source must be provided")

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
