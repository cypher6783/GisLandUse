"""Topography suitability scoring for rice cultivation.

Rice requires relatively flat land for paddy management:
- Slope: 0–3° excellent, 3–8° moderate, >15° unsuitable
- Elevation: 0–500m optimal, up to 1500m possible
- Aspect: flat or gentle slopes, aspect less critical for irrigated rice
"""

import numpy as np
from pathlib import Path
from typing import Optional

import rasterio


def trapezoidal_score(
    value: np.ndarray, x1: float, x2: float, x3: float, x4: float
) -> np.ndarray:
    """Apply trapezoidal membership function."""
    score = np.zeros_like(value, dtype=np.float32)

    mask_rise = (value > x1) & (value < x2)
    score[mask_rise] = (value[mask_rise] - x1) / (x2 - x1)

    mask_flat = (value >= x2) & (value <= x3)
    score[mask_flat] = 1.0

    mask_fall = (value > x3) & (value < x4)
    score[mask_fall] = (x4 - value[mask_fall]) / (x4 - x3)

    return score


def score_slope(slope: np.ndarray) -> np.ndarray:
    """Score slope for rice cultivation.

    Rice paddies need flat land:
        - 0-3°: Excellent (fully mechanized paddy possible)
        - 3-8°: Moderate (terracing needed)
        - 8-15°: Marginal (significant terracing required)
        - >15°: Not suitable
    """
    return trapezoidal_score(slope, 0, 0, 3, 15)


def score_elevation(elevation: np.ndarray) -> np.ndarray:
    """Score elevation for rice.

    Rice grows best at low elevations:
        - 0-500m: Excellent
        - 500-1000m: Good (temperature decreases ~0.6°C per 100m)
        - 1000-1500m: Marginal
        - >1500m: Not suitable (too cold)
    """
    return trapezoidal_score(elevation, 0, 0, 500, 1500)


def score_aspect(aspect: np.ndarray) -> np.ndarray:
    """Score terrain aspect for rice.

    Aspect is less critical for rice with irrigation, but:
        - Flat areas (slope <3°) are best regardless of aspect
        - For sloped areas, north-facing slopes in tropics receive less
          direct sun, which can be beneficial in hot climates
    This function returns a mild modifier (0.8-1.0).
    """
    # Aspect has minimal impact on rice suitability
    # Return a constant mild score
    score = np.full_like(aspect, 0.9, dtype=np.float32)
    return score


def score_topography(
    slope_path: Path,
    elevation_path: Optional[Path] = None,
    aspect_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute combined topography suitability score.

    Slope is the dominant factor. Elevation and aspect are secondary.

    Returns:
        2D numpy array of scores (0-1)
    """
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    s_slope = score_slope(slope)

    # Weighted combination
    if elevation_path and Path(elevation_path).exists():
        with rasterio.open(elevation_path) as src:
            elev = src.read(1).astype(np.float32)
        s_elev = score_elevation(elev)
        combined = 0.6 * s_slope + 0.4 * s_elev
    else:
        combined = s_slope

    if aspect_path and Path(aspect_path).exists():
        with rasterio.open(aspect_path) as src:
            aspect = src.read(1).astype(np.float32)
        s_aspect = score_aspect(aspect)
        # Aspect has minimal weight
        combined = 0.9 * combined + 0.1 * s_aspect

    combined = np.clip(combined, 0.0, 1.0)

    if output_path:
        meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(combined, 1)

    return combined
