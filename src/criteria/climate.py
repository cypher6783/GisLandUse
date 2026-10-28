"""Climate suitability scoring for rice cultivation.

Rice requires:
- Annual rainfall: 1,200–2,000 mm
- Growing season temperature: 20–35°C
- No frost (min temp > 10°C)
- Adequate rainfall during growing season
"""

import numpy as np
from pathlib import Path
from typing import Optional

import rasterio


def gaussian_score(value: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply Gaussian membership function.

    Returns a score between 0 and 1, with maximum at the center.
    width controls how quickly the score drops off.
    """
    return np.exp(-((value - center) ** 2) / (2 * width ** 2))


def trapezoidal_score(
    value: np.ndarray, x1: float, x2: float, x3: float, x4: float
) -> np.ndarray:
    """Apply trapezoidal membership function.

    Scores:
        - 0 where value <= x1 or value >= x4
        - 1 where x2 <= value <= x3
        - Linear ramps between x1-x2 and x3-x4
    """
    score = np.zeros_like(value, dtype=np.float32)

    # Rising ramp
    mask_rise = (value > x1) & (value < x2)
    score[mask_rise] = (value[mask_rise] - x1) / (x2 - x1)

    # Plateau
    mask_flat = (value >= x2) & (value <= x3)
    score[mask_flat] = 1.0

    # Falling ramp
    mask_fall = (value > x3) & (value < x4)
    score[mask_fall] = (x4 - value[mask_fall]) / (x4 - x3)

    return score


def score_annual_rainfall(rainfall: np.ndarray) -> np.ndarray:
    """Score annual rainfall suitability for rice.

    Optimal: 1,200–2,000 mm/year
    """
    return trapezoidal_score(rainfall, 600, 1200, 2000, 3000)


def score_mean_temperature(temperature: np.ndarray) -> np.ndarray:
    """Score mean annual temperature suitability for rice.

    Optimal: 22–30°C
    """
    return trapezoidal_score(temperature, 15, 22, 30, 40)


def score_min_temperature(min_temp: np.ndarray) -> np.ndarray:
    """Score minimum temperature (frost risk).

    Rice is damaged below 10°C. Score decreases sharply below 10°C.
    """
    score = np.ones_like(min_temp, dtype=np.float32)
    score[min_temp < 15] = 0.9
    score[min_temp < 12] = 0.7
    score[min_temp < 10] = 0.3
    score[min_temp < 5] = 0.0
    return score


def score_warmest_month_temp(max_temp: np.ndarray) -> np.ndarray:
    """Score maximum temperature during warmest month.

    Rice grows best with warm (not extreme) temperatures.
    Optimal: 25–35°C
    """
    return trapezoidal_score(max_temp, 20, 25, 35, 42)


def score_rainfall_seasonality(cv_rainfall: np.ndarray) -> np.ndarray:
    """Score rainfall seasonality (coefficient of variation).

    Rice needs distinct wet/dry seasons. Moderate seasonality is good.
    CV of 30-60% indicates good monsoon-type rainfall.
    """
    return trapezoidal_score(cv_rainfall, 10, 30, 60, 90)


def score_climate(
    bio01_path: Path,
    bio05_path: Path,
    bio06_path: Path,
    bio12_path: Path,
    bio15_path: Path,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute combined climate suitability score.

    Combines:
        - Annual precipitation (bio12)
        - Mean annual temperature (bio01)
        - Min temperature of coldest month (bio06)
        - Max temperature of warmest month (bio05)
        - Precipitation seasonality (bio15)

    Returns:
        2D numpy array of scores (0-1)
    """
    with rasterio.open(bio12_path) as src:
        rainfall = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    with rasterio.open(bio01_path) as src:
        temp_mean = src.read(1).astype(np.float32)

    with rasterio.open(bio06_path) as src:
        temp_min = src.read(1).astype(np.float32)

    with rasterio.open(bio05_path) as src:
        temp_max = src.read(1).astype(np.float32)

    with rasterio.open(bio15_path) as src:
        cv_rain = src.read(1).astype(np.float32)

    # Individual scores
    s_rainfall = score_annual_rainfall(rainfall)
    s_temp_mean = score_mean_temperature(temp_mean)
    s_temp_min = score_min_temperature(temp_min)
    s_temp_max = score_warmest_month_temp(temp_max)
    s_rain_season = score_rainfall_seasonality(cv_rain)

    # Combined score (weighted average within climate category)
    weights = {
        "rainfall": 0.35,
        "temp_mean": 0.25,
        "temp_min": 0.15,
        "temp_max": 0.10,
        "rain_season": 0.15,
    }

    combined = (
        weights["rainfall"] * s_rainfall
        + weights["temp_mean"] * s_temp_mean
        + weights["temp_min"] * s_temp_min
        + weights["temp_max"] * s_temp_max
        + weights["rain_season"] * s_rain_season
    )

    # Clip to 0-1
    combined = np.clip(combined, 0.0, 1.0)

    # Save if output path provided
    if output_path:
        meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(combined, 1)

    return combined
