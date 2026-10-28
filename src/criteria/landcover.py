"""Land use/land cover suitability scoring for rice cultivation.

Rice is typically grown on:
- Existing cropland (easiest to convert)
- Grasslands/pastures (moderate conversion effort)
- Wetlands/floodplains (natural rice habitat)
- Avoid: dense forest, urban/built-up, barren, water bodies
"""

import numpy as np
from pathlib import Path
from typing import Optional

import rasterio


# ESA WorldCover 2021 class codes
# https://esa-worldcover.org/
WORLDCOVER_CLASSES = {
    10: ("Tree cover", 0.2),
    20: ("Shrubland", 0.4),
    30: ("Grassland", 0.8),
    40: ("Cropland", 1.0),
    50: ("Built-up", 0.0),
    60: ("Bare / sparse vegetation", 0.1),
    70: ("Snow and Ice", 0.0),
    80: ("Permanent water bodies", 0.0),
    90: ("Herbaceous wetland", 0.9),
    95: ("Mangroves", 0.3),
    100: ("Moss and lichen", 0.2),
}


def score_landcover_class(landcover: np.ndarray) -> np.ndarray:
    """Score land cover using ESA WorldCover class assignments.

    Each class is assigned a suitability score based on how suitable
    the current land use is for rice cultivation.
    """
    score = np.full_like(landcover, 0.5, dtype=np.float32)  # Default for unknown classes

    for class_code, (name, suit_score) in WORLDCOVER_CLASSES.items():
        mask = landcover == class_code
        score[mask] = suit_score

    return score


def score_landcover_continuous(landcover: np.ndarray) -> np.ndarray:
    """Score land cover using continuous value ranges.

    For datasets that don't use ESA class codes, this function
    provides scoring based on typical land cover value ranges.
    """
    # This is a fallback for non-standard land cover datasets
    # Values typically 0-100 or categorical
    score = np.zeros_like(landcover, dtype=np.float32)

    # Assume higher values = more vegetation = better for rice
    score[landcover > 0] = 0.3
    score[landcover > 20] = 0.5
    score[landcover > 40] = 0.7
    score[landcover > 60] = 0.9
    score[landcover > 80] = 1.0

    return score


def score_landcover(
    landcover_path: Path,
    output_path: Optional[Path] = None,
    is_categorical: bool = True,
) -> np.ndarray:
    """Compute land cover suitability score.

    Args:
        landcover_path: Path to land cover raster
        output_path: Optional path to save result
        is_categorical: True if using ESA WorldCover class codes

    Returns:
        2D numpy array of scores (0-1)
    """
    with rasterio.open(landcover_path) as src:
        landcover = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    if is_categorical:
        score = score_landcover_class(landcover)
    else:
        score = score_landcover_continuous(landcover)

    score = np.clip(score, 0.0, 1.0)

    if output_path:
        meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(score, 1)

    return score
