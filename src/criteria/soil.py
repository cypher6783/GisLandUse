"""Soil suitability scoring for rice cultivation.

Rice prefers:
- pH: 5.5–7.0 (slightly acidic to neutral)
- Texture: Clay loam, silty clay loam (good water retention)
- Organic matter: >1.5%
- Drainage: Well-drained to imperfectly drained
- Soil depth: >50 cm
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


def score_ph(ph: np.ndarray) -> np.ndarray:
    """Score soil pH for rice.

    Optimal range: 5.5–7.0
    Rice tolerates slightly acidic to neutral soils.
    """
    return trapezoidal_score(ph, 4.0, 5.5, 7.0, 8.5)


def score_organic_carbon(soc: np.ndarray) -> np.ndarray:
    """Score soil organic carbon content.

    SoilGrids returns values in dg/kg (deci-grams per kg).
    Higher organic matter is better for rice.
    Optimal: >15 dg/kg (1.5%)
    """
    # Convert dg/kg to percentage if needed (depends on SoilGrids version)
    # SoilGrids 250m reports in g/kg * 10
    score = np.zeros_like(soc, dtype=np.float32)

    score[soc < 5] = 0.2      # Very low organic matter
    score[soc >= 5] = 0.4
    score[soc >= 10] = 0.6
    score[soc >= 15] = 0.8
    score[soc >= 25] = 1.0     # Excellent organic matter

    return score


def score_clay_content(clay: np.ndarray) -> np.ndarray:
    """Score clay content for rice.

    Rice needs soil with good water retention.
    Optimal: 20–40% clay (clay loam to silty clay).
    Too much clay (>50%) causes drainage issues.
    Too little (<15%) doesn't retain water.
    """
    return trapezoidal_score(clay, 5, 20, 40, 60)


def score_sand_content(sand: np.ndarray) -> np.ndarray:
    """Score sand content for rice.

    Too much sand means poor water retention.
    Optimal: 15–40% sand.
    """
    # Inverse: less sand is generally better for rice
    score = np.zeros_like(sand, dtype=np.float32)

    score[sand < 15] = 0.9     # Good (mostly clay/silt)
    score[sand >= 15] = 0.8
    score[sand >= 30] = 0.6
    score[sand >= 50] = 0.3    # Too sandy
    score[sand >= 70] = 0.1    # Very sandy

    return score


def score_bulk_density(bdod: np.ndarray) -> np.ndarray:
    """Score bulk density.

    SoilGrids returns in cg/cm³ (centigrams per cubic cm).
    Optimal for rice: 1.0–1.4 g/cm³ (100-140 cg/cm³)
    """
    # Convert from cg/cm³ to g/cm³
    bd_g = bdod / 100.0
    return trapezoidal_score(bd_g, 0.8, 1.0, 1.4, 1.8)


def score_soil(
    ph_path: Path,
    soc_path: Path,
    clay_path: Path,
    sand_path: Optional[Path] = None,
    bdod_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute combined soil suitability score.

    Combines:
        - pH
        - Organic carbon
        - Clay content
        - Sand content (if available)
        - Bulk density (if available)

    Returns:
        2D numpy array of scores (0-1)
    """
    with rasterio.open(ph_path) as src:
        ph = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    with rasterio.open(soc_path) as src:
        soc = src.read(1).astype(np.float32)

    with rasterio.open(clay_path) as src:
        clay = src.read(1).astype(np.float32)

    # Individual scores
    s_ph = score_ph(ph)
    s_soc = score_organic_carbon(soc)
    s_clay = score_clay_content(clay)

    # Weights for soil components
    weights = {"ph": 0.30, "soc": 0.25, "clay": 0.25, "sand": 0.10, "bdod": 0.10}

    combined = weights["ph"] * s_ph + weights["soc"] * s_soc + weights["clay"] * s_clay

    # Add optional components
    available_weight = weights["ph"] + weights["soc"] + weights["clay"]

    if sand_path and Path(sand_path).exists():
        with rasterio.open(sand_path) as src:
            sand = src.read(1).astype(np.float32)
        s_sand = score_sand_content(sand)
        combined += weights["sand"] * s_sand
        available_weight += weights["sand"]

    if bdod_path and Path(bdod_path).exists():
        with rasterio.open(bdod_path) as src:
            bdod = src.read(1).astype(np.float32)
        s_bdod = score_bulk_density(bdod)
        combined += weights["bdod"] * s_bdod
        available_weight += weights["bdod"]

    # Renormalize if some components were missing
    if available_weight > 0:
        combined = combined / available_weight * (weights["ph"] + weights["soc"] + weights["clay"])

    combined = np.clip(combined, 0.0, 1.0)

    if output_path:
        meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(combined, 1)

    return combined
