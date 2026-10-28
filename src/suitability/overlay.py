"""Weighted overlay aggregation engine.

Combines individual criterion scores into a single suitability index
using AHP-derived weights.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from ..ahp.weights import analyze_pairwise_matrix, print_analysis
from .classifier import classify_suitability, compute_area_statistics, print_area_statistics


# Default criterion file mappings
# These files are created by the preprocessing step
DEFAULT_CRITERIA_FILES = {
    "Climate": "climate_combined.tif",
    "Soil": "soil_combined.tif",
    "Topography": "topo_combined.tif",
    "Land Cover": "landcover_score.tif",
    "Water": "water_combined.tif",
    "Socioeconomic": "socio_combined.tif",
}

# Individual climate files needed for scoring
CLIMATE_FILES = {
    "bio_01": "climate_bio_1.tif",    # Annual Mean Temperature
    "bio_05": "climate_bio_5.tif",    # Max Temperature Warmest Month
    "bio_06": "climate_bio_6.tif",    # Min Temperature Coldest Month
    "bio_12": "climate_bio_12.tif",   # Annual Precipitation
    "bio_15": "climate_bio_15.tif",   # Precipitation Seasonality
}

# Individual soil files needed for scoring
SOIL_FILES = {
    "ph": "soil_phh2o_0-5cm_mean.tif",
    "soc": "soil_soc_0-5cm_mean.tif",
    "clay": "soil_clay_0-5cm_mean.tif",
    "sand": "soil_sand_0-5cm_mean.tif",
    "bdod": "soil_bdod_0-5cm_mean.tif",
}


def weighted_overlay(
    criteria_rasters: dict[str, np.ndarray],
    weights: dict[str, float],
    nodata: float = -9999,
) -> np.ndarray:
    """Perform weighted overlay of criterion score rasters.

    Args:
        criteria_rasters: dict mapping criterion name to 2D score array (0-1)
        weights: dict mapping criterion name to weight (must sum to 1.0)
        nodata: nodata value for output

    Returns:
        2D array of combined suitability scores (0-1)
    """
    # Validate inputs
    common_criteria = set(criteria_rasters.keys()) & set(weights.keys())
    if not common_criteria:
        raise ValueError("No matching criteria between rasters and weights")

    missing = set(criteria_rasters.keys()) - set(weights.keys())
    if missing:
        print(f"  Warning: Criteria without weights (will be excluded): {missing}")

    # Normalize weights to sum to 1.0 for available criteria
    available_weights = {k: weights[k] for k in common_criteria}
    total_weight = sum(available_weights.values())
    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}

    # Initialize combined score
    first_raster = list(criteria_rasters.values())[0]
    combined = np.zeros_like(first_raster, dtype=np.float32)
    weight_sum = np.zeros_like(first_raster, dtype=np.float32)

    for name in common_criteria:
        raster = criteria_rasters[name]
        w = normalized_weights[name]

        # Handle nodata pixels
        valid = raster != nodata
        combined[valid] += w * raster[valid]
        weight_sum[valid] += w

    # Normalize by actual weight sum (handles partial coverage)
    valid = weight_sum > 0
    combined[valid] = combined[valid] / weight_sum[valid]
    combined[~valid] = nodata

    return combined


def run_overlay_analysis(
    processed_dir: Path,
    output_dir: Path,
    weights: Optional[dict[str, float]] = None,
    pairwise_matrix: Optional[np.ndarray] = None,
) -> dict:
    """Run the full weighted overlay analysis.

    Steps:
        1. Run scoring functions on individual criterion files
        2. Compute AHP weights (or use provided weights)
        3. Perform weighted overlay
        4. Classify into FAO suitability classes
        5. Compute and print statistics

    Returns:
        dict with results, weights, statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run scoring functions to create combined criterion files
    print("Running criterion scoring functions...")
    criteria_rasters = {}
    meta = None

    # Climate scoring
    climate_path = processed_dir / CLIMATE_FILES["bio_12"]
    if climate_path.exists():
        from ..criteria.climate import score_climate
        bio01 = processed_dir / CLIMATE_FILES["bio_01"]
        bio05 = processed_dir / CLIMATE_FILES["bio_05"]
        bio06 = processed_dir / CLIMATE_FILES["bio_06"]
        bio15 = processed_dir / CLIMATE_FILES["bio_15"]

        if all(p.exists() for p in [bio01, bio05, bio06, climate_path, bio15]):
            print("  Scoring climate...")
            climate_score = score_climate(
                bio01, bio05, bio06, climate_path, bio15,
                output_path=processed_dir / "climate_combined.tif"
            )
            criteria_rasters["Climate"] = climate_score
            with rasterio.open(processed_dir / "climate_combined.tif") as src:
                meta = src.meta.copy()
        else:
            print("  Warning: Not all climate files found, skipping...")
    else:
        print("  Warning: Climate data not found, skipping...")

    # Soil scoring
    ph_path = processed_dir / SOIL_FILES["ph"]
    if ph_path.exists():
        from ..criteria.soil import score_soil
        soc_path = processed_dir / SOIL_FILES["soc"]
        clay_path = processed_dir / SOIL_FILES["clay"]
        sand_path = processed_dir / SOIL_FILES.get("sand", "")
        bdod_path = processed_dir / SOIL_FILES.get("bdod", "")

        if all(p.exists() for p in [ph_path, soc_path, clay_path]):
            print("  Scoring soil...")
            soil_score = score_soil(
                ph_path, soc_path, clay_path,
                sand_path=Path(sand_path) if sand_path and Path(sand_path).exists() else None,
                bdod_path=Path(bdod_path) if bdod_path and Path(bdod_path).exists() else None,
                output_path=processed_dir / "soil_combined.tif"
            )
            criteria_rasters["Soil"] = soil_score
    else:
        print("  Warning: Soil data not found, skipping...")

    # Topography scoring
    slope_path = processed_dir / "slope.tif"
    if slope_path.exists():
        from ..criteria.topography import score_topography
        dem_path = processed_dir / "dem_aligned.tif"
        aspect_path = processed_dir / "aspect.tif"

        print("  Scoring topography...")
        topo_score = score_topography(
            slope_path,
            elevation_path=dem_path if dem_path.exists() else None,
            aspect_path=aspect_path if aspect_path.exists() else None,
            output_path=processed_dir / "topo_combined.tif"
        )
        criteria_rasters["Topography"] = topo_score
    else:
        print("  Warning: Topography data not found, skipping...")

    # Land cover scoring
    landcover_path = processed_dir / "landcover.tif"
    if landcover_path.exists():
        from ..criteria.landcover import score_landcover
        print("  Scoring land cover...")
        lc_score = score_landcover(
            landcover_path,
            output_path=processed_dir / "landcover_score.tif"
        )
        criteria_rasters["Land Cover"] = lc_score
    else:
        print("  Warning: Land cover data not found, skipping...")

    # Water scoring (skip if not available)
    water_distance_path = processed_dir / "water_distance.tif"
    if water_distance_path.exists():
        from ..criteria.water import score_water
        print("  Scoring water availability...")
        water_score = score_water(
            water_distance_path=water_distance_path,
            output_path=processed_dir / "water_combined.tif"
        )
        criteria_rasters["Water"] = water_score
    else:
        print("  Warning: Water data not found, skipping...")

    # Socioeconomic scoring (skip if not available)
    socio_path = processed_dir / "socio_combined.tif"
    if socio_path.exists():
        with rasterio.open(socio_path) as src:
            criteria_rasters["Socioeconomic"] = src.read(1).astype(np.float32)
    else:
        print("  Warning: Socioeconomic data not found, skipping...")

    if not criteria_rasters:
        raise FileNotFoundError("No criterion rasters found in processed directory")

    print(f"\n  Available criteria: {list(criteria_rasters.keys())}")

    # Step 2: Compute AHP weights
    print("\nComputing AHP weights...")
    if pairwise_matrix is not None:
        ahp_result = analyze_pairwise_matrix(pairwise_matrix)
    else:
        ahp_result = analyze_pairwise_matrix()

    print_analysis(ahp_result)

    # Extract weights for available criteria
    ahp_weights = dict(zip(ahp_result["criteria_names"], ahp_result["weights"]))

    if weights is not None:
        # User-provided weights override AHP
        print("\nUsing user-provided weights (overriding AHP):")
        ahp_weights.update(weights)

    # Step 3: Weighted overlay
    print("\nPerforming weighted overlay...")
    combined = weighted_overlay(criteria_rasters, ahp_weights)

    # Save combined score
    score_path = output_dir / "suitability_score.tif"
    meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
    with rasterio.open(score_path, "w", **meta) as dst:
        dst.write(combined, 1)
    print(f"  Saved: {score_path}")

    # Step 4: Classify
    print("\nClassifying suitability...")
    classified, class_info = classify_suitability(combined)

    class_path = output_dir / "suitability_class.tif"
    meta.update({"dtype": "int32", "nodata": -9999})
    with rasterio.open(class_path, "w", **meta) as dst:
        dst.write(classified, 1)
    print(f"  Saved: {class_path}")

    # Step 5: Statistics
    pixel_area_km2 = abs(meta["transform"][0] * meta["transform"][4]) * (111.32 ** 2)
    stats = compute_area_statistics(classified, pixel_area_km2)
    print_area_statistics(stats)

    # Save metadata
    result_metadata = {
        "weights": ahp_weights,
        "ahp_lambda_max": float(ahp_result["lambda_max"]),
        "ahp_consistency_ratio": float(ahp_result["consistency"]["cr"]),
        "ahp_is_consistent": ahp_result["consistency"]["is_consistent"],
        "classification_thresholds": {"S1": 0.75, "S2": 0.50, "S3": 0.25, "N": 0.0},
        "statistics": stats,
    }

    meta_path = output_dir / "analysis_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(result_metadata, f, indent=2, default=str)
    print(f"\n  Metadata saved: {meta_path}")

    return {
        "combined_score": combined,
        "classified": classified,
        "weights": ahp_weights,
        "statistics": stats,
        "metadata": result_metadata,
    }
