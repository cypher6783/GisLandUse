"""FAO Land Suitability Classification.

Implements the standard FAO suitability classes:
    S1 — Highly Suitable
    S2 — Moderately Suitable
    S3 — Marginally Suitable
    N  — Not Suitable

Classification is based on the final combined suitability score (0-1).
"""

import numpy as np
from typing import Optional


# Default classification thresholds
DEFAULT_THRESHOLDS = {
    "S1": 0.75,  # Score >= 0.75
    "S2": 0.50,  # Score >= 0.50
    "S3": 0.25,  # Score >= 0.25
    # Below 0.25 = N (Not Suitable)
}

# Suitability class metadata
SUITABILITY_CLASSES = {
    "S1": {
        "name": "Highly Suitable",
        "description": "Land has no significant limitations for rice cultivation",
        "color": "#1a9641",  # Dark green
        "value": 4,
    },
    "S2": {
        "name": "Moderately Suitable",
        "description": "Land has minor limitations that can be overcome with moderate effort",
        "color": "#a6d96a",  # Light green
        "value": 3,
    },
    "S3": {
        "name": "Marginally Suitable",
        "description": "Land has significant limitations requiring considerable effort to overcome",
        "color": "#fdae61",  # Orange
        "value": 2,
    },
    "N": {
        "name": "Not Suitable",
        "description": "Land has limitations that cannot be overcome for rice cultivation",
        "color": "#d7191c",  # Red
        "value": 1,
    },
}


def classify_suitability(
    score: np.ndarray,
    thresholds: Optional[dict] = None,
) -> tuple[np.ndarray, dict]:
    """Classify continuous suitability scores into FAO classes.

    Args:
        score: 2D array of continuous scores (0-1)
        thresholds: dict mapping class name to minimum score threshold.
                    If None, uses DEFAULT_THRESHOLDS.

    Returns:
        Tuple of:
            - classified: 2D array with integer class values (1-4)
            - metadata: dict with class names and their integer values
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    classified = np.zeros_like(score, dtype=np.int32)

    # Assign classes from highest to lowest
    classified[score >= thresholds["S1"]] = SUITABILITY_CLASSES["S1"]["value"]
    classified[score >= thresholds["S2"]] = SUITABILITY_CLASSES["S2"]["value"]
    classified[score >= thresholds["S3"]] = SUITABILITY_CLASSES["S3"]["value"]
    classified[score < thresholds["S3"]] = SUITABILITY_CLASSES["N"]["value"]

    return classified, SUITABILITY_CLASSES


def classify_to_label(
    score: np.ndarray,
    thresholds: Optional[dict] = None,
) -> np.ndarray:
    """Classify scores and return string labels.

    Returns:
        2D object array of string labels ("S1", "S2", "S3", "N")
    """
    classified, _ = classify_suitability(score, thresholds)

    labels = np.empty(classified.shape, dtype="U2")
    labels[classified == 4] = "S1"
    labels[classified == 3] = "S2"
    labels[classified == 2] = "S3"
    labels[classified == 1] = "N"

    return labels


def compute_area_statistics(
    classified: np.ndarray,
    pixel_area_km2: float = 1.0,
) -> dict:
    """Compute area statistics for each suitability class.

    Args:
        classified: 2D array of integer class values (1-4)
        pixel_area_km2: Area of each pixel in square kilometers

    Returns:
        dict with per-class statistics
    """
    total_pixels = classified.size
    total_area = total_pixels * pixel_area_km2

    stats = {}
    for class_name, info in SUITABILITY_CLASSES.items():
        value = info["value"]
        mask = classified == value
        count = np.sum(mask)
        area = count * pixel_area_km2
        pct = (count / total_pixels) * 100

        stats[class_name] = {
            "name": info["name"],
            "pixel_count": int(count),
            "area_km2": float(area),
            "percentage": float(pct),
        }

    stats["_total"] = {
        "pixel_count": total_pixels,
        "area_km2": float(total_area),
    }

    return stats


def print_area_statistics(stats: dict) -> None:
    """Print formatted area statistics."""
    print("\n" + "=" * 60)
    print("SUITABILITY AREA STATISTICS")
    print("=" * 60)

    total = stats["_total"]["area_km2"]

    print(f"\n{'Class':<15} {'Name':<25} {'Area (km²)':>12} {'Percentage':>12}")
    print("-" * 66)

    for class_name in ["S1", "S2", "S3", "N"]:
        s = stats[class_name]
        print(
            f"{class_name:<15} {s['name']:<25} "
            f"{s['area_km2']:>12,.0f} {s['percentage']:>11.1f}%"
        )

    print("-" * 66)
    print(f"{'TOTAL':<15} {'':<25} {total:>12,.0f} {'100.0%':>12}")
    print("=" * 60)
