"""Sensitivity analysis for AHP weights.

Tests how variations in criteria weights affect the final suitability classification.
Helps identify which criteria most influence the results.
"""

import numpy as np
from typing import Optional
from pathlib import Path

import rasterio
import json

from .classifier import classify_suitability, compute_area_statistics


def perturb_weights(
    base_weights: dict[str, float],
    perturbation: float = 0.2,
    n_steps: int = 5,
) -> list[dict[str, float]]:
    """Generate perturbed weight sets for sensitivity analysis.

    For each criterion, varies its weight from -perturbation to +perturbation
    while proportionally adjusting other weights to maintain sum = 1.0.

    Args:
        base_weights: Original weights dict
        perturbation: Maximum fractional perturbation (0.2 = ±20%)
        n_steps: Number of perturbation levels per criterion

    Returns:
        List of weight dicts, one per perturbation scenario
    """
    scenarios = []
    criteria = list(base_weights.keys())

    for criterion in criteria:
        base_w = base_weights[criterion]
        perturbations = np.linspace(-perturbation, perturbation, n_steps)

        for p in perturbations:
            new_weight = base_w * (1 + p)

            # Don't allow negative weights
            if new_weight <= 0:
                continue

            # Scale other weights proportionally
            other_total = sum(base_weights[c] for c in criteria if c != criterion)
            new_other_total = 1.0 - new_weight

            new_weights = {}
            for c in criteria:
                if c == criterion:
                    new_weights[c] = new_weight
                else:
                    new_weights[c] = base_weights[c] / other_total * new_other_total

            scenarios.append(new_weights)

    return scenarios


def sensitivity_analysis(
    criteria_rasters: dict[str, np.ndarray],
    base_weights: dict[str, float],
    output_dir: Optional[Path] = None,
    perturbation: float = 0.2,
    n_steps: int = 5,
    pixel_area_km2: float = 1.0,
) -> dict:
    """Run full sensitivity analysis on weight perturbations.

    For each perturbation scenario:
        1. Compute weighted overlay with perturbed weights
        2. Classify into FAO classes
        3. Compute area statistics
        4. Track how class areas change

    Returns:
        dict with:
            - base_classification: original classified result
            - base_statistics: original area statistics
            - scenarios: list of scenario results
            - sensitivity_metrics: per-criterion sensitivity metrics
    """
    print("=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Base case
    print("\nComputing base case...")
    base_overlay = np.zeros_like(list(criteria_rasters.values())[0], dtype=np.float32)
    total_weight = sum(base_weights.values())
    for name, raster in criteria_rasters.items():
        if name in base_weights:
            w = base_weights[name] / total_weight
            valid = raster != -9999
            base_overlay[valid] += w * raster[valid]

    base_classified, _ = classify_suitability(base_overlay)
    base_stats = compute_area_statistics(base_classified, pixel_area_km2)

    print(f"Base weights: {base_weights}")
    for cls in ["S1", "S2", "S3", "N"]:
        print(f"  {cls}: {base_stats[cls]['percentage']:.1f}%")

    # Generate perturbation scenarios
    scenarios = perturb_weights(base_weights, perturbation, n_steps)
    print(f"\nGenerated {len(scenarios)} perturbation scenarios")

    # Run each scenario
    scenario_results = []
    for i, weights in enumerate(scenarios):
        # Find which criterion was perturbed and by how much
        changed = None
        for c in base_weights:
            if not np.isclose(weights[c], base_weights[c], rtol=0.01):
                changed = c
                break

        # Compute overlay with perturbed weights
        overlay = np.zeros_like(base_overlay)
        total_w = sum(weights.values())
        for name, raster in criteria_rasters.items():
            if name in weights:
                w = weights[name] / total_w
                valid = raster != -9999
                overlay[valid] += w * raster[valid]

        classified, _ = classify_suitability(overlay)
        stats = compute_area_statistics(classified, pixel_area_km2)

        # Compute change from base
        s1_change = stats["S1"]["percentage"] - base_stats["S1"]["percentage"]

        scenario_results.append({
            "scenario_id": i,
            "perturbed_criterion": changed,
            "weights": weights,
            "statistics": stats,
            "s1_change_pct": s1_change,
        })

    # Compute sensitivity metrics per criterion
    print("\nComputing sensitivity metrics...")
    sensitivity_metrics = {}
    criteria = list(base_weights.keys())

    for criterion in criteria:
        # Find scenarios where this criterion was perturbed
        relevant = [s for s in scenario_results if s["perturbed_criterion"] == criterion]

        if not relevant:
            continue

        s1_changes = [s["s1_change_pct"] for s in relevant]

        sensitivity_metrics[criterion] = {
            "max_s1_change": max(abs(c) for c in s1_changes) if s1_changes else 0,
            "mean_s1_change": float(np.mean(s1_changes)) if s1_changes else 0,
            "base_weight": base_weights[criterion],
            "is_influential": max(abs(c) for c in s1_changes) > 5.0 if s1_changes else False,
        }

    # Print sensitivity summary
    print("\n" + "-" * 50)
    print(f"{'Criterion':<15} {'Weight':>8} {'Max ΔS1%':>10} {'Influential':>12}")
    print("-" * 50)
    for name, metrics in sensitivity_metrics.items():
        influential = "YES" if metrics["is_influential"] else "no"
        print(
            f"{name:<15} {metrics['base_weight']:>8.3f} "
            f"{metrics['max_s1_change']:>10.2f} {influential:>12}"
        )
    print("-" * 50)

    # Identify most influential criterion
    most_influential = max(
        sensitivity_metrics.items(),
        key=lambda x: x[1]["max_s1_change"],
    )
    print(f"\nMost influential criterion: {most_influential[0]} "
          f"(max ΔS1 = {most_influential[1]['max_s1_change']:.2f}%)")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "base_weights": base_weights,
            "base_statistics": {k: v for k, v in base_stats.items() if not k.startswith("_")},
            "sensitivity_metrics": sensitivity_metrics,
            "num_scenarios": len(scenarios),
            "perturbation_range": f"±{perturbation*100:.0f}%",
        }
        results_path = output_dir / "sensitivity_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved: {results_path}")

    return {
        "base_classification": base_classified,
        "base_statistics": base_stats,
        "scenarios": scenario_results,
        "sensitivity_metrics": sensitivity_metrics,
    }
