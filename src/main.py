"""Main CLI entry point for the Rice Land Suitability Analysis system."""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_download():
    """Download all required datasets."""
    from src.data.downloaders import download_all
    download_all()


def cmd_preprocess():
    """Preprocess downloaded data (clip, reproject, align)."""
    from src.data.preprocessors import preprocess_all
    preprocess_all()


def cmd_analyze():
    """Run the full suitability analysis."""
    from src.suitability.overlay import run_overlay_analysis
    from src.output.mapgen import (
        create_suitability_map,
        create_score_map,
        create_bar_chart,
        create_weight_chart,
        create_interactive_map,
    )
    from src.output.report import generate_text_report, generate_html_report
    from pathlib import Path
    import json

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "output"

    # Check if processed data exists
    if not processed_dir.exists() or not list(processed_dir.glob("*.tif")):
        print("ERROR: No processed data found.")
        print("Please run: python -m src.main download")
        print("         then: python -m src.main preprocess")
        return

    # Run overlay analysis
    results = run_overlay_analysis(processed_dir, output_dir)

    # Generate maps
    print("\nGenerating maps...")
    classified_path = output_dir / "suitability_class.tif"
    score_path = output_dir / "suitability_score.tif"

    create_suitability_map(classified_path, output_dir / "suitability_map.png")
    create_score_map(score_path, output_dir / "score_map.png")
    create_bar_chart(results["statistics"], output_dir / "area_distribution.png")
    create_weight_chart(results["weights"], output_dir / "weight_distribution.png")
    create_interactive_map(classified_path, output_dir / "suitability_interactive.html")

    # Generate reports
    print("\nGenerating reports...")
    generate_text_report(
        results["statistics"],
        results["weights"],
        results["metadata"],
        output_dir / "suitability_report.txt",
    )
    generate_html_report(
        results["statistics"],
        results["weights"],
        results["metadata"],
        output_dir / "suitability_report.html",
        score_map_path=str(output_dir / "score_map.png"),
        class_map_path=str(output_dir / "suitability_map.png"),
        weight_chart_path=str(output_dir / "weight_distribution.png"),
        bar_chart_path=str(output_dir / "area_distribution.png"),
    )

    print(f"\nAnalysis complete! Results saved in: {output_dir}")


def cmd_sensitivity():
    """Run sensitivity analysis on AHP weights."""
    from src.suitability.sensitivity import sensitivity_analysis
    from src.suitability.overlay import DEFAULT_CRITERIA_FILES
    from src.ahp.weights import analyze_pairwise_matrix
    from pathlib import Path
    import rasterio
    import numpy as np

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "output"

    # Load criterion rasters
    print("Loading criterion rasters...")
    criteria_rasters = {}
    for name, filename in DEFAULT_CRITERIA_FILES.items():
        filepath = processed_dir / filename
        if filepath.exists():
            with rasterio.open(filepath) as src:
                criteria_rasters[name] = src.read(1).astype(np.float32)
            print(f"  Loaded: {name}")

    if not criteria_rasters:
        print("ERROR: No criterion rasters found. Run analysis first.")
        return

    # Get AHP weights
    ahp_result = analyze_pairwise_matrix()
    weights = dict(zip(ahp_result["criteria_names"], ahp_result["weights"]))

    # Compute pixel area
    with rasterio.open(list(processed_dir.glob("*.tif"))[0]) as src:
        pixel_area = abs(src.transform[0] * src.transform[4]) * (111.32 ** 2)

    # Run sensitivity analysis
    sensitivity_analysis(
        criteria_rasters,
        weights,
        output_dir=output_dir,
        pixel_area_km2=pixel_area,
    )

    print(f"\nSensitivity analysis complete! Results in: {output_dir}")


def cmd_ahp():
    """Display AHP weight analysis."""
    from src.ahp.weights import analyze_pairwise_matrix, print_analysis
    result = analyze_pairwise_matrix()
    print_analysis(result)


def main():
    """CLI entry point."""
    commands = {
        "download": ("Download all required datasets", cmd_download),
        "preprocess": ("Preprocess downloaded data (clip, reproject, align)", cmd_preprocess),
        "analyze": ("Run full suitability analysis and generate outputs", cmd_analyze),
        "sensitivity": ("Run sensitivity analysis on AHP weights", cmd_sensitivity),
        "ahp": ("Display AHP weight analysis", cmd_ahp),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("\nRice Land Suitability Analysis — Nigeria")
        print("=" * 50)
        print("\nUsage: python -m src.main <command>\n")
        print("Commands:")
        for name, (desc, _) in commands.items():
            print(f"  {name:<15} {desc}")
        print("\nTypical workflow:")
        print("  1. python -m src.main download")
        print("  2. python -m src.main preprocess")
        print("  3. python -m src.main analyze")
        print("  4. python -m src.main sensitivity")
        return

    cmd_name = sys.argv[1]
    _, func = commands[cmd_name]
    func()


if __name__ == "__main__":
    main()
