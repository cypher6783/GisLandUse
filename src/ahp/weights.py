"""AHP weight calculation and consistency ratio validation."""

import numpy as np
from typing import Optional

from .pairwise import DEFAULT_MATRIX, DEFAULT_CRITERIA, RI_TABLE, create_pairwise_matrix


def calculate_weights(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate AHP weights from a pairwise comparison matrix.

    Uses the eigenvalue method:
    1. Compute the principal eigenvector
    2. Normalize to sum to 1.0

    Returns:
        Tuple of (weights, lambda_max)
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the index of the largest eigenvalue (should be real and positive)
    max_idx = np.argmax(np.real(eigenvalues))
    lambda_max = np.real(eigenvalues[max_idx])

    # The corresponding eigenvector is the weight vector
    weights = np.real(eigenvectors[:, max_idx])

    # Ensure all weights are positive (they should be for a consistent matrix)
    weights = np.abs(weights)

    # Normalize to sum to 1.0
    weights = weights / weights.sum()

    return weights, lambda_max


def calculate_consistency_ratio(matrix: np.ndarray, weights: np.ndarray, lambda_max: float) -> dict:
    """Calculate the Consistency Ratio (CR) for an AHP matrix.

    CR < 0.10 indicates acceptable consistency.

    Returns:
        dict with:
            - ci: Consistency Index
            - ri: Random Index
            - cr: Consistency Ratio
            - is_consistent: bool (CR < 0.10)
            - message: human-readable result
    """
    n = matrix.shape[0]

    # Consistency Index
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # Random Index from Saaty's table
    ri = RI_TABLE.get(n, 1.49)  # Default to 1.49 for n > 10

    # Consistency Ratio
    cr = ci / ri if ri > 0 else 0.0

    is_consistent = cr < 0.10

    if is_consistent:
        message = f"CR = {cr:.4f} < 0.10 — Matrix is CONSISTENT"
    else:
        message = (
            f"CR = {cr:.4f} >= 0.10 — Matrix is INCONSISTENT. "
            f"Please revise your pairwise comparisons."
        )

    return {
        "ci": ci,
        "ri": ri,
        "cr": cr,
        "is_consistent": is_consistent,
        "message": message,
    }


def analyze_pairwise_matrix(
    matrix: Optional[np.ndarray] = None,
    criteria_names: Optional[list[str]] = None,
) -> dict:
    """Full AHP analysis: compute weights and validate consistency.

    Args:
        matrix: Pairwise comparison matrix. If None, uses default.
        criteria_names: Criterion names. If None, uses defaults.

    Returns:
        dict with:
            - matrix: the (possibly corrected) matrix
            - criteria_names: list of criterion names
            - weights: normalized weight vector
            - lambda_max: maximum eigenvalue
            - consistency: dict from calculate_consistency_ratio()
    """
    if matrix is None:
        matrix, criteria_names = create_pairwise_matrix()
    elif criteria_names is None:
        criteria_names = DEFAULT_CRITERIA[: matrix.shape[0]]

    # Calculate weights
    weights, lambda_max = calculate_weights(matrix)

    # Check consistency
    consistency = calculate_consistency_ratio(matrix, weights, lambda_max)

    return {
        "matrix": matrix,
        "criteria_names": criteria_names,
        "weights": weights,
        "lambda_max": lambda_max,
        "consistency": consistency,
    }


def print_analysis(result: dict) -> None:
    """Print a formatted AHP analysis result."""
    from .pairwise import format_matrix

    print("\n" + "=" * 60)
    print("AHP ANALYSIS RESULTS")
    print("=" * 60)

    print("\nPairwise Comparison Matrix:")
    print(format_matrix(result["matrix"], result["criteria_names"]))

    print(f"\nMaximum Eigenvalue (λmax): {result['lambda_max']:.4f}")

    print("\nWeights:")
    for name, weight in zip(result["criteria_names"], result["weights"]):
        bar = "#" * int(weight * 50)
        print(f"  {name:>15}: {weight:.4f}  {bar}")

    print(f"\n  Total: {sum(result['weights']):.4f}")

    c = result["consistency"]
    print(f"\nConsistency Check:")
    print(f"  Consistency Index (CI):  {c['ci']:.4f}")
    print(f"  Random Index (RI):       {c['ri']:.4f}")
    print(f"  Consistency Ratio (CR):  {c['cr']:.4f}")
    print(f"  Result: {c['message']}")
    print("=" * 60)


if __name__ == "__main__":
    result = analyze_pairwise_matrix()
    print_analysis(result)
