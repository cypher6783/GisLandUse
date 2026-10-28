"""AHP pairwise comparison matrix construction."""

import numpy as np
from typing import List, Optional

# Saaty's Random Index for consistency check
# RI values for matrix sizes 1-15
RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

# Default criteria for rice suitability analysis
DEFAULT_CRITERIA = [
    "Climate",
    "Soil",
    "Topography",
    "Land Cover",
    "Water",
    "Socioeconomic",
]

# Default pairwise comparison matrix for rice suitability
# Based on literature review for rice cultivation requirements:
# - Soil and Climate are most important (rice needs specific water/temp/pH conditions)
# - Water availability is critical (rice is water-intensive)
# - Land cover matters (current use affects conversion feasibility)
# - Topography affects drainage and mechanization
# - Socioeconomic factors affect market access but are secondary
DEFAULT_MATRIX = np.array([
    [1,   1/2, 3,   2,   1,   3],    # Climate
    [2,   1,   4,   3,   2,   4],    # Soil
    [1/3, 1/4, 1,   1/2, 1/2, 2],    # Topography
    [1/2, 1/3, 2,   1,   1,   2],    # Land Cover
    [1,   1/2, 2,   1,   1,   3],    # Water
    [1/3, 1/4, 1/2, 1/2, 1/3, 1],    # Socioeconomic
], dtype=np.float64)


def create_pairwise_matrix(
    comparisons: Optional[List[List[float]]] = None,
    criteria_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, List[str]]:
    """Create or validate a pairwise comparison matrix.

    Args:
        comparisons: NxN matrix of pairwise comparisons (Saaty scale).
                     If None, uses default matrix.
        criteria_names: List of criterion names. If None, uses defaults.

    Returns:
        Tuple of (matrix, criteria_names)
    """
    if comparisons is None:
        return DEFAULT_MATRIX.copy(), DEFAULT_CRITERIA.copy()

    matrix = np.array(comparisons, dtype=np.float64)
    n = matrix.shape[0]

    if matrix.shape != (n, n):
        raise ValueError(f"Matrix must be square, got {matrix.shape}")

    # Validate reciprocal property: a_ij = 1 / a_ji
    for i in range(n):
        for j in range(i + 1, n):
            expected = 1.0 / matrix[j, i]
            if not np.isclose(matrix[i, j], expected, rtol=0.01):
                print(
                    f"  Warning: Matrix[{i},{j}]={matrix[i,j]:.3f} "
                    f"!= 1/Matrix[{j},{i}]={expected:.3f}. "
                    f"Setting Matrix[{i},{j}] = 1/Matrix[{j},{i}]"
                )
                matrix[i, j] = expected

    if criteria_names is None:
        criteria_names = [f"Criterion {i+1}" for i in range(n)]

    return matrix, criteria_names


def get_saaty_scale_description() -> dict:
    """Return the Saaty intensity of importance scale."""
    return {
        1: "Equal importance",
        2: "Equal to moderate importance",
        3: "Moderate importance",
        4: "Moderate to strong importance",
        5: "Strong importance",
        6: "Strong to very strong importance",
        7: "Very strong importance",
        8: "Very to extremely strong importance",
        9: "Extreme importance",
        "1/2": "Inverse of 2",
        "1/3": "Inverse of 3",
        "1/4": "Inverse of 4",
        "1/5": "Inverse of 5",
        "1/7": "Inverse of 7",
        "1/9": "Inverse of 9",
    }


def format_matrix(matrix: np.ndarray, criteria_names: List[str]) -> str:
    """Format a pairwise comparison matrix as a readable string."""
    n = len(criteria_names)
    header = "        " + "  ".join(f"{name[:8]:>8}" for name in criteria_names)
    lines = [header, "-" * len(header)]

    for i in range(n):
        row_str = f"{criteria_names[i]:>10}  "
        for j in range(n):
            val = matrix[i, j]
            if val == int(val):
                row_str += f"{int(val):>8}  "
            else:
                row_str += f"{val:>8.3f}  "
        lines.append(row_str)

    return "\n".join(lines)
