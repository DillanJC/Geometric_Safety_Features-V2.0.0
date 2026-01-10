"""
Geometry Bundle Schema — Phase E Contract (Algebra v2)

Frozen dataclass defining output structure for geometric features.
Single source of truth for output keys.

Design principles:
- Schema version tracking for forward compatibility
- NAME_MAP for key aliasing (swappable keys without refactor)
- No literal output-key strings elsewhere in codebase
- All records include _schema_version

Version History:
- v1.0: Initial release with dark_river_candidate/observer_mode flags
- v2.0: Removed binary flags (dark_river hypothesis falsified), continuous features only
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

__version__ = "2.0.0"

# Schema version for this output format
SCHEMA_VERSION = "geometry_v2.0"


@dataclass(frozen=True)
class GeometryOutputSchema:
    """
    Frozen contract for geometry bundle outputs.

    All geometry computations must produce records matching this schema.

    v2.0: Continuous features only - binary dark river flags removed after
    falsification of discrete region hypothesis.
    """

    # === Core Distance Features ===
    dist_to_ref_mean: float
    """Mean distance from query point to reference set (Centroid Anchor / CA)"""

    dist_to_ref_nearest: float
    """Distance to nearest reference point (1-NN distance)"""

    # === k-NN Statistics (k=50 default) ===
    knn_mean_distance: float
    """Mean distance to k nearest neighbors"""

    knn_std_distance: float
    """Standard deviation of k-NN distances (local uniformity) - STRONGEST borderline predictor"""

    knn_min_distance: float
    """Minimum k-NN distance (already captured in dist_to_ref_nearest, but kept for completeness)"""

    knn_max_distance: float
    """Maximum k-NN distance (neighborhood extent) - STRONGEST overall predictor"""

    # === Geometric Structure Features ===
    local_curvature: float
    """
    Local manifold curvature via SVD singular value ratio: σ_min / σ_max

    Low values (~0.01-0.02) indicate anisotropic neighborhoods (manifold stretching)
    High values (~1.0) indicate spherical neighborhoods

    NOTE: Continuous correlation with boundary distance, not discrete threshold
    """

    ridge_proximity: float
    """
    Coefficient of variation (CV) of neighborhood distances: σ_dist / μ_dist

    Moderate values (~0.2) indicate smooth density (typical for normalized embeddings)
    Higher values (~0.3-0.4) indicate density variations

    NOTE: Continuous correlation with boundary distance, not discrete threshold
    Also referred to as "ridge_proximity" in legacy code
    """

    # === Metadata (Artifact Discipline) ===
    _schema_version: str
    """Schema version for forward compatibility"""

    _ref_hash: str
    """Hash of reference set used (for reproducibility)"""

    _config_hash: str
    """Hash of configuration (k, distance metric, etc.)"""

    k_neighbors: int
    """Number of neighbors used for k-NN statistics"""


# NAME_MAP adapter for swappable keys (batch-order invariant)
NAME_MAP = {
    # Canonical name → Alternative name
    "dist_to_ref_mean": "mean_ref_distance",
    "dist_to_ref_nearest": "nearest_neighbor_dist",
    "local_curvature": "svd_curvature",
    "ridge_proximity": "coefficient_of_variation",
}


def validate_output(record: Dict[str, Any]) -> bool:
    """
    Validate that a record conforms to GeometryOutputSchema v2.0.

    Args:
        record: Dictionary to validate

    Returns:
        True if valid, raises ValueError if not
    """
    required_fields = [
        'dist_to_ref_mean', 'dist_to_ref_nearest',
        'knn_mean_distance', 'knn_std_distance',
        'knn_min_distance', 'knn_max_distance',
        'local_curvature', 'ridge_proximity',
        '_schema_version', '_ref_hash', '_config_hash', 'k_neighbors'
    ]

    missing = [f for f in required_fields if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Type checks
    float_fields = [
        'dist_to_ref_mean', 'dist_to_ref_nearest',
        'knn_mean_distance', 'knn_std_distance',
        'knn_min_distance', 'knn_max_distance',
        'local_curvature', 'ridge_proximity'
    ]

    for field in float_fields:
        if not isinstance(record[field], (float, np.floating)):
            raise ValueError(f"{field} must be float, got {type(record[field])}")

    # Schema version
    if record['_schema_version'] != SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: {record['_schema_version']} != {SCHEMA_VERSION}")

    return True


def create_record(
    dist_to_ref_mean: float,
    dist_to_ref_nearest: float,
    knn_mean_distance: float,
    knn_std_distance: float,
    knn_min_distance: float,
    knn_max_distance: float,
    local_curvature: float,
    ridge_proximity: float,
    ref_hash: str,
    config_hash: str,
    k_neighbors: int
) -> Dict[str, Any]:
    """
    Create a validated geometry output record (v2.0).

    This is the canonical way to create geometry outputs.
    Ensures schema compliance.

    NOTE: v2.0 removed dark_river_candidate and observer_mode boolean flags.
    Use continuous feature values directly for boundary prediction.
    """
    record = {
        'dist_to_ref_mean': float(dist_to_ref_mean),
        'dist_to_ref_nearest': float(dist_to_ref_nearest),
        'knn_mean_distance': float(knn_mean_distance),
        'knn_std_distance': float(knn_std_distance),
        'knn_min_distance': float(knn_min_distance),
        'knn_max_distance': float(knn_max_distance),
        'local_curvature': float(local_curvature),
        'ridge_proximity': float(ridge_proximity),
        '_schema_version': SCHEMA_VERSION,
        '_ref_hash': ref_hash,
        '_config_hash': config_hash,
        'k_neighbors': int(k_neighbors)
    }

    validate_output(record)
    return record
