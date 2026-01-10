"""
Mirrorfield Geometry Module — Phase E

Native k-NN geometric features for AI safety boundary detection.

Validated Performance:
- Baseline (embeddings only): R² ≈ 0.34
- With geometry features: R² ≈ 0.40 (+6.4%, p<10⁻⁶)
- Robust across k ∈ {25, 50, 100}

Main Interface:
    from mirrorfield.geometry import GeometryBundle

    # Initialize with reference set
    bundle = GeometryBundle(reference_embeddings, k=50)

    # Compute for query points
    results = bundle.compute(query_embeddings)

    # Extract features for ML models
    features = bundle.get_feature_matrix(results)
"""

from .schema import (
    GeometryOutputSchema,
    SCHEMA_VERSION,
    validate_output,
    create_record,
    NAME_MAP
)

from .features import (
    compute_knn_features,
    compute_centroid_anchor,
    detect_dark_rivers,
    detect_observer_mode,
    batch_invariance_test,
    FEATURE_NAMES
)

from .bundle import GeometryBundle

__version__ = "1.0.0"

__all__ = [
    # Main interface
    'GeometryBundle',

    # Schema
    'GeometryOutputSchema',
    'SCHEMA_VERSION',
    'validate_output',
    'create_record',
    'NAME_MAP',

    # Features
    'compute_knn_features',
    'compute_centroid_anchor',
    'detect_dark_rivers',
    'detect_observer_mode',
    'batch_invariance_test',
    'FEATURE_NAMES',
]
