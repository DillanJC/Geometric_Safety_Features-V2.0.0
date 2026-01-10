"""
Geometry Bundle — Integration Layer (v2.0)

Combines all geometric features into schema-compliant outputs.

Responsibilities:
- Compute all 7 k-NN features
- Compute centroid anchor
- Package into GeometryOutputSchema v2.0 format (continuous features only)
- Reference-only computation (batch-order invariant)
- Batch-order invariant

v2.0 Changes:
- Removed dark_river_candidate and observer_mode binary detection
- Removed binary threshold parameters from __init__
- get_flags() and summarize() updated to remove dark river stats
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .schema import create_record, SCHEMA_VERSION
from .features import (
    compute_knn_features,
    compute_centroid_anchor,
    compute_reference_hash,
    compute_config_hash
)


class GeometryBundle:
    """
    Main interface for computing geometric features.

    Usage:
        bundle = GeometryBundle(reference_embeddings, k=50)
        results = bundle.compute(query_embeddings)
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        k: int = 50,
        metric: str = 'euclidean'
    ):
        """
        Initialize geometry bundle with reference set.

        Args:
            reference_embeddings: Reference embeddings (N_ref, D)
            k: Number of neighbors (default: 50, validated optimum)
            metric: Distance metric (default: 'euclidean')
        """
        self.reference_embeddings = reference_embeddings
        self.k = k
        self.metric = metric

        # Compute reference statistics (kept for potential future use)
        ref_centroid = reference_embeddings.mean(axis=0)
        ref_distances = np.linalg.norm(reference_embeddings - ref_centroid, axis=1)
        self.reference_std = ref_distances.std()

        # Compute hashes for artifact discipline
        self.ref_hash = compute_reference_hash(reference_embeddings)
        self.config_hash = compute_config_hash(k, metric)

    def compute(
        self,
        query_embeddings: np.ndarray,
        boundary_distances: Optional[np.ndarray] = None,
        check_collapse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compute geometry bundle for query embeddings.

        Args:
            query_embeddings: Query points (N_query, D)
            boundary_distances: Optional boundary distances for collapse check
            check_collapse: If True, check for ridge-boundary correlation

        Returns:
            List of geometry records (one per query point)
        """
        N_query = len(query_embeddings)

        # Compute k-NN features
        knn_features, knn_metadata = compute_knn_features(
            query_embeddings,
            self.reference_embeddings,
            k=self.k,
            metric=self.metric,
            check_collapse=check_collapse,
            boundary_distances=boundary_distances
        )

        # Compute centroid anchor
        centroid_anchors = compute_centroid_anchor(
            query_embeddings,
            self.reference_embeddings
        )

        # Extract individual features
        knn_mean = knn_features[:, 0]
        knn_std = knn_features[:, 1]
        knn_min = knn_features[:, 2]
        knn_max = knn_features[:, 3]
        local_curvature = knn_features[:, 4]
        ridge_proximity = knn_features[:, 5]
        nearest_dist = knn_features[:, 6]

        # Package results (v2.0: no binary flags)
        results = []
        for i in range(N_query):
            record = create_record(
                dist_to_ref_mean=float(centroid_anchors[i]),
                dist_to_ref_nearest=float(nearest_dist[i]),
                knn_mean_distance=float(knn_mean[i]),
                knn_std_distance=float(knn_std[i]),
                knn_min_distance=float(knn_min[i]),
                knn_max_distance=float(knn_max[i]),
                local_curvature=float(local_curvature[i]),
                ridge_proximity=float(ridge_proximity[i]),
                ref_hash=self.ref_hash,
                config_hash=self.config_hash,
                k_neighbors=self.k
            )
            results.append(record)

        # Add metadata to first record (for debugging)
        if results:
            results[0]['_metadata'] = knn_metadata

        return results

    def compute_batch(
        self,
        query_embeddings: np.ndarray,
        batch_size: int = 100,
        boundary_distances: Optional[np.ndarray] = None,
        check_collapse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compute geometry bundle in batches (for large query sets).

        Args:
            query_embeddings: Query points (N_query, D)
            batch_size: Batch size for processing
            boundary_distances: Optional boundary distances
            check_collapse: Check collapse only on first batch

        Returns:
            List of geometry records
        """
        N = len(query_embeddings)
        all_results = []

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_embeddings = query_embeddings[start_idx:end_idx]

            batch_boundary = None
            if boundary_distances is not None:
                batch_boundary = boundary_distances[start_idx:end_idx]

            # Only check collapse on first batch
            batch_check_collapse = check_collapse and (start_idx == 0)

            batch_results = self.compute(
                batch_embeddings,
                batch_boundary,
                batch_check_collapse
            )

            all_results.extend(batch_results)

        return all_results

    def get_feature_matrix(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract 7-feature matrix from geometry results.

        Useful for downstream ML models.

        Args:
            results: List of geometry records from compute()

        Returns:
            features: (N, 7) array of geometric features
        """
        N = len(results)
        features = np.zeros((N, 7), dtype=np.float32)

        for i, record in enumerate(results):
            features[i, 0] = record['knn_mean_distance']
            features[i, 1] = record['knn_std_distance']
            features[i, 2] = record['knn_min_distance']
            features[i, 3] = record['knn_max_distance']
            features[i, 4] = record['local_curvature']
            features[i, 5] = record['ridge_proximity']
            features[i, 6] = record['dist_to_ref_nearest']

        return features

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize geometry bundle results.

        Args:
            results: List of geometry records

        Returns:
            Summary statistics
        """
        N = len(results)

        # Extract features
        features = self.get_feature_matrix(results)

        summary = {
            'n_samples': N,
            'feature_statistics': {
                'knn_mean_distance': {
                    'mean': float(features[:, 0].mean()),
                    'std': float(features[:, 0].std()),
                    'min': float(features[:, 0].min()),
                    'max': float(features[:, 0].max())
                },
                'knn_std_distance': {
                    'mean': float(features[:, 1].mean()),
                    'std': float(features[:, 1].std()),
                    'min': float(features[:, 1].min()),
                    'max': float(features[:, 1].max())
                },
                'knn_max_distance': {
                    'mean': float(features[:, 3].mean()),
                    'std': float(features[:, 3].std()),
                    'min': float(features[:, 3].min()),
                    'max': float(features[:, 3].max())
                },
                'local_curvature': {
                    'mean': float(features[:, 4].mean()),
                    'std': float(features[:, 4].std()),
                    'min': float(features[:, 4].min()),
                    'max': float(features[:, 4].max())
                },
                'ridge_proximity': {
                    'mean': float(features[:, 5].mean()),
                    'std': float(features[:, 5].std()),
                    'min': float(features[:, 5].min()),
                    'max': float(features[:, 5].max())
                }
            },
            'reference_hash': self.ref_hash,
            'config_hash': self.config_hash,
            'schema_version': SCHEMA_VERSION
        }

        # Add metadata from first result if available
        if results and '_metadata' in results[0]:
            summary['computation_metadata'] = results[0]['_metadata']

        return summary
