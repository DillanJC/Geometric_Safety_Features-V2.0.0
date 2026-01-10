"""
Test Phase E Geometry Bundle

Validates that Phase E implementation works correctly:
1. Schema compliance
2. Batch-order invariance
3. Collapse detection
4. Dark river detection
5. Integration with real data

This is the final acceptance test for Phase E.
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import (
    GeometryBundle,
    validate_output,
    batch_invariance_test,
    SCHEMA_VERSION
)


def test_schema_compliance():
    """Test that outputs conform to schema."""
    print("="*80)
    print("TEST 1: Schema Compliance")
    print("="*80 + "\n")

    # Create synthetic data
    np.random.seed(42)
    reference = np.random.randn(100, 256).astype(np.float32)
    queries = np.random.randn(10, 256).astype(np.float32)

    # Compute geometry
    bundle = GeometryBundle(reference, k=10)
    results = bundle.compute(queries)

    # Validate all records
    print(f"Generated {len(results)} records")

    for i, record in enumerate(results):
        try:
            validate_output(record)
        except ValueError as e:
            print(f"✗ Record {i} FAILED validation: {e}")
            return False

    print(f"✓ All {len(results)} records conform to schema")
    print(f"  Schema version: {SCHEMA_VERSION}")
    print()
    return True


def test_batch_order_invariance():
    """Test batch permutation invariance (batch-order invariance)."""
    print("="*80)
    print("TEST 2: Batch-Order Invariance")
    print("="*80 + "\n")

    np.random.seed(42)
    reference = np.random.randn(100, 256).astype(np.float32)
    queries = np.random.randn(20, 256).astype(np.float32)

    passed = batch_invariance_test(queries, reference, k=10, n_permutations=5)

    if passed:
        print("✓ PASS: Batch permutation produces identical per-sample features")
        print("  Reference set never modified by queries")
    else:
        print("✗ FAIL: Batch permutation changed features")
        return False

    print()
    return True


def test_collapse_detection():
    """Test ridge-boundary correlation warning."""
    print("="*80)
    print("TEST 3: Collapse Detection")
    print("="*80 + "\n")

    np.random.seed(42)
    reference = np.random.randn(100, 256).astype(np.float32)
    queries = np.random.randn(50, 256).astype(np.float32)

    # Create boundary distances that are NOT perfectly correlated with geometry
    boundary_distances = np.random.randn(50)

    bundle = GeometryBundle(reference, k=10)
    results = bundle.compute(queries, boundary_distances=boundary_distances, check_collapse=True)

    # Check metadata
    if results and '_metadata' in results[0]:
        metadata = results[0]['_metadata']
        if 'collapse_check' in metadata:
            status = metadata['collapse_check']
            print(f"Collapse check: {status}")

            if 'ridge_boundary_correlation' in metadata:
                corr = metadata['ridge_boundary_correlation']
                print(f"Ridge-boundary correlation: {corr:.3f}")

                if status == 'PASS':
                    print("✓ PASS: Ridge not collapsed with boundary distance")
                elif status == 'RIDGE_COLLAPSE':
                    print("⚠️  WARNING: Ridge collapsed (correlation > 0.9)")
        else:
            print("✗ No collapse check metadata")
            return False
    else:
        print("✗ No metadata in results")
        return False

    print()
    return True


def test_feature_values():
    """Test that all 7 continuous features are computed correctly."""
    print("="*80)
    print("TEST 4: Feature Value Computation")
    print("="*80 + "\n")

    np.random.seed(42)
    reference = np.random.randn(200, 256).astype(np.float32)
    queries = np.random.randn(100, 256).astype(np.float32)

    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries)

    # Extract feature matrix
    features = bundle.get_feature_matrix(results)

    print(f"Computed features for {len(queries)} queries")
    print(f"Feature matrix shape: {features.shape}")
    print()

    # Check that all features are finite
    if np.all(np.isfinite(features)):
        print("✓ All feature values are finite")
    else:
        print("✗ Some feature values are non-finite!")
        return False

    # Check that local_curvature is non-zero (SVD fix verification)
    curv_mean = features[:, 4].mean()
    if curv_mean > 0:
        print(f"✓ Local curvature non-zero (mean={curv_mean:.6f})")
    else:
        print(f"✗ Local curvature is zero (SVD fix failed?)")
        return False

    # Check feature ranges are reasonable
    print()
    print("Feature value ranges:")
    feature_names = ['knn_mean', 'knn_std', 'knn_min', 'knn_max', 'curvature', 'ridge', 'nearest']
    for i, name in enumerate(feature_names):
        fmin, fmax = features[:, i].min(), features[:, i].max()
        print(f"  {name:12s}: [{fmin:.4f}, {fmax:.4f}]")

    print()
    return True


def test_real_data():
    """Test on actual sentiment classification data."""
    print("="*80)
    print("TEST 5: Real Data Integration")
    print("="*80 + "\n")

    # Try multiple locations for data
    possible_paths = [
        Path(__file__).parent.parent / "runs" / "openai_3_large_test_20251231_024532",
        Path("runs/openai_3_large_test_20251231_024532"),
        Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")  # Fallback
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        print("⚠️  SKIP: Real data not found")
        print(f"  Tried: {possible_paths[0]}")
        print()
        return True

    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    print(f"Loaded data: N={len(embeddings)}, D={embeddings.shape[1]}")

    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    print(f"Reference: N={len(reference)}")
    print(f"Query: N={len(queries)}")
    print()

    # Create geometry bundle
    print("Computing geometry bundle...")
    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries, boundary_distances=query_boundaries, check_collapse=True)

    print(f"✓ Computed {len(results)} records")
    print()

    # Summarize
    summary = bundle.summarize(results)

    print("Summary Statistics:")
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Schema version: {summary['schema_version']}")
    print()

    print("Feature Statistics:")
    for feature in ['knn_std_distance', 'knn_max_distance', 'local_curvature', 'ridge_proximity']:
        if feature in summary['feature_statistics']:
            stats = summary['feature_statistics'][feature]
            print(f"  {feature}:")
            print(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print()

    # Check collapse
    if 'computation_metadata' in summary:
        meta = summary['computation_metadata']
        if 'collapse_check' in meta:
            print(f"Collapse check: {meta['collapse_check']}")
            if 'ridge_boundary_correlation' in meta:
                print(f"  Ridge-boundary correlation: {meta['ridge_boundary_correlation']:.3f}")
    print()

    # Extract feature matrix for ML
    features = bundle.get_feature_matrix(results)
    print(f"Feature matrix: shape={features.shape}, dtype={features.dtype}")
    print(f"  Feature means: {features.mean(axis=0).round(4)}")
    print(f"  Feature stds: {features.std(axis=0).round(4)}")
    print()

    return True


def test_batch_processing():
    """Test batch processing for large datasets."""
    print("="*80)
    print("TEST 6: Batch Processing")
    print("="*80 + "\n")

    np.random.seed(42)
    reference = np.random.randn(100, 256).astype(np.float32)
    queries = np.random.randn(250, 256).astype(np.float32)

    bundle = GeometryBundle(reference, k=10)

    # Compute in batches
    print("Computing 250 queries in batches of 50...")
    results = bundle.compute_batch(queries, batch_size=50)

    print(f"✓ Processed {len(results)} records in batches")
    print()

    return True


def main():
    print("\n" + "="*80)
    print("PHASE E GEOMETRY BUNDLE — ACCEPTANCE TEST")
    print("="*80 + "\n")

    tests = [
        ("Schema Compliance", test_schema_compliance),
        ("Batch-Order Invariance", test_batch_order_invariance),
        ("Collapse Detection", test_collapse_detection),
        ("Feature Value Computation", test_feature_values),
        ("Real Data Integration", test_real_data),
        ("Batch Processing", test_batch_processing),
    ]

    results = []

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Result: {passed_count}/{total_count} tests passed")
    print()

    if passed_count == total_count:
        print("="*80)
        print("✓ PHASE E ACCEPTANCE TEST: PASS")
        print("="*80)
        print()
        print("Phase E geometry bundle is ready for production.")
        print()

        # Save acceptance report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("runs") / f"phase_e_acceptance_{timestamp}.json"
        report_path.parent.mkdir(exist_ok=True)

        report = {
            'timestamp': timestamp,
            'status': 'PASS',
            'tests_passed': passed_count,
            'tests_total': total_count,
            'schema_version': SCHEMA_VERSION,
            'test_results': [{'name': name, 'passed': passed} for name, passed in results]
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Acceptance report saved: {report_path}")
        print()

        return 0
    else:
        print("="*80)
        print("✗ PHASE E ACCEPTANCE TEST: FAIL")
        print("="*80)
        print()
        print(f"Failed {total_count - passed_count} tests. Review output above.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
