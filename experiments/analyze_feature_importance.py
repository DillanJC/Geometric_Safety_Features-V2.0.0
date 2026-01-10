"""
Feature Importance Analysis for Geometry Features

Determines which of the 7 k-NN features actually drive the improvement
in boundary distance prediction.

Tests:
1. Pearson correlation with boundary distance
2. Random Forest feature importance
3. Ablation study (drop each feature, measure loss)
4. Borderline-sliced analysis
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def load_data():
    """Load real sentiment data."""
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
        raise FileNotFoundError(
            f"Data not found. Tried:\n" + "\n".join(f"  - {p}" for p in possible_paths) +
            f"\n\nPlease ensure data files are in the runs/ directory."
        )

    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    return embeddings, boundary_distances


def compute_geometry_features(embeddings, boundary_distances):
    """Compute geometry bundle features."""
    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    # Compute geometry
    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries, boundary_distances=query_boundaries)

    # Extract feature matrix
    features = bundle.get_feature_matrix(results)

    return features, query_boundaries


def correlation_analysis(features, boundary_distances, feature_names):
    """Compute correlation of each feature with boundary distance."""
    print("="*80)
    print("CORRELATION ANALYSIS")
    print("="*80 + "\n")

    correlations = []
    for i, name in enumerate(feature_names):
        corr, pval = pearsonr(features[:, i], boundary_distances)
        correlations.append((name, corr, pval))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Features ranked by correlation with boundary distance:")
    print()
    for name, corr, pval in correlations:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:25s}: r = {corr:+.3f}  (p = {pval:.2e}) {sig}")
    print()

    return correlations


def random_forest_importance(features, boundary_distances, feature_names):
    """Random Forest feature importance."""
    print("="*80)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*80 + "\n")

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(features, boundary_distances)

    # Get importances
    importances = rf.feature_importances_

    # Sort by importance
    importance_ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("Features ranked by Random Forest importance:")
    print()
    for name, imp in importance_ranking:
        bar = "█" * int(imp * 100)
        print(f"  {name:25s}: {imp:.3f} {bar}")
    print()

    # R² score
    predictions = rf.predict(features)
    r2 = r2_score(boundary_distances, predictions)
    print(f"Random Forest R²: {r2:.3f}")
    print()

    return importance_ranking, r2


def ablation_analysis(features, boundary_distances, feature_names):
    """Ablation study: drop each feature and measure performance loss."""
    print("="*80)
    print("ABLATION ANALYSIS")
    print("="*80 + "\n")

    # Baseline with all features
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_full.fit(features, boundary_distances)
    baseline_r2 = r2_score(boundary_distances, rf_full.predict(features))

    print(f"Baseline (all 7 features): R² = {baseline_r2:.3f}")
    print()

    # Drop each feature one at a time
    ablation_results = []

    for i, name in enumerate(feature_names):
        # Create feature matrix without feature i
        mask = np.ones(features.shape[1], dtype=bool)
        mask[i] = False
        features_ablated = features[:, mask]

        # Train without this feature
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(features_ablated, boundary_distances)
        r2 = r2_score(boundary_distances, rf.predict(features_ablated))

        # Calculate loss
        loss = baseline_r2 - r2
        ablation_results.append((name, r2, loss))

    # Sort by loss (higher loss = more important)
    ablation_results.sort(key=lambda x: x[2], reverse=True)

    print("Features ranked by performance loss when dropped:")
    print()
    for name, r2, loss in ablation_results:
        pct_loss = 100 * loss / baseline_r2
        print(f"  {name:25s}: R² = {r2:.3f}  (loss = {loss:+.4f}, {pct_loss:+.1f}%)")
    print()

    return ablation_results


def borderline_analysis(features, boundary_distances, feature_names):
    """Analyze features specifically on borderline slice."""
    print("="*80)
    print("BORDERLINE SLICE ANALYSIS")
    print("="*80 + "\n")

    # Define borderline as |boundary_distance| < 0.5
    borderline_mask = np.abs(boundary_distances) < 0.5
    n_borderline = borderline_mask.sum()

    print(f"Borderline slice: {n_borderline}/{len(boundary_distances)} points ({100*n_borderline/len(boundary_distances):.1f}%)")
    print()

    if n_borderline < 10:
        print("⚠️  Too few borderline points for analysis")
        return

    features_borderline = features[borderline_mask]
    boundary_borderline = boundary_distances[borderline_mask]

    # Correlation analysis on borderline
    print("Correlations on borderline slice:")
    correlations = []
    for i, name in enumerate(feature_names):
        corr, pval = pearsonr(features_borderline[:, i], boundary_borderline)
        correlations.append((name, corr, pval))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr, pval in correlations:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:25s}: r = {corr:+.3f}  (p = {pval:.2e}) {sig}")
    print()

    # Random Forest on borderline
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(features_borderline, boundary_borderline)
    r2 = r2_score(boundary_borderline, rf.predict(features_borderline))

    print(f"Random Forest R² on borderline: {r2:.3f}")
    print()

    importances = rf.feature_importances_
    importance_ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("Feature importance on borderline:")
    for name, imp in importance_ranking:
        bar = "█" * int(imp * 100)
        print(f"  {name:25s}: {imp:.3f} {bar}")
    print()


def main():
    print("\n" + "="*80)
    print("GEOMETRY FEATURE IMPORTANCE ANALYSIS")
    print("="*80 + "\n")

    # Load data
    embeddings, boundary_distances = load_data()
    print(f"Loaded: N={len(embeddings)}, D={embeddings.shape[1]}")
    print()

    # Compute geometry features
    print("Computing geometry features...")
    features, query_boundaries = compute_geometry_features(embeddings, boundary_distances)
    print(f"Computed features: {features.shape}")
    print()

    feature_names = [
        'knn_mean_distance',
        'knn_std_distance',
        'knn_min_distance',
        'knn_max_distance',
        'local_curvature',
        'ridge_proximity',
        'dist_to_ref_nearest'
    ]

    # Analysis 1: Correlation
    correlations = correlation_analysis(features, query_boundaries, feature_names)

    # Analysis 2: Random Forest importance
    rf_importance, rf_r2 = random_forest_importance(features, query_boundaries, feature_names)

    # Analysis 3: Ablation study
    ablation_results = ablation_analysis(features, query_boundaries, feature_names)

    # Analysis 4: Borderline slice
    borderline_analysis(features, query_boundaries, feature_names)

    # Summary
    print("="*80)
    print("SUMMARY: TOP FEATURES")
    print("="*80 + "\n")

    print("Top 3 by correlation:")
    for name, corr, _ in correlations[:3]:
        print(f"  {name}: r = {corr:+.3f}")
    print()

    print("Top 3 by Random Forest importance:")
    for name, imp in rf_importance[:3]:
        print(f"  {name}: {imp:.3f}")
    print()

    print("Top 3 by ablation loss:")
    for name, _, loss in ablation_results[:3]:
        print(f"  {name}: loss = {loss:+.4f}")
    print()

    print("="*80)
    print("CONCLUSION")
    print("="*80 + "\n")

    # Identify consensus top features
    top_by_corr = set([name for name, _, _ in correlations[:3]])
    top_by_rf = set([name for name, _ in rf_importance[:3]])
    top_by_ablation = set([name for name, _, _ in ablation_results[:3]])

    consensus = top_by_corr & top_by_rf & top_by_ablation

    if consensus:
        print("Consensus top features (appear in top 3 for all metrics):")
        for name in consensus:
            print(f"  ⭐ {name}")
    else:
        print("No single feature dominates all metrics.")
        union = top_by_corr | top_by_rf | top_by_ablation
        print("\nTop features (appear in at least one top-3 ranking):")
        for name in union:
            in_corr = "✓" if name in top_by_corr else " "
            in_rf = "✓" if name in top_by_rf else " "
            in_abl = "✓" if name in top_by_ablation else " "
            print(f"  [{in_corr}] Corr  [{in_rf}] RF  [{in_abl}] Ablation: {name}")
    print()


if __name__ == "__main__":
    main()
