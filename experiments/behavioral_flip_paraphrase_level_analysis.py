"""
Behavioral Flip Experiment - Paraphrase-Level Analysis

Instead of aggregating flip rate per query (N=30), treats each paraphrase
as an independent observation (N=150), testing whether original query geometry
predicts whether a specific paraphrase will flip.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle


def main():
    # Load flip results
    results_path = Path("experiments/behavioral_flip_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print("=" * 80)
    print("PARAPHRASE-LEVEL ANALYSIS (N=150)")
    print("=" * 80)
    print("\nTreating each paraphrase as independent observation")
    print("Hypothesis: Original query geometry predicts paraphrase flip\n")

    # Load reference embeddings
    data_path = Path("runs/openai_3_large_test_20251231_024532")
    reference_embeddings = np.load(data_path / "embeddings.npy")

    # Build paraphrase-level dataset
    paraphrase_data = []

    for zone_name, zone_samples in samples.items():
        for sample in zone_samples:
            original_label = sample['original_pred']['label']
            original_embedding = np.array(sample['original_pred']['embedding'])
            boundary_distance = sample['boundary_distance']

            for para_pred in sample['paraphrase_preds']:
                para_label = para_pred['label']
                flipped = int(para_label != original_label)

                paraphrase_data.append({
                    'zone': zone_name,
                    'original_label': original_label,
                    'para_label': para_label,
                    'flipped': flipped,
                    'original_embedding': original_embedding,
                    'boundary_distance': boundary_distance
                })

    print(f"Total paraphrases: {len(paraphrase_data)}")

    # Extract arrays
    original_embeddings = np.array([p['original_embedding'] for p in paraphrase_data])
    flip_labels = np.array([p['flipped'] for p in paraphrase_data])
    zones = np.array([p['zone'] for p in paraphrase_data])
    boundary_distances = np.array([p['boundary_distance'] for p in paraphrase_data])

    print(f"Flips: {flip_labels.sum()}/{len(flip_labels)} ({100*flip_labels.mean():.1f}%)")
    print(f"  Safe: {flip_labels[zones == 'safe'].sum()}/{(zones == 'safe').sum()}")
    print(f"  Borderline: {flip_labels[zones == 'borderline'].sum()}/{(zones == 'borderline').sum()}")
    print(f"  Unsafe: {flip_labels[zones == 'unsafe'].sum()}/{(zones == 'unsafe').sum()}\n")

    # Compute geometric features for original queries
    print("Computing geometric features for original queries...")
    bundle = GeometryBundle(reference_embeddings, k=50)
    geometry_results = bundle.compute(original_embeddings)
    geometry_features = bundle.get_feature_matrix(geometry_results)

    feature_names = [
        'knn_mean_distance',
        'knn_std_distance',
        'knn_min_distance',
        'knn_max_distance',
        'local_curvature',
        'ridge_proximity',
        'dist_to_ref_nearest'
    ]

    # ========================================================================
    # Analysis 1: Point-Biserial Correlation (continuous feature vs binary flip)
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS 1: Point-Biserial Correlation (Feature vs Flip)")
    print("=" * 80)
    print(f"{'Feature':<25} {'r_pb':>10} {'p-value':>12} {'Sig':>5}")
    print("-" * 80)

    correlations = {}

    for i, feature_name in enumerate(feature_names):
        feature_values = geometry_features[:, i]
        r_pb, p = pointbiserialr(flip_labels, feature_values)
        correlations[feature_name] = {'r_pb': r_pb, 'p': p}

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feature_name:<25} {r_pb:>10.3f} {p:>12.2e} {sig:>5}")

    # ========================================================================
    # Analysis 2: Logistic Regression (Predict Flip)
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Logistic Regression (Predicting Flip)")
    print("=" * 80)

    # Baseline: boundary distance only
    X_boundary = boundary_distances.reshape(-1, 1)
    lr_boundary = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_boundary.fit(X_boundary, flip_labels)
    pred_proba_boundary = lr_boundary.predict_proba(X_boundary)[:, 1]
    auc_boundary = roc_auc_score(flip_labels, pred_proba_boundary)

    # Geometry: geometry features only
    lr_geometry = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_geometry.fit(geometry_features, flip_labels)
    pred_proba_geometry = lr_geometry.predict_proba(geometry_features)[:, 1]
    auc_geometry = roc_auc_score(flip_labels, pred_proba_geometry)

    # Combined: boundary distance + geometry
    X_combined = np.concatenate([X_boundary, geometry_features], axis=1)
    lr_combined = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_combined.fit(X_combined, flip_labels)
    pred_proba_combined = lr_combined.predict_proba(X_combined)[:, 1]
    auc_combined = roc_auc_score(flip_labels, pred_proba_combined)

    print(f"\nBoundary Distance Only:   AUC = {auc_boundary:.3f}")
    print(f"Geometry Features Only:   AUC = {auc_geometry:.3f}")
    print(f"Combined:                 AUC = {auc_combined:.3f}")

    improvement_vs_boundary = (auc_combined - auc_boundary) / auc_boundary * 100
    improvement_vs_geometry = (auc_combined - auc_geometry) / auc_geometry * 100

    print(f"\nImprovement over boundary alone: {improvement_vs_boundary:+.1f}%")
    print(f"Improvement over geometry alone: {improvement_vs_geometry:+.1f}%")

    # ========================================================================
    # Analysis 3: Zone-Stratified Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Zone-Stratified Correlation")
    print("=" * 80)

    zone_names = ['safe', 'borderline', 'unsafe']

    for zone_name in zone_names:
        mask = zones == zone_name
        n_zone = mask.sum()
        n_flips = flip_labels[mask].sum()
        flip_rate = flip_labels[mask].mean()

        print(f"\n{zone_name.upper()} (N={n_zone}, flips={n_flips}, rate={100*flip_rate:.1f}%):")

        if n_flips > 5:  # Need enough flips for correlation
            # Find best feature
            best_r = -1
            best_feature = None

            for i, feature_name in enumerate(feature_names):
                feature_values = geometry_features[mask, i]
                flip_labels_zone = flip_labels[mask]

                r_pb, p = pointbiserialr(flip_labels_zone, feature_values)
                if abs(r_pb) > abs(best_r):
                    best_r = r_pb
                    best_feature = feature_name

            if best_feature:
                print(f"  Top feature: {best_feature} (r_pb={best_r:.3f})")
        else:
            print(f"  Insufficient flips for correlation analysis")

    # ========================================================================
    # Analysis 4: Feature Importance from Logistic Regression
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Feature Importance (Logistic Regression Coefficients)")
    print("=" * 80)

    coeffs = lr_geometry.coef_[0]
    feature_importance = list(zip(feature_names, coeffs))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<25} {'Coefficient':>12} {'Direction':>10}")
    print("-" * 80)
    for name, coeff in feature_importance:
        direction = "↑ flip" if coeff > 0 else "↓ flip"
        print(f"{name:<25} {coeff:>12.3f} {direction:>10}")

    # ========================================================================
    # Save Results
    # ========================================================================
    analysis_results = {
        'n_paraphrases': len(paraphrase_data),
        'n_flips': int(flip_labels.sum()),
        'flip_rate_overall': float(flip_labels.mean()),
        'correlations': {
            name: {'r_pb': float(correlations[name]['r_pb']),
                   'p_value': float(correlations[name]['p'])}
            for name in feature_names
        },
        'logistic_regression': {
            'auc_boundary_only': float(auc_boundary),
            'auc_geometry_only': float(auc_geometry),
            'auc_combined': float(auc_combined),
            'improvement_vs_boundary_percent': float(improvement_vs_boundary)
        },
        'feature_importance': [
            {'feature': name, 'coefficient': float(coeff)}
            for name, coeff in feature_importance
        ]
    }

    output_path = Path("experiments/behavioral_flip_paraphrase_level_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✓ Saved results to {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find top correlated feature
    top_feature = max(correlations.items(), key=lambda x: abs(x[1]['r_pb']))
    top_name, top_stats = top_feature

    print(f"\nTop correlated feature: {top_name}")
    print(f"  r_pb = {top_stats['r_pb']:.3f}, p = {top_stats['p']:.2e}")
    print(f"  Significant: {'Yes ***' if top_stats['p'] < 0.001 else 'Yes **' if top_stats['p'] < 0.01 else 'Yes *' if top_stats['p'] < 0.05 else 'No'}")

    print(f"\nPredictive power (AUC):")
    print(f"  Boundary distance: {auc_boundary:.3f}")
    print(f"  Geometry features: {auc_geometry:.3f}")
    print(f"  Combined:          {auc_combined:.3f}")

    print(f"\nConclusion:")
    if top_stats['p'] < 0.05:
        print(f"  ✓ Significant correlation found at paraphrase level (N=150)")
        print(f"  ✓ {top_name} predicts flip likelihood")
    else:
        print(f"  ✗ No significant correlation at p < 0.05")
        print(f"  • Increasing sample size did not rescue signal")

    if improvement_vs_boundary > 5:
        print(f"  ✓ Geometry adds value beyond boundary distance ({improvement_vs_boundary:+.1f}%)")
    else:
        print(f"  ✗ Geometry provides minimal added value ({improvement_vs_boundary:+.1f}%)")


if __name__ == "__main__":
    main()
