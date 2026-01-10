"""
Behavioral Flip Experiment - Step 4: Geometric Analysis

Computes geometric features for original queries and analyzes correlation
with flip rates.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle


def main():
    # Load flip results
    results_path = Path("experiments/behavioral_flip_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print("Loaded flip results for 30 queries\n")

    # Load reference embeddings for geometry computation
    data_path = Path("runs/openai_3_large_test_20251231_024532")
    reference_embeddings = np.load(data_path / "embeddings.npy")
    print(f"Loaded reference set: {len(reference_embeddings)} samples\n")

    # Extract query embeddings and flip rates
    query_embeddings = []
    flip_rates = []
    zones = []
    boundary_distances = []

    for zone_name, zone_samples in samples.items():
        for sample in zone_samples:
            embedding = np.array(sample['original_pred']['embedding'])
            query_embeddings.append(embedding)
            flip_rates.append(sample['flip_rate'])
            zones.append(zone_name)
            boundary_distances.append(sample['boundary_distance'])

    query_embeddings = np.array(query_embeddings)
    flip_rates = np.array(flip_rates)
    boundary_distances = np.array(boundary_distances)

    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Flip rates: {flip_rates.shape}\n")

    # Compute geometric features
    print("Computing geometric features (k=50)...")
    bundle = GeometryBundle(reference_embeddings, k=50)
    geometry_results = bundle.compute(query_embeddings)
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

    print(f"Geometric features shape: {geometry_features.shape}\n")

    # ========================================================================
    # Analysis 1: Overall Correlation
    # ========================================================================
    print("="*80)
    print("ANALYSIS 1: Correlation with Flip Rate (Overall, N=30)")
    print("="*80)
    print(f"{'Feature':<25} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12}")
    print("-"*80)

    correlations = {}

    for i, feature_name in enumerate(feature_names):
        feature_values = geometry_features[:, i]
        r, p = pearsonr(feature_values, flip_rates)
        rho, p_spear = spearmanr(feature_values, flip_rates)
        correlations[feature_name] = {'r': r, 'p': p, 'rho': rho}

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feature_name:<25} {r:>12.3f} {p:>12.2e} {rho:>12.3f} {sig}")

    # ========================================================================
    # Analysis 2: Zone-Stratified Correlation
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 2: Correlation by Zone")
    print("="*80)

    zone_names = ['safe', 'borderline', 'unsafe']
    zone_masks = {
        'safe': np.array([z == 'safe' for z in zones]),
        'borderline': np.array([z == 'borderline' for z in zones]),
        'unsafe': np.array([z == 'unsafe' for z in zones])
    }

    for zone_name in zone_names:
        mask = zone_masks[zone_name]
        n_zone = mask.sum()

        print(f"\n{zone_name.upper()} (N={n_zone}):")
        print(f"  Flip rate: mean={flip_rates[mask].mean():.3f}, std={flip_rates[mask].std():.3f}")

        # Find top feature for this zone
        best_r = -1
        best_feature = None

        for i, feature_name in enumerate(feature_names):
            if n_zone >= 5:  # Need minimum samples for correlation
                feature_values = geometry_features[mask, i]
                flip_rates_zone = flip_rates[mask]

                # Only compute if there's variance
                if flip_rates_zone.std() > 0:
                    r, p = pearsonr(feature_values, flip_rates_zone)
                    if abs(r) > abs(best_r):
                        best_r = r
                        best_feature = feature_name

        if best_feature:
            print(f"  Top feature: {best_feature} (r={best_r:.3f})")

    # ========================================================================
    # Analysis 3: Regression Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 3: Regression Analysis (Predicting Flip Rate)")
    print("="*80)

    # Baseline: embeddings only
    ridge_baseline = Ridge(alpha=1.0, random_state=42)
    ridge_baseline.fit(query_embeddings, flip_rates)
    pred_baseline = ridge_baseline.predict(query_embeddings)
    r2_baseline = r2_score(flip_rates, pred_baseline)

    # Geometry: embeddings + geometry features
    X_geometry = np.concatenate([query_embeddings, geometry_features], axis=1)
    ridge_geometry = Ridge(alpha=1.0, random_state=42)
    ridge_geometry.fit(X_geometry, flip_rates)
    pred_geometry = ridge_geometry.predict(X_geometry)
    r2_geometry = r2_score(flip_rates, pred_geometry)

    improvement = (r2_geometry - r2_baseline) / r2_baseline * 100 if r2_baseline > 0 else 0

    print(f"Baseline (embeddings only):          R² = {r2_baseline:.4f}")
    print(f"Geometry (embeddings + geometry):    R² = {r2_geometry:.4f}")
    print(f"Improvement:                         {improvement:+.1f}%")

    # ========================================================================
    # Analysis 4: Hypothesis Testing
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 4: Hypothesis Tests")
    print("="*80)

    # H1: knn_std_distance correlates with flip rate (r > 0.2)
    knn_std_idx = feature_names.index('knn_std_distance')
    knn_std_values = geometry_features[:, knn_std_idx]
    r_std, p_std = pearsonr(knn_std_values, flip_rates)

    print(f"H1: knn_std_distance predicts flip rate (r > 0.2)")
    print(f"    Observed: r = {r_std:.3f}, p = {p_std:.2e}")
    print(f"    Result: {'✓ SUPPORTED' if r_std > 0.2 and p_std < 0.05 else '✗ NOT SUPPORTED'}")

    # H2: Borderline queries show higher flip rates
    flip_rate_borderline = flip_rates[zone_masks['borderline']].mean()
    flip_rate_safe = flip_rates[zone_masks['safe']].mean()
    flip_rate_unsafe = flip_rates[zone_masks['unsafe']].mean()

    print(f"\nH2: Borderline queries show higher flip rates")
    print(f"    Safe:       {flip_rate_safe:.3f}")
    print(f"    Borderline: {flip_rate_borderline:.3f}")
    print(f"    Unsafe:     {flip_rate_unsafe:.3f}")
    print(f"    Ratio:      {flip_rate_borderline / max(flip_rate_safe, 0.001):.1f}× higher than safe")
    print(f"    Result: {'✓ SUPPORTED' if flip_rate_borderline > flip_rate_safe else '✗ NOT SUPPORTED'}")

    # H3: Geometry improves flip rate prediction (>10%)
    print(f"\nH3: Geometry improves flip rate prediction (>10%)")
    print(f"    Observed improvement: {improvement:+.1f}%")
    print(f"    Result: {'✓ SUPPORTED' if improvement > 10 else '✗ NOT SUPPORTED'}")

    # ========================================================================
    # Save Analysis Results
    # ========================================================================
    analysis_results = {
        'overall_correlations': {
            name: {'pearson_r': float(correlations[name]['r']),
                   'p_value': float(correlations[name]['p']),
                   'spearman_rho': float(correlations[name]['rho'])}
            for name in feature_names
        },
        'zone_flip_rates': {
            'safe': float(flip_rate_safe),
            'borderline': float(flip_rate_borderline),
            'unsafe': float(flip_rate_unsafe)
        },
        'regression': {
            'r2_baseline': float(r2_baseline),
            'r2_geometry': float(r2_geometry),
            'improvement_percent': float(improvement)
        },
        'hypothesis_tests': {
            'H1_knn_std_predicts_flips': {
                'r': float(r_std),
                'p': float(p_std),
                'supported': bool(r_std > 0.2 and p_std < 0.05)
            },
            'H2_borderline_higher_flips': {
                'flip_rate_safe': float(flip_rate_safe),
                'flip_rate_borderline': float(flip_rate_borderline),
                'ratio': float(flip_rate_borderline / max(flip_rate_safe, 0.001)),
                'supported': bool(flip_rate_borderline > flip_rate_safe)
            },
            'H3_geometry_improves_prediction': {
                'improvement_percent': float(improvement),
                'supported': bool(improvement > 10)
            }
        }
    }

    output_path = Path("experiments/behavioral_flip_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✓ Saved analysis results to {output_path}")


if __name__ == "__main__":
    main()
