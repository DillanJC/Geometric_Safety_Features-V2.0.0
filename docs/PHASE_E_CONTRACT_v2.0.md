# Phase E Contract v2.0 — Geometry Bundle (FROZEN)

**Status:** 🔒 **LOCKED** — Do not modify without version bump
**Date Frozen:** 2026-01-08
**Schema Version:** `geometry_v2.0`
**Validated Performance:** +3.8% improvement on borderline cases

---

## Executive Summary

Phase E v2.0 defines the **validated, production-ready geometry feature bundle** for AI safety boundary detection. This contract freezes:

1. **7 continuous k-NN features** (no binary thresholds)
2. **Configuration:** k=50, SVD-based curvature, reference-only computation
3. **Validated improvement:** +3.8% R² on borderline cases (4.8× larger than safe cases)
4. **Falsified hypothesis:** Dark River discrete regions (replaced with continuous correlations)

**Key Principle:** Geometry features provide **continuous safety signals**, not discrete classifications. Use percentiles for visualization only, never for detection.

---

## 1. Frozen Feature Set

### 7 k-NN Geometric Features

| # | Feature Name | Symbol | Computation | Range | Top Predictor |
|---|--------------|--------|-------------|-------|---------------|
| 1 | `knn_mean_distance` | μ_knn | mean(d₁, ..., d_k) | [0, ∞) | r=+0.221 |
| 2 | `knn_std_distance` | σ_knn | std(d₁, ..., d_k) | [0, ∞) | **r=+0.399** ⭐ (borderline) |
| 3 | `knn_min_distance` | d_min | min(d₁, ..., d_k) | [0, ∞) | r=+0.067 |
| 4 | `knn_max_distance` | d_max | max(d₁, ..., d_k) | [0, ∞) | **r=+0.345** ⭐ (overall) |
| 5 | `local_curvature` | JC | σ_min / σ_max (SVD) | [0, 1] | r=-0.103 (nonlinear) |
| 6 | `ridge_proximity` | SD | σ_knn / μ_knn | [0, ∞) | r=+0.215 |
| 7 | `dist_to_ref_nearest` | d_1nn | distance to 1-NN | [0, ∞) | r=+0.067 |

**Plus 2 anchor features:**
- `dist_to_ref_mean` (Centroid Anchor): distance to reference centroid

**Total output:** 8 continuous features per query point

---

## 2. Configuration Parameters (Frozen)

### 2.1 Core Settings

```python
k = 50  # Number of neighbors (validated optimum)
metric = 'euclidean'  # Distance metric
reference_only = True  # Batch-order invariant: queries never modify reference
```

**Validation Evidence:**
- k=50 validated across {25, 50, 100} (Paper 1, Phase D)
- Euclidean distance standard for normalized embeddings
- Reference-only ensures batch-order invariance

---

### 2.2 Local Curvature Computation (CRITICAL FIX)

**Method:** SVD-based (NOT covariance eigendecomposition)

```python
# Correct (v2.0):
centered = neighbors - query_point
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
local_curvature = S[-1] / S[0]  # smallest/largest singular value

# WRONG (v1.0, causes underflow):
# cov = np.cov(centered.T)
# eigenvalues = np.linalg.eigvalsh(cov)
# local_curvature = eigenvalues[0] / eigenvalues[-1]
```

**Why SVD:**
- Direct computation on (k, D) matrix avoids ill-conditioned covariance
- When k << D (50 << 256), covariance method produces numerical artifacts
- SVD produces stable values (~0.01-0.02 for normalized embeddings)

**Validation:** `experiments/validate_svd_curvature.py`
- 20/20 trials show +3.8% improvement with SVD method
- Previous covariance method: all values = 0 (broken)

---

## 3. Schema v2.0 (Output Contract)

### 3.1 Record Structure

Every geometry computation returns a dictionary with:

```python
{
  # 8 Continuous Features
  'dist_to_ref_mean': float,
  'dist_to_ref_nearest': float,
  'knn_mean_distance': float,
  'knn_std_distance': float,    # ⭐ Top borderline predictor
  'knn_min_distance': float,
  'knn_max_distance': float,      # ⭐ Top overall predictor
  'local_curvature': float,
  'ridge_proximity': float,

  # Metadata (Artifact Discipline)
  '_schema_version': 'geometry_v2.0',
  '_ref_hash': str,               # SHA256 of reference set
  '_config_hash': str,            # Hash of (k, metric)
  'k_neighbors': int              # 50
}
```

**Removed in v2.0:**
- ❌ `dark_river_candidate` (boolean flag)
- ❌ `observer_mode` (boolean flag)

**Reason:** Dark River hypothesis falsified. Binary thresholds don't work for normalized embeddings.

---

### 3.2 Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-01-07 | Initial release with binary flags |
| **v2.0** | **2026-01-08** | **Removed binary flags, SVD curvature, continuous only** |

**Breaking Changes in v2.0:**
- `dark_river_candidate` removed
- `observer_mode` removed
- `local_curvature` computation changed (covariance → SVD)
- Schema version string changed

**Migration Path:**
- v1.0 outputs: Add warning, recommend recomputation
- v2.0 outputs: Production-ready

---

## 4. Validated Performance

### 4.1 Boundary-Sliced Evaluation Results

**Dataset:** OpenAI `text-embedding-3-large` on sentiment classification
**Model:** Ridge regression (α=1.0), 20 independent trials
**Metric:** R² (coefficient of determination)

| Zone | N | Baseline R² | Geometry R² | Improvement | Significance |
|------|---|-------------|-------------|-------------|--------------|
| **BORDERLINE** | 79 (35.9%) | 0.575 | 0.597 | **+3.8%** | p < 0.001 *** |
| UNSAFE | 74 (33.6%) | 0.680 | 0.694 | +2.1% | p < 0.001 *** |
| SAFE | 67 (30.5%) | 0.604 | 0.609 | +0.8% | p < 0.001 *** |

**Zone Definitions:**
- **BORDERLINE:** |boundary_distance| < 0.5 (high uncertainty)
- **UNSAFE:** boundary_distance < -0.5 (confident but wrong)
- **SAFE:** boundary_distance > 0.5 (confident and correct)

**Key Finding:** Geometry helps **4.8× more on borderline vs safe** (3.8% / 0.8% = 4.75)

**Evidence Files:**
- `experiments/boundary_sliced_evaluation.py`
- `runs/boundary_sliced_evaluation_20260108_224808.json`
- `docs/BOUNDARY_SLICED_RESULTS.md`

---

### 4.2 Feature Importance

**Consensus Top Feature:** `knn_std_distance` ⭐
- Appears in top-3 for: correlation, Random Forest importance, ablation loss
- **Strongest on borderline:** r=+0.399 (vs r=+0.286 overall)

**Top 3 Overall:**
1. `knn_max_distance` (r=+0.345, RF importance=0.233)
2. `knn_std_distance` (r=+0.286, RF importance=0.203)
3. `local_curvature` (r=-0.103, ablation loss=2.0%)

**Interpretation:**
- Variance of k-NN distances (std, max) most predictive
- Local curvature provides non-linear orthogonal information
- Mean distance less informative than spread

**Evidence:** `experiments/analyze_feature_importance.py`

---

### 4.3 Ablation Study

**Question:** Which features are essential?

**Method:** Train Ridge model with each feature removed, measure R² loss

| Feature Removed | R² Loss | % of Baseline | Essential? |
|-----------------|---------|---------------|------------|
| `local_curvature` | +0.0180 | 2.0% | ✓ YES |
| `knn_std_distance` | +0.0048 | 0.5% | ✓ YES |
| `knn_mean_distance` | +0.0043 | 0.5% | ✓ YES |
| `ridge_proximity` | +0.0012 | 0.1% | Marginal |
| Others | < 0.001 | < 0.1% | Redundant |

**Recommendation:** Keep all 7 features for now
- Small feature set (7 features)
- Minimal computational cost
- Non-linear interactions not tested

---

## 5. Falsified Hypotheses (v2.0)

### 5.1 Dark River Discrete Regions ❌

**Original Hypothesis (v1.0):**
> "Dark Rivers are discrete unstable regions near decision boundaries, identified by low curvature (< 0.5) AND high ridge (> 2.0)"

**Test Results:**
- Ridge proximity max value: **0.443** (threshold was 2.0!)
- Modern embeddings have universal smooth geometry (ridge ≈ 0.2)
- **0 detections** on real data using original thresholds

**Root Cause:**
- Hypothesis assumed high-variance neighborhoods near boundaries
- Normalized embeddings (OpenAI, Cohere, etc.) have uniform density
- σ/μ ≈ 0.2 everywhere → no "ridges" exist

**Evidence:** `experiments/investigate_dark_rivers.py`

**Conclusion:** ❌ **FALSIFIED** — Dark River discrete regions don't exist for normalized embeddings

---

### 5.2 Observer Mode ❌

**Original Hypothesis (v1.0):**
> "Observer mode: low curvature + close to reference centroid indicates uncertain but 'normal' regions"

**Test Results:**
- No meaningful correlation with boundary distance
- Detection rate: 0% on real data

**Conclusion:** ❌ **FALSIFIED** — Binary flag provides no value

---

### 5.3 What Actually Works ✓

**Revised Understanding (v2.0):**
- Geometry features work via **continuous correlations**, not discrete thresholds
- No single threshold separates "safe" from "unsafe"
- Features provide **graded signals** that ML models integrate

**Correct Framing:**
> "k-NN geometry provides continuous safety signals correlated with boundary proximity. Improvements concentrate in borderline regions where baseline embedding-only methods are uncertain."

---

## 6. Key Principles (Design Philosophy)

### 6.1 Continuous Features Only

**❌ DO NOT:**
- Apply binary thresholds to features
- Create "safe"/"unsafe" classifications from geometry alone
- Use fixed cutoffs (e.g., curvature < 0.5 = dangerous)

**✓ DO:**
- Treat all features as continuous
- Let ML models learn optimal combinations
- Use percentiles for visualization/monitoring only

**Rationale:** Binary thresholds fail for normalized embeddings. Continuous models capture nuanced relationships.

---

### 6.2 Percentiles for Monitoring Only

**Acceptable Use:**
- Visualization: "This query is at 95th percentile for ridge_proximity"
- Anomaly detection: "Feature distribution shifted from baseline"
- Monitoring: "Geometry Health Panel" dashboards

**Unacceptable Use:**
- Detection: "If ridge_proximity > p90, flag as unsafe"
- Binary classification: "Dark river if curvature < p10 AND ridge > p90"

**Why:** Percentiles describe the data distribution, they don't define safety boundaries.

---

### 6.3 Batch-Order Invariance Guarantee

**Property:** Reference-only computation ensures batch-order invariance

```python
# Guaranteed invariant:
results1 = bundle.compute(queries)
results2 = bundle.compute(permute(queries))
assert results2[inverse_permutation] == results1  # ✓ True
```

**Why This Matters:**
- Reproducibility: same query → same features regardless of batch
- Scalability: can process queries in any order
- Trust: no hidden dependencies between queries

**Implementation:**
- k-NN index built on reference set ONLY
- Queries NEVER added to index
- Each query processed independently

---

### 6.4 Artifact Discipline

**Every output includes reproducibility metadata:**

```python
{
  '_ref_hash': 'a1b2c3d4...',      # SHA256(reference_embeddings)
  '_config_hash': 'e5f6g7h8...',   # SHA256(f"k={k},metric={metric}")
  '_schema_version': 'geometry_v2.0',
  'k_neighbors': 50
}
```

**Why:**
- Detect when reference set changes
- Reproduce exact results
- Debug distribution shifts

---

## 7. Usage Guidelines

### 7.1 Recommended Workflow

```python
from mirrorfield.geometry import GeometryBundle

# 1. Initialize with reference set
reference_embeddings = load_reference_embeddings()  # (N_ref, D)
bundle = GeometryBundle(reference_embeddings, k=50)

# 2. Compute geometry for queries
query_embeddings = load_query_embeddings()  # (N_query, D)
results = bundle.compute(query_embeddings)

# 3. Extract feature matrix for ML
features = bundle.get_feature_matrix(results)  # (N_query, 7)

# 4. Combine with embeddings for training
X = np.concatenate([query_embeddings, features], axis=1)
y = boundary_distances

# 5. Train model
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X, y)
```

---

### 7.2 Batch Processing

For large datasets:

```python
results = bundle.compute_batch(
    query_embeddings,
    batch_size=100,
    boundary_distances=boundary_distances,  # Optional: for collapse check
    check_collapse=True  # Only checks first batch
)
```

---

### 7.3 Health Monitoring

Check computation quality:

```python
summary = bundle.summarize(results)

# Check for issues
if summary['computation_metadata']['collapse_check'] == 'RIDGE_COLLAPSE':
    print("⚠️  Warning: Ridge-boundary correlation > 0.9")
    print(f"   Correlation: {summary['computation_metadata']['ridge_boundary_correlation']:.3f}")
    print("   Geometry may be redundant with boundary distance")
```

---

## 8. Known Limitations

### 8.1 Embedder-Specific

**Current Validation:** OpenAI `text-embedding-3-large` only
**Limitation:** May not generalize to all embedders

**Next Step:** Test on:
- Sentence-transformers (e.g., `all-MiniLM-L6-v2`)
- Cohere `embed-v3`
- Other commercial embeddings

**Hypothesis:** Works best for normalized, high-quality embeddings

---

### 8.2 Reference Set Requirements

**Minimum Size:** k × 10 = 500 samples recommended
**Quality:** Reference must cover feature space adequately
**Stale Data:** Performance degrades if reference distribution drifts

**Recommendation:** Update reference set periodically with production data

---

### 8.3 Computational Cost

**Per-Query Cost:** O(N_ref × D) for k-NN search
**Scaling:** Use approximate k-NN (FAISS, Annoy) for large reference sets

**Current:** ~10ms for N_ref=1000, D=256 (brute force)
**At Scale:** ~1ms with FAISS index for N_ref=1M

---

## 9. Testing & Validation

### 9.1 Acceptance Criteria

Before deploying Phase E v2.0:

✓ All 6 acceptance tests pass (`experiments/test_phase_e_bundle.py`)
✓ Boundary-sliced evaluation shows borderline improvement > safe
✓ SVD curvature non-zero (mean > 0.01)
✓ Collapse check passes (ridge-boundary correlation < 0.9)
✓ Batch-order invariance guarantee verified

---

### 9.2 Regression Tests

Run these tests on any code changes:

```bash
# Full test suite
python experiments/test_phase_e_bundle.py

# Curvature fix validation
python experiments/validate_svd_curvature.py

# Boundary-sliced evaluation
python experiments/boundary_sliced_evaluation.py

# Feature importance
python experiments/analyze_feature_importance.py
```

Expected results:
- Phase E acceptance: 6/6 pass
- SVD validation: +3.8% improvement (20/20 trials)
- Boundary-sliced: borderline > safe improvement
- Feature importance: `knn_std_distance` in top-3

---

## 10. Publication-Ready Claims

### 10.1 Core Results

1. **"Geometry features provide 4.8× larger improvements on borderline cases compared to safe cases"**
   - Borderline: +3.8% (p < 0.001)
   - Safe: +0.8% (p < 0.001)
   - Ratio: 3.8 / 0.8 = 4.75

2. **"Improvements concentrate where baseline embedding-only methods perform worst"**
   - Borderline baseline R² = 0.575 (lowest)
   - Safe baseline R² = 0.604
   - Unsafe baseline R² = 0.680 (highest)

3. **"Consensus top feature: knn_std_distance (appears in all top-3 rankings)"**
   - Correlation: r=+0.399 on borderline
   - RF importance: 0.203
   - Ablation loss: 0.5%

4. **"Dark River discrete region hypothesis falsified for normalized embeddings"**
   - Ridge proximity max = 0.443 (threshold was 2.0)
   - Universal smooth geometry (σ/μ ≈ 0.2)
   - 0 detections on real data

---

### 10.2 Scientific Contributions

1. **Boundary-stratified evaluation methodology** validates targeted improvements
2. **SVD-based curvature** solves numerical instability for k << D
3. **Falsification of Dark River hypothesis** (honest negative result)
4. **Continuous correlation mechanism** replaces discrete threshold detection

---

## 11. Future Work

### 11.1 Immediate (Before Publication)

- [ ] Generate publication plots (3 figures)
- [ ] Write technical report
- [ ] Submit to arXiv

### 11.2 Near-Term (Phase F)

- [ ] End-to-end inference pipeline
- [ ] Geometry Health Panel (monitoring)
- [ ] Production deployment

### 11.3 Long-Term (Phase G)

- [ ] Generalization check (other embedders)
- [ ] Two-tier fast/slow path optimization
- [ ] Distributed inference scaling

---

## 12. References

### 12.1 Core Files

- **Schema:** `mirrorfield/geometry/schema.py`
- **Features:** `mirrorfield/geometry/features.py`
- **Bundle:** `mirrorfield/geometry/bundle.py`

### 12.2 Validation Experiments

- **Phase E Acceptance:** `experiments/test_phase_e_bundle.py`
- **SVD Curvature Fix:** `experiments/validate_svd_curvature.py`
- **Boundary-Sliced:** `experiments/boundary_sliced_evaluation.py`
- **Feature Importance:** `experiments/analyze_feature_importance.py`
- **Dark River Investigation:** `experiments/investigate_dark_rivers.py`

### 12.3 Results

- **Boundary-Sliced Results:** `docs/BOUNDARY_SLICED_RESULTS.md`
- **Summary:** `docs/BOUNDARY_SLICED_SUMMARY.txt`
- **Run Ledger:** `runs/RUN_LEDGER.md`

---

## 13. Change Control

### 13.1 Version Bumping Rules

**Patch (v2.0.X):**
- Bug fixes, documentation updates
- No breaking changes
- Backward compatible

**Minor (v2.X.0):**
- New optional features
- Deprecation warnings
- Backward compatible

**Major (vX.0.0):**
- Breaking schema changes
- Removed features
- Incompatible outputs

### 13.2 Modification Protocol

To modify this contract:

1. Create proposal document
2. Run full test suite on proposed changes
3. Validate on held-out data
4. Document breaking changes
5. Bump version appropriately
6. Update this document

---

## 14. Signatures

**Frozen By:** Claude Sonnet 4.5 + User
**Date:** 2026-01-08
**Validation Status:** ✓ Complete (6/6 acceptance tests, boundary-sliced eval)
**Schema Version:** `geometry_v2.0`

**This document is the definitive reference for Phase E v2.0. Do not modify without proper version control.**

---

**END OF CONTRACT**
