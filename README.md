# Geometric Safety Features for AI Boundary Detection

**Boundary-Stratified Evaluation of k-NN Geometric Features**

[![Status](https://img.shields.io/badge/status-publication--ready-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## Overview

This repository contains the implementation and experimental code for our paper on using k-NN geometric features to detect high-uncertainty regions in AI embedding spaces, with applications to AI safety.

**Key Finding:** Geometric features provide **4.8× larger improvements** on borderline (high-uncertainty) cases compared to safe cases, demonstrating targeted value where baseline methods struggle most.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run boundary-stratified evaluation
python experiments/boundary_sliced_evaluation.py

# Generate publication figures
python experiments/generate_publication_plots_v2.py
```

---

## Paper Abstract

AI models exhibit unpredictable failure modes near decision boundaries, where uncertainty is highest and safety risks concentrate. We propose seven k-NN geometric features and introduce boundary-stratified evaluation to assess performance separately on safe, borderline, and unsafe regions.

Testing on sentiment classification (N=1099), geometric features provide +3.8% R² improvement on borderline cases vs +0.8% on safe cases. The consensus top feature, `knn_std_distance`, shows amplified correlation on borderline cases (r=+0.399 vs r=+0.286 overall). A supplementary behavioral flip experiment validates that geometry predicts robustness under paraphrasing (AUC=0.707, 23% better than boundary distance alone).

**Full paper:** [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)

---

## Repository Structure

```
mirrorfield_publication/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── docs/
│   ├── TECHNICAL_REPORT.md           # Full technical paper (~13,500 words)
│   ├── PHASE_E_CONTRACT_v2.0.md      # Frozen feature specification
│   └── BEHAVIORAL_FLIP_REPORT.md     # Supplementary experiment details
├── mirrorfield/
│   └── geometry/
│       ├── __init__.py
│       ├── bundle.py                  # GeometryBundle (main API)
│       ├── features.py                # 7 k-NN geometric features
│       └── schema.py                  # Schema v2.0 specification
├── experiments/
│   ├── boundary_sliced_evaluation.py  # Main experiment (Section 3.1)
│   ├── analyze_feature_importance.py  # Feature analysis (Section 3.2)
│   ├── generate_publication_plots_v2.py # Figure generation
│   ├── behavioral_flip_*.py           # Supplementary experiment (5 scripts)
│   └── test_phase_e_bundle.py         # Acceptance tests
├── plots/
│   ├── figure1_r2_by_region.png       # Main result visualization
│   ├── figure2_feature_importance.png # Feature correlations
│   └── figure3_ablation_study.png     # Ablation testing
└── data/
    └── README.md                       # Data availability statement
```

---

## Main Results

### Boundary-Stratified Performance

| Zone | N | Baseline R² | Geometry R² | Improvement | Significance |
|------|---|-------------|-------------|-------------|--------------|
| **BORDERLINE** | 79 | 0.575 | 0.597 | **+3.8%** | p < 0.001 *** |
| UNSAFE | 74 | 0.680 | 0.694 | +2.1% | p < 0.001 *** |
| SAFE | 67 | 0.604 | 0.609 | +0.8% | p < 0.001 *** |

**Key Finding:** Geometry provides **4.8× larger improvement** on borderline vs safe cases.

### Behavioral Flip Validation

| Model | AUC | Improvement |
|-------|-----|-------------|
| Boundary Distance Only | 0.574 | — |
| **Geometry Features** | **0.707** | **+23%** |

**Key Finding:** Geometry predicts robustness under paraphrasing significantly better than model uncertainty alone.

### Consensus Top Feature

**`knn_std_distance` (neighborhood standard deviation):**
- Overall correlation: r = +0.286***
- Borderline correlation: r = +0.399*** (amplified)
- Behavioral flip: r_pb = +0.168* (p = 0.040)
- Top-3 across correlation, Random Forest, and ablation

---

## 7 Geometric Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `knn_mean_distance` | Average distance to k=50 nearest neighbors |
| 2 | `knn_std_distance` ⭐ | Std deviation of neighbor distances (top feature) |
| 3 | `knn_min_distance` | Distance to nearest neighbor |
| 4 | `knn_max_distance` | Distance to farthest of k neighbors |
| 5 | `local_curvature` | Manifold anisotropy via SVD (σ_min/σ_max) |
| 6 | `ridge_proximity` | Coefficient of variation of neighborhood distances (σ/μ) |
| 7 | `dist_to_ref_nearest` | Distance to nearest reference point |

---

## Usage Example

```python
from mirrorfield.geometry import GeometryBundle
import numpy as np

# Load your embeddings
reference_embeddings = np.load("reference.npy")  # (N_ref, D)
query_embeddings = np.load("query.npy")          # (N_query, D)

# Initialize geometry bundle
bundle = GeometryBundle(reference_embeddings, k=50)

# Compute geometric features
results = bundle.compute(query_embeddings)
features = bundle.get_feature_matrix(results)  # (N_query, 7)

# Features: [knn_mean, knn_std, knn_min, knn_max, curvature, ridge, 1nn]
print(f"Geometric features shape: {features.shape}")
```

---

## Reproducing Results

### Main Experiment (Boundary-Stratified Evaluation)

```bash
# Run evaluation with embedded data
python experiments/boundary_sliced_evaluation.py

# Outputs:
# - runs/boundary_sliced_evaluation_TIMESTAMP/summary.json
# - Prints zone-stratified R² improvements
```

### Generate Figures

```bash
# Creates figures 1-3 in plots/ directory
python experiments/generate_publication_plots_v2.py

# Outputs:
# - plots/figure1_r2_by_region.png (and .pdf)
# - plots/figure2_feature_importance.png (and .pdf)
# - plots/figure3_ablation_study.png (and .pdf)
```

### Behavioral Flip Experiment

```bash
# Full pipeline (requires OpenAI API key)
export OPENAI_API_KEY="your-key"

# 1. Select samples
python experiments/behavioral_flip_sample_selection.py

# 2. Generate paraphrases (cost: ~$0.90)
python experiments/behavioral_flip_generate_paraphrases.py

# 3. Compute predictions and flip rates
python experiments/behavioral_flip_compute_flips.py

# 4. Analyze correlations
python experiments/behavioral_flip_analyze.py

# 5. Paraphrase-level analysis
python experiments/behavioral_flip_paraphrase_level_analysis.py
```

---

## Installation

### Requirements

- Python 3.9+
- NumPy 1.24+
- SciPy 1.10+
- scikit-learn 1.3+
- matplotlib 3.7+ (for plotting)
- OpenAI API (for behavioral flip experiment only)

### Install

```bash
# Clone repository
git clone https://github.com/DillanJC/mirrorfield_publication.git
cd mirrorfield_publication

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Data Availability

The sentiment classification dataset used in our experiments is available upon request. For reproducibility:

- **Dataset:** Sentiment classification (N=1099)
- **Embedder:** OpenAI `text-embedding-3-large` (D=256)
- **Data files:** `embeddings.npy`, `labels.npy`, `boundary_distances.npy`, `texts.json`
- **Reference split:** 80% (879 samples)
- **Query split:** 20% (220 samples)

See [`data/README.md`](data/README.md) for access instructions.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{coghlan2026geometric,
  title={Boundary-Stratified Evaluation of k-NN Geometric Features for AI Safety Detection},
  author={Dillan John Coghlan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## Key Contributions

1. **Boundary-stratified evaluation methodology** revealing targeted improvements in high-uncertainty regions

2. **7 k-NN geometric features** with validated +3.8% improvement on borderline cases (4.8× larger than safe)

3. **SVD-based curvature computation** solving numerical instability for k << D

4. **Falsification of discrete region hypothesis** for normalized embeddings, replacing with continuous correlation mechanism

5. **Behavioral flip validation** demonstrating geometry predicts robustness under paraphrasing (AUC=0.707)

6. **Production-ready implementation** with frozen schema v2.0 and comprehensive testing

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions or collaboration:
- **Author:** Dillan John Coghlan
- **Email:** DillanJC91@Gmail.com
- **GitHub Issues:** [github.com/DillanJC/mirrorfield_publication/issues]

---

## Acknowledgments

This research was conducted using OpenAI's `text-embedding-3-large` model and GPT-4 for paraphrase generation (behavioral flip experiment). Total experimental cost: ~$0.90.

---

**Last Updated:** January 2026
