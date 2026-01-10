# Data Availability

## Dataset Description

The experiments in this repository use a sentiment classification dataset with the following properties:

- **Task:** Binary sentiment classification (positive/negative)
- **Total samples:** 1,099
- **Train/test split:** 80/20 (879 reference, 220 query)
- **Embedder:** OpenAI `text-embedding-3-large`
- **Embedding dimension:** 256
- **Normalization:** All embeddings have unit L2 norm (||e|| = 1.0)

## Data Files

The dataset consists of:

1. **`embeddings.npy`** - (1099, 256) array of embeddings
2. **`labels.npy`** - (1099,) array of binary labels (0=negative, 1=positive)
3. **`boundary_distances.npy`** - (1099,) array of signed boundary distances
4. **`texts.json`** - JSON file with original text strings

## Boundary Distance

Boundary distance is computed as:

```
boundary_distance(y_true, p_pred) = {
    +2(p - 0.5)  if prediction correct
    -2(p - 0.5)  if prediction wrong
}
```

Where `p` is the model's predicted probability for the positive class.

Range: [-2, +2]
- Positive values = correct predictions
- Negative values = errors
- Magnitude = confidence distance from decision boundary

## Zone Definitions

Samples are stratified into three zones:

| Zone | Criterion | N (query set) | Description |
|------|-----------|---------------|-------------|
| **SAFE** | boundary_distance > 0.5 | 67 (30.5%) | Confident correct predictions |
| **BORDERLINE** | \|boundary_distance\| < 0.5 | 79 (35.9%) | High-uncertainty region |
| **UNSAFE** | boundary_distance < -0.5 | 74 (33.6%) | Confident errors |

## Accessing the Data

The data is embedded directly in the experiment scripts for reproducibility:

- **`experiments/generate_publication_plots_v2.py`** - Contains aggregated statistics
- **`experiments/boundary_sliced_evaluation.py`** - Loads from `runs/` directory

For access to the raw dataset:

1. **Academic researchers:** Contact the authors with your institutional email
2. **Industry researchers:** Contact the authors with use case description
3. **Reproducibility:** Run experiments using provided scripts with embedded data

## Privacy & Ethics

- All text data is from publicly available sentiment datasets
- No personally identifiable information (PII) included
- No sensitive or harmful content filtering required
- Approved for research use under standard academic data sharing agreements

## Citation

If you use this dataset, please cite our paper:

```bibtex
@article{coghlan2026geometric,
  title={Boundary-Stratified Evaluation of k-NN Geometric Features for AI Safety Detection},
  author={Dillan John Coghlan},
  year={2026}
}
```

## Contact

For data access requests:
- **Email:** DillanJC91@Gmail.com
- **GitHub Issues:** [github.com/DillanJC/mirrorfield_publication/issues]

---

**Note:** The embedded data in `generate_publication_plots_v2.py` is sufficient to reproduce all figures and main results without requiring the full dataset.
