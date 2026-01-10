"""
Behavioral Flip Experiment - Step 1: Sample Selection

Selects 30 queries (10 per zone) for paraphrase robustness testing.
"""

import numpy as np
import json
from pathlib import Path

def select_sample_queries(
    embeddings,
    boundary_distances,
    texts,
    labels,
    n_per_zone=10,
    seed=42
):
    """
    Select 30 queries (10 per zone) for flip testing.

    Criteria:
    - Diverse boundary distances within each zone
    - Text length 50-200 characters
    - Clear semantic content

    Args:
        embeddings: (N, D) array
        boundary_distances: (N,) array
        texts: list of strings
        labels: (N,) array
        n_per_zone: samples per zone (default: 10)
        seed: random seed for reproducibility

    Returns:
        sample: dict with selected queries
    """
    np.random.seed(seed)

    # Define zones
    safe_mask = boundary_distances > 0.5
    borderline_mask = (boundary_distances >= -0.5) & (boundary_distances <= 0.5)
    unsafe_mask = boundary_distances < -0.5

    zones = {
        'safe': safe_mask,
        'borderline': borderline_mask,
        'unsafe': unsafe_mask
    }

    samples = {}

    for zone_name, mask in zones.items():
        # Get indices for this zone
        zone_indices = np.where(mask)[0]
        zone_bd = boundary_distances[mask]

        # Filter by text length (50-200 characters)
        valid_indices = []
        for idx in zone_indices:
            text = texts[idx]
            if 50 <= len(text) <= 200:
                valid_indices.append(idx)

        valid_indices = np.array(valid_indices)

        print(f"\n{zone_name.upper()} zone:")
        print(f"  Total samples: {len(zone_indices)}")
        print(f"  Valid length (50-200 chars): {len(valid_indices)}")

        if len(valid_indices) < n_per_zone:
            print(f"  WARNING: Only {len(valid_indices)} valid samples, need {n_per_zone}")
            selected_indices = valid_indices
        else:
            # Sort by boundary distance to get diverse samples
            valid_bd = boundary_distances[valid_indices]
            sorted_order = np.argsort(valid_bd)

            # Select evenly spaced samples across boundary distance range
            step = len(valid_indices) // n_per_zone
            selected_positions = [i * step for i in range(n_per_zone)]
            selected_indices = valid_indices[sorted_order[selected_positions]]

        # Store selected samples
        zone_samples = []
        for idx in selected_indices:
            zone_samples.append({
                'index': int(idx),
                'text': texts[idx],
                'label': int(labels[idx]),
                'boundary_distance': float(boundary_distances[idx]),
                'embedding': embeddings[idx].tolist(),
                'zone': zone_name
            })

        samples[zone_name] = zone_samples

        # Print summary
        bd_values = [s['boundary_distance'] for s in zone_samples]
        print(f"  Selected {len(zone_samples)} samples")
        print(f"  Boundary distance range: [{min(bd_values):.3f}, {max(bd_values):.3f}]")

    return samples


def main():
    # Load data
    data_path = Path("runs/openai_3_large_test_20251231_024532")

    print("Loading data...")
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")
    labels = np.load(data_path / "labels.npy")

    with open(data_path / "texts.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        texts = data['texts'] if isinstance(data, dict) else data

    print(f"Loaded {len(embeddings)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Select samples
    samples = select_sample_queries(
        embeddings,
        boundary_distances,
        texts,
        labels,
        n_per_zone=10,
        seed=42
    )

    # Save samples
    output_path = Path("experiments/behavioral_flip_samples.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f"\n✓ Saved {sum(len(s) for s in samples.values())} samples to {output_path}")

    # Print example from each zone
    print("\n" + "="*80)
    print("EXAMPLE SAMPLES:")
    print("="*80)

    for zone_name in ['safe', 'borderline', 'unsafe']:
        print(f"\n{zone_name.upper()} example:")
        example = samples[zone_name][0]
        print(f"  Text: {example['text']}")
        print(f"  Label: {example['label']} (0=negative, 1=positive)")
        print(f"  Boundary distance: {example['boundary_distance']:.3f}")


if __name__ == "__main__":
    main()
