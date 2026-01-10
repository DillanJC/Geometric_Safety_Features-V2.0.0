"""
Behavioral Flip Experiment - Step 3: Compute Predictions and Flip Rates

1. Embeds all texts (original + paraphrases) using OpenAI
2. Trains sentiment classifier on reference set
3. Predicts on all texts
4. Computes flip rates
5. Computes geometric features for originals
"""

import numpy as np
import json
from pathlib import Path
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def embed_texts(texts, client, model="text-embedding-3-large", dimensions=256):
    """
    Embed texts using OpenAI API.

    Args:
        texts: list of strings
        client: OpenAI client
        model: embedding model
        dimensions: embedding dimension

    Returns:
        embeddings: (N, D) array
    """
    print(f"  Embedding {len(texts)} texts...")

    # Batch embed (OpenAI supports up to 2048 texts per request)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch,
            dimensions=dimensions
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        if (i // batch_size + 1) % 5 == 0:
            print(f"    Embedded {i+len(batch)}/{len(texts)}...")

    embeddings = np.array(all_embeddings)
    print(f"  ✓ Embedded {len(embeddings)} texts (shape: {embeddings.shape})")

    return embeddings


def compute_boundary_distance(y_true, y_pred_proba):
    """
    Compute signed boundary distance.

    Args:
        y_true: (N,) ground truth labels {0, 1}
        y_pred_proba: (N,) predicted probabilities [0, 1]

    Returns:
        boundary_distance: (N,) signed distances
    """
    # Predicted label
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Correct/incorrect
    correct = (y_pred == y_true)

    # Signed distance
    distance = 2 * (y_pred_proba - 0.5)
    distance = np.where(correct, distance, -distance)

    return distance


def compute_flip_rate(original_label, paraphrase_labels):
    """
    Compute flip rate: fraction of paraphrases with different prediction.

    Args:
        original_label: int (0 or 1)
        paraphrase_labels: list of ints

    Returns:
        flip_rate: float [0, 1]
        n_flips: int
    """
    flips = [p_label != original_label for p_label in paraphrase_labels]
    flip_rate = sum(flips) / len(flips) if flips else 0.0
    n_flips = sum(flips)

    return flip_rate, n_flips


def main():
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    # Load samples with paraphrases
    samples_path = Path("experiments/behavioral_flip_samples_with_paraphrases.json")
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"Loaded {sum(len(s) for s in samples.values())} samples with paraphrases\n")

    # Load reference data for training classifier
    print("Loading reference data...")
    data_path = Path("runs/openai_3_large_test_20251231_024532")
    reference_embeddings = np.load(data_path / "embeddings.npy")
    reference_labels = np.load(data_path / "labels.npy")
    print(f"  Reference set: {len(reference_embeddings)} samples\n")

    # Collect all texts to embed (original + paraphrases)
    all_texts = []
    text_indices = {}  # Map text to (zone, sample_idx, type, para_idx)

    for zone_name, zone_samples in samples.items():
        for sample_idx, sample in enumerate(zone_samples):
            # Original text
            original_text = sample['text']
            all_texts.append(original_text)
            text_indices[len(all_texts) - 1] = (zone_name, sample_idx, 'original', None)

            # Paraphrases
            for para_idx, para_text in enumerate(sample.get('paraphrases', [])):
                all_texts.append(para_text)
                text_indices[len(all_texts) - 1] = (zone_name, sample_idx, 'paraphrase', para_idx)

    print(f"Total texts to embed: {len(all_texts)}")
    print(f"  Original: 30")
    print(f"  Paraphrases: {len(all_texts) - 30}\n")

    # Embed all texts
    print("Step 1: Embedding all texts...")
    print(f"  Estimated cost: ~${len(all_texts) * 0.00002:.3f}")
    all_embeddings = embed_texts(all_texts, client)

    print("\nStep 2: Training sentiment classifier...")
    # Train logistic regression on reference set
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(reference_embeddings, reference_labels)

    # Evaluate on reference set
    ref_pred_proba = classifier.predict_proba(reference_embeddings)[:, 1]
    ref_pred = (ref_pred_proba > 0.5).astype(int)
    ref_accuracy = accuracy_score(reference_labels, ref_pred)
    print(f"  Classifier accuracy on reference: {ref_accuracy:.3f}")

    print("\nStep 3: Computing predictions for all texts...")
    pred_proba = classifier.predict_proba(all_embeddings)[:, 1]
    pred_labels = (pred_proba > 0.5).astype(int)

    # Assign predictions back to samples
    for text_idx, (zone_name, sample_idx, text_type, para_idx) in text_indices.items():
        sample = samples[zone_name][sample_idx]

        if text_type == 'original':
            sample['original_pred'] = {
                'probability': float(pred_proba[text_idx]),
                'label': int(pred_labels[text_idx]),
                'embedding': all_embeddings[text_idx].tolist()
            }
        else:  # paraphrase
            if 'paraphrase_preds' not in sample:
                sample['paraphrase_preds'] = []
            sample['paraphrase_preds'].append({
                'probability': float(pred_proba[text_idx]),
                'label': int(pred_labels[text_idx]),
                'paraphrase_idx': para_idx,
                'text': all_texts[text_idx]
            })

    print("\nStep 4: Computing flip rates...")
    flip_stats = {
        'safe': [],
        'borderline': [],
        'unsafe': []
    }

    for zone_name, zone_samples in samples.items():
        for sample in zone_samples:
            original_label = sample['original_pred']['label']
            paraphrase_labels = [p['label'] for p in sample['paraphrase_preds']]

            flip_rate, n_flips = compute_flip_rate(original_label, paraphrase_labels)

            sample['flip_rate'] = flip_rate
            sample['n_flips'] = n_flips

            flip_stats[zone_name].append(flip_rate)

    # Print flip rate statistics
    print("\nFlip Rate Statistics:")
    print("="*60)
    for zone_name in ['safe', 'borderline', 'unsafe']:
        rates = flip_stats[zone_name]
        print(f"{zone_name.upper():12} | Mean: {np.mean(rates):.3f} | Std: {np.std(rates):.3f} | Max: {np.max(rates):.3f}")

    # Save results
    output_path = Path("experiments/behavioral_flip_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f"\n✓ Saved results to {output_path}")

    # Print examples
    print("\n" + "="*80)
    print("EXAMPLE FLIP CASES:")
    print("="*80)

    for zone_name in ['safe', 'borderline', 'unsafe']:
        # Find sample with highest flip rate
        zone_samples = samples[zone_name]
        max_flip_sample = max(zone_samples, key=lambda s: s['flip_rate'])

        if max_flip_sample['flip_rate'] > 0:
            print(f"\n{zone_name.upper()} - Highest flip rate ({max_flip_sample['flip_rate']:.1%}):")
            print(f"  Original: {max_flip_sample['text']}")
            print(f"  Original pred: {max_flip_sample['original_pred']['label']} (p={max_flip_sample['original_pred']['probability']:.3f})")
            print(f"  Flipped paraphrases:")
            orig_label = max_flip_sample['original_pred']['label']
            for p in max_flip_sample['paraphrase_preds']:
                if p['label'] != orig_label:
                    print(f"    - \"{p['text']}\" → {p['label']} (p={p['probability']:.3f})")


if __name__ == "__main__":
    main()
