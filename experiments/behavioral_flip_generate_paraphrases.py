"""
Behavioral Flip Experiment - Step 2: Paraphrase Generation

Generates 5 semantic-preserving paraphrases per query using GPT-4.
"""

import json
import os
from pathlib import Path
from openai import OpenAI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, assume env vars set manually

def generate_paraphrases(text, n=5, model="gpt-4", client=None):
    """
    Generate n paraphrases of text that preserve sentiment.

    Args:
        text: original text
        n: number of paraphrases (default: 5)
        model: OpenAI model to use
        client: OpenAI client instance

    Returns:
        list of paraphrases
    """
    prompt = f"""Generate {n} paraphrases of the following text that:
1. Preserve the original sentiment (positive/negative)
2. Use different words/sentence structure
3. Are natural and fluent
4. Have similar length (±20%)

Original text: "{text}"

Return only the {n} paraphrases, one per line, without numbering or additional text."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates semantic-preserving paraphrases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    # Parse response - split by newlines and filter empty lines
    paraphrases_text = response.choices[0].message.content
    paraphrases = [p.strip() for p in paraphrases_text.split('\n') if p.strip()]

    # Remove numbering if present (e.g., "1. ", "2. ")
    cleaned_paraphrases = []
    for p in paraphrases:
        # Remove leading numbers and punctuation
        cleaned = p.lstrip('0123456789. ').strip()
        if cleaned:
            cleaned_paraphrases.append(cleaned)

    return cleaned_paraphrases[:n]  # Ensure we return exactly n


def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return

    client = OpenAI(api_key=api_key)

    # Load samples
    samples_path = Path("experiments/behavioral_flip_samples.json")
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"Loaded {sum(len(s) for s in samples.values())} samples")
    print(f"Estimated cost: ~$0.90 (30 queries × $0.03)")
    print()

    # Generate paraphrases for each query
    total_queries = sum(len(s) for s in samples.values())
    current_query = 0

    for zone_name, zone_samples in samples.items():
        print(f"\nProcessing {zone_name.upper()} zone ({len(zone_samples)} queries):")

        for i, sample in enumerate(zone_samples):
            current_query += 1
            text = sample['text']

            print(f"  [{current_query}/{total_queries}] Generating paraphrases...")

            try:
                paraphrases = generate_paraphrases(text, n=5, client=client)

                # Store paraphrases
                sample['paraphrases'] = paraphrases

                # Print first paraphrase as preview
                if paraphrases:
                    print(f"      Original: {text}")
                    print(f"      Example:  {paraphrases[0]}")
                else:
                    print(f"      WARNING: No paraphrases generated")

            except Exception as e:
                print(f"      ERROR: {e}")
                sample['paraphrases'] = []

    # Save updated samples with paraphrases
    output_path = Path("experiments/behavioral_flip_samples_with_paraphrases.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f"\n✓ Saved samples with paraphrases to {output_path}")

    # Summary
    total_paraphrases = sum(
        len(sample.get('paraphrases', []))
        for zone_samples in samples.values()
        for sample in zone_samples
    )
    print(f"\nSummary:")
    print(f"  Original queries: {total_queries}")
    print(f"  Total paraphrases: {total_paraphrases}")
    print(f"  Expected: {total_queries * 5}")


if __name__ == "__main__":
    main()
