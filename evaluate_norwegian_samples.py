"""Run a full evaluation on Norwegian medical samples defined in data files."""

import json
from pathlib import Path
from typing import List, Tuple

from medical_embedding_eval import (
    EmbeddingEvaluator,
    DummyEmbedder,
    MedicalSample,
    SamplePair,
    SampleVariation,
)


def load_samples(base_path: Path) -> Tuple[List[MedicalSample], List[SampleVariation]]:
    """Load samples and variations from a JSON definition file."""
    with base_path.open(encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{base_path} is empty or invalid JSON") from exc

    samples = [
        MedicalSample(
            text=item["text"],
            sample_id=item["sample_id"],
            metadata=item.get("metadata", {}),
        )
        for item in payload.get("samples", [])
    ]

    sample_lookup = {sample.sample_id: sample for sample in samples}

    variations: List[SampleVariation] = []
    for item in payload.get("variations", []):
        original_id = item["original_id"]
        if original_id not in sample_lookup:
            raise ValueError(f"Unknown original sample id: {original_id}")
        variations.append(
            SampleVariation(
                original_sample=sample_lookup[original_id],
                variation_text=item["variation_text"],
                variation_type=item["variation_type"],
                variation_id=item["variation_id"],
                changes_applied=item.get("changes_applied", []),
                metadata=item.get("metadata", {}),
            )
        )

    return samples, variations


def main() -> None:
    """Evaluate Norwegian medical samples using the dummy embedder."""
    print("=" * 80)
    print("Norwegian Medical Embedding Evaluation")
    print("=" * 80)
    print()

    data_path = Path("data/general_somatic.json").resolve()
    print(f"Loading samples from {data_path}...")
    samples, variations = load_samples(data_path)
    print(f"Loaded {len(samples)} samples and {len(variations)} variations")
    print()

    print("Creating sample pairs...")
    pairs = [SamplePair(original=variation.original_sample, variation=variation) for variation in variations]
    print(f"Created {len(pairs)} pairs for evaluation")
    print()

    print("Initializing embedding model...")
    embedder = DummyEmbedder(embedding_dim=384, seed=42)
    print(f"Model: {embedder.get_model_name()}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    print()

    print("Running evaluation...")
    evaluator = EmbeddingEvaluator(embedder)
    metrics = evaluator.evaluate_pairs(pairs)
    print("Evaluation complete!")
    print()

    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(metrics)
    print()

    print("=" * 80)
    print("DETAILED ANALYSIS BY VARIATION TYPE")
    print("=" * 80)
    print()

    variation_types = {result.variation_type for result in metrics.results}
    for variation_type in sorted(variation_types):
        type_results = [result for result in metrics.results if result.variation_type == variation_type]
        print(f"\nVariation Type: {variation_type}")
        print(f"Number of samples: {len(type_results)}")
        print(f"Mean similarity: {metrics.similarity_by_type[variation_type]:.4f}")
        print("\nExamples:")
        for idx, result in enumerate(type_results[:2], start=1):
            print(f"\n  Example {idx}:")
            print(f"    Original:  {result.original_text[:60]}...")
            print(f"    Variation: {result.variation_text[:60]}...")
            print(f"    Similarity: {result.cosine_similarity:.4f}")
        if len(type_results) > 2:
            print(f"\n  ... and {len(type_results) - 2} more")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Replace DummyEmbedder with your actual embedding model")
    print("2. Expand data/general_somatic.json with richer sample sets")
    print("3. Compare multiple models with evaluator.compare_models")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
