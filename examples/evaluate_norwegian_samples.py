import json
from medical_embedding_eval import MedicalSample, SampleVariation

"""Complete evaluation example using Norwegian medical samples.

This example demonstrates a complete workflow:
1. Load Norwegian medical samples
2. Evaluate with a dummy embedder (replace with your own model)
3. Generate comprehensive evaluation report
"""

from medical_embedding_eval import (
    EmbeddingEvaluator,
    DummyEmbedder,
    SamplePair,
)
from norwegian_medical_examples import (
    get_norwegian_medical_samples,
    get_norwegian_medical_variations,
)


def main():
    """Run complete evaluation on Norwegian medical samples."""
    print("=" * 80)
    print("Norwegian Medical Embedding Evaluation")
    print("=" * 80)
    print()
    
    # Step 1: Load samples and variations
    # print("Loading Norwegian medical samples...")
    # samples = get_norwegian_medical_samples()
    # variations = get_norwegian_medical_variations()
    # print(f"Loaded {len(samples)} samples with {len(variations)} variations")
    # print()



    # Load from JSON
    with open("data/general_somatic.json", encoding="utf-8") as fh:
        data = json.load(fh)

    samples = [
        MedicalSample(
            text=item["text"],
            sample_id=item["sample_id"],
            metadata=item.get("metadata", {}),
        )
        for item in data["samples"]
    ]

    sample_lookup = {sample.sample_id: sample for sample in samples}

    variations = []
    for item in data["variations"]:
        original = sample_lookup[item["original_id"]]
        variations.append(
            SampleVariation(
                original_sample=original,
                variation_text=item["variation_text"],
                variation_type=item["variation_type"],
                variation_id=item["variation_id"],
                changes_applied=item.get("changes_applied", []),
                metadata=item.get("metadata", {}),
            )
        )
    
    # Step 2: Create sample pairs
    print("Creating sample pairs...")
    # Note: Variations already reference their original samples
    pairs = [
        SamplePair(original=variation.original_sample, variation=variation)
        for variation in variations
    ]
    print(f"Created {len(pairs)} pairs for evaluation")
    print()
    
    # Step 3: Initialize embedding model
    # NOTE: Replace DummyEmbedder with your actual embedding model
    # Examples:
    #   - Sentence-BERT: from sentence_transformers import SentenceTransformer
    #   - OpenAI: from openai import OpenAI
    #   - Custom model: Implement EmbeddingModel interface
    print("Initializing embedding model...")
    print("NOTE: Using DummyEmbedder for demonstration.")
    print("Replace with your actual embedding model for real evaluation.")
    embedder = DummyEmbedder(embedding_dim=384, seed=42)
    print(f"Model: {embedder.get_model_name()}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    print()
    
    # Step 4: Run evaluation
    print("Running evaluation...")
    evaluator = EmbeddingEvaluator(embedder)
    metrics = evaluator.evaluate_pairs(pairs)
    print("Evaluation complete!")
    print()
    
    # Step 5: Display comprehensive results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(metrics)
    print()
    
    # Step 6: Detailed analysis by variation type
    print("=" * 80)
    print("DETAILED ANALYSIS BY VARIATION TYPE")
    print("=" * 80)
    print()
    
    variation_types = set(r.variation_type for r in metrics.results)
    for vtype in sorted(variation_types):
        type_results = [r for r in metrics.results if r.variation_type == vtype]
        print(f"\nVariation Type: {vtype}")
        print(f"Number of samples: {len(type_results)}")
        print(f"Mean similarity: {metrics.similarity_by_type[vtype]:.4f}")
        print("\nExamples:")
        for i, result in enumerate(type_results[:2], 1):  # Show first 2 examples
            print(f"\n  Example {i}:")
            print(f"    Original:  {result.original_text[:60]}...")
            print(f"    Variation: {result.variation_text[:60]}...")
            print(f"    Similarity: {result.cosine_similarity:.4f}")
        if len(type_results) > 2:
            print(f"\n  ... and {len(type_results) - 2} more")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("1. Replace DummyEmbedder with your actual embedding model")
    print("2. Add more Norwegian medical samples for comprehensive evaluation")
    print("3. Consider evaluating multiple models and comparing results")
    print("4. Use the metrics to guide model selection for your specific use case")
    print()
    print("For model comparison, use:")
    print("  comparison = evaluator1.compare_models(evaluator2, pairs)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
