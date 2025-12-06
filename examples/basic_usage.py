"""Basic usage example of the medical embedding evaluation framework.

This example demonstrates how to:
1. Create medical samples and variations
2. Evaluate embedding models
3. Compare results
"""

from medical_embedding_eval import (
    MedicalSample,
    SampleVariation,
    SamplePair,
    EmbeddingEvaluator,
    DummyEmbedder,
)


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("Medical Embedding Evaluation - Basic Usage Example")
    print("=" * 60)
    print()
    
    # Step 1: Create medical samples (Norwegian examples)
    print("Step 1: Creating medical samples...")
    samples = [
        MedicalSample(
            text="Pasienten har høyt blodtrykk og diabetes type 2.",
            sample_id="sample_001",
            metadata={"language": "norwegian", "domain": "cardiology"}
        ),
        MedicalSample(
            text="Undersøkelsen viser symptomer på lungebetennelse.",
            sample_id="sample_002",
            metadata={"language": "norwegian", "domain": "pulmonology"}
        ),
        MedicalSample(
            text="Pasienten rapporterer sterke smerter i høyre kne.",
            sample_id="sample_003",
            metadata={"language": "norwegian", "domain": "orthopedics"}
        ),
    ]
    print(f"Created {len(samples)} medical samples")
    print()
    
    # Step 2: Create semantic variations
    print("Step 2: Creating semantic variations...")
    variations = [
        SampleVariation(
            original_sample=samples[0],
            variation_text="Personen lider av forhøyet blodtrykksmåling og sukkersyke type 2.",
            variation_type="paraphrase",
            variation_id="var_001",
            changes_applied=[
                "Replaced 'pasienten' with 'personen'",
                "Replaced 'høyt blodtrykk' with 'forhøyet blodtrykksmåling'",
                "Replaced 'diabetes' with 'sukkersyke'"
            ]
        ),
        SampleVariation(
            original_sample=samples[1],
            variation_text="Kontrollen indikerer tegn på pneumoni.",
            variation_type="medical_term_variation",
            variation_id="var_002",
            changes_applied=[
                "Replaced 'undersøkelsen' with 'kontrollen'",
                "Replaced 'symptomer' with 'tegn'",
                "Replaced 'lungebetennelse' with 'pneumoni'"
            ]
        ),
        SampleVariation(
            original_sample=samples[2],
            variation_text="Personen melder om kraftige plager i det høyre kneet.",
            variation_type="synonym_replacement",
            variation_id="var_003",
            changes_applied=[
                "Replaced 'pasienten' with 'personen'",
                "Replaced 'rapporterer' with 'melder om'",
                "Replaced 'sterke smerter' with 'kraftige plager'",
                "Replaced 'kne' with 'kneet'"
            ]
        ),
    ]
    print(f"Created {len(variations)} variations")
    print()
    
    # Step 3: Create sample pairs
    print("Step 3: Creating sample pairs...")
    pairs = [
        SamplePair(original=samples[i], variation=variations[i])
        for i in range(len(samples))
    ]
    print(f"Created {len(pairs)} sample pairs")
    print()
    
    # Step 4: Initialize embedding model (using dummy for demonstration)
    print("Step 4: Initializing embedding model...")
    embedder = DummyEmbedder(embedding_dim=384, seed=42)
    print(f"Initialized: {embedder.get_model_name()}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    print()
    
    # Step 5: Create evaluator and run evaluation
    print("Step 5: Running evaluation...")
    evaluator = EmbeddingEvaluator(embedder)
    metrics = evaluator.evaluate_pairs(pairs)
    print()
    
    # Step 6: Display results
    print("Step 6: Evaluation Results")
    print("-" * 60)
    print(metrics)
    print()
    
    # Step 7: Show individual results
    print("Step 7: Individual Sample Results")
    print("-" * 60)
    for i, result in enumerate(metrics.results, 1):
        print(f"\nSample {i}:")
        print(f"  Original: {result.original_text}")
        print(f"  Variation: {result.variation_text}")
        print(f"  Type: {result.variation_type}")
        print(f"  Cosine Similarity: {result.cosine_similarity:.4f}")
    print()
    
    # Step 8: Batch evaluation example
    print("Step 8: Batch Evaluation Example")
    print("-" * 60)
    originals = [s.text for s in samples]
    variation_texts = [v.variation_text for v in variations]
    batch_similarities = evaluator.batch_evaluate(originals, variation_texts)
    print("Batch similarities:", [f"{s:.4f}" for s in batch_similarities])
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
