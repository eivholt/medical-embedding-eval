"""Evaluate Norwegian medical samples using cached embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from medical_embedding_eval import (
    DEFAULT_AZURE_EMBEDDING_CONFIGS,
    EvaluationMetrics,
    SimilarityMetrics,
    SimilarityResult,
    CachedEmbedding,
    compute_text_hash,
    load_samples_from_json,
    resolve_deployment_name,
)
from medical_embedding_eval.embedding_cache import EmbeddingCache

load_dotenv()

DATA_PATH = Path("data/general_somatic.json")
CACHE_DIR = Path("data/embeddings")


def display_results(model_name: str, metrics: EvaluationMetrics) -> None:
    print("=" * 80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 80)
    print()
    print(metrics)
    print()

    print("=" * 80)
    print(f"DETAILED ANALYSIS BY VARIATION TYPE: {model_name}")
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


def evaluate_with_cache(
    model_name: str,
    deployment_name: str,
    cache: EmbeddingCache,
    records: Dict[str, CachedEmbedding],
    variations,
) -> EvaluationMetrics:
    results = []
    for variation in variations:
        sample = variation.original_sample
        sample_key = cache.record_key("sample", sample.sample_id)
        variation_key = cache.record_key("variation", variation.variation_id)

        sample_record = records.get(sample_key)
        variation_record = records.get(variation_key)
        if sample_record is None or variation_record is None:
            raise KeyError(
                f"Missing cached embedding for {sample_key if sample_record is None else variation_key} in {deployment_name}"
            )

        if sample_record.text_hash != compute_text_hash(sample.text):
            raise ValueError(
                f"Sample {sample.sample_id} text changed since embeddings were cached for {deployment_name}."
            )
        if variation_record.text_hash != compute_text_hash(variation.variation_text):
            raise ValueError(
                f"Variation {variation.variation_id} text changed since embeddings were cached for {deployment_name}."
            )

        similarity = SimilarityMetrics.cosine_similarity(
            sample_record.embedding,
            variation_record.embedding,
        )

        results.append(
            SimilarityResult(
                original_text=sample_record.text,
                variation_text=variation_record.text,
                cosine_similarity=similarity,
                sample_id=sample.sample_id,
                variation_id=variation.variation_id,
                variation_type=variation.variation_type,
            )
        )

    metrics = SimilarityMetrics.compute_evaluation_metrics(results, model_name=model_name)
    return metrics


def main() -> None:
    print("=" * 80)
    print("Norwegian Medical Embedding Evaluation")
    print("=" * 80)
    print()

    data_path = DATA_PATH.resolve()
    print(f"Loading samples from {data_path}...")
    samples, variations = load_samples_from_json(data_path)
    print(f"Loaded {len(samples)} samples and {len(variations)} variations")
    print()

    cache = EmbeddingCache(CACHE_DIR)
    any_success = False

    for config in DEFAULT_AZURE_EMBEDDING_CONFIGS:
        deployment_name = resolve_deployment_name(config)
        records = cache.load(deployment_name)
        if not records:
            print(
                f"No cached embeddings found for model {config.display_name} (deployment '{deployment_name}'). "
                "Run generate_embeddings.py first."
            )
            continue

        try:
            metrics = evaluate_with_cache(config.display_name, deployment_name, cache, records, variations)
        except (KeyError, ValueError) as exc:
            print(f"Skipping {config.display_name}: {exc}")
            continue

        display_results(config.display_name, metrics)
        any_success = True

    if not any_success:
        print("No models evaluated. Ensure embeddings are generated and cached before running this script.")


if __name__ == "__main__":
    main()
