"""Evaluate Norwegian medical samples using cached embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from medical_embedding_eval import (
    DEFAULT_AZURE_EMBEDDING_CONFIGS,
    DEFAULT_GEMINI_EMBEDDING_CONFIGS,
    BenchmarkMetrics,
    EvaluationMetrics,
    SimilarityMetrics,
    SimilarityResult,
    CachedEmbedding,
    compute_text_hash,
    load_samples_from_directory,
    resolve_deployment_name,
    resolve_gemini_model_name,
    resolve_gemini_cache_key,
)
from medical_embedding_eval.embedding_cache import EmbeddingCache

load_dotenv()

DATA_DIR = Path("data")
CACHE_DIR = Path("data/embeddings")


def display_results(model_name: str, metrics: EvaluationMetrics, benchmark: BenchmarkMetrics) -> None:
    print("=" * 80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 80)
    print()

    print("=" * 80)
    print(f"RANKING METRICS: {model_name}")
    print("=" * 80)
    print()
    if benchmark.evaluated_queries:
        print(f"Queries with positives: {benchmark.evaluated_queries}/{benchmark.total_queries}")
        print(f"Mean Reciprocal Rank: {benchmark.mean_reciprocal_rank:.4f}")
        for k, value in sorted(benchmark.recall_at_k.items()):
            print(f"Recall@{k}: {value:.4f}")
        print(f"nDCG: {benchmark.ndcg:.4f}")
        print(f"Average Precision: {benchmark.average_precision:.4f}")
        if benchmark.precision_at_k_by_label:
            print("Precision@k by label:")
            for label in ("positive", "related", "negative"):
                values = benchmark.precision_at_k_by_label.get(label)
                if not values:
                    continue
                formatted = ", ".join(
                    f"@{k}: {values[k]:.4f}" for k in sorted(values)
                )
                print(f"  {label.title()}: {formatted}")
    else:
        print("No positive labels available for ranking metrics.")

    print()
    print("Correlation with human labels:")
    if benchmark.pearson is not None:
        print(f"  Pearson:  {benchmark.pearson:.4f}")
    else:
        print("  Pearson:  N/A")
    if benchmark.spearman is not None:
        print(f"  Spearman: {benchmark.spearman:.4f}")
    else:
        print("  Spearman: N/A")
    print()
    print(metrics)
    print()

    if metrics.similarity_by_label:
        print("=" * 80)
        print(f"MEAN COSINE BY LABEL: {model_name}")
        print("=" * 80)
        print()
        for label in ("positive", "related", "negative"):
            value = metrics.similarity_by_label.get(label)
            if value is not None:
                print(f"{label.title()}: {value:.4f}")
        remaining = sorted(
            key for key in metrics.similarity_by_label
            if key not in {"positive", "related", "negative"}
        )
        for label in remaining:
            print(f"{label.title()}: {metrics.similarity_by_label[label]:.4f}")
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
) -> Tuple[EvaluationMetrics, BenchmarkMetrics]:
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
                human_label=variation.similarity_label,
            )
        )

    metrics = SimilarityMetrics.compute_evaluation_metrics(results, model_name=model_name)
    benchmark = SimilarityMetrics.compute_benchmark_metrics(results)
    return metrics, benchmark


def main() -> None:
    print("=" * 80)
    print("Norwegian Medical Embedding Evaluation")
    print("=" * 80)
    print()

    data_dir = DATA_DIR.resolve()
    print(f"Scanning {data_dir} for JSON sample definitions...")
    samples, variations = load_samples_from_directory(data_dir)
    print(f"Loaded {len(samples)} samples and {len(variations)} variations")
    print()

    cache = EmbeddingCache(CACHE_DIR)
    any_success = False
    summary: List[Tuple[str, EvaluationMetrics, BenchmarkMetrics]] = []

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
            metrics, benchmark = evaluate_with_cache(config.display_name, deployment_name, cache, records, variations)
        except (KeyError, ValueError) as exc:
            print(f"Skipping {config.display_name}: {exc}")
            continue

        display_results(config.display_name, metrics, benchmark)
        summary.append((config.display_name, metrics, benchmark))
        any_success = True

    for config in DEFAULT_GEMINI_EMBEDDING_CONFIGS:
        model_name = resolve_gemini_model_name(config)
        cache_key = resolve_gemini_cache_key(config)
        records = cache.load(cache_key)
        if not records:
            print(
                f"No cached embeddings found for model {config.display_name} (Gemini '{model_name}'). "
                "Run generate_embeddings.py first."
            )
            continue

        try:
            metrics, benchmark = evaluate_with_cache(config.display_name, cache_key, cache, records, variations)
        except (KeyError, ValueError) as exc:
            print(f"Skipping {config.display_name}: {exc}")
            continue

        display_results(config.display_name, metrics, benchmark)
        summary.append((config.display_name, metrics, benchmark))
        any_success = True

    if not any_success:
        print("No models evaluated. Ensure embeddings are generated and cached before running this script.")
        return

    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    headers = [
        "Model",
        "MRR",
        "Recall@1",
        "Recall@3",
        "nDCG",
        "AvgPrec",
        "Prec@1 Pos",
        "Prec@1 Rel",
        "Prec@1 Neg",
        "Prec@3 Pos",
        "Prec@3 Rel",
        "Prec@3 Neg",
        "Mean Cosine",
        "Mean Cos Pos",
        "Mean Cos Rel",
        "Mean Cos Neg",
        "Pearson",
        "Spearman",
    ]
    print(" | ".join(headers))
    print("-" * 80)

    for model_name, metrics, benchmark in summary:
        mrr = f"{benchmark.mean_reciprocal_rank:.4f}" if benchmark.evaluated_queries else "N/A"
        recall1 = f"{benchmark.recall_at_k.get(1, 0.0):.4f}" if benchmark.evaluated_queries else "N/A"
        recall3 = f"{benchmark.recall_at_k.get(3, 0.0):.4f}" if benchmark.evaluated_queries else "N/A"
        ndcg = f"{benchmark.ndcg:.4f}" if benchmark.evaluated_queries else "N/A"
        avg_prec = f"{benchmark.average_precision:.4f}" if benchmark.evaluated_queries else "N/A"
        pearson = f"{benchmark.pearson:.4f}" if benchmark.pearson is not None else "N/A"
        spearman = f"{benchmark.spearman:.4f}" if benchmark.spearman is not None else "N/A"
        precision_lookup = benchmark.precision_at_k_by_label or {}

        def fmt_precision(label: str, k: int) -> str:
            value = precision_lookup.get(label, {}).get(k)
            return f"{value:.4f}" if value is not None else "N/A"

        mean_cos = f"{metrics.mean_similarity:.4f}"
        mean_pos = metrics.similarity_by_label.get("positive")
        mean_rel = metrics.similarity_by_label.get("related")
        mean_neg = metrics.similarity_by_label.get("negative")
        mean_pos_str = f"{mean_pos:.4f}" if mean_pos is not None else "N/A"
        mean_rel_str = f"{mean_rel:.4f}" if mean_rel is not None else "N/A"
        mean_neg_str = f"{mean_neg:.4f}" if mean_neg is not None else "N/A"

        row = [
            model_name,
            mrr,
            recall1,
            recall3,
            ndcg,
            avg_prec,
            fmt_precision("positive", 1),
            fmt_precision("related", 1),
            fmt_precision("negative", 1),
            fmt_precision("positive", 3),
            fmt_precision("related", 3),
            fmt_precision("negative", 3),
            mean_cos,
            mean_pos_str,
            mean_rel_str,
            mean_neg_str,
            pearson,
            spearman,
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
