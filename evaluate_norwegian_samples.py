"""Evaluate Norwegian medical samples using cached embeddings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from medical_embedding_eval import (
    DEFAULT_AZURE_EMBEDDING_CONFIGS,
    DEFAULT_GEMINI_EMBEDDING_CONFIGS,
    DEFAULT_NVIDIA_EMBEDDING_CONFIGS,
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
    resolve_nvidia_model_name,
    resolve_nvidia_cache_key,
)
from medical_embedding_eval.embedding_cache import EmbeddingCache

load_dotenv()

DATA_DIR = Path("data")
CACHE_DIR = Path("data/embeddings")

try:  # pragma: no cover - optional dependency
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover - optional dependency
    colorama_init = None
    COLOR_ENABLED = False
    GREEN = CYAN = YELLOW = RED = RESET = ""
else:  # pragma: no cover - optional dependency
    colorama_init(autoreset=False)
    COLOR_ENABLED = True
    GREEN = Fore.GREEN
    CYAN = Fore.CYAN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    RESET = Style.RESET_ALL


def colorize(text: str, color: str) -> str:
    if COLOR_ENABLED and color:
        return f"{color}{text}{RESET}"
    return text


def pad_text(display: str, plain: str, width: int, *, align_right: bool) -> str:
    padding = max(width - len(plain), 0)
    if align_right:
        return " " * padding + display
    return display + " " * padding


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
            print(f"    Original:  {result.original_text}")
            print(f"    Variation: {result.variation_text}")
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
) -> Tuple[EvaluationMetrics, BenchmarkMetrics, Dict[str, Optional[float]]]:
    results = []
    dataset_results: Dict[str, List[SimilarityResult]] = {}
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

        dataset_id = variation.metadata.get("dataset_id") if variation.metadata else None
        if not dataset_id:
            dataset_id = sample.metadata.get("dataset_id")
        dataset_display = variation.metadata.get("dataset_display_name") if variation.metadata else None
        if not dataset_display:
            dataset_display = sample.metadata.get("dataset_display_name")

        similarity = SimilarityMetrics.cosine_similarity(
            sample_record.embedding,
            variation_record.embedding,
        )

        dataset_name = dataset_display or dataset_id or "Uncategorized Dataset"

        result = SimilarityResult(
            original_text=sample_record.text,
            variation_text=variation_record.text,
            cosine_similarity=similarity,
            sample_id=sample.sample_id,
            variation_id=variation.variation_id,
            variation_type=variation.variation_type,
            human_label=variation.similarity_label,
            dataset_id=dataset_id,
            dataset_display_name=dataset_display,
        )

        results.append(result)
        dataset_results.setdefault(dataset_name, []).append(result)

    metrics = SimilarityMetrics.compute_evaluation_metrics(results, model_name=model_name)
    benchmark = SimilarityMetrics.compute_benchmark_metrics(results)
    dataset_mrr: Dict[str, Optional[float]] = {}
    for dataset_name, dataset_res in dataset_results.items():
        dataset_benchmark = SimilarityMetrics.compute_benchmark_metrics(dataset_res)
        value = dataset_benchmark.mean_reciprocal_rank if dataset_benchmark.evaluated_queries else None
        dataset_mrr[dataset_name] = value
    return metrics, benchmark, dataset_mrr


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
    summary: List[Tuple[str, EvaluationMetrics, BenchmarkMetrics, Dict[str, Optional[float]]]] = []
    dataset_comparison: Dict[str, Dict[str, Optional[float]]] = {}

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
            metrics, benchmark, dataset_mrr = evaluate_with_cache(
                config.display_name, deployment_name, cache, records, variations
            )
        except (KeyError, ValueError) as exc:
            print(f"Skipping {config.display_name}: {exc}")
            continue

        display_results(config.display_name, metrics, benchmark)
        summary.append((config.display_name, metrics, benchmark, dataset_mrr))
        for dataset_name, value in dataset_mrr.items():
            dataset_comparison.setdefault(dataset_name, {})[config.display_name] = value
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
            metrics, benchmark, dataset_mrr = evaluate_with_cache(
                config.display_name, cache_key, cache, records, variations
            )
        except (KeyError, ValueError) as exc:
            print(f"Skipping {config.display_name}: {exc}")
            continue

        display_results(config.display_name, metrics, benchmark)
        summary.append((config.display_name, metrics, benchmark, dataset_mrr))
        for dataset_name, value in dataset_mrr.items():
            dataset_comparison.setdefault(dataset_name, {})[config.display_name] = value
        any_success = True

    for config in DEFAULT_NVIDIA_EMBEDDING_CONFIGS:
        model_name = resolve_nvidia_model_name(config)
        cache_key = resolve_nvidia_cache_key(config)
        records = cache.load(cache_key)
        if not records:
            print(
                f"No cached embeddings found for model {config.display_name} (NVIDIA '{model_name}'). "
                "Run generate_embeddings.py first."
            )
            continue

        try:
            metrics, benchmark, dataset_mrr = evaluate_with_cache(
                config.display_name, cache_key, cache, records, variations
            )
        except (KeyError, ValueError) as exc:
            print(f"Skipping {config.display_name}: {exc}")
            continue

        display_results(config.display_name, metrics, benchmark)
        summary.append((config.display_name, metrics, benchmark, dataset_mrr))
        for dataset_name, value in dataset_mrr.items():
            dataset_comparison.setdefault(dataset_name, {})[config.display_name] = value
        any_success = True

    if not any_success:
        print("No models evaluated. Ensure embeddings are generated and cached before running this script.")
        return

    summary_data: List[Dict[str, object]] = []
    for model_name, metrics, benchmark, dataset_mrr in summary:
        precision_lookup = benchmark.precision_at_k_by_label or {}

        def lookup_precision(label: str, k: int) -> Optional[float]:
            return precision_lookup.get(label, {}).get(k)

        values: Dict[str, Optional[float]] = {
            "MRR": benchmark.mean_reciprocal_rank if benchmark.evaluated_queries else None,
            "Recall@1": benchmark.recall_at_k.get(1) if benchmark.evaluated_queries else None,
            "Recall@3": benchmark.recall_at_k.get(3) if benchmark.evaluated_queries else None,
            "nDCG": benchmark.ndcg if benchmark.evaluated_queries else None,
            "AvgPrec": benchmark.average_precision if benchmark.evaluated_queries else None,
            "Prec@1 Pos": lookup_precision("positive", 1),
            "Prec@1 Rel": lookup_precision("related", 1),
            "Prec@1 Neg": lookup_precision("negative", 1),
            "Prec@3 Pos": lookup_precision("positive", 3),
            "Prec@3 Rel": lookup_precision("related", 3),
            "Prec@3 Neg": lookup_precision("negative", 3),
            "Mean Cosine": metrics.mean_similarity,
            "Mean Cos Pos": metrics.similarity_by_label.get("positive"),
            "Mean Cos Rel": metrics.similarity_by_label.get("related"),
            "Mean Cos Neg": metrics.similarity_by_label.get("negative"),
            "Pearson": benchmark.pearson,
            "Spearman": benchmark.spearman,
        }

        summary_data.append(
            {
                "model": model_name,
                "metrics": metrics,
                "benchmark": benchmark,
                "values": values,
            }
        )

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

    column_preferences = {
        "MRR": True,
        "Recall@1": True,
        "Recall@3": True,
        "nDCG": True,
        "AvgPrec": True,
        "Prec@1 Pos": True,
        "Prec@1 Rel": True,
        "Prec@1 Neg": False,
        "Prec@3 Pos": True,
        "Prec@3 Rel": True,
        "Prec@3 Neg": False,
        "Mean Cosine": True,
        "Mean Cos Pos": True,
        "Mean Cos Rel": True,
        "Mean Cos Neg": False,
        "Pearson": True,
        "Spearman": True,
    }

    header_display: Dict[str, str] = {}
    for header in headers:
        if header in column_preferences:
            header_display[header] = f"{header} {'↑' if column_preferences[header] else '↓'}"
        else:
            header_display[header] = header

    best_values: Dict[str, Optional[float]] = {}
    worst_values: Dict[str, Optional[float]] = {}
    for column, higher_is_better in column_preferences.items():
        values = [row["values"].get(column) for row in summary_data if row["values"].get(column) is not None]
        if values:
            if higher_is_better:
                best_values[column] = max(values)
                worst_values[column] = min(values)
            else:
                best_values[column] = min(values)
                worst_values[column] = max(values)
        else:
            best_values[column] = None
            worst_values[column] = None

    tolerance = 1e-9

    def format_metric_cell(column: str, value: Optional[float]) -> Tuple[str, str]:
        if value is None:
            plain_text = "N/A"
            text = plain_text
            best = best_values.get(column)
            worst = worst_values.get(column)
            highlight_best = False
            highlight_worst = False
        else:
            plain_text = f"{value:.4f}"
            text = plain_text
            best = best_values.get(column)
            worst = worst_values.get(column)
            highlight_best = best is not None and math.isclose(value, best, rel_tol=tolerance, abs_tol=tolerance)
            highlight_worst = (
                worst is not None
                and best is not None
                and not math.isclose(best, worst, rel_tol=tolerance, abs_tol=tolerance)
                and math.isclose(value, worst, rel_tol=tolerance, abs_tol=tolerance)
            )
        best = best_values.get(column)
        worst = worst_values.get(column)
        if highlight_best:
            text = colorize(text, GREEN)
        if highlight_worst:
            text = colorize(text, RED)
        return plain_text, text

    column_widths: Dict[str, int] = {header: len(header_display[header]) for header in headers}
    table_rows: List[Tuple[Dict[str, str], Dict[str, str]]] = []

    for row in summary_data:
        model_name = row["model"]
        values = row["values"]
        plain_row: Dict[str, str] = {"Model": model_name}
        display_row: Dict[str, str] = {"Model": model_name}
        column_widths["Model"] = max(column_widths["Model"], len(model_name))
        for header in headers[1:]:
            plain_value, display_value = format_metric_cell(header, values.get(header))
            plain_row[header] = plain_value
            display_row[header] = display_value
            column_widths[header] = max(column_widths[header], len(plain_value))
        table_rows.append((plain_row, display_row))

    header_cells = []
    for idx, header in enumerate(headers):
        align_right = idx != 0
        display_label = header_display[header]
        header_cells.append(pad_text(display_label, display_label, column_widths[header], align_right=align_right))
    print(" | ".join(header_cells))
    print("-" * sum(column_widths[header] + (3 if i < len(headers) - 1 else 0) for i, header in enumerate(headers)))

    for plain_row, display_row in table_rows:
        cells = []
        for idx, header in enumerate(headers):
            align_right = idx != 0
            cells.append(
                pad_text(
                    display_row[header],
                    plain_row[header],
                    column_widths[header],
                    align_right=align_right,
                )
            )
        print(" | ".join(cells))

    if dataset_comparison:
        print()
        print("=" * 80)
        print("DATASET MRR SUMMARY")
        print("=" * 80)
        model_order = [row["model"] for row in summary_data]
        datasets = sorted(dataset_comparison.keys())
        header = ["Model"] + datasets
        model_row_widths: Dict[str, int] = {col: len(col) for col in header}

        dataset_best: Dict[str, Optional[float]] = {}
        dataset_worst: Dict[str, Optional[float]] = {}
        for dataset_name in datasets:
            values = [dataset_comparison[dataset_name].get(model) for model in model_order if dataset_comparison[dataset_name].get(model) is not None]
            if values:
                dataset_best[dataset_name] = max(values)
                dataset_worst[dataset_name] = min(values)
            else:
                dataset_best[dataset_name] = None
                dataset_worst[dataset_name] = None

        table_rows: List[Tuple[Dict[str, str], Dict[str, str]]] = []
        for model_name in model_order:
            plain_row = {"Model": model_name}
            display_row = {"Model": model_name}
            model_row_widths["Model"] = max(model_row_widths["Model"], len(model_name))
            for dataset_name in datasets:
                value = dataset_comparison.get(dataset_name, {}).get(model_name)
                if value is None:
                    plain = "N/A"
                    display = plain
                else:
                    plain = f"{value:.4f}"
                    display = plain
                    best = dataset_best.get(dataset_name)
                    worst = dataset_worst.get(dataset_name)
                    if best is not None and math.isclose(value, best, rel_tol=tolerance, abs_tol=tolerance):
                        display = colorize(display, GREEN)
                    elif (
                        best is not None
                        and worst is not None
                        and not math.isclose(best, worst, rel_tol=tolerance, abs_tol=tolerance)
                        and math.isclose(value, worst, rel_tol=tolerance, abs_tol=tolerance)
                    ):
                        display = colorize(display, RED)
                plain_row[dataset_name] = plain
                display_row[dataset_name] = display
                model_row_widths[dataset_name] = max(model_row_widths.get(dataset_name, len(dataset_name)), len(plain))
            table_rows.append((plain_row, display_row))

        header_cells = []
        for idx, col in enumerate(header):
            align_right = idx != 0
            label = colorize(col, CYAN) if idx != 0 and COLOR_ENABLED else col
            header_cells.append(pad_text(label, col, model_row_widths[col], align_right=align_right))
        print(" | ".join(header_cells))
        print("-" * sum(model_row_widths[col] + (3 if i < len(header) - 1 else 0) for i, col in enumerate(header)))

        for plain_row, display_row in table_rows:
            cells = []
            for idx, col in enumerate(header):
                align_right = idx != 0
                cells.append(
                    pad_text(
                        display_row[col],
                        plain_row[col],
                        model_row_widths[col],
                        align_right=align_right,
                    )
                )
            print(" | ".join(cells))


if __name__ == "__main__":
    main()
