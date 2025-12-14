"""Metrics for evaluating embedding similarity."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import math


@dataclass
class SimilarityResult:
    """Results from a single similarity comparison.
    
    Attributes:
        original_text: Original sample text
        variation_text: Variation sample text
        cosine_similarity: Cosine similarity score
        sample_id: ID of the original sample
        variation_id: ID of the variation
        variation_type: Type of variation applied
        human_label: Optional[float]: Human annotated similarity label
    """
    original_text: str
    variation_text: str
    cosine_similarity: float
    sample_id: str
    variation_id: str
    variation_type: str
    human_label: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from evaluation.
    
    Attributes:
        results: List of individual similarity results
        mean_similarity: Mean cosine similarity across all pairs
        median_similarity: Median cosine similarity
        std_similarity: Standard deviation of similarity scores
        min_similarity: Minimum similarity score
        max_similarity: Maximum similarity score
        similarity_by_type: Mean similarity grouped by variation type
        model_name: Name of the embedding model used
    """
    results: List[SimilarityResult]
    mean_similarity: float
    median_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    similarity_by_type: Dict[str, float] = field(default_factory=dict)
    similarity_by_label: Dict[str, float] = field(default_factory=dict)
    model_name: str = "Unknown"
    
    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Embedding Model: {self.model_name}",
            f"Number of Evaluations: {len(self.results)}",
            f"Mean Cosine Similarity: {self.mean_similarity:.4f}",
            f"Median Cosine Similarity: {self.median_similarity:.4f}",
            f"Std Dev: {self.std_similarity:.4f}",
            f"Range: [{self.min_similarity:.4f}, {self.max_similarity:.4f}]",
        ]
        
        if self.similarity_by_type:
            lines.append("\nSimilarity by Variation Type:")
            for vtype, sim in sorted(self.similarity_by_type.items()):
                lines.append(f"  {vtype}: {sim:.4f}")

        if self.similarity_by_label:
            lines.append("\nSimilarity by Label:")
            for label in ("positive", "related", "negative"):
                if label in self.similarity_by_label:
                    lines.append(f"  {label}: {self.similarity_by_label[label]:.4f}")
            remaining = sorted(
                (key for key in self.similarity_by_label if key not in {"positive", "related", "negative"})
            )
            for label in remaining:
                lines.append(f"  {label}: {self.similarity_by_label[label]:.4f}")
        
        return "\n".join(lines)


class SimilarityMetrics:
    """Utilities for computing similarity metrics."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score in range [-1, 1]
        """
        # Ensure vectors are 1D
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def pairwise_cosine_similarity(embeddings1: np.ndarray, 
                                   embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: Array of shape (n, dim)
            embeddings2: Array of shape (m, dim)
            
        Returns:
            Array of shape (n, m) with pairwise similarities
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        embeddings1_normalized = embeddings1 / (norm1 + 1e-8)
        embeddings2_normalized = embeddings2 / (norm2 + 1e-8)
        
        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(embeddings1_normalized, embeddings2_normalized.T)
        
        return similarities
    
    @staticmethod
    def compute_evaluation_metrics(results: List[SimilarityResult],
                                   model_name: str = "Unknown") -> EvaluationMetrics:
        """Compute aggregated metrics from similarity results.
        
        Args:
            results: List of similarity results
            model_name: Name of the embedding model
            
        Returns:
            EvaluationMetrics object with aggregated statistics
        """
        if not results:
            raise ValueError("Cannot compute metrics from empty results")
        
        similarities = np.array([r.cosine_similarity for r in results])
        
        # Compute basic statistics
        metrics = EvaluationMetrics(
            results=results,
            mean_similarity=float(np.mean(similarities)),
            median_similarity=float(np.median(similarities)),
            std_similarity=float(np.std(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            model_name=model_name,
        )
        
        # Group by variation type
        type_groups: Dict[str, List[float]] = {}
        label_groups: Dict[str, List[float]] = {}

        for result in results:
            if result.variation_type not in type_groups:
                type_groups[result.variation_type] = []
            type_groups[result.variation_type].append(result.cosine_similarity)

            if result.human_label is not None:
                bucket = SimilarityMetrics._label_bucket(result.human_label)
                label_groups.setdefault(bucket, []).append(result.cosine_similarity)
        
        # Compute mean for each type
        metrics.similarity_by_type = {
            vtype: float(np.mean(sims))
            for vtype, sims in type_groups.items()
        }

        metrics.similarity_by_label = {
            bucket: float(np.mean(sims))
            for bucket, sims in label_groups.items()
        }
        
        return metrics

    @staticmethod
    def compute_benchmark_metrics(
        results: List[SimilarityResult],
        positive_label: float = 1.0,
        k_values: Tuple[int, ...] = (1, 3, 5),
    ) -> "BenchmarkMetrics":
        if not results:
            raise ValueError("Cannot compute metrics from empty results")

        by_sample: Dict[str, List[SimilarityResult]] = {}
        for result in results:
            by_sample.setdefault(result.sample_id, []).append(result)

        total_queries = len(by_sample)
        positive_queries = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in k_values}
        ndcg_sum = 0.0
        average_precision_sum = 0.0

        for sample_results in by_sample.values():
            # sort descending by cosine similarity
            sorted_results = sorted(sample_results, key=lambda r: r.cosine_similarity, reverse=True)

            positive_indices = [idx for idx, res in enumerate(sorted_results) if res.human_label == positive_label]
            if not positive_indices:
                continue

            positive_queries += 1
            best_rank = positive_indices[0] + 1  # convert to 1-based
            mrr_sum += 1.0 / best_rank

            for k in k_values:
                if best_rank <= k:
                    recall_hits[k] += 1

            ndcg_sum += SimilarityMetrics._ndcg(sorted_results)
            average_precision_sum += SimilarityMetrics._average_precision(sorted_results, positive_label)

        recall_at_k = {
            k: (recall_hits[k] / positive_queries if positive_queries else 0.0)
            for k in k_values
        }

        labeled_results = [res for res in results if res.human_label is not None]
        if len(labeled_results) >= 2:
            similarities = np.array([res.cosine_similarity for res in labeled_results], dtype=float)
            labels = np.array([res.human_label for res in labeled_results], dtype=float)

            pearson = SimilarityMetrics._pearson_correlation(similarities, labels)
            spearman = SimilarityMetrics._spearman_correlation(similarities, labels)
        else:
            pearson = None
            spearman = None

        return BenchmarkMetrics(
            positive_label=positive_label,
            total_queries=total_queries,
            evaluated_queries=positive_queries,
            mean_reciprocal_rank=(mrr_sum / positive_queries if positive_queries else 0.0),
            recall_at_k=recall_at_k,
            pearson=pearson,
            spearman=spearman,
            ndcg=(ndcg_sum / positive_queries if positive_queries else 0.0),
            average_precision=(average_precision_sum / positive_queries if positive_queries else 0.0),
        )

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        if x.size < 2 or y.size < 2:
            return None
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        if x.size < 2 or y.size < 2:
            return None
        x_ranks = SimilarityMetrics._rankdata(x)
        y_ranks = SimilarityMetrics._rankdata(y)
        return SimilarityMetrics._pearson_correlation(x_ranks, y_ranks)

    @staticmethod
    def _rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        sorted_values = values[order]

        i = 0
        n = len(values)
        while i < n:
            j = i
            while (
                j + 1 < n
                and math.isclose(sorted_values[j + 1], sorted_values[i], rel_tol=1e-9, abs_tol=1e-12)
            ):
                j += 1
            average_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average_rank
            i = j + 1

        return ranks

    @staticmethod
    def _label_bucket(label: float) -> str:
        if math.isclose(label, 1.0, abs_tol=1e-6):
            return "positive"
        if math.isclose(label, 0.5, abs_tol=1e-6):
            return "related"
        if math.isclose(label, 0.0, abs_tol=1e-6):
            return "negative"
        if label >= 0.75:
            return "positive"
        if label >= 0.25:
            return "related"
        return "negative"

    @staticmethod
    def _gain(label: Optional[float]) -> float:
        if label is None:
            return 0.0
        return float(2**label - 1)

    @staticmethod
    def _ndcg(sorted_results: List[SimilarityResult]) -> float:
        gains = [SimilarityMetrics._gain(res.human_label) for res in sorted_results]
        dcg = 0.0
        for idx, gain in enumerate(gains, start=1):
            if gain == 0.0:
                continue
            dcg += gain / math.log2(idx + 1)

        ideal_gains = sorted(gains, reverse=True)
        idcg = 0.0
        for idx, gain in enumerate(ideal_gains, start=1):
            if gain == 0.0:
                continue
            idcg += gain / math.log2(idx + 1)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _average_precision(sorted_results: List[SimilarityResult], positive_label: float) -> float:
        hits = 0
        precision_sum = 0.0
        for idx, res in enumerate(sorted_results, start=1):
            if res.human_label == positive_label:
                hits += 1
                precision_sum += hits / idx
        if hits == 0:
            return 0.0
        return precision_sum / hits


@dataclass
class BenchmarkMetrics:
    positive_label: float
    total_queries: int
    evaluated_queries: int
    mean_reciprocal_rank: float
    recall_at_k: Dict[int, float]
    pearson: Optional[float]
    spearman: Optional[float]
    ndcg: float
    average_precision: float
