"""Metrics for evaluating embedding similarity."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field


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
    """
    original_text: str
    variation_text: str
    cosine_similarity: float
    sample_id: str
    variation_id: str
    variation_type: str


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
        for result in results:
            if result.variation_type not in type_groups:
                type_groups[result.variation_type] = []
            type_groups[result.variation_type].append(result.cosine_similarity)
        
        # Compute mean for each type
        metrics.similarity_by_type = {
            vtype: float(np.mean(sims))
            for vtype, sims in type_groups.items()
        }
        
        return metrics
