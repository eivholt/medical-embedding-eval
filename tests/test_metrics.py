"""Tests for similarity metrics."""

import pytest
import numpy as np

from medical_embedding_eval.metrics import SimilarityMetrics, SimilarityResult


class TestSimilarityMetrics:
    """Tests for SimilarityMetrics utilities."""
    
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = SimilarityMetrics.cosine_similarity(vec, vec)
        assert pytest.approx(similarity, abs=1e-6) == 1.0
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = SimilarityMetrics.cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, abs=1e-6) == 0.0
    
    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = SimilarityMetrics.cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, abs=1e-6) == -1.0
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = SimilarityMetrics.cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_pairwise_cosine_similarity(self):
        """Test pairwise cosine similarity."""
        embeddings1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        embeddings2 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        similarities = SimilarityMetrics.pairwise_cosine_similarity(
            embeddings1, embeddings2
        )
        
        assert similarities.shape == (2, 3)
        # First row: [1, 0, 0] should be most similar to [1, 0, 0]
        assert pytest.approx(similarities[0, 0], abs=1e-6) == 1.0
        assert pytest.approx(similarities[0, 1], abs=1e-6) == 0.0
        # Second row: [0, 1, 0] should be most similar to [0, 1, 0]
        assert pytest.approx(similarities[1, 1], abs=1e-6) == 1.0
    
    def test_compute_evaluation_metrics(self):
        """Test computing evaluation metrics."""
        results = [
            SimilarityResult(
                original_text="Text 1",
                variation_text="Var 1",
                cosine_similarity=0.9,
                sample_id="s1",
                variation_id="v1",
                variation_type="paraphrase"
            ),
            SimilarityResult(
                original_text="Text 2",
                variation_text="Var 2",
                cosine_similarity=0.8,
                sample_id="s2",
                variation_id="v2",
                variation_type="paraphrase"
            ),
            SimilarityResult(
                original_text="Text 3",
                variation_text="Var 3",
                cosine_similarity=0.7,
                sample_id="s3",
                variation_id="v3",
                variation_type="synonym"
            ),
        ]
        
        metrics = SimilarityMetrics.compute_evaluation_metrics(
            results, model_name="TestModel"
        )
        
        assert metrics.model_name == "TestModel"
        assert pytest.approx(metrics.mean_similarity, abs=1e-6) == 0.8
        assert pytest.approx(metrics.median_similarity, abs=1e-6) == 0.8
        assert metrics.min_similarity == 0.7
        assert metrics.max_similarity == 0.9
        assert len(metrics.similarity_by_type) == 2
        assert "paraphrase" in metrics.similarity_by_type
        assert "synonym" in metrics.similarity_by_type
        assert pytest.approx(metrics.similarity_by_type["paraphrase"], abs=1e-6) == 0.85
        assert pytest.approx(metrics.similarity_by_type["synonym"], abs=1e-6) == 0.7
    
    def test_compute_evaluation_metrics_empty_raises_error(self):
        """Test that empty results raise error."""
        with pytest.raises(ValueError, match="Cannot compute metrics from empty results"):
            SimilarityMetrics.compute_evaluation_metrics([])
    
    def test_evaluation_metrics_str(self):
        """Test string representation of evaluation metrics."""
        results = [
            SimilarityResult(
                original_text="Text 1",
                variation_text="Var 1",
                cosine_similarity=0.9,
                sample_id="s1",
                variation_id="v1",
                variation_type="paraphrase"
            ),
        ]
        
        metrics = SimilarityMetrics.compute_evaluation_metrics(
            results, model_name="TestModel"
        )
        
        metrics_str = str(metrics)
        assert "TestModel" in metrics_str
        assert "Mean Cosine Similarity" in metrics_str
        assert "0.9000" in metrics_str
