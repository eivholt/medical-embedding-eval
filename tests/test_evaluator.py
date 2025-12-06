"""Tests for the embedding evaluator."""

import pytest
import numpy as np

from medical_embedding_eval.sample import MedicalSample, SampleVariation, SamplePair
from medical_embedding_eval.embedder import DummyEmbedder
from medical_embedding_eval.evaluator import EmbeddingEvaluator


class TestEmbeddingEvaluator:
    """Tests for EmbeddingEvaluator class."""
    
    @pytest.fixture
    def dummy_embedder(self):
        """Create a dummy embedder for testing."""
        return DummyEmbedder(embedding_dim=384, seed=42)
    
    @pytest.fixture
    def sample_pair(self):
        """Create a sample pair for testing."""
        original = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001"
        )
        variation = SampleVariation(
            original_sample=original,
            variation_text="Personen har forhøyet blodtrykksmåling",
            variation_type="paraphrase",
            variation_id="var_001"
        )
        return SamplePair(original=original, variation=variation)
    
    def test_evaluator_initialization(self, dummy_embedder):
        """Test evaluator initialization."""
        evaluator = EmbeddingEvaluator(dummy_embedder)
        assert evaluator.embedding_model == dummy_embedder
        assert evaluator.model_name == dummy_embedder.get_model_name()
    
    def test_evaluate_single_pair(self, dummy_embedder, sample_pair):
        """Test evaluating a single pair."""
        evaluator = EmbeddingEvaluator(dummy_embedder)
        result = evaluator.evaluate_pair(sample_pair.original, sample_pair.variation)
        
        assert result.sample_id == "sample_001"
        assert result.variation_id == "var_001"
        assert result.variation_type == "paraphrase"
        assert -1.0 <= result.cosine_similarity <= 1.0
    
    def test_evaluate_multiple_pairs(self, dummy_embedder):
        """Test evaluating multiple pairs."""
        evaluator = EmbeddingEvaluator(dummy_embedder)
        
        # Create multiple sample pairs
        pairs = []
        for i in range(3):
            original = MedicalSample(
                text=f"Sample text {i}",
                sample_id=f"sample_{i:03d}"
            )
            variation = SampleVariation(
                original_sample=original,
                variation_text=f"Varied text {i}",
                variation_type="paraphrase",
                variation_id=f"var_{i:03d}"
            )
            pairs.append(SamplePair(original=original, variation=variation))
        
        metrics = evaluator.evaluate_pairs(pairs)
        
        assert len(metrics.results) == 3
        assert metrics.mean_similarity is not None
        assert metrics.median_similarity is not None
        assert metrics.std_similarity is not None
        assert metrics.model_name == dummy_embedder.get_model_name()
    
    def test_evaluate_empty_pairs_raises_error(self, dummy_embedder):
        """Test that evaluating empty pairs raises error."""
        evaluator = EmbeddingEvaluator(dummy_embedder)
        with pytest.raises(ValueError, match="Cannot evaluate empty list"):
            evaluator.evaluate_pairs([])
    
    def test_batch_evaluate(self, dummy_embedder):
        """Test batch evaluation."""
        evaluator = EmbeddingEvaluator(dummy_embedder)
        
        originals = ["Text 1", "Text 2", "Text 3"]
        variations = ["Varied 1", "Varied 2", "Varied 3"]
        
        similarities = evaluator.batch_evaluate(originals, variations)
        
        assert len(similarities) == 3
        assert all(-1.0 <= s <= 1.0 for s in similarities)
    
    def test_batch_evaluate_mismatched_lengths_raises_error(self, dummy_embedder):
        """Test that mismatched lengths raise error."""
        evaluator = EmbeddingEvaluator(dummy_embedder)
        
        originals = ["Text 1", "Text 2"]
        variations = ["Varied 1"]
        
        with pytest.raises(ValueError, match="must have the same length"):
            evaluator.batch_evaluate(originals, variations)
    
    def test_compare_models(self, sample_pair):
        """Test comparing two models."""
        embedder1 = DummyEmbedder(embedding_dim=384, seed=42)
        embedder2 = DummyEmbedder(embedding_dim=384, seed=123)
        
        evaluator1 = EmbeddingEvaluator(embedder1)
        evaluator2 = EmbeddingEvaluator(embedder2)
        
        comparison = evaluator1.compare_models(evaluator2, [sample_pair])
        
        assert 'model1' in comparison
        assert 'model2' in comparison
        assert 'difference' in comparison
        assert comparison['model1']['name'] == embedder1.get_model_name()
        assert comparison['model2']['name'] == embedder2.get_model_name()
