"""Core evaluation engine for embedding models."""

from typing import List, Optional
import numpy as np

from .sample import MedicalSample, SampleVariation, SamplePair
from .embedder import EmbeddingModel
from .metrics import SimilarityMetrics, SimilarityResult, EvaluationMetrics


class EmbeddingEvaluator:
    """Main evaluator for comparing embedding models on medical samples.
    
    This class orchestrates the evaluation process:
    1. Takes pairs of original samples and their variations
    2. Generates embeddings using the provided model
    3. Computes similarity metrics
    4. Aggregates and reports results
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize evaluator with an embedding model.
        
        Args:
            embedding_model: The embedding model to evaluate
        """
        self.embedding_model = embedding_model
        self.model_name = embedding_model.get_model_name()
    
    def evaluate_pair(self, 
                     original: MedicalSample, 
                     variation: SampleVariation) -> SimilarityResult:
        """Evaluate a single sample-variation pair.
        
        Args:
            original: Original medical sample
            variation: Semantic variation of the sample
            
        Returns:
            SimilarityResult with computed metrics
        """
        # Generate embeddings
        embeddings = self.embedding_model.embed([original.text, variation.variation_text])
        
        # Compute cosine similarity
        similarity = SimilarityMetrics.cosine_similarity(embeddings[0], embeddings[1])
        
        # Create result
        result = SimilarityResult(
            original_text=original.text,
            variation_text=variation.variation_text,
            cosine_similarity=similarity,
            sample_id=original.sample_id,
            variation_id=variation.variation_id,
            variation_type=variation.variation_type,
            human_label=variation.similarity_label,
        )
        
        return result
    
    def evaluate_pairs(self, pairs: List[SamplePair]) -> EvaluationMetrics:
        """Evaluate multiple sample-variation pairs.
        
        Args:
            pairs: List of SamplePair objects to evaluate
            
        Returns:
            EvaluationMetrics with aggregated results
        """
        if not pairs:
            raise ValueError("Cannot evaluate empty list of pairs")
        
        results = []
        for pair in pairs:
            result = self.evaluate_pair(pair.original, pair.variation)
            results.append(result)
        
        # Compute aggregated metrics
        metrics = SimilarityMetrics.compute_evaluation_metrics(
            results, 
            model_name=self.model_name
        )
        
        return metrics
    
    def evaluate_samples(self,
                        samples: List[MedicalSample],
                        variations: List[SampleVariation]) -> EvaluationMetrics:
        """Evaluate samples with their variations.
        
        Automatically matches variations to their original samples.
        
        Args:
            samples: List of original medical samples
            variations: List of variations
            
        Returns:
            EvaluationMetrics with aggregated results
        """
        # Create pairs by matching variations to originals
        pairs = []
        for variation in variations:
            # Find the matching original sample
            original = variation.original_sample
            if original in samples:
                pairs.append(SamplePair(original=original, variation=variation))
        
        if not pairs:
            raise ValueError("No matching sample-variation pairs found")
        
        return self.evaluate_pairs(pairs)
    
    def batch_evaluate(self,
                      originals: List[str],
                      variations: List[str]) -> np.ndarray:
        """Batch evaluate cosine similarities for text pairs.
        
        This is a simplified method for quick evaluations without
        creating full Sample objects.
        
        Args:
            originals: List of original texts
            variations: List of variation texts (must match length of originals)
            
        Returns:
            Array of cosine similarity scores
        """
        if len(originals) != len(variations):
            raise ValueError("originals and variations must have the same length")
        
        # Generate embeddings in batches
        original_embeddings = self.embedding_model.embed(originals)
        variation_embeddings = self.embedding_model.embed(variations)
        
        # Compute pairwise similarities (diagonal contains matching pairs)
        similarities = SimilarityMetrics.pairwise_cosine_similarity(
            original_embeddings,
            variation_embeddings
        )
        
        # Extract diagonal (similarities of matching pairs)
        pair_similarities = np.diagonal(similarities)
        
        return pair_similarities
    
    def compare_models(self,
                      other_evaluator: 'EmbeddingEvaluator',
                      pairs: List[SamplePair]) -> dict:
        """Compare this model with another model on the same pairs.
        
        Args:
            other_evaluator: Another EmbeddingEvaluator to compare with
            pairs: Sample pairs to evaluate on
            
        Returns:
            Dictionary with comparison results for both models
        """
        metrics1 = self.evaluate_pairs(pairs)
        metrics2 = other_evaluator.evaluate_pairs(pairs)
        
        return {
            'model1': {
                'name': self.model_name,
                'metrics': metrics1,
            },
            'model2': {
                'name': other_evaluator.model_name,
                'metrics': metrics2,
            },
            'difference': metrics1.mean_similarity - metrics2.mean_similarity,
        }
