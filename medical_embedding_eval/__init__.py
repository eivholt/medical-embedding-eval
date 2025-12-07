"""Medical Embedding Evaluation Framework.

A framework for evaluating embedding models on medical records by comparing
semantic similarity of original samples with their variations.
"""

from .sample import MedicalSample, SampleVariation, SamplePair
from .embedder import EmbeddingModel, DummyEmbedder
from .evaluator import EmbeddingEvaluator
from .metrics import SimilarityMetrics, SimilarityResult, EvaluationMetrics

__version__ = "0.1.0"

__all__ = [
    "MedicalSample",
    "SampleVariation",
    "SamplePair",
    "EmbeddingModel",
    "DummyEmbedder",
    "EmbeddingEvaluator",
    "SimilarityMetrics",
    "SimilarityResult",
    "EvaluationMetrics",
]
