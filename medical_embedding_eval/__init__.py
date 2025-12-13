"""Medical Embedding Evaluation Framework.

A framework for evaluating embedding models on medical records by comparing
semantic similarity of original samples with their variations.
"""

from .sample import MedicalSample, SampleVariation, SamplePair
from .embedder import EmbeddingModel, DummyEmbedder, AzureOpenAIEmbedder, GeminiEmbedder
from .embedding_cache import EmbeddingCache, CachedEmbedding, compute_text_hash
from .data_loader import load_samples_from_json, load_samples_from_directory
from .model_config import (
    AzureEmbeddingConfig,
    DEFAULT_AZURE_EMBEDDING_CONFIGS,
    GeminiEmbeddingConfig,
    DEFAULT_GEMINI_EMBEDDING_CONFIGS,
    resolve_deployment_name,
    resolve_gemini_model_name,
    resolve_gemini_task_type,
    resolve_gemini_api_key,
)
from .evaluator import EmbeddingEvaluator
from .metrics import (
    SimilarityMetrics,
    SimilarityResult,
    EvaluationMetrics,
    BenchmarkMetrics,
)

__version__ = "0.1.0"

__all__ = [
    "MedicalSample",
    "SampleVariation",
    "SamplePair",
    "EmbeddingModel",
    "DummyEmbedder",
    "AzureOpenAIEmbedder",
    "GeminiEmbedder",
    "EmbeddingCache",
    "CachedEmbedding",
    "compute_text_hash",
    "EmbeddingEvaluator",
    "SimilarityMetrics",
    "SimilarityResult",
    "EvaluationMetrics",
    "BenchmarkMetrics",
    "load_samples_from_json",
    "load_samples_from_directory",
    "AzureEmbeddingConfig",
    "DEFAULT_AZURE_EMBEDDING_CONFIGS",
    "GeminiEmbeddingConfig",
    "DEFAULT_GEMINI_EMBEDDING_CONFIGS",
    "resolve_deployment_name",
    "resolve_gemini_model_name",
    "resolve_gemini_task_type",
    "resolve_gemini_api_key",
]
