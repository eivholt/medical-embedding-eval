"""Shared configuration helpers for embedding model deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class AzureEmbeddingConfig:
    """Configuration for a single Azure OpenAI embedding deployment."""

    display_name: str
    deployment_env_var: str
    default_deployment: str
    embedding_dim: int


DEFAULT_AZURE_EMBEDDING_CONFIGS: List[AzureEmbeddingConfig] = [
    AzureEmbeddingConfig(
        display_name="text-embedding-3-large",
        deployment_env_var="AZURE_OPENAI_EMBEDDING_3_LARGE_DEPLOYMENT",
        default_deployment="text-embedding-3-large",
        embedding_dim=3072,
    ),
    AzureEmbeddingConfig(
        display_name="text-embedding-3-small",
        deployment_env_var="AZURE_OPENAI_EMBEDDING_3_SMALL_DEPLOYMENT",
        default_deployment="text-embedding-3-small",
        embedding_dim=1536,
    ),
    AzureEmbeddingConfig(
        display_name="text-embedding-ada-002",
        deployment_env_var="AZURE_OPENAI_EMBEDDING_ADA_DEPLOYMENT",
        default_deployment="text-embedding-ada-002",
        embedding_dim=1536,
    ),
]


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    """Configuration for a Gemini embedding model."""

    display_name: str
    model_env_var: str
    default_model: str
    task_type_env_var: str
    default_task_type: str
    embedding_dim: int
    api_key_env_var: str = "GEMINI_API_KEY"


DEFAULT_GEMINI_EMBEDDING_CONFIGS: List[GeminiEmbeddingConfig] = [
    GeminiEmbeddingConfig(
        display_name="gemini-embedding-001",
        model_env_var="GEMINI_EMBEDDING_MODEL",
        default_model="models/gemini-embedding-001",
        task_type_env_var="GEMINI_EMBEDDING_TASK_TYPE",
        default_task_type="retrieval_document",
        embedding_dim=768,
    )
]


def iter_deployments(configs: Iterable[AzureEmbeddingConfig]) -> Iterable[AzureEmbeddingConfig]:
    """Yield configurations as-is; helper kept for future filtering."""
    return configs


def resolve_deployment_name(config: AzureEmbeddingConfig) -> str:
    """Return the deployment name to use for a given configuration."""
    return os.getenv(config.deployment_env_var, config.default_deployment)


def resolve_gemini_model_name(config: GeminiEmbeddingConfig) -> str:
    """Return the Gemini model identifier for a given configuration."""
    return os.getenv(config.model_env_var, config.default_model)


def resolve_gemini_task_type(config: GeminiEmbeddingConfig) -> str:
    """Return the task type to use for Gemini embeddings."""
    return os.getenv(config.task_type_env_var, config.default_task_type)


def resolve_gemini_api_key(config: GeminiEmbeddingConfig) -> Optional[str]:
    """Return the API key for Gemini embeddings, if configured."""
    return os.getenv(config.api_key_env_var)
