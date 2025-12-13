"""Shared configuration helpers for Azure OpenAI embedding deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List


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


def iter_deployments(configs: Iterable[AzureEmbeddingConfig]) -> Iterable[AzureEmbeddingConfig]:
    """Yield configurations as-is; helper kept for future filtering."""
    return configs


def resolve_deployment_name(config: AzureEmbeddingConfig) -> str:
    """Return the deployment name to use for a given configuration."""
    return os.getenv(config.deployment_env_var, config.default_deployment)
