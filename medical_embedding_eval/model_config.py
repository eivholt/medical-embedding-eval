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
        display_name="text-embedding-3-large 3072",
        deployment_env_var="AZURE_OPENAI_EMBEDDING_3_LARGE_DEPLOYMENT",
        default_deployment="text-embedding-3-large",
        embedding_dim=3072,
    ),
    AzureEmbeddingConfig(
        display_name="text-embedding-3-small 1536",
        deployment_env_var="AZURE_OPENAI_EMBEDDING_3_SMALL_DEPLOYMENT",
        default_deployment="text-embedding-3-small",
        embedding_dim=1536,
    ),
    AzureEmbeddingConfig(
        display_name="text-embedding-ada-002 1536",
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
    task_type_env_var: Optional[str]
    default_task_type: str
    embedding_dim: int
    output_dimensionality: Optional[int] = None
    api_key_env_var: str = "GEMINI_API_KEY"


DEFAULT_GEMINI_EMBEDDING_CONFIGS: List[GeminiEmbeddingConfig] = [
    GeminiEmbeddingConfig(
        display_name="gemini-embedding-001 768",
        model_env_var="GEMINI_EMBEDDING_MODEL",
        default_model="models/gemini-embedding-001",
        task_type_env_var="GEMINI_EMBEDDING_TASK_TYPE",
        default_task_type="retrieval_document",
        embedding_dim=768,
        output_dimensionality=768,
    ),
    GeminiEmbeddingConfig(
        display_name="gemini-embedding-001 3072,retrieval",
        model_env_var="GEMINI_EMBEDDING_MODEL",
        default_model="models/gemini-embedding-001",
        task_type_env_var="GEMINI_EMBEDDING_TASK_TYPE",
        default_task_type="retrieval_document",
        embedding_dim=3072,
        output_dimensionality=3072,
    ),
    GeminiEmbeddingConfig(
        display_name="gemini-embedding-001 3072,similarity",
        model_env_var="GEMINI_EMBEDDING_MODEL",
        default_model="models/gemini-embedding-001",
        task_type_env_var=None,
        default_task_type="semantic_similarity",
        embedding_dim=3072,
        output_dimensionality=3072,
    ),
]


@dataclass(frozen=True)
class NvidiaEmbeddingConfig:
    """Configuration for an NVIDIA embedding model."""

    display_name: str
    model_env_var: str
    default_model: str
    embedding_dim: int
    api_key_env_var: str = "NVIDIA_API_KEY"
    base_url_env_var: str = "NVIDIA_API_BASE_URL"
    default_base_url: str = "https://integrate.api.nvidia.com/v1"
    input_type: str = "query"
    input_type_env_var: Optional[str] = "NVIDIA_EMBED_INPUT_TYPE"
    truncate: str = "NONE"
    truncate_env_var: Optional[str] = "NVIDIA_EMBED_TRUNCATE"
    encoding_format: str = "float"
    encoding_format_env_var: Optional[str] = "NVIDIA_EMBED_ENCODING"


DEFAULT_NVIDIA_EMBEDDING_CONFIGS: List[NvidiaEmbeddingConfig] = [
    NvidiaEmbeddingConfig(
        display_name="nvidia/nv-embed-v1 1024",
        model_env_var="NVIDIA_EMBED_MODEL",
        default_model="nvidia/nv-embed-v1",
        embedding_dim=1024,
    ),
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
    if config.task_type_env_var:
        return os.getenv(config.task_type_env_var, config.default_task_type)
    return config.default_task_type


def resolve_gemini_api_key(config: GeminiEmbeddingConfig) -> Optional[str]:
    """Return the API key for Gemini embeddings, if configured."""
    return os.getenv(config.api_key_env_var)


def resolve_gemini_output_dimensionality(config: GeminiEmbeddingConfig) -> Optional[int]:
    """Return the requested output dimensionality for Gemini embeddings."""
    return config.output_dimensionality


def resolve_gemini_cache_key(config: GeminiEmbeddingConfig) -> str:
    """Return cache key combining model and output dimensionality."""
    model_name = resolve_gemini_model_name(config)
    task_type = resolve_gemini_task_type(config)
    suffix = config.output_dimensionality if config.output_dimensionality is not None else "default"
    safe_model = model_name.replace(":", "-")
    safe_task = task_type.replace(":", "-").replace("/", "-")
    return f"{safe_model}-{safe_task}-{suffix}"


def resolve_nvidia_model_name(config: NvidiaEmbeddingConfig) -> str:
    """Return the NVIDIA model identifier for a given configuration."""
    return os.getenv(config.model_env_var, config.default_model)


def resolve_nvidia_api_key(config: NvidiaEmbeddingConfig) -> Optional[str]:
    """Return the NVIDIA API key for embeddings, if configured."""
    return os.getenv(config.api_key_env_var)


def resolve_nvidia_base_url(config: NvidiaEmbeddingConfig) -> str:
    """Return the NVIDIA endpoint base URL."""
    return os.getenv(config.base_url_env_var, config.default_base_url)


def resolve_nvidia_input_type(config: NvidiaEmbeddingConfig) -> str:
    """Return the input_type parameter for the NVIDIA embedding request."""
    if config.input_type_env_var:
        return os.getenv(config.input_type_env_var, config.input_type)
    return config.input_type


def resolve_nvidia_truncate(config: NvidiaEmbeddingConfig) -> str:
    """Return the truncate parameter for the NVIDIA embedding request."""
    if config.truncate_env_var:
        return os.getenv(config.truncate_env_var, config.truncate)
    return config.truncate


def resolve_nvidia_encoding_format(config: NvidiaEmbeddingConfig) -> str:
    """Return the encoding_format parameter for the NVIDIA embedding request."""
    if config.encoding_format_env_var:
        return os.getenv(config.encoding_format_env_var, config.encoding_format)
    return config.encoding_format


def resolve_nvidia_cache_key(config: NvidiaEmbeddingConfig) -> str:
    """Return cache key combining model and request modifiers."""
    model_name = resolve_nvidia_model_name(config)
    input_type = resolve_nvidia_input_type(config)
    truncate = resolve_nvidia_truncate(config)
    encoding_format = resolve_nvidia_encoding_format(config)
    safe_model = model_name.replace("/", "-").replace(":", "-")
    safe_input = input_type.replace("/", "-")
    safe_truncate = truncate.replace("/", "-")
    safe_format = encoding_format.replace("/", "-")
    return f"{safe_model}-{safe_input}-{safe_truncate}-{safe_format}"
