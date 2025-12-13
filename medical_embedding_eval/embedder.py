"""Abstract interface for embedding models."""

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from openai import AzureOpenAI


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.
    
    This interface allows plugging in different embedding models
    (e.g., Sentence-BERT, OpenAI, custom models) for evaluation.
    """
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim) containing embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors.
        
        Returns:
            Integer representing the embedding dimension
        """
        pass
    
    def get_model_name(self) -> str:
        """Get a human-readable name for the model.
        
        Returns:
            String name of the model
        """
        return self.__class__.__name__

    def get_cache_key(self) -> str:
        """Return identifier used for embedding cache segregation."""
        return self.get_model_name()


class DummyEmbedder(EmbeddingModel):
    """Simple dummy embedder for testing purposes.
    
    Generates random embeddings of specified dimension.
    Useful for framework testing without requiring actual embedding models.
    """
    
    def __init__(self, embedding_dim: int = 384, seed: int = 42):
        """Initialize dummy embedder.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate random embeddings.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate random embeddings and normalize them
        embeddings = self._rng.randn(len(texts), self.embedding_dim)
        # Normalize to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def get_model_name(self) -> str:
        """Get model name.
        
        Returns:
            Model name string
        """
        return f"DummyEmbedder(dim={self.embedding_dim})"


class AzureOpenAIEmbedder(EmbeddingModel):
    """Embedding model backed by an Azure OpenAI deployment."""

    def __init__(
        self,
        deployment_name: str,
        embedding_dim: int,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional["AzureOpenAI"] = None,
    ) -> None:
        if client is None:
            endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

            if not endpoint or not api_key:
                raise ValueError("Azure OpenAI endpoint and API key must be provided")

            try:
                from openai import AzureOpenAI  # Imported lazily to avoid hard dependency by default
            except ImportError as exc:  # pragma: no cover - import error is configuration specific
                raise ImportError(
                    "The openai package is required to use AzureOpenAIEmbedder. Install it via 'pip install openai'."
                ) from exc

            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )

        self._client = client
        self._deployment_name = deployment_name
        self.embedding_dim = embedding_dim
        self._model_name = model_name or deployment_name

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        response = self._client.embeddings.create(
            model=self._deployment_name,
            input=texts,
        )

        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    def get_model_name(self) -> str:
        return self._model_name

    def get_cache_key(self) -> str:
        return self._deployment_name
