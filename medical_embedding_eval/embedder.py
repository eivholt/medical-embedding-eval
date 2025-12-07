"""Abstract interface for embedding models."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


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
